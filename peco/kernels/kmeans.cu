#include <cuda_runtime.h>
#include <math_constants.h>
#include <cublas_v2.h>
#include <cstdint>
#include <map>
#include <memory>

namespace {
constexpr int THREADS = 256;

// Reduce squared norms for contiguous FP32 centers [B,C,D] into norms [B,C].
__global__ void compute_center_norms(const float* centers, float* norms,
                                     int clusters, int dims, int batch) {
    int cluster = blockIdx.x;
    if (cluster >= batch * clusters) return;
    __shared__ float partial[THREADS];
    float sum = 0.f;
    for (int d = threadIdx.x; d < dims; d += blockDim.x) {
        float v = centers[static_cast<int64_t>(cluster) * dims + d];
        sum += v * v;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = THREADS / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
        __syncthreads();
    }
    if (!threadIdx.x) norms[cluster] = partial[0];
}

// Select one center per row from -2*X*C^T [rows,C] plus squared center norms.
// Token norms do not affect the argmin; equal distances choose the lowest ID.
__global__ void select_nearest_centers(const float* dot, const float* center_norm,
                               int32_t* labels, int rows, int clusters) {
    int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float distances[THREADS];
    __shared__ int ids[THREADS];
    float best = CUDART_INF_F;
    int best_id = 0;
    for (int c = threadIdx.x; c < clusters; c += THREADS) {
        float distance = dot[static_cast<int64_t>(row) * clusters + c] + center_norm[c];
        if (distance < best || (distance == best && c < best_id)) {
            best = distance;
            best_id = c;
        }
    }
    distances[threadIdx.x] = best;
    ids[threadIdx.x] = best_id;
    __syncthreads();
    for (int offset = THREADS / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset) {
            float other = distances[threadIdx.x + offset];
            int other_id = ids[threadIdx.x + offset];
            if (other < distances[threadIdx.x] ||
                (other == distances[threadIdx.x] && other_id < ids[threadIdx.x])) {
                distances[threadIdx.x] = other;
                ids[threadIdx.x] = other_id;
            }
        }
        __syncthreads();
    }
    if (!threadIdx.x) labels[row] = ids[0];
}

// Accumulate features [B,N,D] into sums [B,C,D], counting each token once.
__global__ void accumulate_cluster_statistics(const float* x, const int32_t* labels,
                                  float* sums, int32_t* counts,
                                  int tokens, int dims, int clusters, int batch) {
    int64_t total = static_cast<int64_t>(batch) * tokens * dims;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int d = index % dims;
        int64_t row = index / dims;
        int b = row / tokens;
        int c = labels[row];
        atomicAdd(sums + (static_cast<int64_t>(b) * clusters + c) * dims + d, x[index]);
        if (!d) atomicAdd(counts + static_cast<int64_t>(b) * clusters + c, 1);
    }
}

// Normalize each cluster sum; an empty cluster keeps its previous center.
__global__ void normalize_cluster_sums(const float* sums, const int32_t* counts,
                            const float* previous, float* means, int64_t total, int dims) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int count = counts[index / dims];
        means[index] = count ? sums[index] / count : previous[index];
    }
}

template <typename T, bool INVERSE>
// Copy feature bits using original token IDs [B*H,N], without dtype conversion.
// Forward gathers into sorted order; inverse scatters back to original order.
__global__ void remap_token_features(const T* input, const int32_t* indices, T* output,
                                   int tokens, int dims, int64_t total) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int d = i % dims;
        int64_t row = i / dims;
        int64_t head = row / tokens;
        int token = indices[row];
        int64_t mapped = (head * tokens + token) * dims + d;
        if (INVERSE) output[mapped] = input[i];
        else output[i] = input[mapped];
    }
}

// Bound the grid while grid-stride loops cover all requested feature elements.
int launch_block_count(int64_t n) {
    return static_cast<int>((n + THREADS - 1) / THREADS < 65535
                            ? (n + THREADS - 1) / THREADS : 65535);
}

// Cache a cuBLAS handle per host thread and device so stream updates are isolated.
cublasHandle_t thread_blas_context(int* status) {
    struct BlasContext {
        cublasHandle_t handle = nullptr;
        cublasStatus_t status = cublasCreate(&handle);
        ~BlasContext() { if (status == CUBLAS_STATUS_SUCCESS) cublasDestroy(handle); }
    };
    thread_local std::map<int, std::unique_ptr<BlasContext>> contexts;
    int device;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) {
        *status = CUBLAS_STATUS_NOT_INITIALIZED;
        return nullptr;
    }
    auto& context = contexts[device];
    if (!context) context = std::make_unique<BlasContext>();
    *status = context->status;
    return context->handle;
}
}  // namespace

// Enqueue nearest-center assignment on the supplied CUDA stream. Inputs are
// contiguous FP32 X [B,N,D], centers [B,C,D], and int32 output labels [B,N].
// Scratch [chunk_size,C] is reused sequentially across batches and chunks.
extern "C" int peco_assign_tokens_to_centers(
    const float* x, const float* centers, int32_t* labels, float* scratch,
    float* center_norms, int batch, int tokens, int dims, int clusters,
    int chunk_size, void* stream_ptr) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    int status;
    cublasHandle_t handle = thread_blas_context(&status);
    if (status) return 10000 + status;
    status = cublasSetStream(handle, stream);
    if (status) return 10000 + status;
    status = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    if (status) return 10000 + status;
    compute_center_norms<<<batch * clusters, THREADS, 0, stream>>>(
        centers, center_norms, clusters, dims, batch);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    const float alpha = -2.f, beta = 0.f;
    for (int b = 0; b < batch; ++b) {
        for (int start = 0; start < tokens; start += chunk_size) {
            int rows = tokens - start < chunk_size ? tokens - start : chunk_size;
            // cuBLAS reads row-major centers/X as column-major [D,C]/[D,rows].
            // Its column-major [C,rows] result occupies row-major [rows,C] scratch.
            status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                clusters, rows, dims, &alpha,
                                centers + static_cast<int64_t>(b) * clusters * dims, dims,
                                x + (static_cast<int64_t>(b) * tokens + start) * dims, dims,
                                &beta, scratch, clusters);
            if (status) return 10000 + status;
            select_nearest_centers<<<rows, THREADS, 0, stream>>>(
                scratch, center_norms + b * clusters,
                labels + static_cast<int64_t>(b) * tokens + start, rows, clusters);
            error = cudaGetLastError();
            if (error != cudaSuccess) return error;
        }
    }
    return 0;
}

// Recompute FP32 centers [B,C,D] and int32 counts [B,C] from labels [B,N].
// Clear reusable accumulation buffers; preserve the previous center when empty.
extern "C" int peco_recompute_cluster_centers(
    const float* x, const int32_t* labels, const float* previous, float* sums,
    int32_t* counts, float* means, int batch, int tokens, int dims, int clusters,
    void* stream_ptr) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    int64_t center_elements = static_cast<int64_t>(batch) * clusters * dims;
    cudaError_t error = cudaMemsetAsync(sums, 0, center_elements * sizeof(float), stream);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(counts, 0, static_cast<int64_t>(batch) * clusters * sizeof(int32_t), stream);
    if (error != cudaSuccess) return error;
    accumulate_cluster_statistics<<<launch_block_count(static_cast<int64_t>(batch) * tokens * dims), THREADS, 0, stream>>>(
        x, labels, sums, counts, tokens, dims, clusters, batch);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    normalize_cluster_sums<<<launch_block_count(center_elements), THREADS, 0, stream>>>(
        sums, counts, previous, means, center_elements, dims);
    return cudaGetLastError();
}

// Reorder contiguous [B*H,N,D] features using permutation indices [B*H,N].
// Integer element types preserve the bit patterns of FP16, BF16, and FP32 inputs.
extern "C" int peco_reorder_token_features(
    const void* input, const int32_t* indices, void* output, int batch_heads,
    int tokens, int dims, int item_size, int inverse, void* stream_ptr) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    int64_t total = static_cast<int64_t>(batch_heads) * tokens * dims;
    if (item_size == 2) {
        if (inverse) remap_token_features<uint16_t, true><<<launch_block_count(total), THREADS, 0, stream>>>(
            static_cast<const uint16_t*>(input), indices, static_cast<uint16_t*>(output), tokens, dims, total);
        else remap_token_features<uint16_t, false><<<launch_block_count(total), THREADS, 0, stream>>>(
            static_cast<const uint16_t*>(input), indices, static_cast<uint16_t*>(output), tokens, dims, total);
    } else if (item_size == 4) {
        if (inverse) remap_token_features<uint32_t, true><<<launch_block_count(total), THREADS, 0, stream>>>(
            static_cast<const uint32_t*>(input), indices, static_cast<uint32_t*>(output), tokens, dims, total);
        else remap_token_features<uint32_t, false><<<launch_block_count(total), THREADS, 0, stream>>>(
            static_cast<const uint32_t*>(input), indices, static_cast<uint32_t*>(output), tokens, dims, total);
    } else return static_cast<int>(cudaErrorInvalidValue);
    return cudaGetLastError();
}

// CUDA codes pass through unchanged; cuBLAS statuses use a 10000 offset.
extern "C" const char* peco_cluster_error_message(int code) {
    if (code >= 10000) return "cuBLAS failure (subtract 10000 for cublasStatus_t)";
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
