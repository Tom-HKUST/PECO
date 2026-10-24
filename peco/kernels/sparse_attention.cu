// Cluster-sparse attention using standard WMMA and block synchronization.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <mma.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr int kQueryTile = 16;
constexpr int kKeyTile = 32;
constexpr int kThreads = 128;

// The counts stay on the device. Each cluster can add at most one partial tile,
// so ceil(sequence/16) + clusters is a safe, small upper bound on work items.
__global__ void make_work_map(const int* q_sizes, const int* k_sizes,
                             int* workspace, int sq, int sk, int cq, int ck,
                             int capacity) {
    if (threadIdx.x != 0) return;
    const int bh = blockIdx.x;
    const int stride = ck + 2 + 3 * capacity;
    int* head = workspace + static_cast<int64_t>(bh) * stride;
    int* k_prefix = head + 1;
    int* tiles = k_prefix + ck + 1;
    int offset = 0;
    k_prefix[0] = 0;
    for (int c = 0; c < ck; ++c) {
        const int size = max(0, min(k_sizes[static_cast<int64_t>(bh) * ck + c], sk - offset));
        offset += size;
        k_prefix[c + 1] = offset;
    }
    int count = 0;
    offset = 0;
    for (int c = 0; c < cq; ++c) {
        const int size = max(0, min(q_sizes[static_cast<int64_t>(bh) * cq + c], sq - offset));
        for (int start = 0; start < size; start += kQueryTile) {
            tiles[3 * count] = offset + start;
            tiles[3 * count + 1] = min(kQueryTile, size - start);
            tiles[3 * count + 2] = c;
            ++count;
        }
        offset += size;
    }
    head[0] = count;
}

template <typename T> __device__ __forceinline__ float as_float(T value);
template <> __device__ __forceinline__ float as_float(__half value) { return __half2float(value); }
template <> __device__ __forceinline__ float as_float(__nv_bfloat16 value) { return __bfloat162float(value); }
template <typename T> __device__ __forceinline__ T from_float(float value);
template <> __device__ __forceinline__ __half from_float(float value) { return __float2half_rn(value); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float(float value) { return __float2bfloat16_rn(value); }

template <typename T, int D>
__global__ void sparse_attention_kernel(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    const uint8_t* __restrict__ selected, const int* __restrict__ workspace,
    T* __restrict__ output, int sq, int sk, int cq, int ck, int capacity) {
    using namespace nvcuda;
    const int bh = blockIdx.y;
    const int* head = workspace + static_cast<int64_t>(bh) * (ck + 2 + 3 * capacity);
    if (blockIdx.x >= head[0]) return;
    const int* k_prefix = head + 1;
    const int* tile = k_prefix + ck + 1 + 3 * blockIdx.x;
    const int q_start = tile[0];
    const int q_count = tile[1];
    const int q_cluster = tile[2];
    const int tid = threadIdx.x;

    __shared__ __align__(32) T q_tile[kQueryTile * D];
    // K is consumed before V is loaded, allowing both to use the same buffer.
    __shared__ __align__(32) T kv_tile[kKeyTile * D];
    __shared__ __align__(32) float scores[kQueryTile * kKeyTile];
    __shared__ float row_max[kQueryTile];
    __shared__ float row_sum[kQueryTile];
    __shared__ float row_alpha[kQueryTile];
    constexpr int kOutputElements = kQueryTile * D / kThreads;
    float accumulated[kOutputElements];
    #pragma unroll
    for (int i = 0; i < kOutputElements; ++i) accumulated[i] = 0.0f;
    for (int i = tid; i < kQueryTile * D; i += kThreads) {
        const int row = i / D;
        q_tile[i] = row < q_count
            ? q[(static_cast<int64_t>(bh) * sq + q_start + row) * D + i % D]
            : from_float<T>(0.0f);
    }
    if (tid < kQueryTile) {
        row_max[tid] = -CUDART_INF_F;
        row_sum[tid] = 0.0f;
    }
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(D));
    for (int c = 0; c < ck; ++c) {
        if (!selected[(static_cast<int64_t>(bh) * cq + q_cluster) * ck + c]) continue;
        const int key_end = k_prefix[c + 1];
        for (int key_start = k_prefix[c]; key_start < key_end; key_start += kKeyTile) {
            const int key_count = min(kKeyTile, key_end - key_start);
            for (int i = tid; i < kKeyTile * D; i += kThreads) {
                const int row = i / D;
                kv_tile[i] = row < key_count
                    ? k[(static_cast<int64_t>(bh) * sk + key_start + row) * D + i % D]
                    : from_float<T>(0.0f);
            }
            __syncthreads();
            // One warp computes the 16x32 score tile with WMMA tensor cores.
            // K is row-major [32,D], which is column-major [D,32] for QK^T.
            if (tid < 32) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
                wmma::fill_fragment(c0, 0.0f);
                wmma::fill_fragment(c1, 0.0f);
                #pragma unroll
                for (int d = 0; d < D; d += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b0, b1;
                    wmma::load_matrix_sync(a, q_tile + d, D);
                    wmma::load_matrix_sync(b0, kv_tile + d, D);
                    wmma::load_matrix_sync(b1, kv_tile + 16 * D + d, D);
                    wmma::mma_sync(c0, a, b0, c0);
                    wmma::mma_sync(c1, a, b1, c1);
                }
                wmma::store_matrix_sync(scores, c0, kKeyTile, wmma::mem_row_major);
                wmma::store_matrix_sync(scores + 16, c1, kKeyTile, wmma::mem_row_major);
            }
            __syncthreads();

            // Eight lanes reduce each row. Keep both softmax state and PV in
            // FP32; probability rounding to fp16/bf16 is avoided in this version.
            const int row = tid / 8;
            const int lane = tid % 8;
            float maximum = -CUDART_INF_F;
            #pragma unroll
            for (int j = lane; j < kKeyTile; j += 8) {
                const float score = j < key_count ? scores[row * kKeyTile + j] * scale : -CUDART_INF_F;
                scores[row * kKeyTile + j] = score;
                maximum = fmaxf(maximum, score);
            }
            #pragma unroll
            for (int delta = 4; delta > 0; delta /= 2)
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta, 8));
            maximum = fmaxf(maximum, row_max[row]);
            const float alpha = expf(row_max[row] - maximum);
            float denominator = 0.0f;
            #pragma unroll
            for (int j = lane; j < kKeyTile; j += 8) {
                const float probability = expf(scores[row * kKeyTile + j] - maximum);
                scores[row * kKeyTile + j] = probability;
                denominator += probability;
            }
            #pragma unroll
            for (int delta = 4; delta > 0; delta /= 2)
                denominator += __shfl_xor_sync(0xffffffff, denominator, delta, 8);
            if (lane == 0) {
                row_max[row] = maximum;
                row_sum[row] = row_sum[row] * alpha + denominator;
                row_alpha[row] = alpha;
            }
            for (int i = tid; i < kKeyTile * D; i += kThreads) {
                const int key_row = i / D;
                kv_tile[i] = key_row < key_count
                    ? v[(static_cast<int64_t>(bh) * sk + key_start + key_row) * D + i % D]
                    : from_float<T>(0.0f);
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < kOutputElements; ++i) {
                const int index = tid + i * kThreads;
                const int output_row = index / D;
                const int d = index % D;
                float value = accumulated[i] * row_alpha[output_row];
                #pragma unroll
                for (int j = 0; j < kKeyTile; ++j)
                    value = fmaf(scores[output_row * kKeyTile + j], as_float(kv_tile[j * D + d]), value);
                accumulated[i] = value;
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for (int i = 0; i < kOutputElements; ++i) {
        const int index = tid + i * kThreads;
        const int row = index / D;
        if (row < q_count) {
            const float denominator = row_sum[row];
            output[(static_cast<int64_t>(bh) * sq + q_start + row) * D + index % D] =
                from_float<T>(denominator > 0.0f ? accumulated[i] / denominator : 0.0f);
        }
    }
}

template <typename T>
void launch_attention(const void* q, const void* k, const void* v, const uint8_t* selected,
                      const int* workspace, void* output, int bh, int sq, int sk,
                      int d, int cq, int ck, int capacity, cudaStream_t stream) {
    const dim3 grid(capacity, bh);
    if (d == 64)
        sparse_attention_kernel<T, 64><<<grid, kThreads, 0, stream>>>(
            static_cast<const T*>(q), static_cast<const T*>(k), static_cast<const T*>(v),
            selected, workspace, static_cast<T*>(output), sq, sk, cq, ck, capacity);
    else
        sparse_attention_kernel<T, 128><<<grid, kThreads, 0, stream>>>(
            static_cast<const T*>(q), static_cast<const T*>(k), static_cast<const T*>(v),
            selected, workspace, static_cast<T*>(output), sq, sk, cq, ck, capacity);
}
}  // namespace

// dtype: 0=fp16, 1=bf16. Workspace length (int32 elements):
// B*H * (CK+2 + 3*(ceil(SQ/16)+CQ)). All work uses the caller's CUDA stream.
extern "C" int peco_sparse_attention(
    const void* q, const void* k, const void* v, const uint8_t* selected,
    const int* q_sizes, const int* k_sizes, void* output, int* workspace,
    int bh, int sq, int sk, int d, int cq, int ck, int dtype, void* stream_ptr) {
    if (bh < 1 || bh > 65535 || sq < 0 || sk < 0 || cq < 1 || ck < 1 ||
        (d != 64 && d != 128) || (dtype != 0 && dtype != 1)) return cudaErrorInvalidValue;
    if (sq == 0) return cudaSuccess;
    const int64_t capacity64 = (static_cast<int64_t>(sq) + 15) / 16 + cq;
    if (capacity64 > 2147483647LL || ck + 2LL + 3 * capacity64 > 2147483647LL)
        return cudaErrorInvalidValue;
    const int capacity = static_cast<int>(capacity64);
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    make_work_map<<<bh, 1, 0, stream>>>(q_sizes, k_sizes, workspace, sq, sk, cq, ck, capacity);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    if (dtype == 0)
        launch_attention<__half>(q, k, v, selected, workspace, output, bh, sq, sk, d, cq, ck, capacity, stream);
    else
        launch_attention<__nv_bfloat16>(q, k, v, selected, workspace, output, bh, sq, sk, d, cq, ck, capacity, stream);
    return cudaGetLastError();
}

extern "C" const char* peco_attention_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
