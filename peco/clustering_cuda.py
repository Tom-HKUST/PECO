"""Build and call PECO's CUDA clustering kernels through a stable C ABI.

Clustering uses contiguous FP32 features and centers with int32 labels/counts.
Token reordering also preserves FP16 and BF16 feature bits without conversion.
"""

import ctypes
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch


_LIBRARY = None
ROOT = Path(__file__).resolve().parents[1]


def build_cluster_library():
    """Build the CUDA library when missing or older than its CUDA source."""
    source = ROOT / "peco" / "kernels" / "kmeans.cu"
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    destination = build / "libpeco_clustering.so"
    # Serialize worker builds by locking the source, without a persistent lock file.
    with source.open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return destination
        nvcc = os.environ.get("NVCC")
        if not nvcc and os.environ.get("CUDA_HOME"):
            nvcc = str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc")
        if not nvcc:
            preferred = Path("/usr/local/cuda-12.8/bin/nvcc")
            nvcc = str(preferred) if preferred.exists() else shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError("nvcc is required to build the CUDA clustering kernels")
        nvcc = shutil.which(nvcc) or nvcc
        cuda_root = Path(nvcc).resolve().parents[1]
        fd, temporary = tempfile.mkstemp(prefix="clustering-", suffix=".so", dir=build)
        os.close(fd)
        try:
            command = [
                nvcc, "-std=c++17", "-O3", "-shared", "-Xcompiler=-fPIC",
                "-arch=sm_89", str(source), "-lcublas", "-o", temporary,
                "-Xlinker", "-rpath", "-Xlinker", str(cuda_root / "lib64"),
            ]
            subprocess.run(command, check=True)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return destination


def _load_cluster_library():
    """Load the library once and declare the pointer/scalar types of its C ABI."""
    global _LIBRARY
    if _LIBRARY is None:
        library = ctypes.CDLL(str(build_cluster_library()))
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        library.peco_assign_tokens_to_centers.argtypes = [pointer] * 5 + [integer] * 5 + [pointer]
        library.peco_recompute_cluster_centers.argtypes = [pointer] * 6 + [integer] * 4 + [pointer]
        library.peco_reorder_token_features.argtypes = [pointer] * 3 + [integer] * 5 + [pointer]
        for name in ("peco_assign_tokens_to_centers", "peco_recompute_cluster_centers", "peco_reorder_token_features"):
            getattr(library, name).restype = integer
        library.peco_cluster_error_message.argtypes = [integer]
        library.peco_cluster_error_message.restype = ctypes.c_char_p
        _LIBRARY = library
    return _LIBRARY


def _raise_cluster_error(library, status):
    """Raise for an immediate CUDA launch or cuBLAS status failure."""
    if status:
        message = library.peco_cluster_error_message(status).decode("utf-8")
        raise RuntimeError(f"CUDA clustering failed: {message} (status {status})")


def _retain_on_current_stream(device, *tensors):
    """Record native tensor accesses and return the current CUDA stream pointer."""
    stream = torch.cuda.current_stream(device)
    # ctypes bypasses PyTorch's dispatcher; retain every allocation until native
    # work on this stream finishes, including temporary contiguous copies.
    for tensor in tensors:
        tensor.record_stream(stream)
    return ctypes.c_void_p(stream.cuda_stream)


def assign_tokens_to_centers(x, centers, labels, scratch, center_norms, chunk_size):
    """Assign FP32 x [B,N,D] to centers [B,C,D], writing int32 labels [B,N].

    Scratch [chunk_size,C] bounds distance storage; center_norms [B,C] is reused.
    The caller supplies contiguous tensors on one device and valid dimensions.
    """
    library = _load_cluster_library()
    batch, tokens, dims = x.shape
    clusters = centers.shape[1]
    with torch.cuda.device(x.device):
        status = library.peco_assign_tokens_to_centers(
            x.data_ptr(), centers.data_ptr(), labels.data_ptr(), scratch.data_ptr(),
            center_norms.data_ptr(), batch, tokens, dims, clusters, chunk_size,
            _retain_on_current_stream(x.device, x, centers, labels, scratch, center_norms),
        )
    _raise_cluster_error(library, status)


def recompute_cluster_centers(x, labels, previous, sums, counts, means):
    """Write FP32 means [B,C,D] and int32 counts [B,C] for labels [B,N].

    Sums [B,C,D] are cleared and reused; empty clusters retain previous centers.
    All buffers must be contiguous, on the same CUDA device, and correctly sized.
    """
    library = _load_cluster_library()
    batch, tokens, dims = x.shape
    with torch.cuda.device(x.device):
        status = library.peco_recompute_cluster_centers(
            x.data_ptr(), labels.data_ptr(), previous.data_ptr(), sums.data_ptr(),
            counts.data_ptr(), means.data_ptr(), batch, tokens, dims, previous.shape[1],
            _retain_on_current_stream(x.device, x, labels, previous, sums, counts, means),
        )
    _raise_cluster_error(library, status)


def reorder_token_features(tensor, indices, inverse=False):
    """Reorder features [B,H,N,D] using original token IDs [B*H,N].

    Forward gathers original IDs into sorted order; inverse scatters sorted rows
    back to original order. Indices must be a valid permutation on tensor.device.
    """
    library = _load_cluster_library()
    tensor = tensor.contiguous()
    indices = indices.to(dtype=torch.int32).contiguous()
    batch, heads, tokens, dims = tensor.shape
    result = torch.empty_like(tensor)
    with torch.cuda.device(tensor.device):
        status = library.peco_reorder_token_features(
            tensor.data_ptr(), indices.data_ptr(), result.data_ptr(), batch * heads,
            tokens, dims, tensor.element_size(), int(inverse),
            _retain_on_current_stream(tensor.device, tensor, indices, result),
        )
    _raise_cluster_error(library, status)
    return result


if __name__ == "__main__":
    print(build_cluster_library())
