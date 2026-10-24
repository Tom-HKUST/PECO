"""CUDA sparse attention for cluster-sorted Q/K/V, without a torch C++ ABI.

QK uses standard WMMA tensor cores; online softmax and PV accumulate in
FP32. This first version favors a transparent, accurate PV implementation over
an additional tensor-core rounding step. Backward is intentionally unsupported.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import threading

import torch

_ROOT = Path(__file__).resolve().parent.parent
_SOURCE = _ROOT / "peco" / "kernels" / "sparse_attention.cu"
_LIBRARY = _ROOT / "build" / "libpeco_attention.so"
_BUILD_LOCK = threading.Lock()
_LIB = None


def _nvcc() -> str:
    if os.environ.get("NVCC"):
        return os.environ["NVCC"]
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        return str(Path(cuda_home) / "bin" / "nvcc")
    compatible = Path("/usr/local/cuda-12.8/bin/nvcc")
    return str(compatible) if compatible.is_file() else "nvcc"


def _library():
    global _LIB
    if _LIB is not None:
        return _LIB
    with _BUILD_LOCK:
        if _LIB is not None:
            return _LIB
        _LIBRARY.parent.mkdir(parents=True, exist_ok=True)
        # A process-specific temporary file avoids exposing a partially built .so
        # when multiple inference workers start together.
        if not _LIBRARY.exists() or _LIBRARY.stat().st_mtime_ns < _SOURCE.stat().st_mtime_ns:
            temporary = _LIBRARY.with_suffix(f".{os.getpid()}.so")
            command = [
                _nvcc(), "-O3", "-std=c++17", "-arch=sm_89", "--shared",
                "-Xcompiler", "-fPIC", str(_SOURCE), "-o", str(temporary),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                os.replace(temporary, _LIBRARY)
            except subprocess.CalledProcessError as error:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"Failed to build PECO CUDA attention:\n{error.stderr}") from error
        library = ctypes.CDLL(str(_LIBRARY))
        library.peco_sparse_attention.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
        library.peco_sparse_attention.restype = ctypes.c_int
        library.peco_attention_error_string.argtypes = [ctypes.c_int]
        library.peco_attention_error_string.restype = ctypes.c_char_p
        _LIB = library
        return library


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dynamic_map: torch.Tensor,
    q_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    *,
    validate_sizes: bool = True,
) -> torch.Tensor:
    """Return cluster-masked softmax(QK^T/sqrt(D))V.

    Inputs must be contiguous CUDA tensors: Q [B,H,SQ,D], K/V [B,H,SK,D],
    bool map [B,H,CQ,CK], and int32 sizes [B,H,CQ/CK]. Tokens must already
    be grouped in cluster order. Empty clusters and completely masked rows are
    supported; a completely masked row returns zero. Sizes are checked on the
    CUDA stream without copying them to the host. Set validate_sizes=False only
    when the caller already guarantees nonnegative counts and exact sums.
    """
    tensors = (q, k, v, dynamic_map, q_sizes, k_sizes)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("All sparse attention inputs must be torch tensors")
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("Expected Q [B,H,SQ,D] and equal-shaped K/V [B,H,SK,D]")
    batch, heads, sq, dim = q.shape
    if batch < 1 or heads < 1 or batch * heads > 65535:
        raise ValueError("B*H must be between 1 and 65535")
    if k.shape[:2] != q.shape[:2] or k.shape[-1] != dim:
        raise ValueError("Q and K/V batch, head, and feature dimensions must match")
    if dim not in (64, 128):
        raise ValueError("CUDA attention supports head dimensions 64 and 128")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Q/K/V must share dtype torch.float16 or torch.bfloat16")
    if not q.is_cuda or any(t.device != q.device for t in tensors):
        raise ValueError("All sparse attention inputs must reside on the same CUDA device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("All sparse attention inputs must be contiguous")
    if q_sizes.ndim != 3 or k_sizes.ndim != 3 or q_sizes.shape[:2] != (batch, heads) or k_sizes.shape[:2] != (batch, heads):
        raise ValueError("Cluster sizes must have shape [B,H,C]")
    cq, ck = q_sizes.shape[-1], k_sizes.shape[-1]
    if cq < 1 or ck < 1 or dynamic_map.shape != (batch, heads, cq, ck):
        raise ValueError("Expected a nonempty cluster axis and dynamic_map [B,H,CQ,CK]")
    if dynamic_map.dtype != torch.bool or q_sizes.dtype != torch.int32 or k_sizes.dtype != torch.int32:
        raise TypeError("dynamic_map must be bool and cluster sizes must be int32")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise RuntimeError("PECO CUDA sparse attention supports inference only; use torch.no_grad()")
    sk = k.shape[-2]
    capacity = (sq + 15) // 16 + cq
    stride = ck + 2 + 3 * capacity
    if max(sq, sk, cq, ck, capacity, stride) > 2**31 - 1:
        raise ValueError("Sequence/cluster metadata exceeds the CUDA int32 indexing limit")
    with torch.cuda.device(q.device):
        major, minor = torch.cuda.get_device_capability(q.device)
        if (major, minor) < (8, 9):
            raise RuntimeError("This CUDA build requires compute capability 8.9 or higher")
        if validate_sizes:
            torch._assert_async((q_sizes >= 0).all(), "q_sizes must be nonnegative")
            torch._assert_async((k_sizes >= 0).all(), "k_sizes must be nonnegative")
            torch._assert_async((q_sizes.sum(-1) == sq).all(), "q_sizes must sum to the Q sequence length")
            torch._assert_async((k_sizes.sum(-1) == sk).all(), "k_sizes must sum to the K sequence length")
        output = torch.empty_like(q)
        if sq == 0:
            return output
        workspace = torch.empty(batch * heads * stride, dtype=torch.int32, device=q.device)
        library = _library()
        stream = torch.cuda.current_stream(q.device)
        error = library.peco_sparse_attention(
            *(ctypes.c_void_p(t.data_ptr()) for t in tensors[:4]),
            ctypes.c_void_p(q_sizes.data_ptr()), ctypes.c_void_p(k_sizes.data_ptr()),
            ctypes.c_void_p(output.data_ptr()), ctypes.c_void_p(workspace.data_ptr()),
            batch * heads, sq, sk, dim, cq, ck, 0 if q.dtype == torch.float16 else 1,
            ctypes.c_void_p(stream.cuda_stream),
        )
        if error:
            message = library.peco_attention_error_string(error).decode("utf-8", errors="replace")
            raise RuntimeError(f"PECO CUDA sparse attention launch failed: {message} ({error})")
        # The caching allocator cannot see ctypes kernel accesses. Preserve all
        # input allocations until the caller's stream has consumed them.
        for tensor in (*tensors, workspace, output):
            tensor.record_stream(stream)
        return output
