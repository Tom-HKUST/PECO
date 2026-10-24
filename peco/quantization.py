"""Cluster-centroid Base-Offset quantization for Q/K attention inputs.

PECO (2026291592.pdf), page 8 and Section V-C on page 9, subtracts the
assigned token-cluster centroid and quantizes the remaining offset with a
4-bit Min-Max quantizer. ``PE`` means processing element, not positional
encoding. This software implementation reconstructs floating-point Q/K
before attention; it does not require an INT4 attention kernel.

The paper does not specify the quantizer granularity or zero-point rule.
Here each token/head has one affine quantizer over its D channels:

    r = x - C[label]
    low = min(r); scale = (max(r) - low) / (2**bits - 1)
    code = clamp(round((r - low) / scale), 0, 2**bits - 1)
    x_hat = C[label] + low + scale * code

Constant residuals use scale=1 and code=0, preserving their value. Arithmetic
uses FP32, and reconstructed tensors use the original input dtype. Codes
occupy uint8 containers even at 4 bits; they are not nibble-packed. The
centroids are D-dimensional vectors. Consequently the valid QK expansion
is Cq.dot(Ck) + Cq.dot(rk) + rq.dot(Ck) + rq.dot(rk); replacing its first
term with D*Cq*Ck would be incorrect for general KMeans vector centroids.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BaseOffsetQuantized:
    """Quantized payload with flattened batch/head axes.

    ``centroids`` is FP32 [BH,K,D], ``labels`` is int64 [BH,N], ``codes``
    is uint8 [BH,N,D], and ``offset_min``/``scale`` are FP32 [BH,N,1].
    ``original_shape`` restores either [BH,N,D] or [B,H,N,D]. Tensors are
    held by reference; callers must not mutate a payload before decoding it.
    """

    centroids: torch.Tensor
    labels: torch.Tensor
    codes: torch.Tensor
    offset_min: torch.Tensor
    scale: torch.Tensor
    original_dtype: torch.dtype
    original_shape: tuple[int, ...]
    bits: int

    def dequantize(self) -> torch.Tensor:
        return dequantize_base_offset(self)


def _validate_bits(bits: int, *, allow_bypass: bool) -> None:
    allowed = (0, 4, 8) if allow_bypass else (4, 8)
    if type(bits) is not int or bits not in allowed:
        suffix = "; use fake_quantize_base_offset(..., bits=0) to bypass" if not allow_bypass else ""
        raise ValueError(f"bits must be one of {allowed}{suffix}")


def _flatten_inputs(
    x: torch.Tensor, labels: torch.Tensor, centroids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim not in (3, 4):
        raise ValueError("x must have shape [BH,N,D] or [B,H,N,D]")
    if any(size == 0 for size in x.shape):
        raise ValueError("x dimensions must be nonempty")
    if not x.is_floating_point() or not centroids.is_floating_point():
        raise TypeError("x and centroids must have floating-point dtypes")
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError("labels must have dtype int32 or int64")
    if x.device != labels.device or x.device != centroids.device:
        raise ValueError("x, labels and centroids must be on the same device")

    n, d = x.shape[-2:]
    flat = x.reshape(-1, n, d)
    if labels.shape != flat.shape[:2]:
        raise ValueError(f"labels must have shape {tuple(flat.shape[:2])}")
    if (
        centroids.ndim != 3
        or centroids.shape[0] != flat.shape[0]
        or centroids.shape[1] == 0
        or centroids.shape[2] != d
    ):
        raise ValueError(f"centroids must have shape [{flat.shape[0]},K,{d}] with K > 0")
    # torch.gather checks label bounds. Avoid a separate GPU reduction and
    # host synchronization on every attention layer merely to repeat it.
    return flat, labels.to(torch.int64), centroids.float()


def _assigned_centroids(centroids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return centroids.gather(1, labels.unsqueeze(-1).expand(-1, -1, centroids.shape[-1]))


@torch.no_grad()
def quantize_base_offset(
    x: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    bits: int = 4,
) -> BaseOffsetQuantized:
    """Encode Q or K using its assigned cluster's vector centroid.

    ``bits`` is 4 or 8. For an allocation-free disabled path, use
    :func:`fake_quantize_base_offset` with ``bits=0``. All floating inputs
    are expected to be finite, as for the downstream attention operation.
    """
    _validate_bits(bits, allow_bypass=False)
    flat, labels, centroids = _flatten_inputs(x, labels, centroids)
    residual = flat.float() - _assigned_centroids(centroids, labels)
    low = residual.amin(dim=-1, keepdim=True)
    high = residual.amax(dim=-1, keepdim=True)
    levels = (1 << bits) - 1
    raw_scale = (high - low) / levels
    # Also protects FP32 scale underflow for tiny residual ranges.
    scale = torch.where(raw_scale > 0, raw_scale, torch.ones_like(raw_scale))
    codes = ((residual - low) / scale).round().clamp_(0, levels).to(torch.uint8)
    return BaseOffsetQuantized(
        centroids=centroids,
        labels=labels,
        codes=codes,
        offset_min=low,
        scale=scale,
        original_dtype=x.dtype,
        original_shape=tuple(x.shape),
        bits=bits,
    )


@torch.no_grad()
def dequantize_base_offset(encoded: BaseOffsetQuantized) -> torch.Tensor:
    """Reconstruct a quantized payload to the input shape and dtype."""
    base = _assigned_centroids(encoded.centroids, encoded.labels)
    residual = encoded.offset_min + encoded.scale * encoded.codes.float()
    return (base + residual).to(encoded.original_dtype).reshape(encoded.original_shape)


@torch.no_grad()
def fake_quantize_base_offset(
    x: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    bits: int = 4,
) -> torch.Tensor:
    """Quantize/dequantize Q or K; bits=0 returns the original tensor directly."""
    _validate_bits(bits, allow_bypass=True)
    if bits == 0:
        return x
    return quantize_base_offset(x, labels, centroids, bits).dequantize()
