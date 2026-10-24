"""Group similar tokens and select the cluster pairs used by PECO attention.

Each batch/head is clustered independently. The resulting labels define a
contiguous token layout, while centroid attention estimates determine which
query and key clusters can interact. CUDA inputs use the native CUDA/cuBLAS
backend; CPU inputs use a floating-point reference implementation.
"""

import math

import torch

from . import clustering_cuda


@torch.no_grad()
def fit_token_clusters(
    x, n_clusters, max_iters=100, tol=1e-4, init_centroids=None, *, chunk_size=1024,
):
    """Run Euclidean Lloyd updates independently for each batch/head.

    Input features have shape [BH,N,D]. Optional initial centers [BH,K,D]
    allow reuse across denoising steps. Distances and center sums use FP32;
    assignment processes token chunks instead of storing an N-by-K matrix.

    Return labels [BH,N], centers [BH,K,D], counts [BH,K], and the iteration
    count. Each nonempty center is the mean of its returned token partition;
    an empty cluster keeps its previous center. A positive ``tol`` enables
    early stopping by maximum center displacement. Set ``tol=0`` for fixed
    iterations without a host synchronization after each center update.
    """
    if x.ndim != 3 or not x.is_floating_point():
        raise ValueError("x must be a floating tensor with shape [batch_heads, tokens, dims]")
    batch, tokens, dims = x.shape
    if min(batch, tokens, dims) <= 0:
        raise ValueError("x dimensions must be nonzero")
    if not isinstance(n_clusters, int) or not 1 <= n_clusters <= tokens:
        raise ValueError("n_clusters must be an integer in [1, tokens]")
    if not isinstance(max_iters, int) or max_iters < 1:
        raise ValueError("max_iters must be a positive integer")
    if tol < 0 or not math.isfinite(tol):
        raise ValueError("tol must be finite and nonnegative")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    x = x.float().contiguous()
    # Keep the caller's cached centers unchanged while producing a new partition.
    if init_centroids is None:
        indices = torch.randint(tokens, (batch, n_clusters), device=x.device)
        centers = x.gather(1, indices[..., None].expand(-1, -1, dims)).contiguous()
    else:
        if init_centroids.shape != (batch, n_clusters, dims) or init_centroids.device != x.device:
            raise ValueError("init_centroids must match [batch_heads, n_clusters, dims] and x.device")
        centers = init_centroids.float().contiguous().clone()
    labels = torch.empty((batch, tokens), dtype=torch.int32, device=x.device)
    counts = torch.empty((batch, n_clusters), dtype=torch.int32, device=x.device)
    sums = torch.empty_like(centers)
    means = torch.empty_like(centers)
    chunk_size = min(chunk_size, tokens)
    if x.is_cuda:
        scratch = torch.empty((chunk_size, n_clusters), dtype=torch.float32, device=x.device)
        center_norms = torch.empty((batch, n_clusters), dtype=torch.float32, device=x.device)
    for iteration in range(1, max_iters + 1):
        if x.is_cuda:
            clustering_cuda.assign_tokens_to_centers(x, centers, labels, scratch, center_norms, chunk_size)
            clustering_cuda.recompute_cluster_centers(x, labels, centers, sums, counts, means)
        else:
            center_norms = centers.square().sum(-1).unsqueeze(1)
            for start in range(0, tokens, chunk_size):
                # The token norm is identical for all candidate centers and
                # therefore does not affect the nearest-center label.
                distance = -2 * torch.bmm(x[:, start:start + chunk_size], centers.transpose(1, 2))
                distance += center_norms
                labels[:, start:start + chunk_size] = distance.argmin(-1).to(torch.int32)
            sums.zero_()
            counts.zero_()
            long_labels = labels.long()
            sums.scatter_add_(1, long_labels[..., None].expand(-1, -1, dims), x)
            counts.scatter_add_(1, long_labels, torch.ones_like(labels))
            means.copy_(torch.where(counts[..., None] > 0,
                                    sums / counts[..., None].clamp_min(1), centers))
        shift = None
        if tol > 0:
            shift = torch.linalg.vector_norm(means - centers, dim=-1).max().item()
        # Swap only after the new means have been completed, including the
        # iteration that satisfies the stopping criterion.
        centers, means = means, centers
        if tol > 0 and shift <= tol:
            break
    return labels, centers, counts, iteration


def _prepare_sequence_indices(tensor, labels, dim, sorted_indices):
    """Validate token reordering and return contiguous int32 indices [BH,N]."""
    if tensor.ndim != 4 or dim not in (2, -2):
        raise ValueError("permutation expects [batch, heads, tokens, dims] with dim=2")
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("permutation supports float16, bfloat16, and float32")
    batch, heads, tokens, _ = tensor.shape
    if min(tensor.shape) <= 0:
        raise ValueError("permutation tensor dimensions must be nonzero")
    if sorted_indices is None:
        if labels is None or labels.numel() != batch * heads * tokens or labels.device != tensor.device:
            raise ValueError("labels must contain one cluster ID per token on the same device")
        # Stable sorting preserves the original order within each cluster.
        sorted_indices = torch.argsort(labels.reshape(batch * heads, tokens), dim=-1, stable=True)
    if sorted_indices.numel() != batch * heads * tokens or sorted_indices.device != tensor.device:
        raise ValueError("sorted_indices must contain one index per token on the same device")
    return sorted_indices.reshape(batch * heads, tokens).to(torch.int32).contiguous()


@torch.no_grad()
def gather_clustered_tokens(tensor, labels, dim=2, sorted_indices=None):
    """Pack [B,H,N,D] features into cluster order and return the gather indices.

    ``indices[bh,j]`` gives the original token at packed position ``j``.
    Pass the returned indices when packing V to reuse exactly the K ordering.
    """
    indices = _prepare_sequence_indices(tensor, labels, dim, sorted_indices)
    if tensor.is_cuda:
        result = clustering_cuda.reorder_token_features(tensor, indices)
    else:
        batch, heads, tokens, dims = tensor.shape
        result = tensor.gather(2, indices.long().view(batch, heads, tokens, 1).expand(-1, -1, -1, dims))
    return result, indices


@torch.no_grad()
def restore_sequence_order(tensor, sorted_indices, dim=2):
    """Scatter packed features back to the original sequence using gather indices."""
    indices = _prepare_sequence_indices(tensor, None, dim, sorted_indices)
    if tensor.is_cuda:
        return clustering_cuda.reorder_token_features(tensor, indices, inverse=True)
    batch, heads, tokens, dims = tensor.shape
    result = torch.empty_like(tensor)
    result.scatter_(2, indices.long().view(batch, heads, tokens, 1).expand(-1, -1, -1, dims), tensor)
    return result


@torch.no_grad()
def compute_cluster_weights(scores, weights):
    """Normalize centroid scores using key-cluster token counts in FP32.

    This computes softmax(scores + log(counts)) over the key-cluster axis.
    Empty clusters contribute zero mass; an entirely empty row returns zeros.
    """
    scores, weights = scores.float(), weights.float()
    logits = scores + torch.where(weights > 0, weights.log(), -torch.inf)
    row_max = logits.max(dim=-1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    mass = torch.exp(logits - row_max)
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)


@torch.no_grad()
def select_cluster_connections(
    query_centroids, key_centroids, q_cluster_sizes, k_cluster_sizes, p, min_kc_ratio=0,
):
    """Choose nonempty query/key cluster pairs from estimated attention mass.

    Centroids have shape [B,H,C,D]; sizes contain token counts [B,H,C].
    For each query cluster, retain the smallest descending-probability prefix
    reaching ``p``, subject to the minimum key-cluster ratio. Return a boolean
    [B,H,CQ,CK] mask. Empty query or key clusters never produce active pairs.
    """
    if not 0 <= p <= 1 or not 0 <= min_kc_ratio <= 1:
        raise ValueError("p and min_kc_ratio must be in [0, 1]")
    scores = torch.matmul(query_centroids.float(), key_centroids.float().transpose(-1, -2))
    scores *= query_centroids.shape[-1] ** -0.5
    probabilities = compute_cluster_weights(scores, k_cluster_sizes.unsqueeze(-2))
    sorted_probs, sorted_indices = probabilities.sort(dim=-1, descending=True, stable=True)
    cumulative_before = sorted_probs.cumsum(-1) - sorted_probs
    # Compare the mass before each candidate so the crossing cluster is kept.
    keep = cumulative_before < p
    if p == 1:
        # Avoid dropping the last cluster because of cumulative rounding error.
        keep.fill_(True)
    # A zero threshold still selects the most likely nonempty key cluster.
    keep[..., 0] = True
    minimum = math.ceil(min_kc_ratio * key_centroids.shape[-2])
    if minimum:
        keep[..., :minimum] = True
    dynamic_map = torch.zeros_like(probabilities, dtype=torch.bool)
    dynamic_map.scatter_(-1, sorted_indices, keep)
    dynamic_map &= k_cluster_sizes.unsqueeze(-2) > 0
    dynamic_map &= q_cluster_sizes.unsqueeze(-1) > 0
    return dynamic_map
