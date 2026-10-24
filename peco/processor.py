"""HunyuanVideo projections, centroid reuse, and CUDA sparse attention."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .quantization import fake_quantize_base_offset
from .attention_cuda import sparse_attention
from .clustering import (
    restore_sequence_order,
    fit_token_clusters,
    select_cluster_connections,
    gather_clustered_tokens,
)


@dataclass(frozen=True)
class PecoConfig:
    num_q_centroids: int = 100
    num_k_centroids: int = 500
    top_p: float = 0.9
    min_kc_ratio: float = 0.1
    kmeans_iter_init: int = 50
    kmeans_iter_step: int = 2
    warmup_steps: int = 2
    warmup_layers: int = 0
    zero_step_kmeans_init: bool = True
    base_offset_bits: int = 4

    def __post_init__(self):
        if min(self.num_q_centroids, self.num_k_centroids, self.kmeans_iter_init, self.kmeans_iter_step) < 1:
            raise ValueError("Cluster counts and KMeans iteration counts must be positive")
        if not 0 < self.top_p <= 1 or not 0 <= self.min_kc_ratio <= 1:
            raise ValueError("top_p must be in (0, 1], min_kc_ratio in [0, 1]")
        if self.base_offset_bits not in (0, 4, 8):
            raise ValueError("base_offset_bits must be 0, 4 or 8")
        if self.warmup_steps < 0 or self.warmup_layers < 0:
            raise ValueError("Warmup counts must be nonnegative")


@dataclass
class InferenceContext:
    """Shared by processors in one pipeline; filled once per transformer call."""

    step: int = -1
    text_length: int = 0
    valid_text_length: int = 0


def dense_attention(query, key, value, valid_length):
    """Match the original varlen split without allocating an S-by-S mask."""
    output = torch.empty_like(query)
    backend = SDPBackend.FLASH_ATTENTION if query.is_cuda else SDPBackend.MATH
    with sdpa_kernel(backend):
        for start, end in ((0, valid_length), (valid_length, query.shape[2])):
            if start < end:
                output[:, :, start:end] = F.scaled_dot_product_attention(
                    query[:, :, start:end].contiguous(),
                    key[:, :, start:end].contiguous(),
                    value[:, :, start:end].contiguous(),
                    dropout_p=0.0,
                )
    return output


def extend_text_partition(dynamic_map, q_sizes, k_sizes, valid_text, text_length):
    """Append valid-text and padding groups to the video-cluster partition.

    Valid text and video tokens can attend to each other. Padding forms a
    separate sequence and is excluded from every video/text interaction.
    """
    dynamic_map = F.pad(dynamic_map, (0, 2, 0, 2), value=False)
    dynamic_map[..., -2, :-1] = True
    dynamic_map[..., :-1, -2] = True
    dynamic_map[..., -1, -1] = True
    extra = torch.tensor([valid_text, text_length - valid_text], device=q_sizes.device, dtype=torch.int32)
    extra = extra.expand(*q_sizes.shape[:-1], 2)
    return (
        dynamic_map.contiguous(),
        torch.cat((q_sizes, extra), -1).to(torch.int32).contiguous(),
        torch.cat((k_sizes, extra), -1).to(torch.int32).contiguous(),
    )


class PecoAttention:
    def __init__(self, layer_idx, config: PecoConfig, context: InferenceContext):
        self.layer_idx = layer_idx
        self.config = config
        self.context = context
        self.q_centroids = None
        self.k_centroids = None

    def reset(self):
        self.q_centroids = self.k_centroids = None

    def _update_cluster_state(self, query, key):
        """Fit Q/K partitions, reusing this layer's centers after initialization."""
        batch, heads, length, dim = query.shape
        initializing = self.q_centroids is None
        iterations = self.config.kmeans_iter_init if initializing else self.config.kmeans_iter_step
        q_labels, self.q_centroids, q_sizes, _ = fit_token_clusters(
            query.reshape(batch * heads, length, dim).contiguous(),
            self.config.num_q_centroids, max_iters=iterations, tol=0,
            init_centroids=self.q_centroids,
        )
        k_labels, self.k_centroids, k_sizes, _ = fit_token_clusters(
            key.reshape(batch * heads, length, dim).contiguous(),
            self.config.num_k_centroids, max_iters=iterations, tol=0,
            init_centroids=self.k_centroids,
        )
        return q_labels, k_labels, q_sizes.reshape(batch, heads, -1), k_sizes.reshape(batch, heads, -1)

    def attention(self, query, key, value):
        batch, heads, length, dim = query.shape
        if batch != 1:
            raise ValueError("This minimal HunyuanVideo entry supports one prompt per call")
        text_length = self.context.text_length
        video_length = length - text_length
        valid_text = self.context.valid_text_length
        if video_length <= 0 or not 0 <= valid_text <= text_length:
            raise ValueError("Invalid video/text lengths in transformer context")
        warmup = self.context.step < self.config.warmup_steps or self.layer_idx < self.config.warmup_layers
        if warmup:
            if self.config.zero_step_kmeans_init and self.layer_idx >= self.config.warmup_layers:
                self._update_cluster_state(query[:, :, :video_length], key[:, :, :video_length])
            return dense_attention(query, key, value, video_length + valid_text)

        q_video = query[:, :, :video_length].contiguous()
        k_video = key[:, :, :video_length].contiguous()
        v_video = value[:, :, :video_length].contiguous()
        q_labels, k_labels, q_sizes, k_sizes = self._update_cluster_state(q_video, k_video)
        # Estimate cluster connectivity from the original floating-point centers.
        dynamic_map = select_cluster_connections(
            self.q_centroids.reshape(batch, heads, -1, dim),
            self.k_centroids.reshape(batch, heads, -1, dim),
            q_sizes, k_sizes, self.config.top_p, self.config.min_kc_ratio,
        )
        # Reconstruct Q/K from quantized centroid offsets before packing tokens.
        q_video = fake_quantize_base_offset(q_video, q_labels, self.q_centroids, self.config.base_offset_bits)
        k_video = fake_quantize_base_offset(k_video, k_labels, self.k_centroids, self.config.base_offset_bits)
        q_video, q_indices = gather_clustered_tokens(q_video, q_labels, dim=2)
        k_video, k_indices = gather_clustered_tokens(k_video, k_labels, dim=2)
        v_video, _ = gather_clustered_tokens(v_video, k_labels, dim=2, sorted_indices=k_indices)

        q_perm = torch.cat((q_video, query[:, :, video_length:]), 2).contiguous()
        k_perm = torch.cat((k_video, key[:, :, video_length:]), 2).contiguous()
        v_perm = torch.cat((v_video, value[:, :, video_length:]), 2).contiguous()
        dynamic_map, q_sizes, k_sizes = extend_text_partition(
            dynamic_map, q_sizes, k_sizes, valid_text, text_length,
        )
        text_indices = torch.arange(video_length, length, device=query.device, dtype=torch.int32)
        indices = torch.cat((q_indices.reshape(batch * heads, video_length), text_indices.expand(batch * heads, -1)), -1)
        output = sparse_attention(q_perm, k_perm, v_perm, dynamic_map, q_sizes, k_sizes)
        output = restore_sequence_order(output, indices, dim=2)
        return output

    @torch.no_grad()
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        from diffusers.models.embeddings import apply_rotary_emb

        text_length = 0 if encoder_hidden_states is None else encoder_hidden_states.shape[1]
        if attn.add_q_proj is None and text_length:
            hidden_states = torch.cat((hidden_states, encoder_hidden_states), 1)

        def split_heads(x):
            return x.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        query, key, value = (split_heads(proj(hidden_states)) for proj in (attn.to_q, attn.to_k, attn.to_v))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if image_rotary_emb is not None:
            if attn.add_q_proj is None and text_length:
                query = torch.cat((apply_rotary_emb(query[:, :, :-text_length], image_rotary_emb), query[:, :, -text_length:]), 2)
                key = torch.cat((apply_rotary_emb(key[:, :, :-text_length], image_rotary_emb), key[:, :, -text_length:]), 2)
            else:
                query, key = (apply_rotary_emb(x, image_rotary_emb) for x in (query, key))
        if attn.add_q_proj is not None and text_length:
            eq, ek, ev = (split_heads(proj(encoder_hidden_states)) for proj in (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj))
            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)
            query, key, value = torch.cat((query, eq), 2), torch.cat((key, ek), 2), torch.cat((value, ev), 2)
        output = self.attention(query.contiguous(), key.contiguous(), value.contiguous())
        output = output.transpose(1, 2).flatten(2, 3).to(query.dtype)
        if not text_length:
            return output, encoder_hidden_states
        output, encoder_output = output[:, :-text_length], output[:, -text_length:]
        if getattr(attn, "to_out", None) is not None:
            output = attn.to_out[1](attn.to_out[0](output))
        if getattr(attn, "to_add_out", None) is not None:
            encoder_output = attn.to_add_out(encoder_output)
        return output, encoder_output
