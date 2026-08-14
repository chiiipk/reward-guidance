"""Dual-timestep embedding patch for FLUX Flow Map LoRA inference.

Adapted from Holderrieth et al.'s Diamond Maps repository
(https://github.com/PeterHolderrieth/diamond_maps), file
weighted_diamond_maps/utils/dual_time_embedder.py. The Flow Map LoRA was
fine-tuned with this wrapper in place; loading the LoRA without the patch
will give garbage results. The forwarding signature also supports Diffusers
models configured without guidance embeddings.
"""

from __future__ import annotations

import copy
from typing import Any

import torch


class DualTimeEmbedder(torch.nn.Module):
    """Wrap a FLUX time-text embedder so it can accept `[t, t_next]` inputs."""

    def __init__(self, original_embedder: Any):
        super().__init__()
        self.original_embedder = original_embedder
        self.second_embedder = copy.deepcopy(original_embedder)

    def forward(self, timestep, *embedder_args, **embedder_kwargs):
        """Forward either FLUX time-text embedder signature.

        Diffusers calls the guidance-aware embedder as
        ``(timestep, guidance, pooled_projections)`` and the regular embedder as
        ``(timestep, pooled_projections)``. Forwarding the remaining arguments
        verbatim keeps the dual-time patch compatible with both configs.
        """
        if len(timestep.shape) >= 2 and timestep.shape[-1] == 2:
            t1, t2 = timestep.unbind(dim=-1)
            emb1 = self.original_embedder(t1, *embedder_args, **embedder_kwargs)
            emb2 = self.second_embedder(t2, *embedder_args, **embedder_kwargs)
            return (emb1 + emb2) / 2
        return self.original_embedder(timestep, *embedder_args, **embedder_kwargs)


def add_dual_time_embedder(single_time_flux_transformer):
    """Patch a FLUX transformer in-place for two-timestep Flow Map inputs."""
    single_time_flux_transformer.time_text_embed = DualTimeEmbedder(
        single_time_flux_transformer.time_text_embed
    )
    return single_time_flux_transformer
