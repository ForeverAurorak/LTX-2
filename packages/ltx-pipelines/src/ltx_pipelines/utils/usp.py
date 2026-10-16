"""Sequence-parallel (xDiT USP: Ulysses + Ring) multi-GPU support for ``torchrun``-launched pipelines.

Video tokens are sharded across the ranks of xDiT's sequence-parallel (SP) group inside the
transformer forward. Everything else -- noise, audio latents, text contexts, conditionings, the
VAE/upsampler -- stays replicated, so every rank holds the full latents after each denoising step
and, for a given seed, sees exactly the same noise as a single-GPU run.

Inside the transformer, with ``T`` video tokens padded to ``T_pad`` (a multiple of the SP world size):

* video tokens, timesteps and RoPE positions are sliced with the same per-rank range, so each
  rank's positional embeddings line up with its tokens;
* video self-attention (``attn1``) runs xDiT's ``xFuserLongContextAttention`` (Ulysses + Ring) when
  there is no mask. With a mask (padding, or conditioning attention masks) keys/values are
  all-gathered over the SP group instead and the mask's query rows are sliced to the local shard;
* video-to-audio attention has replicated audio queries and *sharded* video keys/values, so K/V are
  all-gathered to the global sequence (padded keys are masked out);
* audio-to-video and text cross-attention have sharded queries and replicated keys/values and run
  locally unchanged;
* video outputs are all-gathered and un-padded, so the wrapper returns the same shapes as the
  unwrapped model.

Launch (see ``ltx_pipelines.a2vid_two_stage``)::

    torchrun --nproc_per_node=N -m ltx_pipelines.a2vid_two_stage --ulysses-degree U --ring-degree R ...

Single-GPU runs (``WORLD_SIZE`` unset or 1) never import ``xfuser``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.model.transformer.attention import Attention, AttentionCallable, MaskedAttentionCallable
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
from ltx_core.multigpu.transformer.sequence_parallel import (
    gather_output_tokens,
    pad_modality_for_uniform_sharding,
    tile_modality_for_rank,
)

if TYPE_CHECKING:
    from ltx_pipelines.utils.blocks import DiffusionStage

logger = logging.getLogger(__name__)


@dataclass
class UspContext:
    """The SP group plus per-forward padding state shared by the patched attention modules."""

    group: dist.ProcessGroup
    rank: int
    world_size: int
    ulysses_degree: int = 1
    ring_degree: int = 1
    # Set by UspModelWrapper for the duration of one forward: real / padded video token counts.
    valid_tokens: int | None = None
    padded_tokens: int | None = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def has_padding(self) -> bool:
        return self.valid_tokens is not None and self.padded_tokens != self.valid_tokens


def resolve_usp_degrees(ulysses_degree: int | None, ring_degree: int, world_size: int) -> tuple[int, int]:
    """Validate ``ulysses * ring == world_size``; ``ulysses_degree=None`` means ``world_size // ring``."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if ring_degree < 1:
        raise ValueError(f"--ring-degree must be >= 1, got {ring_degree}")
    if ulysses_degree is None:
        if world_size % ring_degree != 0:
            raise ValueError(f"--ring-degree ({ring_degree}) must divide WORLD_SIZE ({world_size})")
        ulysses_degree = world_size // ring_degree
    if ulysses_degree < 1:
        raise ValueError(f"--ulysses-degree must be >= 1, got {ulysses_degree}")
    if ulysses_degree * ring_degree != world_size:
        raise ValueError(
            f"--ulysses-degree ({ulysses_degree}) * --ring-degree ({ring_degree}) must equal "
            f"WORLD_SIZE ({world_size}), got {ulysses_degree * ring_degree}"
        )
    return ulysses_degree, ring_degree


def init_usp(ulysses_degree: int | None = None, ring_degree: int = 1) -> UspContext | None:
    """Initialize xDiT's sequence-parallel groups for a ``torchrun`` launch.

    Returns ``None`` (without importing ``xfuser``) when ``WORLD_SIZE`` is 1 or unset.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ulysses_degree, ring_degree = resolve_usp_degrees(ulysses_degree, ring_degree, world_size)
    if world_size == 1:
        return None
    try:
        from xfuser.core.distributed import (  # noqa: PLC0415
            get_sp_group,
            init_distributed_environment,
            initialize_model_parallel,
        )
    except ImportError as e:
        raise RuntimeError("Multi-GPU sequence parallelism requires xDiT: `pip install xfuser`") from e

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)  # before any process group so NCCL binds to the right device
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=local_rank)
    initialize_model_parallel(
        sequence_parallel_degree=world_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
    )
    sp = get_sp_group()
    return UspContext(
        group=sp.device_group,
        rank=sp.rank_in_group,
        world_size=sp.world_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
    )


def shutdown_usp(ctx: UspContext | None) -> None:
    """Barrier all ranks, then destroy xDiT's groups and the process group. No-op without a context."""
    if ctx is None or not dist.is_initialized():
        return
    dist.barrier()
    from xfuser.core.distributed.parallel_state import (  # noqa: PLC0415
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()


# -- sequence shard / gather helpers ---------------------------------------------------------------


def padded_length(length: int, world_size: int) -> int:
    """Smallest multiple of ``world_size`` that is >= ``length``."""
    return length + (-length) % world_size


def shard_bounds(length: int, world_size: int, rank: int) -> tuple[int, int]:
    """``[start, end)`` of ``rank``'s shard of a sequence padded to a multiple of ``world_size``."""
    per_rank = padded_length(length, world_size) // world_size
    return rank * per_rank, (rank + 1) * per_rank


def shard_sequence(x: torch.Tensor, rank: int, world_size: int, dim: int = 1) -> tuple[torch.Tensor, int]:
    """Zero-pad ``dim`` to a multiple of ``world_size`` and return ``(local_shard, original_length)``."""
    length = x.shape[dim]
    pad = padded_length(length, world_size) - length
    if pad:
        pad_shape = list(x.shape)
        pad_shape[dim] = pad
        x = torch.cat([x, x.new_zeros(pad_shape)], dim=dim)
    start, end = shard_bounds(length, world_size, rank)
    return x.narrow(dim, start, end - start).contiguous(), length


def gather_sequence(
    local: torch.Tensor, original_length: int, group: dist.ProcessGroup | None = None, dim: int = 1
) -> torch.Tensor:
    """All-gather per-rank shards along ``dim`` and drop the padding added by :func:`shard_sequence`."""
    world_size = dist.get_world_size(group)
    parts = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(parts, local.contiguous(), group=group)
    return torch.cat(parts, dim=dim).narrow(dim, 0, original_length)


# -- attention -------------------------------------------------------------------------------------


def _all_gather_seq(x: torch.Tensor, ctx: UspContext) -> torch.Tensor:
    """(B, T_local, C) per rank -> (B, T_pad, C) on every rank."""
    parts = [torch.empty_like(x) for _ in range(ctx.world_size)]
    dist.all_gather(parts, x.contiguous(), group=ctx.group)
    return torch.cat(parts, dim=1)


def _padded_key_bias(ctx: UspContext, total_keys: int, like: torch.Tensor) -> torch.Tensor:
    """Additive ``(1, 1, 1, T_pad)`` bias hiding the padded video keys."""
    bias = torch.zeros(1, 1, 1, total_keys, dtype=like.dtype, device=like.device)
    if ctx.valid_tokens is None:
        raise RuntimeError("padded-key bias requested outside a UspModelWrapper forward")
    bias[..., ctx.valid_tokens :] = torch.finfo(like.dtype).min
    return bias


class UspSelfAttention(AttentionCallable):
    """Unmasked video self-attention via xDiT's Ulysses + Ring ``xFuserLongContextAttention``."""

    def __init__(self, ctx: UspContext, attn_type: str = "TORCH") -> None:
        self._ctx = ctx
        self._attn_type = attn_type
        self._usp: torch.nn.Module | None = None

    def _build(self) -> torch.nn.Module:
        from xfuser.core.long_ctx_attention import xFuserLongContextAttention  # noqa: PLC0415
        from yunchang.kernels import AttnType  # noqa: PLC0415

        return xFuserLongContextAttention(attn_type=getattr(AttnType, self._attn_type))

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
        if self._usp is None:
            self._usp = self._build()
        b, t, c = q.shape
        q, k, v = (x.view(b, t, heads, c // heads) for x in (q, k, v))
        out = self._usp(None, q, k, v)  # (B, T_local, H, D); Ulysses all2all + ring attention
        return out.reshape(b, t, c)


class UspMaskedSelfAttention(MaskedAttentionCallable):
    """Masked video self-attention: local queries, all-gathered K/V, query rows of the mask sliced."""

    def __init__(self, ctx: UspContext, original: MaskedAttentionCallable) -> None:
        self._ctx = ctx
        self._original = original

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor
    ) -> torch.Tensor:
        k, v = _all_gather_seq(k, self._ctx), _all_gather_seq(v, self._ctx)
        total, local = k.shape[1], q.shape[1]
        if mask.shape[-2] == total and total != 1:  # dense (.., T_q, T_k) mask: keep this rank's query rows
            mask = mask.narrow(-2, self._ctx.rank * local, local)
        return self._original(q, k, v, heads, mask)


class UspGatherKVAttention(AttentionCallable):
    """Video-to-audio attention: replicated audio queries, sharded video keys/values.

    K/V are gathered to the global (padded) sequence; padded keys are masked out. Every rank computes
    the identical output.
    """

    def __init__(self, ctx: UspContext, original: AttentionCallable, original_masked: MaskedAttentionCallable) -> None:
        self._ctx = ctx
        self._original = original
        self._original_masked = original_masked

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
        k, v = _all_gather_seq(k, self._ctx), _all_gather_seq(v, self._ctx)
        if not self._ctx.has_padding:
            return self._original(q, k, v, heads)
        return self._original_masked(q, k, v, heads, _padded_key_bias(self._ctx, k.shape[1], q))


class UspMaskedGatherKVAttention(MaskedAttentionCallable):
    """Masked counterpart of :class:`UspGatherKVAttention` (no current caller passes a mask)."""

    def __init__(self, ctx: UspContext, original_masked: MaskedAttentionCallable) -> None:
        self._ctx = ctx
        self._original_masked = original_masked

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor
    ) -> torch.Tensor:
        k, v = _all_gather_seq(k, self._ctx), _all_gather_seq(v, self._ctx)
        if self._ctx.has_padding:
            if not torch.is_floating_point(mask):
                raise NotImplementedError("padded video keys require an additive (floating point) mask")
            mask = mask + _padded_key_bias(self._ctx, k.shape[1], mask)
        return self._original_masked(q, k, v, heads, mask)


def create_usp_module_ops(ctx: UspContext, attn_type: str = "TORCH") -> ModuleOps:
    """ModuleOps patching video self-attention (``attn1``) and video-to-audio attention for SP."""

    def mutator(module: torch.nn.Module) -> torch.nn.Module:
        for block in module.transformer_blocks:
            if not isinstance(block, BasicAVTransformerBlock):
                continue
            attn1 = getattr(block, "attn1", None)
            if isinstance(attn1, Attention):
                attn1.attention_function = UspSelfAttention(ctx, attn_type)
                attn1.masked_attention_function = UspMaskedSelfAttention(ctx, attn1.masked_attention_function)
            v2a = getattr(block, "video_to_audio_attn", None)
            if isinstance(v2a, Attention):
                v2a.attention_function = UspGatherKVAttention(
                    ctx, v2a.attention_function, v2a.masked_attention_function
                )
                v2a.masked_attention_function = UspMaskedGatherKVAttention(ctx, v2a.masked_attention_function)
        return module

    return ModuleOps(name="usp_attention", matcher=lambda module: isinstance(module, LTXModel), mutator=mutator)


class UspModelWrapper(torch.nn.Module):
    """Shards the video modality across the SP group around ``model`` and gathers the video output."""

    def __init__(self, model: torch.nn.Module, ctx: UspContext) -> None:
        super().__init__()
        self.model = model
        self.ctx = ctx

    @property
    def num_blocks(self) -> int:
        return self.model.num_blocks

    def forward(
        self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if video is None:
            return self.model(video=video, audio=audio, perturbations=perturbations)
        ctx = self.ctx
        video, t_orig = pad_modality_for_uniform_sharding(video, ctx.world_size)
        video_tile, token_counts = tile_modality_for_rank(video, ctx.rank, ctx.world_size)
        ctx.valid_tokens, ctx.padded_tokens = t_orig, sum(token_counts)
        try:
            video_out, audio_out = self.model(video=video_tile, audio=audio, perturbations=perturbations)
        finally:
            ctx.valid_tokens = ctx.padded_tokens = None
        if video_out is not None:
            video_out = gather_output_tokens(video_out, token_counts, ctx.group)[:, :t_orig]
        return video_out, audio_out


def enable_usp(stage: DiffusionStage, ctx: UspContext, attn_type: str = "TORCH") -> DiffusionStage:
    """Return a copy of a :class:`~ltx_pipelines.utils.blocks.DiffusionStage` running with SP."""
    heads = stage._transformer_builder.model_config().get("transformer", {}).get("num_attention_heads")
    if heads is not None and heads % ctx.ulysses_degree != 0:
        raise ValueError(f"num_attention_heads ({heads}) must be divisible by --ulysses-degree ({ctx.ulysses_degree})")
    builder = stage._transformer_builder
    builder = builder.with_module_ops((*builder.module_ops, create_usp_module_ops(ctx, attn_type)))
    return stage.with_builder(builder).with_model_wrapper(lambda model, _tools: UspModelWrapper(model, ctx))


__all__ = [
    "UspContext",
    "UspModelWrapper",
    "create_usp_module_ops",
    "enable_usp",
    "gather_sequence",
    "init_usp",
    "padded_length",
    "resolve_usp_degrees",
    "shard_bounds",
    "shard_sequence",
    "shutdown_usp",
]
