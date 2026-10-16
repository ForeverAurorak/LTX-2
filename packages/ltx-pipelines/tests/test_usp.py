"""Tests for xDiT-USP sequence parallelism helpers (CPU / gloo; no GPU or xfuser needed)."""

import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ltx_core.model.transformer.attention import PytorchAttention
from ltx_core.model.transformer.modality import Modality
from ltx_pipelines.a2vid_two_stage import build_a2vid_arg_parser, resolve_a2vid_cli_duration
from ltx_pipelines.utils.constants import PipelineParams
from ltx_pipelines.utils.usp import (
    UspContext,
    UspGatherKVAttention,
    UspMaskedGatherKVAttention,
    UspMaskedSelfAttention,
    UspModelWrapper,
    UspSelfAttention,
    gather_sequence,
    init_usp,
    padded_length,
    resolve_usp_degrees,
    shard_bounds,
    shard_sequence,
)

HEADS, DIM = 2, 8


# -- degree validation / CLI ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ulysses", "ring", "world", "expected"),
    [(None, 1, 4, (4, 1)), (None, 2, 4, (2, 2)), (2, 2, 4, (2, 2)), (1, 1, 1, (1, 1)), (4, 1, 4, (4, 1))],
)
def test_resolve_usp_degrees_ok(ulysses: int | None, ring: int, world: int, expected: tuple[int, int]) -> None:
    assert resolve_usp_degrees(ulysses, ring, world) == expected


@pytest.mark.parametrize(
    ("ulysses", "ring", "world"),
    [(2, 1, 4), (3, 2, 4), (None, 3, 4), (0, 1, 1), (1, 0, 1), (2, 1, 1)],
)
def test_resolve_usp_degrees_rejects(ulysses: int | None, ring: int, world: int) -> None:
    with pytest.raises(ValueError, match=r"ulysses|ring|WORLD_SIZE|world_size"):
        resolve_usp_degrees(ulysses, ring, world)


def test_init_usp_single_gpu_is_noop_and_does_not_import_xfuser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert init_usp() is None
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert init_usp(1, 1) is None
    assert "xfuser" not in sys.modules
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        init_usp(2, 1)


def test_init_usp_validates_before_importing_xfuser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(ValueError, match="must equal WORLD_SIZE"):
        init_usp(2, 1)


def test_a2vid_duration_resolution_unchanged() -> None:
    assert resolve_a2vid_cli_duration(
        num_frames=None, audio_max_duration=2.0, frame_rate=24.0, default_num_frames=121
    ) == (None, 2.0)
    assert resolve_a2vid_cli_duration(
        num_frames=None, audio_max_duration=None, frame_rate=24.0, default_num_frames=48
    ) == (
        48,
        2.0,
    )
    with pytest.raises(ValueError, match="not allowed with"):
        resolve_a2vid_cli_duration(num_frames=9, audio_max_duration=1.0, frame_rate=24.0, default_num_frames=121)


def test_a2vid_parser_has_usp_flags() -> None:
    parser = build_a2vid_arg_parser(PipelineParams())
    actions = {opt for a in parser._actions for opt in a.option_strings}
    assert {"--ulysses-degree", "--ring-degree", "--num-frames", "--audio-max-duration"} <= actions


# -- shard / gather helpers (no process group) ------------------------------------------------------


@pytest.mark.parametrize(("length", "world"), [(6, 2), (5, 2), (7, 3), (1, 4), (13, 4), (8, 1)])
def test_shard_covers_exact_global_indices_and_roundtrips(length: int, world: int) -> None:
    x = torch.arange(length).view(1, length, 1).expand(2, length, 3).clone()
    shards = [shard_sequence(x, r, world)[0] for r in range(world)]
    assert all(s.shape[1] == padded_length(length, world) // world for s in shards)
    cat = torch.cat(shards, dim=1)
    assert torch.equal(cat[:, :length], x)  # shards in rank order cover exactly indices 0..length-1
    assert torch.count_nonzero(cat[:, length:]) == 0  # padding is zeros
    for r in range(world):
        start, end = shard_bounds(length, world, r)
        assert (start, end) == (r * shards[0].shape[1], (r + 1) * shards[0].shape[1])
        assert torch.equal(shards[r], cat[:, start:end])


# -- multi-process (gloo) ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _launch(fn, world: int, *args) -> None:  # noqa: ANN001
    port = _free_port()
    mp.spawn(_worker, args=(world, port, fn, *args), nprocs=world, join=True)


def _worker(rank: int, world: int, port: int, fn, *args) -> None:  # noqa: ANN001
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        fn(rank, world, *args)
    finally:
        dist.destroy_process_group()


def _gather_roundtrip(rank: int, world: int, length: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, length, 3)
    local, orig = shard_sequence(x, rank, world)
    assert torch.equal(gather_sequence(local, orig), x)


@pytest.mark.parametrize(("world", "length"), [(2, 5), (2, 6), (3, 7)])
def test_gather_sequence_roundtrip_gloo(world: int, length: int) -> None:
    _launch(_gather_roundtrip, world, length)


class _FakeUsp(torch.nn.Module):
    """Stands in for xFuserLongContextAttention (needs GPU): same (attn, q, k, v) -> (B, T_local, H, D) contract."""

    def __init__(self, ctx: UspContext) -> None:
        super().__init__()
        self.ctx = ctx

    def forward(self, _attn: None, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ks = [torch.empty_like(k) for _ in range(self.ctx.world_size)]
        vs = [torch.empty_like(v) for _ in range(self.ctx.world_size)]
        dist.all_gather(ks, k.contiguous(), group=self.ctx.group)
        dist.all_gather(vs, v.contiguous(), group=self.ctx.group)
        k, v = torch.cat(ks, dim=1), torch.cat(vs, dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)))
        return out.transpose(1, 2)


def _bias(mask: torch.Tensor | None) -> torch.Tensor | None:
    """[0, 1] mask -> additive bias, as TransformerArgsPreprocessor._prepare_self_attention_mask."""
    if mask is None:
        return None
    bias = torch.full_like(mask, torch.finfo(torch.float32).min)
    bias[mask > 0] = torch.log(mask[mask > 0])
    return bias.unsqueeze(1)


class _FakeAVModel(torch.nn.Module):
    """Video self-attn (position-dependent input) + video-to-audio attn, built from attention callables."""

    num_blocks = 1

    def __init__(self, attn, masked_attn, v2a, masked_v2a) -> None:  # noqa: ANN001
        super().__init__()
        torch.manual_seed(1)
        self.vproj, self.aproj = torch.nn.Linear(DIM, DIM), torch.nn.Linear(DIM, DIM)
        self.attn, self.masked_attn, self.v2a, self.masked_v2a = attn, masked_attn, v2a, masked_v2a

    def forward(self, video: Modality, audio: Modality, perturbations: None) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ARG002
        x = self.vproj(video.latent) + video.positions[:, 0, :, 0].unsqueeze(-1)  # RoPE-like position dependence
        mask = _bias(video.attention_mask)
        out = self.attn(x, x, x, HEADS) if mask is None else self.masked_attn(x, x, x, HEADS, mask)
        a_out = self.v2a(self.aproj(audio.latent), x, x, HEADS)
        return out, a_out


def _modality(t: int, dim: int, mask: torch.Tensor | None, seed: int) -> Modality:
    g = torch.Generator().manual_seed(seed)
    pos = torch.arange(t, dtype=torch.float32).view(1, 1, t, 1).expand(1, 3, t, 2).clone()
    return Modality(
        latent=torch.randn(1, t, dim, generator=g),
        sigma=torch.ones(1),
        timesteps=torch.zeros(1, t),
        positions=pos,
        context=torch.zeros(1, 1, dim),
        attention_mask=mask,
    )


def _wrapper_matches_single_rank(rank: int, world: int, t: int, with_mask: bool) -> None:
    ctx = UspContext(group=dist.group.WORLD, rank=rank, world_size=world, ulysses_degree=world)
    g = torch.Generator().manual_seed(7)
    mask = (torch.rand(1, t, t, generator=g) > 0.3).float() if with_mask else None
    if mask is not None:
        mask = mask.clone()
        mask.diagonal(dim1=1, dim2=2).fill_(1.0)
    video, audio = _modality(t, DIM, mask, 3), _modality(4, DIM, None, 4)

    ref = PytorchAttention()
    ref_model = _FakeAVModel(ref, ref, ref, ref)
    expected_v, expected_a = ref_model(video, audio, None)

    usp_attn = UspSelfAttention(ctx)
    usp_attn._usp = _FakeUsp(ctx)
    sp_model = _FakeAVModel(
        usp_attn,
        UspMaskedSelfAttention(ctx, ref),
        UspGatherKVAttention(ctx, ref, ref),
        UspMaskedGatherKVAttention(ctx, ref),
    )
    out_v, out_a = UspModelWrapper(sp_model, ctx)(video, audio, None)
    assert out_v.shape == expected_v.shape
    dv, da = (out_v - expected_v).abs().max().item(), (out_a - expected_a).abs().max().item()
    if rank == 0:
        print(f"world={world} T={t} mask={with_mask}: max|dv|={dv:.2e} max|da|={da:.2e}")  # noqa: T201
    assert dv < 1e-5, dv
    assert da < 1e-5, da
    assert ctx.valid_tokens is None  # per-forward padding state is reset


@pytest.mark.parametrize(("world", "t"), [(2, 6), (2, 5), (3, 7), (3, 9), (4, 6)])
@pytest.mark.parametrize("with_mask", [False, True])
def test_wrapper_matches_single_rank_gloo(world: int, t: int, with_mask: bool) -> None:
    """Sharded video attention + gathered-KV video-to-audio match the unsharded run, with/without padding."""
    _launch(_wrapper_matches_single_rank, world, t, with_mask)
