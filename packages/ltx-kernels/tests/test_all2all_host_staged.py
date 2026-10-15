"""Byte-exact comparison of the CUDA IPC path and the host-staged path of ``All2All``.

Requires two CUDA GPUs with peer access and a built ``all2all_cpp`` extension. Each rank
runs in its own process (``torch.multiprocessing.spawn``) and joins an NCCL group.

Every rank builds two ``All2All`` instances in the same order on all ranks:
  * ``ipc``    -- default constructor, uses CUDA IPC.
  * ``staged`` -- ``force_host_staging=True``, uses D2H -> gloo -> H2D.

Checks (all byte-exact):
  * ``send_recv_heads`` output is identical on both paths.
  * ``gather_heads(send_recv_heads(x))`` returns ``x`` on both paths.
  * ``allgather`` output is identical on both paths.
"""

import socket

import pytest
import torch
import torch.distributed as dist

all2all_cpp = pytest.importorskip("all2all_cpp")

WORLD_SIZE = 2
BATCH, TOKENS, HEADS, HEAD_DIM = 2, 16, 8, 64


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    # Compare raw bytes so the check is exact for every dtype, including float8.
    return t.contiguous().view(torch.uint8)


def _assert_bytes_equal(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
    assert a.shape == b.shape, f"{what}: shape {tuple(a.shape)} != {tuple(b.shape)}"
    assert torch.equal(_as_bytes(a), _as_bytes(b)), f"{what}: outputs differ"


def _worker(rank: int, dtype_name: str, port: int) -> None:
    from ltx_kernels.all_to_all import All2All

    dtype = getattr(torch, dtype_name)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    try:
        num_sms = torch.cuda.get_device_properties(rank).multi_processor_count
        hidden_dim = HEADS * HEAD_DIM
        # Buffer must hold the largest output (allgather yields TOKENS * WORLD_SIZE tokens).
        max_tokens = TOKENS * WORLD_SIZE

        # Construction order must match on every rank: each instance performs collectives.
        ipc = All2All(rank, WORLD_SIZE, max_tokens, hidden_dim, num_sms, dtype)
        staged = All2All(rank, WORLD_SIZE, max_tokens, hidden_dim, num_sms, dtype, force_host_staging=True)
        assert not ipc.uses_host_staging
        assert staged.uses_host_staging

        gen = torch.Generator(device="cuda").manual_seed(1000 + rank)
        x = torch.randn(BATCH, TOKENS, HEADS, HEAD_DIM, device=f"cuda:{rank}", generator=gen).to(dtype)

        # send_recv_heads: identical on both paths.
        redist_ipc = ipc.send_recv_heads(x, copy_out=True)
        redist_staged = staged.send_recv_heads(x)
        _assert_bytes_equal(redist_ipc, redist_staged, "send_recv_heads")

        # gather_heads: the round trip restores x on both paths.
        back_ipc = ipc.gather_heads(redist_ipc, copy_out=True)
        back_staged = staged.gather_heads(redist_staged)
        _assert_bytes_equal(back_ipc, x, "gather_heads(IPC) round trip")
        _assert_bytes_equal(back_staged, x, "gather_heads(staged) round trip")

        # allgather: identical on both paths.
        ag_ipc = ipc.allgather(x, copy_out=True)
        ag_staged = staged.allgather(x)
        _assert_bytes_equal(ag_ipc, ag_staged, "allgather")

        dist.barrier()
        staged.destroy()
        ipc.destroy()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float8_e4m3fn"])
def test_host_staged_matches_ipc(dtype_name: str) -> None:
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"requires at least {WORLD_SIZE} CUDA devices")
    if not hasattr(torch, dtype_name):
        pytest.skip(f"torch has no {dtype_name}")

    torch.multiprocessing.spawn(
        _worker,
        args=(dtype_name, _free_port()),
        nprocs=WORLD_SIZE,
        join=True,
    )
