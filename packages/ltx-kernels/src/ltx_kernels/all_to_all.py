"""All2All communication primitives for distributed attention.
The C++ kernels are exposed via ``torch.library.custom_op`` so that
``torch.compile`` (and CUDA Graph capture under ``mode="reduce-overhead"``)
can trace through them without a graph break. Each :class:`All2All`
instance registers itself in a class-level registry indexed by an integer
``comm_id``; the custom op takes that id, an input tensor, and the instance's
IPC buffer, and dispatches to the appropriate C++ runtime.

When peer device access is unavailable on any rank (no NVLink/PCIe P2P), CUDA IPC
cannot be opened. In that case :class:`All2All` runs ``allgather`` through host
memory instead (D2H -> gloo exchange -> H2D). The decision is made collectively, so
every rank takes the same path. ``send_recv_heads`` and ``gather_heads`` still
require P2P.
"""

import math
import weakref
from typing import Any, ClassVar

import torch
import torch.distributed as dist
from all2all_cpp import All2All as All2AllCpp
from torch.library import custom_op

# Output shapes and kernel offsets are derived from the input tensor under uniform
# sharding. Callers pad before sharding, so every rank has the same token count.
# Each op writes its result into the instance's local IPC buffer, which it takes as a
# mutated argument, and returns nothing; All2All returns a view of that buffer. Returning
# the view from the op instead would declare storage the compiler may reuse for other
# tensors, while the next op on the same instance, from any rank, overwrites it.


@custom_op("ltx_kernels::send_recv_heads", mutates_args=("buffer",), device_types="cuda")
def _send_recv_heads_op(x: torch.Tensor, buffer: torch.Tensor, comm_id: int) -> None:  # noqa: ARG001
    All2All._runtime_registry[comm_id].send_recv_heads(x, False)


@_send_recv_heads_op.register_fake
def _send_recv_heads_fake(x: torch.Tensor, buffer: torch.Tensor, comm_id: int) -> None:  # noqa: ARG001
    return None


@custom_op("ltx_kernels::gather_heads", mutates_args=("buffer",), device_types="cuda")
def _gather_heads_op(x: torch.Tensor, buffer: torch.Tensor, comm_id: int) -> None:  # noqa: ARG001
    All2All._runtime_registry[comm_id].gather_heads(x, False)


@_gather_heads_op.register_fake
def _gather_heads_fake(x: torch.Tensor, buffer: torch.Tensor, comm_id: int) -> None:  # noqa: ARG001
    return None


def _local_peer_access_ok(rank: int, world_size: int) -> bool:
    """True if this rank's GPU can access every other rank's GPU directly (P2P)."""
    if not torch.cuda.is_available():
        return False
    return all(torch.cuda.can_device_access_peer(rank, peer) for peer in range(world_size) if peer != rank)


class All2All:
    """IPC-based All2All communication for distributed head-parallel attention.
    This class manages GPU memory buffers and IPC handles to enable efficient
    cross-GPU communication for attention head redistribution. If any rank lacks
    peer access, ``allgather`` falls back to a host-staged path and the head
    redistribution ops are unavailable.
    Args:
        rank: Local rank of this process.
        world_size: Total number of processes in the distributed group.
        seqlen: Maximum sequence length to allocate buffers for.
        hidden_dim: Hidden dimension size (num_heads * head_dim).
        num_sms: Number of SMs to use for kernel execution.
        tensor_dtype: Data type for tensors (e.g., torch.bfloat16).
        group: PyTorch distributed process group.
    """

    _next_id: ClassVar[int] = 0
    _runtime_registry: ClassVar[dict[int, "All2AllCpp"]] = {}

    @staticmethod
    def _release_comm(comm_id: int, runtime: "All2AllCpp") -> None:
        All2All._runtime_registry.pop(comm_id, None)
        runtime.destroy()

    def __init__(
        self,
        rank: int,
        world_size: int,
        seqlen: int,
        hidden_dim: int,
        num_sms: int,
        tensor_dtype: torch.dtype,
        group: torch.distributed.ProcessGroup | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.num_sms = num_sms
        self.tensor_dtype = tensor_dtype
        self.buffer_size = int(seqlen * hidden_dim * tensor_dtype.itemsize)

        # Initialize the C++ runtime. Omit timeout_seconds to keep the kernel's default
        # (DEFAULT_BARRIER_TIMEOUT_SECONDS); it can still be changed later via set_timeout_seconds.
        if timeout_seconds is None:
            self.runtime = All2AllCpp(rank, world_size, seqlen, hidden_dim, num_sms, tensor_dtype)
        else:
            if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
                raise ValueError(f"all2all timeout seconds must be finite and non-negative, got {timeout_seconds}")
            self.runtime = All2AllCpp(rank, world_size, seqlen, hidden_dim, num_sms, tensor_dtype, timeout_seconds)

        # Register in the class-level table so the custom ops can find us.
        # Auto-cleanup via weakref.finalize covers the case where destroy()
        # isn't called explicitly — without it the registry would keep the
        # runtime (and CUDA/IPC resources) alive for the process lifetime.
        self._comm_id = All2All._next_id
        All2All._next_id += 1
        All2All._runtime_registry[self._comm_id] = self.runtime
        self._finalizer = weakref.finalize(self, All2All._release_comm, self._comm_id, self.runtime)

        # Exchange IPC handles and peer-access capability across all ranks. The capability
        # flag is gathered too, so every rank picks the same path.
        local_handle = self.runtime.get_local_ipc_handle()
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, (_local_peer_access_ok(rank, world_size), local_handle), group)
        self._staged = not all(ok for ok, _ in gathered)

        if self._staged:
            # IPC mappings cannot be opened without peer access, so skip sync() and the
            # IPC buffer. allgather goes through host memory on a CPU (gloo) group.
            self._buffer = None
            pg = group if group is not None else dist.group.WORLD
            self._cpu_group = dist.new_group(ranks=dist.get_process_group_ranks(pg), backend="gloo")
        else:
            self.runtime.sync([handle for _, handle in gathered])
            self._buffer = self.runtime.local_buffer()

    @property
    def uses_host_staging(self) -> bool:
        """True when allgather runs through host memory (no peer access on some rank)."""
        return self._staged

    def set_timeout_seconds(self, seconds: float) -> None:
        """Set the all2all barrier (deadlock-detection) timeout, in seconds.
        The C++ runtime converts to clock cycles using the device's peak SM clock. Raise it
        during the first ``torch.compile`` forward, where one rank's recompile can delay its
        all2all launch past the steady-state timeout; reset for steady state.
        """
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"all2all timeout seconds must be finite and non-negative, got {seconds}")
        self.runtime.set_timeout_seconds(seconds)

    def _require_p2p(self, op: str) -> None:
        if self._staged:
            raise RuntimeError(
                f"All2All.{op} requires peer (P2P) access, which is unavailable on at least one rank. "
                "Only allgather has a host-staged fallback."
            )

    def send_recv_heads(self, x: torch.Tensor, *, copy_out: bool = False) -> torch.Tensor:
        """Exchange attention heads across ranks (All2All pattern).
        Args:
            x: Input tensor of shape [batch, tokens, heads, head_dim].
            copy_out: If True, copy result to a new tensor instead of using buffer.
        Returns:
            Output tensor with redistributed heads. Unless ``copy_out``, a view of the IPC
            buffer that the next call on this instance overwrites.
        """
        self._require_p2p("send_recv_heads")
        torch.ops.ltx_kernels.send_recv_heads(x, self._buffer, self._comm_id)
        batch, tokens, heads, head_dim = x.shape
        out = self._buffer[: x.numel()].view(batch, tokens * self.world_size, heads // self.world_size, head_dim)
        return out.clone() if copy_out else out

    def gather_heads(self, x: torch.Tensor, *, copy_out: bool = False) -> torch.Tensor:
        """Gather heads back to original distribution (reverse All2All).
        Args:
            x: Input tensor with distributed heads.
            copy_out: If True, copy result to a new tensor instead of using buffer.
        Returns:
            Output tensor with gathered heads. Unless ``copy_out``, a view of the IPC buffer
            that the next call on this instance overwrites.
        """
        self._require_p2p("gather_heads")
        torch.ops.ltx_kernels.gather_heads(x, self._buffer, self._comm_id)
        batch, tokens, heads, head_dim = x.shape
        out = self._buffer[: x.numel()].view(batch, tokens // self.world_size, heads * self.world_size, head_dim)
        return out.clone() if copy_out else out

    def allgather(self, x: torch.Tensor, *, copy_out: bool = False) -> torch.Tensor:
        """Allgather operation across all ranks.
        Uses CUDA IPC when peer access is available. Otherwise stages through host memory
        (D2H -> gloo all_gather -> H2D). The host path always returns a new tensor.
        Args:
            x: Input tensor of shape [batch, seqlen, heads, head_dim], contiguous, on this rank's GPU.
            copy_out: If True, copy result to a new tensor instead of using buffer.
        Returns:
            Gathered tensor of shape [batch, seqlen * world_size, heads, head_dim].
        """
        if self._staged:
            return self._allgather_host_staged(x)
        return self.runtime.allgather(x, copy_out)

    def _allgather_host_staged(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or not x.is_contiguous() or not x.is_cuda or x.dtype != self.tensor_dtype:
            raise ValueError("host-staged allgather expects a contiguous 4-D CUDA tensor of the configured dtype")
        if x.device.index != self.rank:
            raise ValueError(f"input must live on cuda:{self.rank}, got {x.device}")

        batch, seqlen, heads, head_dim = x.shape
        # D2H: transport raw bytes so every dtype (including float8) goes through gloo.
        host = x.reshape(-1).view(torch.uint8).cpu()
        gathered = [torch.empty_like(host) for _ in range(self.world_size)]
        dist.all_gather(gathered, host, group=self._cpu_group)

        # Reassemble to [batch, world * seqlen, heads, head_dim]: rank r's tokens go at
        # offset r * seqlen, matching the IPC allgather kernel layout.
        parts = torch.stack([g.view(x.dtype).view(x.shape) for g in gathered])  # [world, B, S, H, D]
        out = parts.permute(1, 0, 2, 3, 4).reshape(batch, self.world_size * seqlen, heads, head_dim)
        # H2D
        return out.to(x.device)

    def destroy(self) -> None:
        """Release IPC handles and GPU memory buffers."""
        self._finalizer()
