"""All2All communication primitives for distributed attention.
The C++ kernels are exposed via ``torch.library.custom_op`` so that
``torch.compile`` (and CUDA Graph capture under ``mode="reduce-overhead"``)
can trace through them without a graph break. Each :class:`All2All`
instance registers itself in a class-level registry indexed by an integer
``comm_id``; the custom op takes that id, an input tensor, and the instance's
IPC buffer, and dispatches to the appropriate C++ runtime.

When peer device access is unavailable on any rank (no NVLink/PCIe P2P), CUDA IPC
cannot be opened. In that case :class:`All2All` runs every operation through host
memory instead: D2H -> gloo collective -> H2D. The decision is made collectively, so
every rank takes the same path. Host-staged ops bypass the custom ops, so they are not
traced by ``torch.compile`` and are not CUDA-Graph captured.
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
    peer access (or ``force_host_staging`` is set on any rank), all operations run
    through host memory instead.
    Args:
        rank: Local rank of this process.
        world_size: Total number of processes in the distributed group.
        seqlen: Maximum sequence length to allocate buffers for.
        hidden_dim: Hidden dimension size (num_heads * head_dim).
        num_sms: Number of SMs to use for kernel execution.
        tensor_dtype: Data type for tensors (e.g., torch.bfloat16).
        group: PyTorch distributed process group.
        timeout_seconds: Optional initial barrier timeout (IPC path only).
        force_host_staging: Use the host-staged path even when P2P is available.
            Intended for testing; must be consistent across ranks (any rank setting it
            switches the whole group).
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
        force_host_staging: bool = False,
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

        # Exchange IPC handles, peer-access capability and the force flag across all ranks.
        # Gathering everything in one collective keeps the path decision identical on every rank.
        local_handle = self.runtime.get_local_ipc_handle()
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(
            gathered, (_local_peer_access_ok(rank, world_size), bool(force_host_staging), local_handle), group
        )
        all_p2p = all(ok for ok, _, _ in gathered)
        any_forced = any(forced for _, forced, _ in gathered)
        self._staged = any_forced or not all_p2p

        if self._staged:
            # IPC mappings are not opened on this path. Host-staged ops use a CPU (gloo)
            # group. The group is created here, on every rank, so the collective is matched.
            self._buffer = None
            pg = group if group is not None else dist.group.WORLD
            self._cpu_group = dist.new_group(ranks=dist.get_process_group_ranks(pg), backend="gloo")
        else:
            self.runtime.sync([handle for _, _, handle in gathered])
            self._buffer = self.runtime.local_buffer()

    @property
    def uses_host_staging(self) -> bool:
        """True when operations run through host memory instead of CUDA IPC."""
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

    def _host_all_to_all(self, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
        """Exchange equally-shaped chunks through host memory.

        ``chunks[d]`` is sent to rank ``d``. The returned list holds, at index ``s``, the
        chunk that rank ``s`` sent to this rank. All chunks share one shape, and every
        tensor is moved as raw bytes so that every dtype (including float8) works.
        """
        shape = chunks[0].shape
        device = chunks[0].device
        send = [c.contiguous().reshape(-1).view(torch.uint8).cpu() for c in chunks]
        recv = [torch.empty_like(s) for s in send]
        dist.all_to_all(recv, send, group=self._cpu_group)  # D2H above, H2D below
        return [r.view(self.tensor_dtype).view(shape).to(device) for r in recv]

    def _host_allgather(self, x: torch.Tensor) -> torch.Tensor:
        host = x.contiguous().reshape(-1).view(torch.uint8).cpu()
        gathered = [torch.empty_like(host) for _ in range(self.world_size)]
        dist.all_gather(gathered, host, group=self._cpu_group)
        parts = [g.view(self.tensor_dtype).view(x.shape).to(x.device) for g in gathered]
        return torch.cat(parts, dim=1)

    def _validate(self, x: torch.Tensor, op: str) -> None:
        if x.dim() != 4 or not x.is_contiguous() or not x.is_cuda or x.dtype != self.tensor_dtype:
            raise ValueError(f"{op} expects a contiguous 4-D CUDA tensor of dtype {self.tensor_dtype}")
        if x.device.index != self.rank:
            raise ValueError(f"{op} input must live on cuda:{self.rank}, got {x.device}")

    def send_recv_heads(self, x: torch.Tensor, *, copy_out: bool = False) -> torch.Tensor:
        """Exchange attention heads across ranks (All2All pattern).
        Args:
            x: Input tensor of shape [batch, tokens, heads, head_dim].
            copy_out: If True, copy result to a new tensor instead of using buffer.
        Returns:
            Output tensor with redistributed heads. Unless ``copy_out``, a view of the IPC
            buffer that the next call on this instance overwrites. The host-staged path
            always returns a new tensor.
        """
        if self._staged:
            return self._send_recv_heads_host(x)
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
            that the next call on this instance overwrites. The host-staged path always
            returns a new tensor.
        """
        if self._staged:
            return self._gather_heads_host(x)
        torch.ops.ltx_kernels.gather_heads(x, self._buffer, self._comm_id)
        batch, tokens, heads, head_dim = x.shape
        out = self._buffer[: x.numel()].view(batch, tokens // self.world_size, heads * self.world_size, head_dim)
        return out.clone() if copy_out else out

    def allgather(self, x: torch.Tensor, *, copy_out: bool = False) -> torch.Tensor:
        """Allgather operation across all ranks.
        Args:
            x: Input tensor of shape [batch, seqlen, heads, head_dim].
            copy_out: If True, copy result to a new tensor instead of using buffer.
        Returns:
            Gathered tensor of shape [batch, seqlen * world_size, heads, head_dim].
        """
        if self._staged:
            self._validate(x, "allgather")
            return self._host_allgather(x)
        return self.runtime.allgather(x, copy_out)

    def _send_recv_heads_host(self, x: torch.Tensor) -> torch.Tensor:
        self._validate(x, "send_recv_heads")
        heads = x.size(2)
        if heads % self.world_size != 0:
            raise ValueError(f"heads ({heads}) must be divisible by world_size ({self.world_size})")
        hpr = heads // self.world_size
        # Send to rank d the heads [d*hpr:(d+1)*hpr] of every local token.
        chunks = [x[:, :, d * hpr : (d + 1) * hpr, :] for d in range(self.world_size)]
        recv = self._host_all_to_all(chunks)  # recv[s]: [B, T, hpr, D] from rank s
        # Rank s's tokens go at offset s * T, matching the IPC kernel layout.
        return torch.cat(recv, dim=1)

    def _gather_heads_host(self, x: torch.Tensor) -> torch.Tensor:
        self._validate(x, "gather_heads")
        total_tokens = x.size(1)
        if total_tokens % self.world_size != 0:
            raise ValueError(f"tokens ({total_tokens}) must be divisible by world_size ({self.world_size})")
        tgt = total_tokens // self.world_size
        # Send to rank d its token slice [d*tgt:(d+1)*tgt] (all local heads).
        chunks = [x[:, d * tgt : (d + 1) * tgt] for d in range(self.world_size)]
        recv = self._host_all_to_all(chunks)  # recv[s]: [B, tgt, hpr, D] from rank s
        # Rank s's heads go at offset s * hpr, matching the IPC kernel layout.
        return torch.cat(recv, dim=2)

    def destroy(self) -> None:
        """Release IPC handles and GPU memory buffers."""
        self._finalizer()
