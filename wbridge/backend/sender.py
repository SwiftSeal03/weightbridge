"""Trainer Worker side of WeightBridge: :class:`WeightSender` coordinates HTTP with a :class:`WeightReceiverController`.

A Trainer Worker rank joins a merged process group with Rollout Workers, posts ``/wbridge/connect`` and
``/wbridge/receive`` on the Rollout Engine HTTP APIs (from rank 0), then exchanges tensor chunks in
P2P rounds defined by a shared :class:`~wbridge.backend.router.WeightRouter`. :meth:`WeightSender.send`
packs local shards via :attr:`WeightSender.save_weights` and issues ``isend``; behavior matches
:meth:`WeightReceiver._receive_weights` on the peer side. See :class:`SenderArgs` for transport
options.
"""

from collections.abc import Callable
from dataclasses import dataclass

import requests
import torch
import torch.distributed as dist

from wbridge.utils.data import ShardSpec
from wbridge.backend.router import WBEndpoint


@dataclass
class SenderArgs:
    """Transport args forwarded to :class:`WeightSender`.

    Attributes:
        world_size: Number of Trainer Worker ranks participating in the process group.
        transfer_mode: ``"gpu_direct"`` (NCCL, CUDA wire buffers) or ``"cpu_direct"``
            (Gloo, CPU wire buffers; :meth:`~WeightSender.send` may return before ``isend`` completes).
        receiver_urls: HTTP base URLs of the Rollout Engine controllers, one per rollout engine.
        master_addr: Host/IP of the Trainer Worker rank-0 process used for rendezvous.
        master_port: TCP port for the rendezvous group.
    """

    world_size: int
    transfer_mode: str
    receiver_urls: list[str]
    master_addr: str
    master_port: int


class WeightSender(WBEndpoint):
    """Packs and P2P-sends sharded weights from Trainer Workers to :class:`WeightReceiver` peers.

    Rank 0 must call :meth:`connect` before :meth:`send`; that establishes the process group
    (and rank-0 only drives the Rollout Engine HTTP endpoints). Each round, :attr:`save_weights` fills
    the logical chunk buffers that :attr:`load_weights` on the receiver will consume (same
    :class:`~wbridge.utils.data.ShardSpec` layout).
    """

    def __init__(
        self,
        args: SenderArgs,
        rank: int,
        shard_spec: ShardSpec,
        save_weights: Callable[[ShardSpec, dict[str, torch.Tensor]], None],
    ) -> None:
        self.cuda_device = f"cuda:{torch.cuda.current_device()}"

        self.transfer_mode = args.transfer_mode
        self.receiver_urls = args.receiver_urls
        self.world_size = args.world_size
        self.init_method = f"tcp://{args.master_addr}:{args.master_port}"

        self.rank = rank
        self.shard_spec = shard_spec
        self.dtype_spec = {}
        self.save_weights = save_weights

        self.connected = False
        self._pending_comm_handles: list = []

    def _drain_pending_comm(self) -> None:
        for h in self._pending_comm_handles:
            h.wait()
        self._pending_comm_handles.clear()


    def connect(self) -> None:
        """Rendezvous senders, query receiver world sizes, then form the merged P2P process group.

        Uses :attr:`init_method` (``tcp://{master_addr}:{master_port}``) from :class:`SenderArgs`
        so all Trainer Worker ranks align before rank 0 calls each ``receiver_urls`` host’s
        ``/wbridge/receiver_world`` and ``/wbridge/connect`` with a contiguous ``rank`` base
        for Rollout Workers. :meth:`set_up_connection` is then called on every Trainer Worker rank with
        the same ``init_method``, total ``world_size`` (trainers + all Rollout Workers), and
        transfer mode.
        """
        self._drain_pending_comm()

        if self.group is not None:
            dist.destroy_process_group(self.group)

        resps = [requests.get(f"{url}/wbridge/receiver_world") for url in self.receiver_urls]
        assert all(resp.status_code == 200 for resp in resps), "Failed to get receiver world size"
        rollout_num_workers = [resp.json()["world_size"] for resp in resps]
        total_world_size = self.world_size + sum(rollout_num_workers)

        pg_init_args = {
            "transfer_mode": self.transfer_mode,
            "init_method": self.init_method,
            "world_size": total_world_size,
            "rank": self.rank,
            "group_name": "wbridge",
            "sender_world_size": self.world_size,
        }

        if self.rank == 0:
            base_rank = self.world_size
            for url, num_workers in zip(self.receiver_urls, rollout_num_workers):
                connect_args = {
                    **pg_init_args,
                    "rank": base_rank,
                }
                resp = requests.post(f"{url}/wbridge/connect", json=connect_args)
                resp.raise_for_status()
                base_rank += num_workers

        self.set_up_connection(**pg_init_args)
        self.connected = True


    def send(self) -> None:
        """Send weights in router rounds. :attr:`save_weights` fills each round's 1D buffers (per name).

        This mirrors :meth:`WeightReceiver._receive_weights`: the receiver calls :attr:`WeightReceiver.load_weights`
        after recv+unpack; here we call :attr:`save_weights` to pack the logical buffer from the
        model, then ``isend`` the wire chunks to each peer.

        With ``transfer_mode == "cpu_direct"``, posted ``isend`` handles are drained at the start of the
        next :meth:`connect` or :meth:`send`, and :meth:`send` returns before those handles complete so
        Trainer Workers can return before the Rollout Worker finishes CPU staging.
        """
        if (
            not self.connected
            or self.shard_spec is None
            or self.router is None
            or self.group is None
            or self.device is None
        ):
            raise RuntimeError("WeightSender.send requires connect() first")
        self._drain_pending_comm()

        router = self.router
        group = self.group

        if self.rank == 0:
            for url in self.receiver_urls:
                resp = requests.post(f"{url}/wbridge/receive")
                resp.raise_for_status()

        for full_spec, overlap_specs in router.local_rounds:
            if not full_spec:
                dist.barrier(group=group)
                continue
            buf = full_spec.make_named_buffer(self.dtype_spec, self.device)
            self.save_weights(full_spec, buf)
            chunks = full_spec(buf)[overlap_specs]

            ops = [
                dist.P2POp(dist.isend, chunk, peer_rank, group)
                for peer_rank, chunk in chunks.items()
            ]
            dist.barrier(group=group)
            handles = list(dist.batch_isend_irecv(ops))
            if self.transfer_mode == "cpu_direct":
                self._pending_comm_handles.extend(handles)
            elif self.transfer_mode == "gpu_direct":
                for h in handles:
                    h.wait()
                del buf
