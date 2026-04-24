from collections.abc import Callable
from dataclasses import dataclass

import requests
import torch
import torch.distributed as dist

from wbridge.utils.data import ShardSpec
from wbridge.backend.router import WeightRouter, WBEndpoint
from wbridge.utils.distributed import init_custom_process_group

import logging
logger = logging.getLogger(__name__)


@dataclass
class SenderArgs:
    """Transport args forwarded to :class:`WeightSender`.

    Attributes:
        world_size: Number of sender ranks participating in the process group.
        transfer_mode: ``"gpu_direct"`` (NCCL, CUDA wire buffers) or ``"cpu_direct"``
            (Gloo, CPU wire buffers; :meth:`~WeightSender.send` may return before ``isend`` completes).
        receiver_urls: HTTP base URLs of the receiver controllers, one per receiver engine.
        master_addr: Host/IP of the sender-side rank-0 process used for rendezvous.
        master_port: TCP port for the rendezvous group.
    """

    world_size: int
    transfer_mode: str
    receiver_urls: list[str]
    master_addr: str
    master_port: int


class WeightSender(WBEndpoint):
    """Sends weight rounds to receivers. ``save_weights`` fills each round's buffers (inverse of the receiver's ``load_weights``)."""

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
        """Join receivers over NCCL after a short-lived Gloo group for sender coordination.

        The Gloo process group uses ``tcp://{master_addr}:{master_port}`` from
        :meth:`__init__` so all sender ranks rendezvous before rank 0 drives
        HTTP receiver_world/connect and broadcasts rendezvous info for the main group.
        """
        self._drain_pending_comm()

        if self.group is not None:
            dist.destroy_process_group(self.group)

        rollout_num_workers = []
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
        trainers can overlap communication with receiver staging.
        """
        if not self.connected or self.shard_spec is None or self.router is None or self.group is None:
            raise RuntimeError("WeightSender.send requires connect() first")
        self._drain_pending_comm()

        if self.rank == 0:
            for url in self.receiver_urls:
                resp = requests.post(f"{url}/wbridge/receive")
                resp.raise_for_status()

        for full_spec, overlap_specs in self.router.local_rounds:
            if full_spec:
                buf = full_spec.make_named_buffer(self.dtype_spec, self.device)
                self.save_weights(full_spec, buf)
                chunks = full_spec(buf)[overlap_specs]

                ops = [
                    dist.P2POp(dist.isend, chunk, peer_rank, self.group)
                    for peer_rank, chunk in chunks.items()
                ]
            dist.barrier(group=self.group)
            if full_spec:
                handles = list(dist.batch_isend_irecv(ops))
                if self.transfer_mode == "cpu_direct":
                    self._pending_comm_handles.extend(handles)
                elif self.transfer_mode == "gpu_direct":
                    for h in handles:
                        h.wait()
                    del buf
