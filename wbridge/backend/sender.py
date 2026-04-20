from dataclasses import dataclass

import requests
import torch
import torch.distributed as dist

from wbridge.utils.data import ShardSpec
from wbridge.utils.distributed import init_custom_process_group


@dataclass
class SenderArgs:
    """Transport args forwarded to :class:`WeightSender`.

    Attributes:
        world_size: Number of sender ranks participating in the process group.
        transfer_mode: ``"gpu_direct"`` (NCCL/cuda) or any other mode supported by
            :class:`WeightSender`.
        receiver_urls: HTTP base URLs of the receiver controllers, one per receiver engine.
        master_addr: Host/IP of the sender-side rank-0 process used for rendezvous.
        master_port: TCP port for the rendezvous group.
    """

    world_size: int
    transfer_mode: str
    receiver_urls: list[str]
    master_addr: str
    master_port: int


class WeightSender:
    def __init__(self, rank: int, args: SenderArgs):
        self.rank = rank
        self.transfer_mode = args.transfer_mode
        self.receiver_urls = args.receiver_urls
        self.world_size = args.world_size
        self.init_method = f"tcp://{args.master_addr}:{args.master_port}"

        if args.transfer_mode == "gpu_direct":
            self.device = "cuda"
            self.backend = "nccl"
        else:
            self.device = "cpu"
            self.backend = "gloo"

        self.connected = False
        self.group: dist.ProcessGroup | None = None
        self.overlaps: dict[int, ShardSpec] = {}
        self.shard_spec: ShardSpec | None = None
        self.dtype_spec: dict[str, torch.dtype] | None = None

    def connect(self, shard_spec: ShardSpec) -> None:
        """Join receivers over NCCL after a short-lived Gloo group for sender coordination.

        The Gloo process group uses ``tcp://{master_addr}:{master_port}`` from
        :meth:`__init__` so all sender ranks rendezvous before rank 0 drives
        HTTP receiver_world/connect and broadcasts rendezvous info for the main group.
        """
        self.shard_spec = shard_spec
        if self.group is not None:
            dist.destroy_process_group(self.group)

        rollout_num_workers = []
        resps = [requests.get(f"{url}/wbridge/receiver_world") for url in self.receiver_urls]
        assert all(resp.status_code == 200 for resp in resps), "Failed to get receiver world size"
        rollout_num_workers = [resp.json()["world_size"] for resp in resps]
        total_world_size = self.world_size + sum(rollout_num_workers)

        pg_init_args = {
            "backend": self.backend,
            "init_method": self.init_method,
            "world_size": total_world_size,
            "rank": self.rank,
            "group_name": "wbridge",
        }

        if self.rank == 0:
            base_rank = self.world_size
            for url, num_workers in zip(self.receiver_urls, rollout_num_workers):
                connect_args = {
                    **pg_init_args,
                    "rank": base_rank,
                    "sender_world_size": self.world_size,
                }
                resp = requests.post(f"{url}/wbridge/connect", json=connect_args)
                resp.raise_for_status()
                base_rank += num_workers

        self.group = init_custom_process_group(**pg_init_args)

        all_specs = [None] * total_world_size
        dist.all_gather_object(all_specs, self.shard_spec, group=self.group)
        self.overlaps = {
            rank: overlap
            for rank, tensor in enumerate(all_specs)
            if rank >= self.world_size and (overlap := ShardSpec.compute_overlap(self.shard_spec, tensor))
        }

        all_dtype_specs = [None] * total_world_size
        dist.all_gather_object(all_dtype_specs, self.dtype_spec, group=self.group)

        if self.dtype_spec is None:
            self.dtype_spec = {}

        for dtype_spec in all_dtype_specs:
            if dtype_spec is not None:
                for name, dtype in dtype_spec.items():
                    if name in self.dtype_spec:
                        assert self.dtype_spec[name] == dtype, f"Dtype mismatch for {name} on rollout rank {self.rank}"
                    else:
                        self.dtype_spec[name] = dtype

        self.connected = True

    def send(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.transfer_mode == "gpu_direct":
            if not self.connected or self.shard_spec is None:
                raise RuntimeError("WeightSender.send requires connect() first")
            if self.rank == 0:
                for url in self.receiver_urls:
                    resp = requests.post(f"{url}/wbridge/receive")
                    resp.raise_for_status()

            chunks = self.shard_spec(state_dict)[self.overlaps]
            ops = [
                dist.P2POp(dist.isend, chunk, receiver_rank, self.group)
                for receiver_rank, chunk in chunks.items()
            ]
            if ops:
                for h in dist.batch_isend_irecv(ops):
                    h.wait()
        else:
            pass  # TODO: Implement CPU transfer send
