"""Multi-round P2P routing plans under per-rank byte caps."""

from __future__ import annotations

import logging
from dataclasses import field

import torch
import torch.distributed as dist

from wbridge.utils.data import ShardSpec, shards_nbytes
from wbridge.utils.distributed import init_custom_process_group

# Per-round cap for total bytes a receiver may take (sum over all senders).
RECEIVER_ROUND_CAP_BYTES = 2 * 1024**3

logger = logging.getLogger(__name__)

type CommRoundPlan = tuple[ShardSpec, dict[int, ShardSpec]]

class WeightRouter:
    """Computes transmission plan for weight transfer between senders and receivers."""

    def __init__(
        self,
        rank: int,
        sender_ws: int,
        all_specs: list[ShardSpec],
        dtype_spec: dict[str, torch.dtype],
    ) -> None:
        self.rank = rank
        self.sender_ws = sender_ws
        self.world_size = len(all_specs)
        self.receiver_ws = self.world_size - sender_ws
        assert all(isinstance(spec, ShardSpec) for spec in all_specs), "all_specs must be a list of ShardSpec"
        self.send_specs = all_specs[:sender_ws]
        self.recv_specs = all_specs[sender_ws:]
        self.dtype_spec = dtype_spec
        self.global_rounds = self.compute_global_rounds()
        self.local_rounds = self.compute_local_rounds()
        
    def compute_global_rounds(self) -> list[dict[tuple[int, int], ShardSpec]]:
        """Schedule rounds in **name-priority** order (sorted tensor names, then sorted pairs per name).

        Each round: walk ``sorted(dtype_spec)`` (tensor names); for each name with remaining work, walk
        pairs ``(si, ri)`` in sorted order; if that slice fits caps for sender ``si`` and receiver
        ``sender_ws + ri``, add it and remove it from the remaining structure.
        """
        all_overlaps = {
            (si, ri): ShardSpec.compute_overlap(self.send_specs[si], self.recv_specs[ri])
            for si in range(self.sender_ws)
            for ri in range(self.receiver_ws)
        }
        
        remaining_names = set(self.dtype_spec)

        rounds: list[set[str]] = []

        while remaining_names:
            send_caps = [RECEIVER_ROUND_CAP_BYTES * self.receiver_ws // self.sender_ws] * self.sender_ws
            recv_caps = [RECEIVER_ROUND_CAP_BYTES] * self.receiver_ws
            round_plan: set[str] = set()
            
            for name in sorted(remaining_names):
                dtype = self.dtype_spec[name]
                send_caps_old = send_caps.copy()
                recv_caps_old = recv_caps.copy()
                
                for (si, ri), overlap in all_overlaps.items():
                    if name not in overlap:
                        continue
                    shards = overlap[name]
                    nbytes = shards_nbytes(shards, dtype)
                    send_caps[si] -= nbytes
                    recv_caps[ri] -= nbytes
                
                if any(c < 0 for c in send_caps + recv_caps):
                    send_caps = send_caps_old
                    recv_caps = recv_caps_old
                    continue
                
                round_plan.add(name)
            
            if not round_plan:
                raise RuntimeError(
                    "routing deadlock: no tensor fits per-round caps (try smaller per-tensor overlap or raise caps)"
                )
            remaining_names -= round_plan
            rounds.append(round_plan)

        return rounds
    
    def compute_local_rounds(self) -> list[CommRoundPlan]:
        """
        Plan for *rank*: round *i* is :class:`CommRoundPlan` — ``peer_specs[peer]`` is the wire overlap
        with global rank *peer*; on receivers ``recv_subspec`` is this round's buffer layout (merged overlaps).
        """
        out: list[tuple[ShardSpec, dict[int, ShardSpec]]] = []
        is_sender = self.rank < self.sender_ws
        if is_sender:
            full_spec = self.send_specs[self.rank]
            overlaps = {
                ri + self.sender_ws: ShardSpec.compute_overlap(full_spec, self.recv_specs[ri])
                for ri in range(self.receiver_ws)
            }
        else:
            full_spec = self.recv_specs[self.rank - self.sender_ws]
            overlaps = {
                si: ShardSpec.compute_overlap(self.send_specs[si], full_spec)
                for si in range(self.sender_ws)
            }
        for round_plan in self.global_rounds:
            round_overlaps = {
                rank: overlap.subset(round_plan)
                for rank, overlap in overlaps.items()
            }
            out.append((full_spec.subset(round_plan), round_overlaps))
        return out
    

class WBEndpoint:
    """Endpoint for weight bridge communication."""
    shard_spec: ShardSpec
    dtype_spec: dict[str, torch.dtype]
    router: WeightRouter | None = None
    group: dist.ProcessGroup | None = None
    
    def set_up_connection(self, **pg_args) -> None:
        """Set up the connection to the other endpoint."""
        sender_ws = pg_args.pop("sender_world_size")
        rank = pg_args["rank"]
        ws = pg_args["world_size"]
        self.group = init_custom_process_group(**pg_args)
        
        all_shard_specs = [None] * ws
        dist.all_gather_object(all_shard_specs, self.shard_spec, group=self.group)

        all_dtype_specs = [None] * ws
        dist.all_gather_object(all_dtype_specs, self.dtype_spec, group=self.group)

        for dtype_spec in all_dtype_specs:
            for name, dtype in dtype_spec.items():
                if name in self.dtype_spec:
                    assert self.dtype_spec[name] == dtype, f"Dtype mismatch for {name} on rollout rank {rank}"
                else:
                    self.dtype_spec[name] = dtype

        self.router = WeightRouter(rank, sender_ws, all_shard_specs, self.dtype_spec)
        
        logger.error(f"Rank: {rank} group initialized")
        