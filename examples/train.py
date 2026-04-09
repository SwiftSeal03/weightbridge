"""Minimal WeightBridge example: Ray node pinning + GPU-direct transfer.

Uses Ray to place the trainer (2 sender workers) and rollout engine
(1 controller + 2 receiver workers) on separate nodes (alive Ray nodes
``[0]`` and ``[1]``).

Architecture::

    driver  (ray.init, first alive node = rollout, second = trainer)
    │
    ├── RolloutEngine  (Ray actor, no GPU — HTTP server + controller)
    │   └── spawns RolloutWorker × NUM_RECEIVER_WORKERS  (Ray actors, 1 GPU each)
    │       └── WeightReceiver (NCCL recv) + local verification
    │
    └── TrainerEngine  (non-Ray manager in driver process)
        └── TrainerWorker × NUM_SENDER_WORKERS  (Ray actors, 1 GPU each)
            └── WeightSender (NCCL isend to receivers)

Tensors (``float32``, shape ``[ROWS, COLS]`` = ``[4, 8]``):

    1. ``uneven_weight`` — row-sharded unevenly: rank 0 → rows ``[0, 1)``,
       rank 1 → rows ``[1, 4)``.
    2. ``col_weight`` — column-sharded: rank 0 → cols ``[0, 4)``, rank 1 →
       cols ``[4, 8)``.
    3. ``dup_weight`` — full tensor duplicated on both sender ranks (deduped on
       the sender side when shards match across ranks).

Usage::

    ray start --head          # on first node (2 GPUs)
    ray start --address=...   # on second node (2 GPUs)
    python examples/train.py
"""

import logging
from functools import partial

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from wbridge import ShardSpec

from utils import get_ray_nodes, generate_local_tensors
from workers import EngineArgs, RolloutEngine, TrainerEngine

logger = logging.getLogger("example")

DTYPE = torch.float32
ROWS, COLS = 4, 8


def _build_sender_shard_spec(rank: int) -> ShardSpec:
    """Shard spec for the given sender rank (``connect``)."""
    if rank == 0:
        entries = {
            "uneven_weight": {
                "shard": [(0, 1, ROWS), (0, COLS, COLS)],
                "dtype": DTYPE,
            },
            "col_weight": {
                "shard": [(0, ROWS, ROWS), (0, COLS // 2, COLS)],
                "dtype": DTYPE,
            },
            "dup_weight": {
                "shard": [(0, ROWS, ROWS), (0, COLS, COLS)],
                "dtype": DTYPE,
            },
        }
    else:
        entries = {
            "uneven_weight": {
                "shard": [(1, ROWS, ROWS), (0, COLS, COLS)],
                "dtype": DTYPE,
            },
            "col_weight": {
                "shard": [(0, ROWS, ROWS), (COLS // 2, COLS, COLS)],
                "dtype": DTYPE,
            },
            "dup_weight": {
                "shard": [(0, ROWS, ROWS), (0, COLS, COLS)],
                "dtype": DTYPE,
            },
        }
    return ShardSpec(entries)


def _build_receiver_shard_spec(rank: int) -> ShardSpec:
    """Build :class:`~wbridge.ShardSpec` for a receiver worker."""
    mid = ROWS // 2
    if rank == 0:
        shard = [(0, mid, ROWS), (0, COLS, COLS)]
    else:
        shard = [(mid, ROWS, ROWS), (0, COLS, COLS)]
    entries = {
        name: {"shard": shard, "dtype": DTYPE}
        for name in ("uneven_weight", "col_weight", "dup_weight")
    }
    return ShardSpec(entries)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s")
    
    # rollout_ip, trainer_ip, rollout_node_id, trainer_node_id = get_ray_nodes()
    # engine_args = EngineArgs(
    #     rollout_host=rollout_ip,
    #     rollout_port=15000,
    #     rollout_scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=rollout_node_id, soft=False),
    #     num_rollout_workers=2,
    #     rollout_shard_spec_generator=_build_receiver_shard_spec,
    #     trainer_host=trainer_ip,
    #     trainer_pg_port=60010,
    #     trainer_scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=trainer_node_id, soft=False),
    #     num_trainer_workers=2,
    #     trainer_shard_spec_generator=_build_sender_shard_spec,
    #     tensor_generator=partial(generate_local_tensors, device="cuda", seed=42),
    # )

    # trainer_engine = TrainerEngine(engine_args)
    # rollout_engine = RolloutEngine.options(scheduling_strategy=engine_args.rollout_scheduling_strategy).remote()
    # ray.get(rollout_engine.init.remote(engine_args))
    
    # recv_future = rollout_engine.recv_weights.remote()
    # trainer_engine.send_weights()
    # ray.get(recv_future)
    # logger.info("Weights received")
    
    # results = ray.get(rollout_engine.verify_all.remote())
    # logger.info(results)
    
    # ray.shutdown()
    
    ip, _, id1, id2 = get_ray_nodes()
    ids = [id1, id2]
    
    @ray.remote(num_gpus=1)
    class NodeManager:
        def init(self, rank: int, world_size: int, master_ip: str):
            self.rank = rank
            self.world_size = world_size
            self.master_ip = master_ip
            
        def run(self):
            import torch.distributed as dist
            import socket
            
            import os
            # from wbridge.utils.distributed import init_custom_process_group
            ip = ray._private.services.get_node_ip_address()
            print(f"ip: {ip}")
            
            os.environ["NCCL_SOCKET_IFNAME"] = "eno1"
                
            group = dist.init_process_group(
                backend="nccl", 
                init_method=f"tcp://{self.master_ip}:60011", 
                world_size=self.world_size,
                rank=self.rank, 
                group_name="test"
            )
            print(f"Group {self.rank} initialized")
            
            tensor = torch.ones(1, dtype=torch.uint8, device="cuda:0")
            dist.broadcast(tensor, src=0, group=group)
            print(f"Tensor {self.rank} broadcasted")

    nodes = [
        NodeManager.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=i, soft=False)).remote() 
        for i in ids
    ]
    ray.get([node.init.remote(i, 2, ip) for i, node in enumerate(nodes)])
    print("Nodes initialized")
    ray.get([node.run.remote() for node in nodes])
    print("Nodes ran")
    ray.shutdown()

if __name__ == "__main__":
    main()
