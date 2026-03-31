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
    └── TrainerWorker × NUM_SENDER_WORKERS  (Ray actors, 1 GPU each)
        └── WeightSender (NCCL isend to receivers)

Tensors (``float32``, shape ``[ROWS, COLS]`` = ``[4, 8]``):

    1. ``uneven_weight`` — row-sharded unevenly: rank 0 → rows ``[0, 1)``,
       rank 1 → rows ``[1, 4)``.
    2. ``col_weight`` — column-sharded: rank 0 → cols ``[0, 4)``, rank 1 →
       cols ``[4, 8)``.
    3. ``dup_weight`` — full tensor duplicated on both sender ranks (deduped in
       ``DirectSender._dedup_sender_metadata``).

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

from wbridge import WeightData

from utils import get_ray_nodes, generate_local_tensors
from workers import RolloutEngine, TrainerWorker

logger = logging.getLogger("example")

NUM_SENDER_WORKERS = 2
NUM_RECEIVER_WORKERS = 2
DTYPE = torch.float32
ROWS, COLS = 4, 8
DEVICE = "cuda"

ROLLOUT_SERVER_PORT = 15000
SENDER_PG_PORT = 60010


# ── Helpers ────────────────────────────────────────────────────────


def _build_sender_metadata(rank: int) -> WeightData:
    """Shard metadata for the given sender rank (``connect``)."""
    if rank == 0:
        meta_dict = {
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
        meta_dict = {
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
    return WeightData(meta_dict)


def _build_receiver_metadata(rank: int) -> WeightData:
    """Build metadata-only WeightData for a receiver worker."""
    mid = ROWS // 2
    if rank == 0:
        shard = [(0, mid, ROWS), (0, COLS, COLS)]
    else:
        shard = [(mid, ROWS, ROWS), (0, COLS, COLS)]
    meta_dict = {
        name: {"shard": shard, "dtype": DTYPE}
        for name in ("uneven_weight", "col_weight", "dup_weight")
    }
    return WeightData(meta_dict)


# ── Entry point ───────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s  %(message)s")

    rollout_ip, trainer_ip, rollout_node_id, trainer_node_id = get_ray_nodes()
    logger.info("Rollout node IP: %s", rollout_ip)
    logger.info("Trainer node IP: %s", trainer_ip)

    seed = torch.randint(0, 2**31, (1,)).item()
    logger.info("Tensor seed: %d (each worker generates [%d, %d] %s locally)", seed, ROWS, COLS, DTYPE)
    tensor_gen = partial(generate_local_tensors, device=DEVICE, seed=seed)

    # 1. Start trainer workers
    trainer_workers = [
        TrainerWorker.options(
            num_cpus=1, num_gpus=1,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=trainer_node_id, soft=False),
        ).remote(
            NUM_SENDER_WORKERS, rank, trainer_ip, SENDER_PG_PORT,
            _build_sender_metadata,
        )
        for rank in range(NUM_SENDER_WORKERS)
    ]

    # 2. Start rollout engine (spawns receiver workers internally)
    rollout_engine = RolloutEngine.options(
        num_cpus=1,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=rollout_node_id, soft=False),
    ).remote(
        rollout_ip, ROLLOUT_SERVER_PORT, rollout_node_id, NUM_RECEIVER_WORKERS,
        _build_receiver_metadata, tensor_gen,
    )

    # 3. Send weights to rollout workers
    receiver_url = f"http://{rollout_ip}:{ROLLOUT_SERVER_PORT}"
    ray.get([w.send_weights.remote(tensor_gen, receiver_url) for w in trainer_workers])

    # 4. Each rollout worker verifies its own shard; engine gathers results
    results = ray.get(rollout_engine.gather_results.remote())
    for r in results:
        if not r["ok"]:
            raise AssertionError(f"RolloutWorker rank {r['rank']} failed: {r['detail']}")
    logger.info("All %d rollout workers verified their shards independently.", len(results))
    ray.shutdown()


if __name__ == "__main__":
    main()
