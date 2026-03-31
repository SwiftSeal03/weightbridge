"""Minimal WeightBridge example: Ray node pinning + GPU-direct transfer.

Uses Ray to place the trainer (2 sender workers) and rollout engine
(1 controller + 2 receiver workers) on separate nodes (alive Ray nodes
``[0]`` and ``[1]``).

Architecture::

    driver  (ray.init, first alive node = rollout, second = trainer)
    │
    ├── RolloutEngine  (Ray actor, no GPU — HTTP server + controller)
    │
    ├── RolloutWorker × NUM_RECEIVER_WORKERS  (Ray actors, 1 GPU each)
    │   └── WeightReceiver (NCCL recv) + local verification
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
import threading
import time

import ray
import torch
import uvicorn
from fastapi import FastAPI
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from wbridge import WeightData, WeightReceiver, WeightReceiverController, WeightSender

from utils import init_ray_and_get_rollout_trainer, generate_local_tensors

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


# ── Rollout engine + workers (Ray actors) ─────────────────────────


@ray.remote
class RolloutWorker:
    """Ray actor — one per receiver GPU.  Receives weights via NCCL and
    verifies them against pre-loaded ground truth."""

    def __init__(self, ipc_name: str, rank: int):
        self.rank = rank
        self.metadata = _build_receiver_metadata(rank)
        self.receiver = WeightReceiver(
            controller_ipc_name=ipc_name,
            rank=rank,
            metadata=self.metadata,
        )
        self.expected: dict[str, torch.Tensor] | None = None

    def ready(self):
        return True

    def set_expected(self, seed: int):
        """Generate ground-truth tensors from *seed* and slice into this worker's expected shard."""
        self.expected = generate_local_tensors(self.metadata, device=DEVICE, seed=seed)

    def receive_and_verify(self) -> dict:
        """Block until weights arrive, then verify against expected shard."""
        state_dict = generate_local_tensors(self.metadata, device=DEVICE)
        for _ in range(200):
            if self.receiver.request_update(state_dict):
                break
            time.sleep(0.5)
        else:
            return {"rank": self.rank, "ok": False, "detail": "timeout waiting for weights"}

        if self.expected is None:
            return {"rank": self.rank, "ok": True, "detail": "no expected tensors set, skipped"}

        for name, exp in self.expected.items():
            got = state_dict[name]
            if not torch.allclose(exp, got, rtol=1e-5, atol=1e-6):
                max_err = float((exp - got).abs().max().item())
                return {
                    "rank": self.rank, "ok": False,
                    "detail": f"{name}: max abs err {max_err}",
                }
        return {"rank": self.rank, "ok": True, "detail": "all tensors match"}


@ray.remote
class RolloutEngine:
    """Ray actor that hosts the receiver-side HTTP server and the
    WeightReceiverController.  Workers are separate RolloutWorker actors."""

    def __init__(self, addr, port):
        app = FastAPI()
        self.controller = WeightReceiverController(app)

        config = uvicorn.Config(app, host=addr, port=port, log_level="warning")
        server = uvicorn.Server(config)
        threading.Thread(target=server.run, daemon=True).start()
        while not server.started:
            time.sleep(0.1)

        self._workers: list = []

    def get_ipc_name(self) -> str:
        return self.controller.ipc_name

    def set_workers(self, workers: list):
        self._workers = workers
        self.controller.set_worker_num(len(workers))

    def gather_results(self) -> list[dict]:
        results = ray.get([w.receive_and_verify.remote() for w in self._workers])
        return sorted(results, key=lambda r: r["rank"])


# ── Trainer worker (Ray actor) ────────────────────────────────────

@ray.remote
class TrainerWorker:
    """Ray actor — one per sender GPU.  Sends its weight shard via
    WeightBridge (no default torch.distributed group required)."""

    def __init__(self, world_size: int, rank: int,
                 master_addr: str = None, master_port: int = None):
        self.world_size = world_size
        self.rank = rank
        self.sender_init_method = f"tcp://{master_addr}:{master_port}"
        print(f"Trainer worker {rank} initialized")

    def send_weights(self, seed: int, receiver_url: str):
        meta = _build_sender_metadata(self.rank)
        local_tensors = generate_local_tensors(meta, device=DEVICE, seed=seed)

        sender = WeightSender(
            "gpu_direct", receiver_urls=[receiver_url],
            rank=self.rank, world_size=self.world_size,
        )
        start_time = time.time()
        sender.connect(meta, sender_init_method=self.sender_init_method)
        print(f"Trainer worker {self.rank} connected to receiver in {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        sender.send(local_tensors)
        print(f"Trainer worker {self.rank} sent weights in {time.time() - start_time:.2f} seconds")


# ── Entry point ───────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s  %(message)s")

    rollout_ip, trainer_ip, rollout_node_id, trainer_node_id = (
        init_ray_and_get_rollout_trainer()
    )
    logger.info("Rollout node IP: %s", rollout_ip)
    logger.info("Trainer node IP: %s", trainer_ip)

    rollout_sched = NodeAffinitySchedulingStrategy(node_id=rollout_node_id, soft=False)
    trainer_sched = NodeAffinitySchedulingStrategy(node_id=trainer_node_id, soft=False)

    # 1. Start rollout engine (HTTP server only, no GPUs needed)
    rollout_engine = RolloutEngine.options(
        num_cpus=1,
        scheduling_strategy=rollout_sched,
    ).remote(rollout_ip, ROLLOUT_SERVER_PORT)
    ipc_name = ray.get(rollout_engine.get_ipc_name.remote())

    # 2. Start rollout workers (one Ray actor per receiver GPU)
    rollout_workers = [
        RolloutWorker.options(
            num_cpus=1,
            num_gpus=1,
            scheduling_strategy=rollout_sched,
        ).remote(ipc_name, rank)
        for rank in range(NUM_RECEIVER_WORKERS)
    ]
    ray.get([w.ready.remote() for w in rollout_workers])
    ray.get(rollout_engine.set_workers.remote(rollout_workers))

    # 3. Share a seed so all workers generate identical ground-truth tensors locally
    seed = torch.randint(0, 2**31, (1,)).item()
    logger.info("Tensor seed: %d (each worker generates [%d, %d] %s locally)", seed, ROWS, COLS, DTYPE)
    ray.get([w.set_expected.remote(seed) for w in rollout_workers])

    # 4. Start trainer workers and send weights
    trainer_workers = [
        TrainerWorker.options(
            num_cpus=1,
            num_gpus=1,
            scheduling_strategy=trainer_sched,
        ).remote(NUM_SENDER_WORKERS, rank, trainer_ip, SENDER_PG_PORT)
        for rank in range(NUM_SENDER_WORKERS)
    ]

    receiver_url = f"http://{rollout_ip}:{ROLLOUT_SERVER_PORT}"
    ray.get(rollout_engine.ready.remote())
    ray.get([w.send_weights.remote(seed, receiver_url) for w in trainer_workers])

    # 5. Each rollout worker verifies its own shard; engine gathers results
    results = ray.get(rollout_engine.gather_results.remote())
    for r in results:
        if not r["ok"]:
            raise AssertionError(f"RolloutWorker rank {r['rank']} failed: {r['detail']}")
    logger.info("All %d rollout workers verified their shards independently.", len(results))

    ray.shutdown()


if __name__ == "__main__":
    main()
