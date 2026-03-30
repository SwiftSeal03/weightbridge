"""Minimal WeightBridge example: Ray node pinning + GPU-direct transfer.

Uses Ray to place the trainer (2 sender workers) and rollout engine
(2 receiver workers) on separate nodes (alive Ray nodes ``[0]`` and ``[1]``),
each requiring 2 GPUs.

Architecture::

    driver  (ray.init, first alive node = rollout, second = trainer)
    │
    ├── RolloutEngine  (Ray actor pinned to rollout node IP)
    │   ├── FastAPI + WeightReceiverController
    │   ├── receiver_worker 0  (child process, WeightReceiver)
    │   └── receiver_worker 1  (child process, WeightReceiver)
    │
    └── TrainerWorker × NUM_SENDER_WORKERS  (Ray actors pinned to trainer node IP)
        └── torch.distributed (NCCL) + WeightSender.connect / .send
            (GPUDirectSender → NCCL isend to receivers)

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
import multiprocessing as mp
import requests
import threading
import time

import ray
import torch
import torch.distributed as dist
import uvicorn
from fastapi import FastAPI
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from wbridge.utils.distributed import get_local_ip
from wbridge import WeightData, WeightReceiver, WeightReceiverController, WeightSender
from wbridge.utils.data import shards_iterator, shards_to_numel

from utils import init_ray_and_get_rollout_trainer

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


def _build_local_tensors(meta: WeightData, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Create tensor shards from either provided tensors or zeros"""
    local_tensors = {}
    for name, shards, dtype in meta:
        slices = []
        local_tensors[name] = torch.zeros(shards_to_numel(shards), dtype=dtype, device=DEVICE)
        if name in tensors:
            for start, end, shard in shards_iterator(meta[name]):
                slices = [slice(l, r) for l, r, _ in shard]
                local_tensors[name][start:end] = tensors[name][tuple(slices)].reshape(-1)
    return local_tensors


# ── Rollout engine (Ray actor) ────────────────────────────────────


def _receiver_worker(ipc_name: str, rank: int, ready_event: mp.Event):
    """Child process entry — creates a WeightReceiver and blocks."""
    metadata = _build_receiver_metadata(rank)
    state_dict = _build_local_tensors(metadata, {})
    receiver = WeightReceiver(
        controller_ipc_name=ipc_name,
        rank=rank,
        metadata=metadata
    )
    ready_event.set()
    for _ in range(100):
        if receiver.request_update(state_dict):
            print(f"Receiver worker {rank} received weights")
            break
        print(f"Receiver worker {rank} waiting for weights")
        time.sleep(2)
    receiver.stop()

@ray.remote
class RolloutEngine:
    """Ray actor that hosts the receiver-side HTTP server and spawns
    receiver worker child processes (analogous to SGLang schedulers)."""

    def __init__(self):
        app = FastAPI()
        self.controller = WeightReceiverController(app)

        ready_events = [mp.Event() for _ in range(NUM_RECEIVER_WORKERS)]
        for rank in range(NUM_RECEIVER_WORKERS):
            p = mp.Process(
                target=_receiver_worker,
                args=(self.controller.ipc_name, rank, ready_events[rank]),
                daemon=True,
            )
            p.start()
        for ready_event in ready_events:
            ready_event.wait()
        self.controller.set_worker_num(NUM_RECEIVER_WORKERS)

        self._host = get_local_ip()
        self._port = ROLLOUT_SERVER_PORT
        config = uvicorn.Config(app, host=self._host, port=self._port, log_level="warning")
        server = uvicorn.Server(config)
        threading.Thread(target=server.run, daemon=True).start()
        while not server.started:
            time.sleep(0.1)

    def ready(self):
        return True


# ── Trainer worker (Ray actor) ────────────────────────────────────

@ray.remote
class TrainerWorker:
    """Ray actor — one per sender GPU.  Initialises torch.distributed
    with the other TrainerWorkers, then sends its weight shard."""

    def __init__(self, world_size: int, rank: int,
                 master_addr: str = None, master_port: int = None):
        self.world_size = world_size
        self.rank = rank
        self.master_addr = master_addr
        self.master_port = master_port
        print(f"Trainer worker {rank} initializing with master {master_addr}:{master_port}")
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
        )
        print(f"Trainer worker {rank} initialized")

    def send_weights(self, tensors: dict, receiver_url: str):
        meta = _build_sender_metadata(self.rank)
        local_tensors = _build_local_tensors(meta, tensors)

        sender = WeightSender("gpu_direct", receiver_urls=[receiver_url])
        sender.connect(meta)
        print(f"Trainer worker {self.rank} connected to receiver")
        sender.send(local_tensors)
        print(f"Trainer worker {self.rank} sent weights")
        dist.destroy_process_group()


# ── Entry point ───────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s  %(message)s")

    rollout_ip, trainer_ip, rollout_node_id, trainer_node_id = (
        init_ray_and_get_rollout_trainer()
    )
    logger.info("Rollout node IP: %s", rollout_ip)
    logger.info("Trainer node IP: %s", trainer_ip)

    # 1. Start rollout engine on the rollout node
    rollout_engine = RolloutEngine.options(
        num_cpus=1,
        num_gpus=NUM_RECEIVER_WORKERS,
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=rollout_node_id,
            soft=False,
        ),
    ).remote()

    # 2. Create tensors in the driver and dispatch to trainer workers
    tensors = {
        "uneven_weight": torch.randn(ROWS, COLS, dtype=DTYPE),
        "col_weight": torch.randn(ROWS, COLS, dtype=DTYPE),
        "dup_weight": torch.randn(ROWS, COLS, dtype=DTYPE),
    }
    logger.info("Created 3 tensors (each [%d, %d] %s)", ROWS, COLS, DTYPE)

    workers = [
        TrainerWorker.options(
            num_cpus=1,
            num_gpus=1,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=trainer_node_id,
                soft=False,
            ),
        ).remote(NUM_SENDER_WORKERS, rank, trainer_ip, SENDER_PG_PORT)
        for rank in range(NUM_SENDER_WORKERS)
    ]

    # 3. All trainer workers send weights
    receiver_url = f"http://{rollout_ip}:{ROLLOUT_SERVER_PORT}"
    ray.get(rollout_engine.ready.remote())
    ray.get([w.send_weights.remote(tensors, receiver_url) for w in workers])

    logger.info("All senders finished. Done.")
    ray.shutdown()


if __name__ == "__main__":
    main()
