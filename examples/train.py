"""Minimal WeightBridge example: Ray placement groups + GPU-direct transfer.

Uses Ray to place the trainer (2 sender workers) and rollout engine
(2 receiver workers) on separate nodes, each requiring 2 GPUs.

Architecture::

    driver  (ray.init, creates tensors, orchestrates)
    │
    ├── placement group: 4 GPU bundles (PACK strategy)
    │   bundles 0-1 → trainer node
    │   bundles 2-3 → rollout node
    │
    ├── RolloutEngine  (Ray actor on rollout node)
    │   ├── FastAPI + WeightReceiverController
    │   ├── receiver_worker 0  (child process, WeightReceiver)
    │   └── receiver_worker 1  (child process, WeightReceiver)
    │
    └── TrainerWorker × NUM_SENDER_WORKERS  (Ray actors on trainer bundles)
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
import os
import threading
import time

import ray
import torch
import torch.distributed as dist
import uvicorn
from fastapi import FastAPI
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from wbridge.utils.distributed import get_local_ip
from wbridge import WeightData, WeightReceiver, WeightReceiverController, WeightSender, dtype_str_to_torch

logger = logging.getLogger("example")

NUM_SENDER_WORKERS = 2
NUM_RECEIVER_WORKERS = 2
DTYPE_STR = "float32"
DTYPE = torch.float32
ROWS, COLS = 4, 8


# ── Helpers ────────────────────────────────────────────────────────


def _build_sender_metadata(rank: int) -> WeightData:
    """Shard metadata for the given sender rank (``connect``)."""
    if rank == 0:
        meta_dict = {
            "uneven_weight": {
                "shard": [(0, 1, ROWS), (0, COLS, COLS)],
                "dtype": DTYPE_STR,
            },
            "col_weight": {
                "shard": [(0, ROWS, ROWS), (0, COLS // 2, COLS)],
                "dtype": DTYPE_STR,
            },
            "dup_weight": {
                "shard": [(0, ROWS, ROWS), (0, COLS, COLS)],
                "dtype": DTYPE_STR,
            },
        }
    else:
        meta_dict = {
            "uneven_weight": {
                "shard": [(1, ROWS, ROWS), (0, COLS, COLS)],
                "dtype": DTYPE_STR,
            },
            "col_weight": {
                "shard": [(0, ROWS, ROWS), (COLS // 2, COLS, COLS)],
                "dtype": DTYPE_STR,
            },
            "dup_weight": {
                "shard": [(0, ROWS, ROWS), (0, COLS, COLS)],
                "dtype": DTYPE_STR,
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
        name: {"shard": shard, "dtype": DTYPE_STR}
        for name in ("uneven_weight", "col_weight", "dup_weight")
    }
    return WeightData(meta_dict)


def _build_local_tensors(rank: int, meta: WeightData, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Create tensor shards from either provided tensors or zeros"""
    local_tensors = {}
    for name, meta_entry in meta:
        slices = [
            slice(start, end)
            for start, end, _ in meta_entry["shard"]
        ]
        if name in tensors:
            local_tensors[name] = tensors[name][slices].contiguous()
        else:
            local_tensors[name] = torch.zeros(meta_entry["shard"], dtype=dtype_str_to_torch(meta_entry["dtype"]))
    return local_tensors


# ── Placement group ───────────────────────────────────────────────


@ray.remote(num_cpus=0)
class _InfoActor:
    """Lightweight probe to discover which node/GPU a bundle landed on."""

    def get_ip_and_gpu_id(self):
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        return get_local_ip(), gpu_ids.split(",")[0]


def create_placement_group():
    """Reserve GPUs and return (pg, trainer_bundle_indices, rollout_bundle_indices).

    Bundles are sorted by (node_ip, gpu_id) so that consecutive logical
    ranks map to consecutive GPUs on the same node — the first
    NUM_SENDER_WORKERS go to the trainer, the rest to the rollout engine.
    """
    total = NUM_SENDER_WORKERS + NUM_RECEIVER_WORKERS
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(total)]
    pg = placement_group(bundles, strategy="PACK")  
    ray.get(pg.ready())

    probes = [
        _InfoActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=i,
            ),
        ).remote()
        for i in range(total)
    ]
    infos = ray.get([p.get_ip_and_gpu_id.remote() for p in probes])
    for p in probes:
        ray.kill(p)

    ordered = sorted(range(total), key=lambda i: (infos[i][0], infos[i][1]))
    for rank, bundle_idx in enumerate(ordered):
        ip, gpu = infos[bundle_idx]
        logger.info(
            "  rank %d → bundle %d  (node %s, gpu %s)", rank, bundle_idx, ip, gpu
        )

    trainer_bundles = ordered[:NUM_SENDER_WORKERS]
    rollout_bundles = ordered[NUM_SENDER_WORKERS:]
    return pg, trainer_bundles, rollout_bundles


# ── Rollout engine (Ray actor) ────────────────────────────────────


def _receiver_worker(ipc_name: str, rank: int):
    """Child process entry — creates a WeightReceiver and blocks."""
    metadata = _build_receiver_metadata(rank)
    state_dict = _build_local_tensors(rank, metadata, {})
    receiver = WeightReceiver(
        controller_ipc_name=ipc_name,
        rank=rank,
        metadata=metadata,
        state_dict=state_dict
    )
    while True:
        if receiver.request_update()["success"]:
            break
        time.sleep(1)


@ray.remote
class RolloutEngine:
    """Ray actor that hosts the receiver-side HTTP server and spawns
    receiver worker child processes (analogous to SGLang schedulers)."""

    def init(self):
        app = FastAPI()
        self.controller = WeightReceiverController(app)

        for rank in range(NUM_RECEIVER_WORKERS):
            p = mp.Process(
                target=_receiver_worker,
                args=(self.controller.ipc_name, rank),
                daemon=True,
            )
            p.start()
        self.controller.set_worker_num(NUM_RECEIVER_WORKERS)

        self._host = get_local_ip()
        self._port = 15000
        config = uvicorn.Config(app, host=self._host, port=self._port, log_level="warning")
        server = uvicorn.Server(config)
        threading.Thread(target=server.run, daemon=True).start()
        while not server.started:
            time.sleep(0.1)

    def get_server_info(self):
        return self._host, self._port


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
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )

    def send_weights(self, tensors: dict, receiver_url: str):
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            rank=self.rank,
            world_size=self.world_size,
        )

        meta = _build_sender_metadata(self.rank)
        local_tensors = _build_local_tensors(self.rank, meta, tensors)

        sender = WeightSender("gpu_direct", receiver_urls=[receiver_url])
        sender.connect(meta)
        sender.send(local_tensors)

        dist.destroy_process_group()


# ── Entry point ───────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s  %(message)s")
    ray.init(address="auto")

    # 1. Reserve GPUs via placement group
    pg, trainer_bundles, rollout_bundles = create_placement_group()

    # 2. Start rollout engine on the rollout node
    rollout_engine = RolloutEngine.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=rollout_bundles[0],
            placement_group_capture_child_tasks=True,
        ),
    ).remote()
    ray.get(rollout_engine.init.remote())
    host, port = ray.get(rollout_engine.get_server_info.remote())
    receiver_url = f"http://{host}:{port}"
    logger.info("Rollout engine ready at %s", receiver_url)

    # 3. Create tensors in the driver and dispatch to trainer workers
    tensors = {
        "uneven_weight": torch.randn(ROWS, COLS, dtype=DTYPE),
        "col_weight": torch.randn(ROWS, COLS, dtype=DTYPE),
        "dup_weight": torch.randn(ROWS, COLS, dtype=DTYPE),
    }
    logger.info("Created 3 tensors (each [%d, %d] %s)", ROWS, COLS, DTYPE)

    workers = []
    master_addr = get_local_ip()
    master_port = 60010
    for rank in range(NUM_SENDER_WORKERS):
        worker = TrainerWorker.options(
            num_cpus=1,
            num_gpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=trainer_bundles[rank],
            ),
        ).remote(NUM_SENDER_WORKERS, rank, master_addr, master_port)
        workers.append(worker)

    # 4. All trainer workers send weights
    ray.get([w.send_weights.remote(tensors, receiver_url) for w in workers])

    logger.info("All senders finished. Done.")
    ray.shutdown()


if __name__ == "__main__":
    main()
