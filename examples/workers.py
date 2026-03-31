"""Reusable Ray actor definitions for WeightBridge sender / receiver pipelines.

The workers are parameterised by two callables so that callers can plug in
their own sharding logic without touching this file:

* ``metadata_generator(rank) -> WeightData``
* ``tensor_generator(metadata) -> dict[str, Tensor]``

``tensor_generator`` should be a partial with ``device`` and ``seed``
already bound, e.g.
``functools.partial(generate_local_tensors, device="cuda", seed=42)``.
"""

import threading
import time
from typing import Callable

import ray
import torch
import uvicorn
from fastapi import FastAPI
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from wbridge import WeightData, WeightReceiver, WeightReceiverController, WeightSender

MetadataGenerator = Callable[[int], WeightData]
TensorGenerator = Callable[[WeightData], dict[str, torch.Tensor]]


@ray.remote
class RolloutWorker:
    """One per receiver GPU.  Receives weights via NCCL and optionally
    verifies them against ground-truth tensors."""

    def __init__(
        self,
        ipc_name: str,
        rank: int,
        metadata_generator: MetadataGenerator,
        tensor_generator: TensorGenerator,
    ):
        self.rank = rank
        self.tensor_generator = tensor_generator
        self.metadata = metadata_generator(rank)
        self.receiver = WeightReceiver(
            controller_ipc_name=ipc_name,
            rank=rank,
            metadata=self.metadata,
        )
    def ready(self):
        return True

    def receive_and_verify(self) -> dict:
        """Block until weights arrive, then verify against expected shard."""
        expected = self.tensor_generator(self.metadata)
        state_dict = {name: t.clone() for name, t in expected.items()}
        for _ in range(200):
            if self.receiver.request_update(state_dict):
                break
            time.sleep(0.5)
        else:
            return {"rank": self.rank, "ok": False, "detail": "timeout waiting for weights"}

        for name, exp in expected.items():
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
    """Hosts the receiver-side HTTP server and spawns :class:`RolloutWorker`
    actors on the same node."""

    def __init__(
        self,
        addr: str,
        port: int,
        node_id: str,
        num_workers: int,
        metadata_generator: MetadataGenerator,
        tensor_generator: TensorGenerator,
    ):
        app = FastAPI()
        self.controller = WeightReceiverController(app)

        config = uvicorn.Config(app, host=addr, port=port, log_level="warning")
        server = uvicorn.Server(config)
        threading.Thread(target=server.run, daemon=True).start()
        while not server.started:
            time.sleep(0.1)

        sched = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        self._workers = [
            RolloutWorker.options(
                num_cpus=1, num_gpus=1, scheduling_strategy=sched,
            ).remote(
                self.controller.ipc_name, rank,
                metadata_generator, tensor_generator,
            )
            for rank in range(num_workers)
        ]
        ray.get([w.ready.remote() for w in self._workers])
        self.controller.set_worker_num(num_workers)

    def gather_results(self) -> list[dict]:
        results = ray.get([w.receive_and_verify.remote() for w in self._workers])
        return sorted(results, key=lambda r: r["rank"])


@ray.remote
class TrainerWorker:
    """One per sender GPU.  Sends its weight shard via WeightBridge
    (no default ``torch.distributed`` group required)."""

    def __init__(
        self,
        world_size: int,
        rank: int,
        master_addr: str,
        master_port: int,
        metadata_generator: MetadataGenerator,
    ):
        self.world_size = world_size
        self.rank = rank
        self.metadata_generator = metadata_generator
        self.sender_init_method = f"tcp://{master_addr}:{master_port}"

    def send_weights(self, tensor_generator: TensorGenerator, receiver_url: str):
        meta = self.metadata_generator(self.rank)
        local_tensors = tensor_generator(meta)

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
