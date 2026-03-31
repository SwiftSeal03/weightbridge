"""Reusable Ray actor definitions for WeightBridge sender / receiver pipelines.

Configuration is passed as a single :class:`EngineArgs` instance to every
engine and worker ``init`` path.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import ray
import torch
import uvicorn
from fastapi import FastAPI
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from wbridge import WeightData, WeightReceiver, WeightReceiverController, WeightSender

MetadataGenerator = Callable[[int], WeightData]
TensorGenerator = Callable[[WeightData], dict[str, torch.Tensor]]


@dataclass
class EngineArgs:
    rollout_host: str
    rollout_port: int
    rollout_scheduling_strategy: NodeAffinitySchedulingStrategy
    num_rollout_workers: int
    rollout_metadata_generator: MetadataGenerator

    trainer_host: str
    trainer_pg_port: int
    trainer_scheduling_strategy: NodeAffinitySchedulingStrategy
    num_trainer_workers: int
    trainer_metadata_generator: MetadataGenerator

    tensor_generator: TensorGenerator
    rollout_controller_ipc_name: str = field(init=False)


@ray.remote(num_gpus=1, num_cpus=1)
class RolloutWorker:
    """One per receiver GPU.  Receives weights via NCCL and optionally
    verifies them against ground-truth tensors."""

    def init(self, rank: int, args: EngineArgs):
        self.rank = rank
        self.args = args
        self.metadata = args.rollout_metadata_generator(rank)
        self.state_dict = args.tensor_generator(self.metadata)
        self.receiver = WeightReceiver(
            controller_ipc_name=args.rollout_controller_ipc_name,
            rank=rank,
            metadata=self.metadata,
        )

    def receive_and_verify(self) -> dict:
        """Block until weights arrive, then verify against expected shard."""
        recv_state_dict = {name: t.clone() for name, t in self.state_dict.items()}
        for _ in range(10):
            if self.receiver.request_update(recv_state_dict):
                break
            time.sleep(1)
        else:
            return {"rank": self.rank, "ok": False, "detail": "timeout waiting for weights"}

        if all(
            torch.allclose(exp, got, rtol=1e-5, atol=1e-6) 
            for exp, got in zip(self.state_dict.values(), recv_state_dict.values(), strict=True)
        ):
            return {"rank": self.rank, "ok": True, "detail": "all tensors match"}
        else:
            return {"rank": self.rank, "ok": False, "detail": "some tensors do not match"}


@ray.remote(num_cpus=1)
class RolloutEngine:
    """Hosts the receiver-side HTTP server and spawns :class:`RolloutWorker`
    actors on the same node."""

    def init(self, args: EngineArgs):
        app = FastAPI()
        self.controller = WeightReceiverController(app)
        args.rollout_controller_ipc_name = self.controller.ipc_name

        # Start the HTTP server
        config = uvicorn.Config(app, host=args.rollout_host, port=args.rollout_port)
        server = uvicorn.Server(config)
        threading.Thread(target=server.run, daemon=True).start()
        while not server.started:
            time.sleep(0.1)
        print(f"RolloutEngine started on {args.rollout_host}:{args.rollout_port}")

        # Spawn RolloutWorkers
        n = args.num_rollout_workers
        self._workers = [RolloutWorker.options(scheduling_strategy=args.rollout_scheduling_strategy).remote() for _ in range(n)]
        ray.get([w.init.remote(rank, args)for rank, w in enumerate(self._workers)])
        
        self.controller.set_worker_num(n)

    def receive_and_verify_all(self) -> str:
        results = ray.get([w.receive_and_verify.remote() for w in self._workers])
        results = sorted(results, key=lambda r: r["rank"])
        for r in results:
            if not r["ok"]:
                return f"RolloutWorker rank {r['rank']} failed: {r['detail']}"
        return "All RolloutWorkers verified their shards independently."


@ray.remote(num_gpus=1, num_cpus=1)
class TrainerWorker:
    """One per sender GPU.  Sends its weight shard via WeightBridge
    (no default ``torch.distributed`` group required)."""

    def init(self, rank: int, args: EngineArgs):
        self.args = args
        self.rank = rank
        self.metadata = args.trainer_metadata_generator(rank)
        self.state_dict = args.tensor_generator(self.metadata)
        self.sender = WeightSender(
            transfer_mode="gpu_direct",
            receiver_urls=[f"http://{args.rollout_host}:{args.rollout_port}"],
            rank=rank,
            world_size=args.num_trainer_workers,
            master_addr=args.trainer_host,
            master_port=args.trainer_pg_port,
        )

    def send_weights(self):
        self.sender.connect(self.metadata)
        self.sender.send(self.state_dict)


class TrainerEngine:
    """Non-Ray manager that creates and drives :class:`TrainerWorker` actors."""

    def __init__(self, args: EngineArgs):
        # Spawn TrainerWorkers
        n = args.num_trainer_workers
        self._workers = [TrainerWorker.options(scheduling_strategy=args.trainer_scheduling_strategy).remote() for _ in range(n)]
        ray.get([w.init.remote(rank, args) for rank, w in enumerate(self._workers)])
        print(f"TrainerEngine started on {args.trainer_host}:{args.trainer_pg_port}")

    def send_weights(self):
        ray.get([w.send_weights.remote() for w in self._workers])
