"""Reusable Ray actor definitions for WeightBridge sender / receiver pipelines.

Configuration is passed as a single :class:`EngineArgs` instance. HF weights are not serialized in
``EngineArgs``; each worker calls ``build_checkpoint()`` locally so checkpoints are identical across
nodes without shipping CPU tensors through Ray. Trainer (**actor**) workers use a TP shard of HF
names in ``wksd``; **rollout** workers use merged names (``qkv_proj``, ``gate_up_proj``, ...).
HF shard layout on the wire is defined by :meth:`~wbridge.utils.data.LoadSpec.src_spec` after
LoadSpec inference inside :class:`~wbridge.frontend.adapters.SenderAdapter` /
:class:`~wbridge.frontend.adapters.ReceiverAdapter`.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import ray
import torch
import uvicorn
from fastapi import FastAPI
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from wbridge import WeightReceiverController
from wbridge.backend.sender import SenderArgs
from wbridge.frontend.adapters import AdapterContext, ReceiverAdapter, SenderAdapter

from qwen_tiny import (
    QwenTinyConfig,
    actor_load_spec_path,
    build_actor_wksd,
    build_rollout_wksd,
    make_actor_load_weights,
    make_rollout_load_weights,
    rollout_load_spec_path,
)
from utils import make_hf_iter_factory

CheckpointBuilder = Callable[[], dict[str, torch.Tensor]]


def _apply_network_interface_for_process_group(iface: str) -> None:
    if not iface:
        return
    os.environ.setdefault("NCCL_SOCKET_IFNAME", iface)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", iface)


@dataclass
class EngineArgs:
    """``build_checkpoint`` must be picklable (e.g. ``functools.partial`` of a module-level function)."""

    rollout_host: str
    rollout_port: int
    rollout_scheduling_strategy: NodeAffinitySchedulingStrategy
    num_rollout_workers: int

    trainer_host: str
    trainer_pg_port: int
    trainer_scheduling_strategy: NodeAffinitySchedulingStrategy
    num_trainer_workers: int

    model_config: QwenTinyConfig
    build_checkpoint: CheckpointBuilder
    load_spec_dir: str
    dtype: torch.dtype = torch.float32

    rollout_controller_ipc_name: str = ""
    network_interface: str = "eno1"


@ray.remote(num_gpus=1, num_cpus=1)
class RolloutWorker:
    """One per receiver GPU. Receives weights via NCCL and verifies against a pre-recv backup."""

    def init(self, rank: int, args: EngineArgs):
        _apply_network_interface_for_process_group(args.network_interface)
        self.rank = rank
        self.args = args
        cfg = args.model_config
        hf_cpu = args.build_checkpoint()
        hf_iter_factory = make_hf_iter_factory(hf_cpu)
        self.state_dict = build_rollout_wksd(
            cfg,
            device="cuda",
            dtype=args.dtype,
            tp_rank=rank,
            tp_size=args.num_rollout_workers,
        )
        load_weights = make_rollout_load_weights(
            self.state_dict,
            cfg,
            device="cuda",
            dtype=args.dtype,
            tp_rank=rank,
            tp_size=args.num_rollout_workers,
        )
        load_weights(hf_iter_factory())
        ctx = AdapterContext(
            hf_iter_factory=hf_iter_factory,
            wksd=self.state_dict,
            load_weights=load_weights,
            load_spec_path=rollout_load_spec_path(args.load_spec_dir, rank),
            rank=rank,
        )
        self.adapter = ReceiverAdapter(ctx, args.rollout_controller_ipc_name)
        # Snapshot the loaded weights before the first recv so verify() can diff backup vs received.
        self.state_dict_backup = {name: t.detach().clone() for name, t in self.state_dict.items()}

    def recv_weights(self) -> None:
        for _ in range(500):
            if self.adapter.try_receive_weights():
                return
            time.sleep(0.05)
        raise TimeoutError("receiver never became ready for weights")

    def verify(self) -> dict:
        for name, backup in self.state_dict_backup.items():
            received = self.state_dict[name]
            if not torch.allclose(backup, received):
                return {
                    "rank": self.rank,
                    "name": name,
                    "ok": False,
                    "detail": (
                        f"value mismatch for {name} on rank {self.rank}, "
                        f"expected: {backup[:, :1].view(-1)}, got: {received[:, :1].view(-1)}"
                    ),
                }
        return {"rank": self.rank, "ok": True, "detail": "all values match"}


@ray.remote(num_cpus=1)
class RolloutEngine:
    """Hosts the receiver-side HTTP server and spawns :class:`RolloutWorker` actors."""

    def init(self, args: EngineArgs):
        app = FastAPI()
        self.controller = WeightReceiverController(app)
        args.rollout_controller_ipc_name = self.controller.ipc_name

        config = uvicorn.Config(app, host=args.rollout_host, port=args.rollout_port)
        server = uvicorn.Server(config)
        threading.Thread(target=server.run, daemon=True).start()
        while not server.started:
            time.sleep(0.1)
        print(f"RolloutEngine started on {args.rollout_host}:{args.rollout_port}")

        n = args.num_rollout_workers
        self._workers = [
            RolloutWorker.options(scheduling_strategy=args.rollout_scheduling_strategy).remote() for _ in range(n)
        ]
        ray.get([w.init.remote(rank, args) for rank, w in enumerate(self._workers)])

        self.controller.set_worker_num(n)

    def recv_weights(self) -> None:
        ray.get([w.recv_weights.remote() for w in self._workers])

    def verify_all(self) -> str:
        results = ray.get([w.verify.remote() for w in self._workers])
        results = sorted(results, key=lambda r: r["rank"])
        for r in results:
            if not r["ok"]:
                return f"RolloutWorker rank {r['rank']} failed: {r['detail']}"
        return "All RolloutWorkers verified their shards independently."


@ray.remote(num_gpus=1, num_cpus=1)
class TrainerWorker:
    """One per sender GPU. Sends shards via :class:`~wbridge.frontend.adapters.SenderAdapter`."""

    def init(self, rank: int, args: EngineArgs):
        _apply_network_interface_for_process_group(args.network_interface)
        self.args = args
        self.rank = rank
        cfg = args.model_config
        hf_cpu = args.build_checkpoint()
        hf_iter_factory = make_hf_iter_factory(hf_cpu)
        self.state_dict = build_actor_wksd(
            cfg,
            device="cuda",
            dtype=args.dtype,
            tp_rank=rank,
            tp_size=args.num_trainer_workers,
        )
        load_weights = make_actor_load_weights(
            self.state_dict,
            cfg,
            device="cuda",
            dtype=args.dtype,
            tp_rank=rank,
            tp_size=args.num_trainer_workers,
        )
        load_weights(hf_iter_factory())
        ctx = AdapterContext(
            hf_iter_factory=hf_iter_factory,
            wksd=self.state_dict,
            load_weights=load_weights,
            load_spec_path=actor_load_spec_path(args.load_spec_dir, rank),
            rank=rank,
        )
        sender_args = SenderArgs(
            world_size=args.num_trainer_workers,
            transfer_mode="gpu_direct",
            receiver_urls=[f"http://{args.rollout_host}:{args.rollout_port}"],
            master_addr=args.trainer_host,
            master_port=args.trainer_pg_port,
        )
        self.adapter = SenderAdapter(ctx, sender_args)

    def send_weights(self):
        self.adapter.connect()
        self.adapter.send_weights()


class TrainerEngine:
    """Spawns :class:`TrainerWorker` actors (must run before rollout so LoadSpec exists on disk)."""

    def __init__(self, args: EngineArgs):
        n = args.num_trainer_workers
        self._workers = [
            TrainerWorker.options(scheduling_strategy=args.trainer_scheduling_strategy).remote() for _ in range(n)
        ]
        ray.get([w.init.remote(rank, args) for rank, w in enumerate(self._workers)])
        print(f"TrainerEngine started on {args.trainer_host}:{args.trainer_pg_port}")

    def send_weights(self):
        ray.get([w.send_weights.remote() for w in self._workers])
