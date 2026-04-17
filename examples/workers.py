"""Reusable Ray actor definitions for WeightBridge sender / receiver pipelines.

Configuration is passed as a single :class:`EngineArgs` instance. HF weights are not serialized in
``EngineArgs``; each worker calls ``build_checkpoint()`` locally so checkpoints are identical across
nodes without shipping CPU tensors through Ray. Trainer (**actor**) workers use a TP shard of HF
names in ``wksd``; **rollout** workers use merged names (``qkv_proj``, ``gate_up_proj``, …).
Wire :class:`~wbridge.ShardSpec` layouts (per-rank HF boxes) are built here from
:class:`~qwen_tiny.QwenTinyConfig`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import ray
import torch
import uvicorn
from fastapi import FastAPI
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from wbridge import ShardSpec, WeightReceiverController

from adapters import ExampleReceiverAdapter, ExampleSenderAdapter
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


def _shard_col_major(rows: int, cols: int, tp_rank: int, tp_size: int) -> list[tuple[int, int, int]]:
    part = rows // tp_size
    lo = tp_rank * part
    hi = lo + part
    return [(lo, hi, rows), (0, cols, cols)]


def _shard_row_major(rows: int, cols: int, tp_rank: int, tp_size: int) -> list[tuple[int, int, int]]:
    part = cols // tp_size
    lo = tp_rank * part
    hi = lo + part
    return [(0, rows, rows), (lo, hi, cols)]


def _shard_1d(n: int) -> list[tuple[int, int, int]]:
    return [(0, n, n)]


def _shard_embed_rows(cfg: QwenTinyConfig, rank: int, world: int) -> list[tuple[int, int, int]]:
    part = cfg.vocab_size // world
    lo = rank * part
    hi = lo + part
    return [(lo, hi, cfg.vocab_size), (0, cfg.hidden_size, cfg.hidden_size)]


def actor_wire_shard_spec(cfg: QwenTinyConfig, tp_rank: int, tp_size: int) -> ShardSpec:
    """HF regions this actor rank sends (split Q/K/V on the wire; packing is in ``actor_load_weights``)."""
    h, qo, kvo, iq = cfg.hidden_size, cfg.q_out, cfg.kv_out, cfg.intermediate_size
    entries = {
        "model.embed_tokens.weight": _shard_embed_rows(cfg, tp_rank, tp_size),
        "self_attn.q_proj.weight": _shard_col_major(qo, h, tp_rank, tp_size),
        "self_attn.k_proj.weight": _shard_col_major(kvo, h, tp_rank, tp_size),
        "self_attn.v_proj.weight": _shard_col_major(kvo, h, tp_rank, tp_size),
        "self_attn.o_proj.weight": _shard_row_major(h, qo, tp_rank, tp_size),
        "mlp.gate_proj.weight": _shard_col_major(iq, h, tp_rank, tp_size),
        "mlp.up_proj.weight": _shard_col_major(iq, h, tp_rank, tp_size),
        "mlp.down_proj.weight": _shard_row_major(h, iq, tp_rank, tp_size),
        "input_layernorm.weight": _shard_1d(h),
        "post_attention_layernorm.weight": _shard_1d(h),
    }
    return ShardSpec(entries)


def rollout_wire_shard_spec(cfg: QwenTinyConfig, rollout_rank: int, rollout_size: int) -> ShardSpec:
    """HF regions per rollout rank (TP-aligned, same box pattern as :func:`actor_wire_shard_spec`)."""
    return actor_wire_shard_spec(cfg, rollout_rank, rollout_size)


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


@ray.remote(num_gpus=1, num_cpus=1)
class RolloutWorker:
    """One per receiver GPU. Receives weights via NCCL and optionally verifies."""

    def init(self, rank: int, args: EngineArgs):
        self.rank = rank
        self.args = args
        cfg = args.model_config
        hf_cpu = args.build_checkpoint()
        self.hf_iter_factory = make_hf_iter_factory(hf_cpu)
        self.shard_spec = rollout_wire_shard_spec(cfg, rank, args.num_rollout_workers)
        self.state_dict = build_rollout_wksd(
            cfg,
            device="cuda",
            dtype=args.dtype,
            tp_rank=rank,
            tp_size=args.num_rollout_workers,
        )
        self.load_weights = make_rollout_load_weights(
            self.state_dict,
            cfg,
            device="cuda",
            dtype=args.dtype,
            tp_rank=rank,
            tp_size=args.num_rollout_workers,
        )
        self.load_weights(self.hf_iter_factory())
        self.adapter = ExampleReceiverAdapter(
            self.hf_iter_factory,
            self.state_dict,
            self.load_weights,
            self.shard_spec,
            rollout_load_spec_path(args.load_spec_dir, rank),
            rank=rank,
        )
        self.receiver = self.adapter.make_receiver(args.rollout_controller_ipc_name)

    def recv_weights(self) -> None:
        for _ in range(500):
            if self.receiver.is_weights_ready:
                break
            time.sleep(0.05)
        else:
            raise TimeoutError("receiver never became ready for weights")
        self.recv_state_dict = {name: torch.zeros_like(t) for name, t in self.state_dict.items()}
        self.receiver.request_update()
        self.adapter.apply_recv_buffer(self.receiver.recv_buffer, self.recv_state_dict)

    def verify(self) -> dict:
        if all(
            torch.allclose(exp, got, rtol=1e-5, atol=1e-6)
            for exp, got in zip(self.state_dict.values(), self.recv_state_dict.values(), strict=True)
        ):
            return {"rank": self.rank, "ok": True, "detail": "all tensors match"}
        return {"rank": self.rank, "ok": False, "detail": "some tensors do not match"}


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
    """One per sender GPU. Sends shards via :class:`~adapters.ExampleSenderAdapter`."""

    def init(self, rank: int, args: EngineArgs):
        self.args = args
        self.rank = rank
        cfg = args.model_config
        hf_cpu = args.build_checkpoint()
        self.hf_iter_factory = make_hf_iter_factory(hf_cpu)
        self.shard_spec = actor_wire_shard_spec(cfg, rank, args.num_trainer_workers)
        self.state_dict = build_actor_wksd(
            cfg,
            device="cuda",
            dtype=args.dtype,
            tp_rank=rank,
            tp_size=args.num_trainer_workers,
        )
        self.load_weights = make_actor_load_weights(
            self.state_dict,
            cfg,
            device="cuda",
            dtype=args.dtype,
            tp_rank=rank,
            tp_size=args.num_trainer_workers,
        )
        self.load_weights(self.hf_iter_factory())
        self.adapter = ExampleSenderAdapter(
            self.hf_iter_factory,
            self.state_dict,
            self.load_weights,
            self.shard_spec,
            actor_load_spec_path(args.load_spec_dir, rank),
            rank=rank,
        )

    def send_weights(self):
        self.adapter.connect(
            transfer_mode="gpu_direct",
            receiver_urls=[f"http://{self.args.rollout_host}:{self.args.rollout_port}"],
            world_size=self.args.num_trainer_workers,
            master_addr=self.args.trainer_host,
            master_port=self.args.trainer_pg_port,
        )
        self.adapter.send()


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
