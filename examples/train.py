"""Minimal WeightBridge example: Ray node pinning + weight transfer.

Uses a single-layer Qwen2-style HF checkpoint built on each worker via ``build_checkpoint``.
Trainer Workers hold TP shards
of HF names in ``wksd``; Rollout Workers hold merged weights (``qkv_proj``, ``gate_up_proj``, ...).
:class:`~wbridge.frontend.adapters.SenderAdapter` / :class:`~wbridge.frontend.adapters.ReceiverAdapter`
infer per-rank :class:`~wbridge.utils.data.LoadSpec` JSON under *load_spec_dir*.

Use ``--transfer-mode gpu_direct`` (NCCL, default) or ``--transfer-mode cpu_direct`` (Gloo, CPU wire
buffers; single router round; sender return can be decoupled from receive completion). ``wksd`` tensors stay on GPU in both modes.

Usage::

    ray start --head          # node A (GPUs for trainer + rollout workers)
    ray start --address=...   # node B
    python examples/train.py [--transfer-mode gpu_direct|cpu_direct]
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from functools import partial
from pathlib import Path

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from qwen_tiny import DEFAULT_QWEN_TINY_CONFIG, build_qwen_tiny_hf_checkpoint
from utils import get_ray_nodes
from workers import EngineArgs, RolloutEngine, TrainerEngine

logger = logging.getLogger("example")

DTYPE = torch.float32


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s",
    )

    parser = argparse.ArgumentParser(description="WeightBridge train/rollout example")
    parser.add_argument(
        "--transfer-mode",
        choices=("gpu_direct", "cpu_direct"),
        default="gpu_direct",
        help="gpu_direct: NCCL GPU buffers; cpu_direct: Gloo CPU buffers with overlapped send/recv",
    )
    cli = parser.parse_args()

    rollout_ip, trainer_ip, rollout_node_id, trainer_node_id = get_ray_nodes()
    # Bump dirname if layout / LoadSpec format changes (stale JSON under old dirs still hurts until deleted).
    load_spec_dir = str(Path(tempfile.gettempdir()) / "wbridge_example_qwen_loadspec_v3")
    Path(load_spec_dir).mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_QWEN_TINY_CONFIG
    build_checkpoint = partial(build_qwen_tiny_hf_checkpoint, cfg, dtype=DTYPE, seed=42, device="cpu")

    engine_args = EngineArgs(
        rollout_host=rollout_ip,
        rollout_port=15000,
        rollout_scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=rollout_node_id, soft=False),
        num_rollout_workers=2,
        trainer_host=trainer_ip,
        trainer_pg_port=60010,
        trainer_scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=trainer_node_id, soft=False),
        num_trainer_workers=2,
        model_config=cfg,
        build_checkpoint=build_checkpoint,
        load_spec_dir=load_spec_dir,
        dtype=DTYPE,
        transfer_mode=cli.transfer_mode,
    )

    trainer_engine = TrainerEngine(engine_args)
    rollout_engine = RolloutEngine.options(
        scheduling_strategy=engine_args.rollout_scheduling_strategy
    ).remote()
    ray.get(rollout_engine.init.remote(engine_args))

    recv_future = rollout_engine.recv_weights.remote()
    trainer_engine.send_weights()
    ray.get(recv_future)
    logger.info("Weights received")

    results = ray.get(rollout_engine.verify_all.remote())
    logger.info(results)

    ray.shutdown()


if __name__ == "__main__":
    main()
