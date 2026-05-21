"""Minimal WeightBridge example: Ray node pinning + weight transfer.

Uses a single-layer Qwen2-style HF checkpoint built on each worker via ``build_checkpoint``.
Trainer Workers hold TP shards
of HF names in ``wksd``; Rollout Workers hold merged weights (``qkv_proj``, ``gate_up_proj``, ...).
:class:`~wbridge.frontend.adapters.SenderAdapter` / :class:`~wbridge.frontend.adapters.ReceiverAdapter`
infer per-rank :class:`~wbridge.utils.data.LoadSpec` JSON under *load_spec_dir*.

Use ``--transfer-mode gpu_direct`` (NCCL, default) or ``--transfer-mode cpu_direct`` (Gloo, CPU wire
buffers; single router round; sender return can be decoupled from receive completion). ``wksd`` tensors stay on GPU in both modes.
Use ``--network-provider efa`` on AWS EFA clusters, or leave the default ``tcp`` for regular TCP/IP
NCCL/Gloo networking.

Usage::

    ray start --head          # node A (GPUs for trainer + rollout workers)
    ray start --address=...   # node B
    python examples/train.py [--transfer-mode gpu_direct|cpu_direct]
"""

from __future__ import annotations

import argparse
import logging
import os
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


def configure_network_env(network_provider: str, network_interface: str) -> None:
    """Configure process-group networking for the selected provider."""
    if network_interface:
        os.environ.setdefault("NCCL_SOCKET_IFNAME", network_interface)
        os.environ.setdefault("GLOO_SOCKET_IFNAME", network_interface)

    if network_provider == "tcp":
        return
    if network_provider != "efa":
        raise ValueError(f"Unsupported network provider: {network_provider}")

    ld_paths = ["/opt/amazon/ofi-nccl/lib", "/opt/amazon/efa/lib"]
    existing_ld = os.environ.get("LD_LIBRARY_PATH")
    if existing_ld:
        ld_paths.append(existing_ld)
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_paths)

    os.environ.setdefault("FI_PROVIDER", "efa")
    os.environ.setdefault("FI_EFA_USE_DEVICE_RDMA", "0")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,NET")
    os.environ.setdefault("NCCL_NET_GDR_LEVEL", "SYS")
    os.environ.setdefault("NCCL_NET_GDR_READ", "1")


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
    parser.add_argument("--rollout-ip", default=os.environ.get("WB_ROLLOUT_IP"))
    parser.add_argument("--trainer-ip", default=os.environ.get("WB_TRAINER_IP"))
    parser.add_argument(
        "--network-provider",
        choices=("tcp", "efa"),
        default=os.environ.get("WB_NETWORK_PROVIDER", "tcp"),
        help="Process-group network provider. Use efa on AWS EFA clusters; tcp only sets socket IFNAME.",
    )
    parser.add_argument("--network-interface", default=os.environ.get("WB_NETWORK_INTERFACE", "ens6"))
    parser.add_argument("--rollout-port", type=int, default=int(os.environ.get("WB_ROLLOUT_PORT", "15000")))
    parser.add_argument("--trainer-pg-port", type=int, default=int(os.environ.get("WB_TRAINER_PG_PORT", "60010")))
    parser.add_argument(
        "--load-spec-dir",
        default=os.environ.get("WB_LOAD_SPEC_DIR", str(Path(tempfile.gettempdir()) / "wbridge_example_qwen_loadspec_v1")),
    )
    cli = parser.parse_args()
    configure_network_env(cli.network_provider, cli.network_interface)

    rollout_ip, trainer_ip, rollout_node_id, trainer_node_id = get_ray_nodes(cli.rollout_ip, cli.trainer_ip)
    logger.info(
        "Using rollout node %s, trainer node %s, network provider %s, network interface %s, transfer mode %s",
        rollout_ip,
        trainer_ip,
        cli.network_provider,
        cli.network_interface,
        cli.transfer_mode,
    )
    # Bump dirname if layout / LoadSpec format changes (stale JSON under old dirs still hurts until deleted).
    load_spec_dir = cli.load_spec_dir
    Path(load_spec_dir).mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_QWEN_TINY_CONFIG
    build_checkpoint = partial(build_qwen_tiny_hf_checkpoint, cfg, dtype=DTYPE, seed=42, device="cpu")

    engine_args = EngineArgs(
        rollout_host=rollout_ip,
        rollout_port=cli.rollout_port,
        rollout_scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=rollout_node_id, soft=False),
        num_rollout_workers=2,
        trainer_host=trainer_ip,
        trainer_pg_port=cli.trainer_pg_port,
        trainer_scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=trainer_node_id, soft=False),
        num_trainer_workers=2,
        model_config=cfg,
        build_checkpoint=build_checkpoint,
        load_spec_dir=load_spec_dir,
        dtype=DTYPE,
        transfer_mode=cli.transfer_mode,
        network_provider=cli.network_provider,
        network_interface=cli.network_interface,
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
