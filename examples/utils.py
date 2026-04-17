import ray
import torch
from collections.abc import Callable, Iterator


def get_ray_nodes():
    """``ray.init()`` then return rollout (``alive[0]``) and trainer (``alive[1]``).

    Returns ``(rollout_ip, trainer_ip, rollout_node_id, trainer_node_id)``.
    """
    ray.init()
    ray_nodes = [n for n in ray.nodes() if n["Alive"]]
    if len(ray_nodes) < 2:
        raise RuntimeError("Need at least two alive Ray nodes.")
    rollout, trainer = ray_nodes[0], ray_nodes[1]
    return (
        str(rollout["NodeManagerAddress"]),
        str(trainer["NodeManagerAddress"]),
        str(rollout["NodeID"]),
        str(trainer["NodeID"]),
    )


def make_hf_iter_factory(
    full_cpu: dict[str, torch.Tensor],
) -> Callable[[], Iterator[tuple[str, torch.Tensor]]]:
    """Factory of CPU tensor iterators (each call is a fresh pass for verify / infer)."""

    def factory() -> Iterator[tuple[str, torch.Tensor]]:
        for name, t in full_cpu.items():
            yield name, t.contiguous()

    return factory
