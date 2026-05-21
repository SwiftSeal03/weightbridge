from collections.abc import Callable, Iterator

import ray
import torch


def get_ray_nodes(rollout_ip: str | None = None, trainer_ip: str | None = None):
    """``ray.init()`` then return rollout and trainer nodes.

    Returns ``(rollout_ip, trainer_ip, rollout_node_id, trainer_node_id)``.
    """
    ray.init(address="auto")
    ray_nodes = [n for n in ray.nodes() if n["Alive"]]
    if len(ray_nodes) < 2:
        raise RuntimeError("Need at least two alive Ray nodes.")

    by_ip = {str(n["NodeManagerAddress"]): n for n in ray_nodes}
    if rollout_ip is not None or trainer_ip is not None:
        missing = [ip for ip in (rollout_ip, trainer_ip) if ip is not None and ip not in by_ip]
        if missing:
            alive = ", ".join(sorted(by_ip))
            raise RuntimeError(f"Requested Ray node IP(s) not alive: {missing}. Alive Ray IPs: {alive}")
        if rollout_ip is None or trainer_ip is None:
            raise RuntimeError("Pass both rollout_ip and trainer_ip, or neither.")
        rollout, trainer = by_ip[rollout_ip], by_ip[trainer_ip]
    else:
        rollout, trainer = sorted(ray_nodes, key=lambda n: str(n["NodeManagerAddress"]))[:2]

    return (
        str(rollout["NodeManagerAddress"]),
        str(trainer["NodeManagerAddress"]),
        str(rollout["NodeID"]),
        str(trainer["NodeID"]),
    )


def make_hf_iter_factory(
    full_cpu: dict[str, torch.Tensor],
) -> Callable[[], Iterator[tuple[str, torch.Tensor]]]:
    """Factory of CPU tensor iterators (each call is a fresh pass for verify / infer).

    Yields **clones** so :func:`~wbridge.utils.specgen.infer_load_spec` can mutate probe tensors
    (``fill_``, etc.) without corrupting the shared *full_cpu* checkpoint dict.
    """

    def factory() -> Iterator[tuple[str, torch.Tensor]]:
        for name, t in full_cpu.items():
            yield name, t.clone().contiguous()

    return factory
