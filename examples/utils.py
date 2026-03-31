import ray

import torch
from wbridge.utils.data import WeightData, shards_iterator, shards_to_numel


def init_ray_and_get_rollout_trainer():
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

def build_local_tensors(meta: WeightData, tensors: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Create tensor shards from either provided tensors or zeros"""
    local_tensors = {}
    for name, shards, dtype in meta:
        slices = []
        local_tensors[name] = torch.zeros(shards_to_numel(shards), dtype=dtype, device=device)
        if name in tensors:
            for start, end, shard in shards_iterator(meta[name]):
                slices = [slice(l, r) for l, r, _ in shard]
                local_tensors[name][start:end] = tensors[name][tuple(slices)].reshape(-1)
    return local_tensors