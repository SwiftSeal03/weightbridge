import ray

import torch
from wbridge.utils.data import WeightData, shards_iterator, shards_to_numel


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

def generate_local_tensors(
    metadata: WeightData, device: str, seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Create flattened shard tensors described by *metadata*.

    When *seed* is given, full tensors are generated deterministically
    (shape inferred from the ``w`` values of each shard spec) and sliced
    into local shards.  When *seed* is ``None``, zero-filled tensors are
    returned.
    """
    full_tensors: dict[str, torch.Tensor] = {}
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(seed)
        for name, shards, dtype in metadata:
            shape = tuple(w for _, _, w in shards[0])
            full_tensors[name] = torch.randn(*shape, dtype=dtype, device=device, generator=g)

    local_tensors: dict[str, torch.Tensor] = {}
    for name, shards, dtype in metadata:
        local_tensors[name] = torch.zeros(shards_to_numel(shards), dtype=dtype, device=device)
        if name in full_tensors:
            for start, end, shard in shards_iterator(metadata[name]):
                slices = tuple(slice(l, r) for l, r, _ in shard)
                local_tensors[name][start:end] = full_tensors[name][slices].reshape(-1)
    return local_tensors