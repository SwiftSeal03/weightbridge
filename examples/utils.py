import ray


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
