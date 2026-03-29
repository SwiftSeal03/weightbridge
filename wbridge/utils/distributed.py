from datetime import timedelta
from typing import Any
import socket
import os

import torch
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)


def get_local_ip() -> str:
    """Return the local IP address and a new port."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    
    
def get_full_group_port() -> int:
    return 60000 + int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")[0]) * 100


def init_custom_process_group(
    backend: str | Backend = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store=None,
    group_name: str = None,
    pg_options: Any | None = None,
):
    """Create a named process group without touching the default group.

    Mirrors ``slime.utils.distributed_utils.init_process_group`` and
    ``sglang.srt.utils.common.init_custom_process_group``.
    """
    assert (store is None) or (init_method is None), (
        "Cannot specify both init_method and store."
    )

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        store = PrefixStore(group_name, store)

    pg_options_param_name = (
        "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg
