# Integration Guide

A WeightBridge integration has three pieces:

- A receiver controller owned by the Rollout Engine.
- One `ReceiverAdapter` per Rollout Worker.
- One `SenderAdapter` per Trainer Worker.

The examples below show the generic API. Framework-specific adapters such as `WBMegatronAdapter` and `WBSGLangAdapter` build the same lower-level pieces for supported runtimes.

## Receiver Controller

The controller owns the HTTP API that sender rank 0 calls and the ZMQ endpoint used by local Rollout Workers.

```python
import threading
import time

import uvicorn
from fastapi import FastAPI

from wbridge import WeightReceiverController

app = FastAPI()
controller = WeightReceiverController(app)

server = uvicorn.Server(uvicorn.Config(app, host=rollout_host, port=rollout_port))
threading.Thread(target=server.run, daemon=True).start()
while not server.started:
    time.sleep(0.1)

controller_ipc_name = controller.ipc_name
controller.set_worker_num(num_rollout_workers)
```

Call `set_worker_num(n)` after all Rollout Workers have been launched and before sender rank 0 posts `/wbridge/connect` or `/wbridge/receive`.

## Rollout Worker

Each Rollout Worker exposes its runtime state dict, existing framework `load_weights` function, and a fresh HuggingFace tensor iterator through `AdapterContext`.

```python
from wbridge.frontend.adapters import AdapterContext, ReceiverAdapter

ctx = AdapterContext(
    hf_iter_factory=hf_iter_factory,
    wksd=model_state_dict,
    load_weights=model_load_weights,
    load_spec_path=f"/tmp/wbridge_loadspec/rollout_rank{rank}.json",
    rank=rank,
)
adapter = ReceiverAdapter(ctx, controller_ipc_name)

if adapter.request_update():
    pass  # new weights were applied into model_state_dict
```

`request_update()` returns `True` when a pending update was consumed and loaded into runtime tensors. It returns `False` when no update is ready.

In Async mode, Rollout Workers should call `request_update()` from the beginning of the event loop or scheduler tick. In practice, the caller should only do cleanup and load coordination when the receiver reports that a weight update is ready.

## Trainer Worker

Each Trainer Worker builds an `AdapterContext` for local runtime weights, creates `SenderArgs`, connects once, then sends whenever the Rollout Engine should receive a new policy version.

```python
from wbridge.backend.sender import SenderArgs
from wbridge.frontend.adapters import AdapterContext, SenderAdapter

ctx = AdapterContext(
    hf_iter_factory=hf_iter_factory,
    wksd=model_state_dict,
    load_weights=model_load_weights,
    load_spec_path=f"/tmp/wbridge_loadspec/actor_rank{rank}.json",
    rank=rank,
)

sender_args = SenderArgs(
    world_size=num_trainer_workers,
    transfer_mode="gpu_direct",
    receiver_urls=[f"http://{rollout_host}:{rollout_port}"],
    master_addr=trainer_host,
    master_port=trainer_pg_port,
)

adapter = SenderAdapter(ctx, sender_args)
adapter.connect()
adapter.send_weights()
```

Call `connect()` before the first send. Call `send_weights()` for each weight update.

## Sync Mode And Async Mode

WeightBridge exposes the same adapter calls for Sync mode and Async mode, but the integration point differs:

- **Sync mode**: Trainer Workers call `send_weights()` at the update barrier, and Rollout Workers load the update before continuing rollout.
- **Async mode**: Trainer Workers can return after offloading weights, and Rollout Workers call `request_update()` opportunistically from their event loop.

The current public API expresses this through sender calls and receiver polling. Higher-level engines decide where those calls sit in their training or rollout loop.

## Integration Requirements

- `hf_iter_factory` must return a fresh `(name, tensor)` iterator each time it is called.
- `wksd` tensors are expected to be CUDA tensors for `LoadSpec` inference.
- `load_weights` must consume HuggingFace-style `(name, tensor)` pairs and write into `wksd`.
- `load_spec_path` should be unique per rank and invalidated when model layout, tensor names, tensor-parallel size, or inference logic changes.
