# Integration Guide

A WeightBridge integration has three pieces:

- A receiver controller owned by the Rollout Engine.
- One `ReceiverAdapter` per Rollout Worker.
- One `SenderAdapter` per Trainer Worker.

The examples below show the generic API. Integrations should keep WeightBridge at the engine and worker boundaries: initialize once, reuse the same controller and adapters, and avoid changing internal scheduler messaging when a simple event-loop poll is enough.

## Minimal Lifecycle

1. During Rollout Engine initialization, attach `WeightReceiverController` to the engine's existing HTTP app.
2. During Rollout Worker initialization, create one long-lived `ReceiverAdapter`.
3. During Trainer Worker initialization, create one long-lived `SenderAdapter`.
4. After receiver workers are registered, call `SenderAdapter.connect()` once.
5. For each real weight update, call `SenderAdapter.send_weights()`.
6. In each Rollout Worker event-loop or scheduler tick, call `ReceiverAdapter.is_update_ready()`. Call `request_update()` only when it returns `True`.

## Receiver Controller

The controller registers WeightBridge HTTP routes on an existing Rollout Engine app and owns the ZMQ endpoint used by local Rollout Workers. Do not create a second API server when the engine already has one.

```python
from fastapi import FastAPI

from wbridge import WeightReceiverController

app = existing_rollout_engine_app  # or FastAPI() for a standalone example
controller = WeightReceiverController(app)

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

if adapter.is_update_ready():
    adapter.request_update()
    pass  # new weights were applied into model_state_dict
```

`is_update_ready()` is a lightweight readiness check. It may complete deferred connection setup, but it does not receive tensors or call `load_weights`. `request_update()` performs the actual receive/load work and returns `True` when a pending update was consumed and loaded into runtime tensors.

In Async mode, Rollout Workers should poll `is_update_ready()` from the beginning of the event loop or scheduler tick, then call `request_update()` only when the receiver reports that a weight update is ready. This keeps the integration localized to worker initialization and event-loop logic.

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

Call `connect()` once before the first send. It creates the process group and exchanges receiver metadata. Reuse that connection for later updates; call only `send_weights()` for each weight update.

## Sync Mode And Async Mode

WeightBridge exposes the same adapter calls for Sync mode and Async mode, but the integration point differs:

- **Sync mode**: Trainer Workers call `send_weights()` at the update barrier, and Rollout Workers load the update before continuing rollout.
- **Async mode**: Trainer Workers can return after offloading weights, and Rollout Workers poll `is_update_ready()` from their event loop before calling `request_update()`.

The current public API expresses this through sender calls and receiver polling. Higher-level engines decide where those calls sit in their training or rollout loop.

## Redundancy Checklist

- Do not call `connect()` before every send.
- Do not create a dedicated WeightBridge API server when the Rollout Engine already owns an HTTP app.
- Do not initialize adapters or controllers lazily from request handlers.
- Do not route weight readiness through an engine's internal messaging system when the Rollout Worker event loop can poll `is_update_ready()`.

## Integration Requirements

- `hf_iter_factory` must return a fresh `(name, tensor)` iterator each time it is called.
- `wksd` tensors are expected to be CUDA tensors for `LoadSpec` inference.
- `load_weights` must consume HuggingFace-style `(name, tensor)` pairs and write into `wksd`.
- `load_spec_path` should be unique per rank and invalidated when model layout, tensor names, tensor-parallel size, or inference logic changes.
