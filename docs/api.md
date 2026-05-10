# API Reference

This page summarizes the practical integration API. See the source modules for lower-level transport details.

## `SenderArgs`

Transport configuration for `WeightSender` and `SenderAdapter`.

Important fields:

- `world_size`: number of Trainer Worker sender ranks.
- `transfer_mode`: either `"gpu_direct"` or `"cpu_direct"`.
- `receiver_urls`: HTTP base URLs for receiver controllers.
- `master_addr` and `master_port`: rendezvous address used to create the custom PyTorch process group.

## `AdapterContext`

Framework-neutral inputs required to infer or load a `LoadSpec`.

Important fields:

- `hf_iter_factory`: callable returning a fresh iterator of `(name, cpu_tensor)` HuggingFace weights.
- `wksd`: runtime worker state dict. Values are expected to be CUDA tensors for spec inference.
- `load_weights`: framework loader callable that consumes HuggingFace-style weights and writes into `wksd`.
- `load_spec_path`: per-rank JSON cache path.
- `rank`: local adapter rank.

## `SenderAdapter`

High-level Trainer Worker integration. It infers or loads `LoadSpec`, constructs a `WeightSender`, packs local runtime tensors into communication buffers, and sends them to Rollout Workers.

```python
adapter = SenderAdapter(ctx, sender_args)
adapter.connect()
adapter.send_weights()
```

## `ReceiverAdapter`

High-level Rollout Worker integration. It infers or loads `LoadSpec`, constructs a `WeightReceiver`, receives communication buffers, and applies them into local runtime tensors.

```python
adapter = ReceiverAdapter(ctx, controller_ipc_name)
updated = adapter.request_update()
```

`request_update()` returns `True` when an update was consumed and `False` when nothing is ready.

## `WeightReceiverController`

Receiver-side control surface. It registers these routes on the provided FastAPI app:

- `GET /wbridge/receiver_world`
- `POST /wbridge/connect`
- `POST /wbridge/receive`

Create one controller per Rollout Engine, pass `controller.ipc_name` to local Rollout Workers, and call `set_worker_num(n)` once the workers are launched.

## `WBMegatronAdapter`

Framework-specific sender adapter for Megatron-Bridge. It builds the HuggingFace tensor iterator, discovers Megatron conversion tasks, exposes Megatron parameter tensors as `wksd`, and reuses `SenderAdapter` for transfer.

Import it from its specific module so environments without Megatron-Bridge do not need to import that dependency:

```python
from wbridge.frontend.megatron_adapter import WBMegatronAdapter
```

## `WBSGLangAdapter`

Framework-specific receiver adapter for SGLang. It uses SGLang's model loader and `model.load_weights` path to infer the receiver-side `LoadSpec`, then reuses `ReceiverAdapter` for transfer.

Import it from its specific module so environments without SGLang do not need to import that dependency:

```python
from wbridge.frontend.sglang_adapter import WBSGLangAdapter
```
