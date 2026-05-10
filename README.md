# WeightBridge

WeightBridge is an RL weight transfer library for moving sharded model weights from Trainer Workers to Rollout Workers when the two sides use different parameter names, tensor layouts, or parallelism strategies. It infers how HuggingFace checkpoint tensors map into each runtime, computes sender/receiver shard overlaps, and transfers only the needed slices through PyTorch P2P.

## Why WeightBridge?

- Replaces wasteful all-gather/broadcast weight sync with fine-grained peer-to-peer transfer.
- Supports flexible RL configurations where trainer and rollout runtimes use different sharding layouts.
- Avoids configuration-specific hard-coding by inferring `LoadSpec` metadata from existing `load_weights` functions.
- Organizes weight transfer across a Data Plane, Metadata Plane, and Control Plane.
- Provides one integration path for Sync mode and Async mode weight updates.

## Installation

Install the package from the repository root:

```bash
pip install -e .
```

WeightBridge requires Python `>=3.10` and depends on `torch`, `pyzmq`, and `fastapi`. Some examples or framework adapters may additionally require `ray`, `uvicorn`, `sglang`, `megatron.bridge`, or `safetensors`.

## Tiny Integration Sketch

A rollout engine starts a `WeightReceiverController`, each Rollout Worker owns a `ReceiverAdapter`, and each Trainer Worker owns a `SenderAdapter`.

```python
# Rollout Worker
receiver = ReceiverAdapter(receiver_ctx, controller_ipc_name)
if receiver.request_update():
    pass  # new weights were loaded into the rollout runtime
```

```python
# Trainer Worker
sender = SenderAdapter(sender_ctx, sender_args)
sender.connect()
sender.send_weights()
```

See [Integration Guide](docs/integration.md) for the complete setup.

## Documentation

- [Motivation](docs/motivation.md): RL weight transfer background and why WeightBridge uses P2P.
- [Architecture](docs/architecture.md): Data Plane, Metadata Plane, and Control Plane design.
- [Integration Guide](docs/integration.md): controller, receiver, and sender setup.
- [API Reference](docs/api.md): practical reference for the public integration classes.
- [Examples](docs/examples.md): Ray toy Qwen walkthrough and transfer modes.
- [Limitations](docs/limitations.md): assumptions, caveats, and current simplifications.

## Development

The main tests cover receiver metadata routes, shard compatibility, overlap packing/unpacking, and `LoadSpec` inference:

```bash
python -m pytest tests/test_query_receivers_metadata.py
python -m pytest tests/test_shard_compatibility.py
python -m pytest tests/test_specgen.py
```
