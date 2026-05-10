# Examples

The `examples/` directory contains a minimal Ray-based transfer pipeline built around a toy Qwen2-style model layout.

## Running The Example

Start Ray, then run the trainer script:

```bash
ray start --head
ray start --address=...
python examples/train.py --transfer-mode gpu_direct
```

Use `--transfer-mode cpu_direct` to run the same example with Gloo and CPU wire buffers:

```bash
python examples/train.py --transfer-mode cpu_direct
```

## Toy Qwen Layout

The example is intentionally small, but it exercises production-like layout issues:

- `examples/qwen_tiny.py` builds a one-block Qwen-style HuggingFace checkpoint.
- Trainer Workers use an actor/Megatron-like layout with packed `self_attention.linear_qkv.weight` and `mlp.linear_fc1.weight` tensors.
- Rollout Workers use an SGLang-like layout with `self_attn.qkv_proj.weight` and `mlp.gate_up_proj.weight` tensors.
- Tensor-parallel ranks own different row or column slices.
- `LoadSpec` inference discovers QKV merge, gate/up merge, row-parallel slices, column-parallel slices, and vocab slices by probing the example loaders.

## Worker Flow

`examples/workers.py` shows the full integration pattern:

- `RolloutEngine` starts a FastAPI server and `WeightReceiverController`.
- `RolloutWorker` creates a `ReceiverAdapter` and polls `request_update()`.
- `TrainerWorker` creates a `SenderAdapter`, calls `connect()`, then calls `send_weights()`.
- The example verifies that every rollout shard matches the expected pre-send checkpoint values after transfer.

## Transfer Modes

- `gpu_direct`: uses NCCL and CUDA communication buffers. It is the main mode for GPU-to-GPU P2P transfer.
- `cpu_direct`: uses Gloo and CPU communication buffers. Model weights still live in GPU runtime tensors, but wire buffers are staged on CPU.
