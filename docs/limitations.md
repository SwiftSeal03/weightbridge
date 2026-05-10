# Simplifications, Assumptions, and Limitations

This page collects current WeightBridge caveats for performance, algorithm behavior, `LoadSpec` inference, and operation.

## Performance

- Data packing and reconstruction are currently sequential, tensor by tensor.
- Weight chunks are transferred with `torch.batch_isend_irecv`.
- In the face of rollout replicas, Rollout Worker bandwidth may be underutilized.
- In chunked sending, GPU communication and weight loading are not overlapped.

## Algorithm

- Senders currently do not perform data deduplication.
- For DP, the user is responsible for matching rollout engines with workers.
- `WeightRouter` schedules communication from sender and receiver `ShardSpec` overlaps; it does not infer higher-level replica equivalence.

## `LoadSpec` Inference

`LoadSpec` inference assumes the framework `load_weights` function can be probed symbolically and that the resulting mapping can be represented as rectangular shard mappings.

Current assumptions and limitations:

- `load_weights` performs no numerical changes other than PyTorch type casts.
- `ShardMapping`s only exist between tensors with the same dimensionality.
- A component is a maximal continuous set of elements, connected in the grid graph, in a source tensor that loads into the same destination tensor while preserving relative positions after `load_weights` is called.
- All such components must be rectangular.
- The width in all dimensions of <=2D components must be greater than 1.
- Only one component is supported in the mapping between any two >2D tensors.

## Operational Caveats

- `infer_load_spec` expects worker state dict tensors to be CUDA tensors. The probing logic mutates runtime tensors during inference, then restores them.
- `hf_iter_factory` must return a fresh iterator each time. If underlying HuggingFace tensors are reused, yield clones so probing does not corrupt shared CPU state.
- `gpu_direct` uses NCCL and CUDA wire buffers.
- `cpu_direct` uses Gloo and CPU wire buffers; model weights still live in GPU runtime tensors.
- Network interface selection may matter. The example sets `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` from `EngineArgs.network_interface`.
- `LoadSpec` JSON files are cached per rank. Delete stale cache files when model layout, tensor names, tensor-parallel size, or spec inference logic changes.
- Framework-specific adapters import heavy optional dependencies only from their own modules. Import `wbridge.frontend.adapters` for the generic API.
- The Ray example assumes a two-node style layout and builds identical toy checkpoints locally on each worker instead of shipping checkpoint tensors through Ray.
- `WeightReceiverController.set_worker_num(n)` must be called before sender rank 0 posts `/wbridge/connect` or `/wbridge/receive`.
