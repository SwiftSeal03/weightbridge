# Motivation

Modern RL training systems repeatedly transfer model weights from a training runtime to rollout or inference runtimes. Those runtimes often do not store parameters in the same format: the Trainer Workers may use Megatron-style tensor-parallel shards, while Rollout Workers may use SGLang or vLLM-style merged runtime tensors.

WeightBridge exists to make this transfer efficient and reusable across flexible RL configurations.

## RL Weight Transfer Workflow

A typical RL workflow has a Trainer Engine that advances the policy and a Rollout Engine that serves prompts, generates samples, and periodically refreshes its model weights. The weight update path connects Trainer Workers, which own updated parameters, to Rollout Workers, which need those parameters loaded into their serving runtime.

The transfer mode changes the system pressure:

- **On-policy**: rollout waits for the latest policy, so weight transfer sits directly on the critical path.
- **1-off-policy**: rollout can use a recent policy version, but stale weights still affect convergence and scheduling.
- **Unbounded off-policy**: rollout can lag further behind, making asynchronous update mechanics more important.

Collocated and Sync modes mainly care about weight transfer efficiency for training throughput. Async mode also cares about convergence, because update delay changes the policy version used by rollout.

## Why All-Gather/Broadcast Is Wasteful

Many existing RL frameworks use all-gather plus broadcast to move weights. This is easy to implement because every side reconstructs broad tensor views and can reuse standard collective primitives, but it wastes bandwidth when the sender and receiver need only overlapping shards.

The waste grows with model parallelism:

- TP8 can cause roughly 8x traffic for tensors that should be routed to a smaller set of peers.
- EP256 can cause roughly 256x traffic for expert weights when only a subset is needed by each destination.
- Larger sparse or MoE models make this more visible because model size grows while activated parameters per token remain comparatively stable.

WeightBridge uses peer-to-peer routing so each receiver gets only the tensor slices it needs.

## Why P2P Is Harder

P2P transfer requires more metadata and scheduling than all-gather/broadcast:

- **Data matching**: the system must know which sender owns each source slice and which receiver needs it.
- **Name mappings**: runtime parameter names may not match HuggingFace checkpoint names.
- **Tensor transformations**: loaders may merge, split, or shard tensors differently on each side.
- **Communication scheduling**: message sizes can be unequal, and GPU buffers are limited.
- **Packetization limits**: arbitrary slicing is not always natural for model parameters or loader logic.

WeightBridge handles this through `LoadSpec` inference, `ShardSpec` overlap computation, and a routed P2P Data Plane.

## Why Existing Frameworks Are Hard To Reuse

Some production systems implement P2P weight transfer, but their logic is often hard-coded for one model layout, one sharding strategy, or one training mode. That is fragile because the RL weight transfer configuration space is large:

- Different model families and sharding configs change parameter format translation.
- On-policy and off-policy execution change Sync mode versus Async mode requirements.
- Framework choices such as SLIME, VeRL, AReaL, StreamRL, and Laminar expose different assumptions about collocation, sync, async, CPU staging, or GPU buffers.

WeightBridge separates format translation, shard matching, and control flow so a small integration layer can support many runtime combinations without rewriting the transfer path for each configuration.
