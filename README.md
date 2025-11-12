### PyTorch Transformer Inference Engine

GPT-2 inference implementation with KV caching for efficient generation.

## Implementation

- Base architecture: Adapted from [nanoGPT](https://github.com/karpathy/nanoGPT)
- **KV caching**: My implementation for 4x speedup
- **Profiling & benchmarking**: My performance analysis

## My Contributions

1. Implemented KV cache mechanism in attention layers
2. Profiled inference bottlenecks
3. Benchmarked speedup (4.2x on 512-token sequences)
