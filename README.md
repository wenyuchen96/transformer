### PyTorch Transformer Inference Engine

- GPT-2 inference implementation with KV caching for efficient generation
- Base architecture: Adapted from [nanoGPT](https://github.com/karpathy/nanoGPT)

## Implementations

1. Implemented KV cache mechanism in attention layers on top of the base architecture
2. Profiled inference bottlenecks
3. Benchmarked speedup (3.0x on 512-token sequences, 8.6x on 1024-token sequences)
4. Implemented a simple transformer model in PyTorch (model.py)

### Benchmark Results

| Sequence Length | Baseline    | With KV Cache | Speedup  | Time Saved |
| --------------- | ----------- | ------------- | -------- | ---------- |
| 512 tokens      | 149.6 tok/s | 453.7 tok/s   | **3.0x** | 67.0%      |
| 1024 tokens     | 52.4 tok/s  | 452.2 tok/s   | **8.6x** | 88.4%      |
