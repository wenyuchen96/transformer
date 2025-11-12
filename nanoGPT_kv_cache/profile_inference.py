import torch
from torch.profiler import profile, record_function, ProfilerActivity
from model_nanogpt import GPT
from tiktoken import get_encoding

print("Loading GPT-2...")
model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Prepare input
enc = get_encoding("gpt2")
prompt = "Hello, I am a language model"
x = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)[None, ...]

print(f"\nProfiling on device: {device}")
print("=" * 70)

# Profile WITHOUT KV cache
print("\n1. Profiling WITHOUT KV Cache (baseline)")
print("-" * 70)

with profile(
    activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == 'cuda' else []),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("generate_no_cache"):
        with torch.no_grad():
            _ = model.generate(x, 100, temperature=1.0, top_k=200)

# Print results
print(prof.key_averages().table(
    sort_by="cpu_time_total" if device == 'cpu' else "cuda_time_total",
    row_limit=15
))

# Save detailed results
prof.export_chrome_trace("trace_no_cache.json")
print("\n✓ Detailed trace saved to: trace_no_cache.json")

print("\n" + "=" * 70)

# Profile WITH KV cache
print("\n2. Profiling WITH KV Cache (optimized)")
print("-" * 70)

with profile(
    activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == 'cuda' else []),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("generate_with_cache"):
        with torch.no_grad():
            _ = model.generate_with_cache(x, 100, temperature=1.0, top_k=200)

# Print results
print(prof.key_averages().table(
    sort_by="cpu_time_total" if device == 'cpu' else "cuda_time_total",
    row_limit=15
))

# Save detailed results
prof.export_chrome_trace("trace_with_cache.json")
print("\n✓ Detailed trace saved to: trace_with_cache.json")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)