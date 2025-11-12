import torch
import time
from contextlib import nullcontext
from model_nanogpt import GPT, GPTConfig

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

# Load model
print("Loading GPT-2...")
model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
model.eval()
model.to(device)

# Test prompt
prompt = "Hello, I am a language model and I will tell you a story about " * 10
from tiktoken import get_encoding
enc = get_encoding("gpt2")
start_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

print(f"\nPrompt: {prompt}")
print(f"Generating 512 tokens...\n")

# Benchmark WITHOUT cache
print("=" * 60)
print("WITHOUT KV Cache (baseline)")
print("=" * 60)
torch.cuda.synchronize() if device == 'cuda' else None
start = time.time()

with torch.no_grad():
    with ctx:
        y1 = model.generate(x, 512, temperature=0.8, top_k=200)

torch.cuda.synchronize() if device == 'cuda' else None
time_no_cache = time.time() - start

print(f"Time: {time_no_cache:.2f}s")
print(f"Speed: {512 / time_no_cache:.1f} tokens/sec")

# Benchmark WITH cache
print("\n" + "=" * 60)
print("WITH KV Cache (optimized)")
print("=" * 60)
torch.cuda.synchronize() if device == 'cuda' else None
start = time.time()

with torch.no_grad():
    with ctx:
        y2 = model.generate_with_cache(x, 512, temperature=0.8, top_k=200)

torch.cuda.synchronize() if device == 'cuda' else None
time_with_cache = time.time() - start

print(f"Time: {time_with_cache:.2f}s")
print(f"Speed: {512 / time_with_cache:.1f} tokens/sec")

# Results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Speedup: {time_no_cache / time_with_cache:.2f}x")
print(f"Time saved: {time_no_cache - time_with_cache:.2f}s ({100 * (1 - time_with_cache/time_no_cache):.1f}%)")

# Verify outputs match
print(f"\nOutputs match: {torch.equal(y1, y2)}")