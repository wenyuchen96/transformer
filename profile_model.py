import torch
from torch.profiler import profile, record_function, ProfilerActivity
from model import GPT, Config, Util

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

config = Config()
util = Util(text, config)
model = GPT(config).to(config.device)

model.eval()

print(f"Profiling model generation on {config.device}")

activities = [ProfilerActivity.CPU]
if config.device == 'cuda':
    activities.append(ProfilerActivity.CUDA)

context = torch.zeros((1,1), dtype=torch.long, device=config.device)

with torch.no_grad():
    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=True
    ) as prof:
        with record_function("model_generation"):
            output = model.generate(context, max_new_tokens=100)

print("\n--- Profiler Results (Grouped by Stack) ---")
print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=20))