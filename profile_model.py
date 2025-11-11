import torch
import cProfile
import pstats
import io
from model import GPT, Config, Util

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

config = Config()
util = Util(text, config)
model = GPT(config).to(config.device)

model.eval()

print("Profiling model generation...")
profiler = cProfile.Profile()
profiler.enable()

context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
with torch.no_grad():
    output = model.generate(context, max_new_tokens=100)

profiler.disable()

s = io.StringIO()
stats = pstats.Stats(profiler, stream=s)
stats.sort_stats('cumulative')
stats.print_stats(20)
print(s.getvalue())