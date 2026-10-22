from pathlib import Path
import json
import numpy as np

base = Path('sparc_performance_evaluation/results/raw/step6_baseline_comparison')
ep_73300 = list(base.rglob('episode_80001'))[0]
ep_73303 = list(base.rglob('episode_80010'))[0]

# Check matlab.log in both episode folders
log_300 = (ep_73300 / 'matlab.log').read_text(encoding='utf-8', errors='ignore') if (ep_73300 / 'matlab.log').exists() else ""
log_303 = (ep_73303 / 'matlab.log').read_text(encoding='utf-8', errors='ignore') if (ep_73303 / 'matlab.log').exists() else ""

print(f"Log 73300 lines: {len(log_300.splitlines())}")
print(f"Log 73303 lines: {len(log_303.splitlines())}")

# Check any other files in ep_73303
print("Files in ep_73303:", [p.name for p in ep_73303.iterdir()])
