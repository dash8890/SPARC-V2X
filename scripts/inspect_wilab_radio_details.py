from pathlib import Path
import pandas as pd

base = Path('sparc_performance_evaluation/results/raw/step6_baseline_comparison')
ep_73300 = list(base.rglob('episode_80001'))[0]
ep_73303 = list(base.rglob('episode_80010'))[0]

def inspect_tsv(p):
    if not p.exists():
        return f"File not found: {p}"
    lines = p.read_text(encoding='utf-8', errors='ignore').splitlines()
    return f"{p.name} ({len(lines)} lines):\n" + "\n".join(lines[:10])

print("--- SEED 73300 ---")
print(inspect_tsv(ep_73300 / 'wireless_blind_spot_1_5G.xls'))
print(inspect_tsv(ep_73300 / 'packet_reception_ratio_1_5G.xls'))

print("\n--- SEED 73303 ---")
print(inspect_tsv(ep_73303 / 'wireless_blind_spot_1_5G.xls'))
print(inspect_tsv(ep_73303 / 'packet_reception_ratio_1_5G.xls'))
