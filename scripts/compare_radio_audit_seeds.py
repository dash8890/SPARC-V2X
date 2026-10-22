from pathlib import Path
import json
import numpy as np

base = Path('sparc_performance_evaluation/results/raw/step6_baseline_comparison')
ep_73300 = list(base.rglob('episode_80001'))[0]
ep_73303 = list(base.rglob('episode_80010'))[0]

audit_300 = json.loads((ep_73300 / 'bridge_audit.json').read_text(encoding='utf-8'))
audit_303 = json.loads((ep_73303 / 'bridge_audit.json').read_text(encoding='utf-8'))

valid_300 = [e for e in audit_300 if e.get('cbr_valid') and e.get('delay_valid')]
valid_303 = [e for e in audit_303 if e.get('cbr_valid') and e.get('delay_valid')]

prr_300 = [float(e.get('window_prr', e.get('prr', 1.0))) for e in valid_300]
prr_303 = [float(e.get('window_prr', e.get('prr', 1.0))) for e in valid_303]

cbr_300 = [float(e.get('cbr', 0.0)) for e in valid_300]
cbr_303 = [float(e.get('cbr', 0.0)) for e in valid_303]

resel_300 = [float(e.get('resel_t', 0.0)) for e in valid_300]
resel_303 = [float(e.get('resel_t', 0.0)) for e in valid_303]

print("=== SEED 73300 (P_viol = 0.02) ===")
print(f"Mean PRR: {np.mean(prr_300):.4f} | Min PRR: {np.min(prr_300):.4f} | Max PRR: {np.max(prr_300):.4f}")
print(f"Mean CBR: {np.mean(cbr_300):.4f} | Min CBR: {np.min(cbr_300):.4f} | Max CBR: {np.max(cbr_300):.4f}")
print(f"Mean Resel: {np.mean(resel_300):.2f} reassignments/window")
print(f"Windows with PRR < 0.95: {sum(1 for p in prr_300 if p < 0.95)} / {len(prr_300)}")

print("\n=== SEED 73303 (P_viol = 0.56) ===")
print(f"Mean PRR: {np.mean(prr_303):.4f} | Min PRR: {np.min(prr_303):.4f} | Max PRR: {np.max(prr_303):.4f}")
print(f"Mean CBR: {np.mean(cbr_303):.4f} | Min CBR: {np.min(cbr_303):.4f} | Max CBR: {np.max(cbr_303):.4f}")
print(f"Mean Resel: {np.mean(resel_303):.2f} reassignments/window")
print(f"Windows with PRR < 0.95: {sum(1 for p in prr_303 if p < 0.95)} / {len(prr_303)}")

# Check first 20 window PRRs
print("\nFirst 20 windows PRR comparison (Seed 73300 vs Seed 73303):")
for i in range(min(20, len(prr_300), len(prr_303))):
    print(f"  Epoch {i:2d}: Seed 73300 PRR={prr_300[i]:.4f} | Seed 73303 PRR={prr_303[i]:.4f}")
