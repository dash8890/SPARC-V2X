from pathlib import Path
import json
import numpy as np

runs_dir = Path('sparc_performance_evaluation/results/raw/step6_baseline_comparison')
episodes = sorted(list(runs_dir.rglob('episode_*')))
print(f"Total episodes found: {len(episodes)}")

for ep in episodes:
    audit_file = ep / 'bridge_audit.json'
    if not audit_file.exists():
        continue
    audit = json.loads(audit_file.read_text(encoding='utf-8'))
    valid = [e for e in audit if e.get('cbr_valid') and e.get('delay_valid')]
    prr = [float(e.get('window_prr', e.get('prr', 1.0))) for e in valid]
    cbr = [float(e.get('cbr', 0.0)) for e in valid]
    p95 = [float(e.get('p95_delay_ms', 0.0)) for e in valid]
    viols = sum(1 for p in prr if p < 0.95)
    v_done = sum(float(e.get('reward', 0.0)) for e in audit)
    print(f"{ep.parent.name:20s} | {ep.name}: V={v_done:6.2f} | Viols={viols:2d}/{len(valid):3d} ({viols/max(1,len(valid)):.4f}) | Mean PRR={np.mean(prr):.4f} | CBR={np.mean(cbr):.4f}")
