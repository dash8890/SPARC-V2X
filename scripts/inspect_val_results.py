from pathlib import Path
import json

runs_dir = Path('sparc_performance_evaluation/results/raw/pb_crl_train/runs')
ep_dirs = sorted([p for p in runs_dir.glob('episode_*') if p.is_dir()])
print(f"Total episodes run: {len(ep_dirs)}")

# Last 4 episodes correspond to the final validation evaluation
print("\n--- Final Validation Episodes ---")
for ep in ep_dirs[-4:]:
    audit_path = ep / 'bridge_audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        valid = [e for e in audit if e.get('cbr_valid') and e.get('delay_valid')]
        prr = [float(e.get('window_prr', e.get('prr', 1.0))) for e in valid]
        viols = sum(1 for p in prr if p < 0.95)
        v_done = sum(float(e.get('reward', 0.0)) for e in audit)
        cbr = [float(e.get('cbr', 0.0)) for e in valid]
        print(f"{ep.name}: V_done={v_done:6.2f} | Viols={viols:2d}/{len(valid):3d} ({viols/max(1,len(valid)):.4f}) | Mean PRR={sum(prr)/len(prr):.4f} | Mean CBR={sum(cbr)/len(cbr):.4f}")
