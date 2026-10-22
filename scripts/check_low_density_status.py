from pathlib import Path
import json

runs = Path('sparc_performance_evaluation/results/raw/low_density_audit/runs')
for ep in sorted(runs.glob('episode_*')):
    audit = ep / 'bridge_audit.json'
    matlab_log = ep / 'matlab.log'
    if audit.exists():
        data = json.loads(audit.read_text(encoding='utf-8'))
        valid = [e for e in data if e.get('cbr_valid') and e.get('delay_valid')]
        prr = [float(e.get('window_prr', e.get('prr', 1.0))) for e in valid]
        viols = sum(1 for p in prr if p < 0.95)
        print(f"{ep.name}: SUCCEEDED! viols={viols}/{len(valid)} | mean_prr={sum(prr)/len(prr):.4f}")
    else:
        err = matlab_log.read_text(encoding='utf-8', errors='ignore') if matlab_log.exists() else 'no log'
        err_line = err.strip().splitlines()[0] if err.strip() else 'empty'
        print(f"{ep.name}: FAILED - {err_line}")
