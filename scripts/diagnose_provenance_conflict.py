from pathlib import Path
import json

runs = Path('sparc_performance_evaluation/results/raw/low_density_audit/runs')
print("Listing all episodes in:", runs)

for ep in sorted(runs.glob('episode_*')):
    audit_file = ep / 'bridge_audit.json'
    matlab_log = ep / 'matlab.log'
    summary_file = ep / 'episode_summary.json'
    cfg_file = ep / 'test_trace_runtime.cfg'
    
    print(f"\n=== {ep.name} ===")
    if cfg_file.exists():
        cfg_lines = [l for l in cfg_file.read_text().splitlines() if '[rho]' in l or '[seed]' in l]
        print("  Config:", cfg_lines)
    if summary_file.exists():
        s = json.loads(summary_file.read_text())
        print(f"  Summary: seed={s.get('seed')}, completed_value={s.get('completed_value')}")
    if audit_file.exists():
        audit = json.loads(audit_file.read_text(encoding='utf-8'))
        valid = [e for e in audit if e.get('cbr_valid') and e.get('delay_valid')]
        prr = [float(e.get('window_prr', e.get('prr', 1.0))) for e in valid]
        cbr = [float(e.get('cbr', 0.0)) for e in valid]
        viols = sum(1 for p in prr if p < 0.95)
        print(f"  Audit: len={len(audit)}, valid={len(valid)}, viols={viols}, p_viol={viols/max(1,len(valid)):.4f}, mean_prr={sum(prr)/len(prr):.4f}, mean_cbr={sum(cbr)/len(cbr):.4f}")
    if matlab_log.exists():
        log_text = matlab_log.read_text(encoding='utf-8', errors='ignore')
        # Check if error or seed
        print(f"  matlab.log: {len(log_text.splitlines())} lines | first line: {log_text.splitlines()[0] if log_text.splitlines() else 'empty'}")
        if 'error' in log_text.lower() or 'unable' in log_text.lower():
            print(f"  matlab.log ERRORS found:")
            for l in log_text.splitlines():
                if any(k in l.lower() for k in ['error', 'unable', 'seed', 'fail']):
                    print("    ", l)
