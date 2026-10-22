"""Summarize isolated WiLab zero-computation safety-payload calibration."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/wilab_payload_calibration_final')
    args = ap.parse_args()
    root = Path(args.root)
    rows=[]
    for pdir in sorted(root.glob('P*')):
        if not pdir.is_dir() or not pdir.name[1:].isdigit(): continue
        payload=int(pdir.name[1:])
        for audit in sorted(pdir.glob('seed*_N50_B0/episode_0000/bridge_audit.json')):
            seed=int(audit.parts[-3].split('_')[0].replace('seed',''))
            data=json.loads(audit.read_text())
            if not data: continue
            r=data[-1]
            prr=r.get('safety_prr_cumulative', r.get('prr'))
            p95=r.get('safety_p95_delay_cumulative_ms', r.get('p95_delay_ms'))
            rows.append({'payload_bytes':payload,'seed':seed,'prr':prr,'p95_delay_ms':p95,
                         'cbr':r.get('cbr'),'safety_rx_cumulative':r.get('safety_rx_cumulative'),
                         'safety_error_cumulative':r.get('safety_error_cumulative'),
                         'safe': bool(prr is not None and p95 is not None and prr>=0.95 and p95<=50)})
    rows.sort(key=lambda x:(x['payload_bytes'],x['seed']))
    out=root/'payload_calibration_rows.csv'; out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ['payload_bytes','seed']); w.writeheader(); w.writerows(rows)
    summary=[]
    for payload in sorted({r['payload_bytes'] for r in rows}):
        z=[r for r in rows if r['payload_bytes']==payload]
        summary.append({'payload_bytes':payload,'n':len(z),'safe_n':sum(r['safe'] for r in z),
                        'mean_prr':sum(r['prr'] for r in z)/len(z) if z else None,
                        'min_prr':min(r['prr'] for r in z) if z else None,
                        'max_p95_delay_ms':max(r['p95_delay_ms'] for r in z) if z else None})
    (root/'payload_calibration_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# WiLab Safety-Payload Calibration (N=50, B=0)','',
           'Configuration: 100 ms safety generation, 20 ms allocation, 300 controller epochs, 8 calibration seeds. The table uses the final cumulative WiLab safety counters from each run; PRR is authoritative cumulative success/(success+error), and P95 is the exact cumulative delay histogram metric.','',
           '| Payload (B) | Seeds | Safe seeds | Mean PRR | Min PRR | Max P95 (ms) |','|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['payload_bytes']} | {s['n']} | {s['safe_n']} | {s['mean_prr']:.4f} | {s['min_prr']:.4f} | {s['max_p95_delay_ms']:.1f} |")
    lines += ['', 'Safety gate: PRR >= 0.95 and exact P95 delay <= 50 ms for every valid calibration seed.']
    (root/'WILAB_PAYLOAD_CALIBRATION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__': main()
