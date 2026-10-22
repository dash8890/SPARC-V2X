"""Summarize 300-byte, 1000-epoch WiLab byte-budget calibration."""
import csv,json
from pathlib import Path

def main():
    root=Path('results/wilab_byte_budget_calibration_300B_long1000')
    rerun=Path('results/wilab_byte_budget_calibration_300B_long1000_rerun')
    rows=[]
    for base in [root,rerun]:
        for d in base.glob('seed*_N50_B*'):
            if not d.is_dir() or d.name.endswith('_bridge'): continue
            p=d/'episode_0000'/'bridge_audit.json'
            if not p.exists(): continue
            seed=int(d.name.split('_')[0][4:]); budget=int(d.name.split('_B')[1]); data=json.loads(p.read_text())
            if not data: continue
            r=data[-1]; prr=r.get('safety_prr_cumulative',r.get('prr')); p95=r.get('safety_p95_delay_cumulative_ms',r.get('p95_delay_ms'))
            rows.append({'byte_budget':budget,'seed':seed,'epochs':len(data),'prr':prr,'p95_delay_ms':p95,'cbr':r.get('cbr'),'scheduled_packets':r.get('wilab_scheduled_attempts'),'pending_packets':r.get('wilab_pending_packets'),'admitted_bytes':r.get('computation_bytes_offered'),'safe':prr>=.95 and p95<=50})
    # retain one row per seed/budget, preferring the rerun if duplicate
    uniq={(r['byte_budget'],r['seed']):r for r in rows}; rows=sorted(uniq.values(),key=lambda x:(x['byte_budget'],x['seed']))
    # add the validated zero-budget rows
    b0=[]
    for r in csv.DictReader((Path('results/wilab_payload_calibration_final/P300_long1000_confirmation_rows.csv')).open()):
        b0.append({'byte_budget':0,'seed':int(r['seed']),'epochs':int(r['epochs']),'prr':float(r['prr']),'p95_delay_ms':float(r['p95_delay_ms']),'cbr':float(r['cbr']),'scheduled_packets':None,'pending_packets':None,'admitted_bytes':0,'safe':r['safe']=='True'})
    rows=sorted(b0+rows,key=lambda x:(x['byte_budget'],x['seed']))
    out=Path('results/wilab_byte_budget_calibration_300B_long1000_summary.csv')
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    lines=['# WiLab 300-B Computation-Budget Calibration','', 'N=50, 300-byte safety payload, 100 ms generation, 20 ms allocation, 1000 controller epochs, simulation horizon 12 s, eight calibration seeds. The zero-budget row is the validated 1000-epoch confirmation.', '', '| Budget (kB) | Seeds | Safe seeds | Mean PRR | Min PRR | Max P95 (ms) | Mean CBR |','|---:|---:|---:|---:|---:|---:|---:|']
    for b in sorted({r['byte_budget'] for r in rows}):
        z=[r for r in rows if r['byte_budget']==b]; lines.append(f"| {b/1000:g} | {len(z)} | {sum(r['safe'] for r in z)} | {sum(r['prr'] for r in z)/len(z):.4f} | {min(r['prr'] for r in z):.4f} | {max(r['p95_delay_ms'] for r in z):.1f} | {sum(r['cbr'] for r in z)/len(z):.4f} |")
    lines += ['', 'Safety gate: PRR >= 0.95 and exact P95 delay <= 50 ms for every calibration seed.']
    Path('results/wilab_byte_budget_calibration_300B_long1000/WILAB_300B_BYTE_BUDGET_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
if __name__=='__main__': main()
