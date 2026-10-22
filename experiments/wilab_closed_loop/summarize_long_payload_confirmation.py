"""Summarize the completed 1000-epoch 300-byte WiLab confirmation."""
import csv, json
from pathlib import Path

def main():
    root=Path('results/wilab_payload_calibration_final')
    sources={'101':root/'P300_long1000_retry'/'seed101_N50_B0',
             **{s:root/'P300_long1000_horizon2'/f'seed{s}_N50_B0' for s in ['202','303','404','505','606','707','808']}}
    rows=[]
    for seed,base in sources.items():
        p=base/'episode_0000'/'bridge_audit.json'
        if not p.exists(): continue
        d=json.loads(p.read_text()); r=d[-1]
        prr=r.get('safety_prr_cumulative',r.get('prr')); p95=r.get('safety_p95_delay_cumulative_ms',r.get('p95_delay_ms'))
        rows.append({'seed':int(seed),'epochs':len(d),'prr':prr,'p95_delay_ms':p95,'cbr':r.get('cbr'),'safe':prr>=.95 and p95<=50})
    rows.sort(key=lambda x:x['seed'])
    out=root/'P300_long1000_confirmation_rows.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    safe=sum(x['safe'] for x in rows); mean=sum(x['prr'] for x in rows)/len(rows)
    lines=['# 300-byte WiLab Long-Horizon Confirmation','',
           'Configuration: N=50, zero computation, 300 B safety payload, 100 ms generation, 20 ms allocation, 1000 controller epochs, simulation horizon 12 s (1 s warm-up plus 10 s controller horizon), eight calibration seeds.','',
           '| Seeds | Safe seeds | Mean PRR | Min PRR | Max P95 delay (ms) |','|---:|---:|---:|---:|---:|',
           f'| {len(rows)} | {safe} | {mean:.4f} | {min(x["prr"] for x in rows):.4f} | {max(x["p95_delay_ms"] for x in rows):.1f} |','',
           'Gate: every seed must satisfy PRR >= 0.95 and exact P95 delay <= 50 ms.']
    (root/'WILAB_300B_LONG_CONFIRMATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
if __name__=='__main__': main()
