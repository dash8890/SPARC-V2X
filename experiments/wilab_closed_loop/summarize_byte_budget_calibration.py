import csv, json, re
from pathlib import Path

def collect(root, phase):
    rows=[]
    for d in sorted(Path(root).glob('seed*_N*_B*')):
        m=re.match(r'seed(\d+)_N(\d+)_B(\d+)$', d.name)
        if not m: continue
        files=list(d.glob('episode_*/bridge_audit.json'))
        if not files: continue
        audit=json.loads(files[0].read_text())
        if not audit: continue
        def avg(k):
            x=[float(e[k]) for e in audit if isinstance(e.get(k),(int,float))]
            return sum(x)/len(x) if x else None
        last=audit[-1]
        mainout=list(d.glob('episode_*/MainOut.xls'))
        main_prr=None; main_cbr=None
        if mainout:
            lines=mainout[0].read_text(errors='ignore').splitlines()
            if len(lines)>=2:
                hdr=lines[0].split('\t'); vals=lines[-1].split('\t')
                def field(name):
                    try: return float(vals[hdr.index(name)])
                    except Exception: return None
                main_prr=field('Packet reception ratio C-V2X')
                main_cbr=field('Median CBR')
        rows.append({'phase':phase,'seed':int(m.group(1)),'density':int(m.group(2)),'byte_budget':int(m.group(3)),
            'epochs':len(audit),'prr_mean':avg('prr'),'p95_delay_mean_ms':avg('p95_delay_ms'),'cbr_mean':avg('cbr'),
            'wilab_mainout_prr':main_prr,'wilab_mainout_median_cbr':main_cbr,
            'p95_delay_max_ms':max(float(e.get('p95_delay_ms',0)) for e in audit),
            'offered_bytes_last_epoch':last.get('computation_bytes_offered'),'offered_packets_last_epoch':last.get('computation_packets_offered'),
            'enqueued_bytes_cumulative':last.get('wilab_enqueued_bytes'),'enqueued_packets_cumulative':last.get('wilab_enqueued_packets'),
            'scheduled_attempts_cumulative':last.get('wilab_scheduled_attempts'),'pending_packets_final':last.get('wilab_pending_packets'),
            'selected_mean':avg('num_selected'),'return':sum(float(e.get('reward',0)) for e in audit),
            'safe_set_min':min(int(e.get('K_t',0)) for e in audit),'safe_set_max':max(int(e.get('K_t',0)) for e in audit)})
    return rows

cal=collect('results/wilab_byte_budget_calibration','calibration')
hold=collect('results/wilab_byte_budget_heldout','heldout')
rows=cal+hold
out=Path('results/wilab_byte_budget_calibration'); out.mkdir(exist_ok=True)
with (out/'byte_budget_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
def grouped(data):
    g={}
    for r in data:
        key=(r['density'],r['byte_budget']); g.setdefault(key,[]).append(r)
    return g
lines=['# WiLab state-conditioned byte-budget calibration','',f'Calibration runs: {len(cal)} (8 seeds); held-out runs: {len(hold)} (2 seeds).','', 'The action is an aggregate admitted computation-byte budget. Each row aggregates the 20 controller epochs of one seed/density/budget run. Bridge epoch PRR is retained for diagnosis, but the authoritative safety PRR below is the cumulative C-V2X PRR reported by WiLab in `MainOut.xls`.','', '## Held-out summary','', '|N|Budget (B)|WiLab PRR|Bridge PRR mean|P95 delay mean (ms)|CBR mean|Enqueued MB|Scheduled attempts|Pending packets|','|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for (n,b),rs in sorted(grouped(hold).items()):
    def a(k): return sum(float(x[k]) for x in rs)/len(rs)
    lines.append(f'|{n}|{b}|{a("wilab_mainout_prr"):.4f}|{a("prr_mean"):.4f}|{a("p95_delay_mean_ms"):.2f}|{a("cbr_mean"):.4f}|{a("enqueued_bytes_cumulative")/1e6:.3f}|{a("scheduled_attempts_cumulative"):.1f}|{a("pending_packets_final"):.1f}|')
lines += ['', '## Calibration summary', '', '|N|Budget (B)|WiLab PRR|Bridge PRR mean|P95 delay mean (ms)|CBR mean|Enqueued MB|Scheduled attempts|Pending packets|','|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for (n,b),rs in sorted(grouped(cal).items()):
    def ac(k): return sum(float(x[k]) for x in rs)/len(rs)
    lines.append(f'|{n}|{b}|{ac("wilab_mainout_prr"):.4f}|{ac("prr_mean"):.4f}|{ac("p95_delay_mean_ms"):.2f}|{ac("cbr_mean"):.4f}|{ac("enqueued_bytes_cumulative")/1e6:.3f}|{ac("scheduled_attempts_cumulative"):.1f}|{ac("pending_packets_final"):.1f}|')
lines += ['', '## PRR audit finding','', 'The bridge-level `prr` field is 1.0 in these runs because it reflects the valid-window/controller result rather than the full WiLab safety denominator. It must not be used as the safety KPI. `MainOut.xls` reports the cumulative C-V2X packet reception ratio and is the authoritative PRR used here. Shield fitting must use this reconciled metric.','', '## Interpretation','', 'The calibration measures how the WiLab packet/resource process responds to state-conditioned computation-byte budgets. No shield was fit from the unreconciled bridge PRR.','', '## Files','', '- `byte_budget_summary.csv`: per-seed, per-density, per-budget summary.', '- `wilab_byte_budget_calibration/`: 200 calibration run directories.', '- `wilab_byte_budget_heldout/`: 50 held-out run directories.']
(out/'WILAB_BYTE_BUDGET_CALIBRATION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(f'wrote {len(rows)} rows')
