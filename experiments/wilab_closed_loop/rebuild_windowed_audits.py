import json,re
from pathlib import Path
for root in ['results/wilab_windowed_calibration','results/wilab_windowed_heldout','results/wilab_density_window_calibration','results/wilab_density_window_calibration_retry','results/wilab_density_window_calibration_retry2']:
  for d in Path(root).glob('seed*_N*_B*'):
    if d.name.endswith('_bridge'): continue
    af=next(d.glob('episode_*/bridge_audit.json'),None)
    if not af: continue
    bridge=Path(str(d)+'_bridge')/'wilab_closed_loop_bridge'
    if not bridge.exists(): continue
    a=json.loads(af.read_text())
    changed=False
    for e in a:
      rf=bridge/f'result_000000_{int(e["epoch"]):06d}.json'
      if rf.exists():
        r=json.loads(rf.read_text())
        for k in ('safety_rx_cumulative','safety_error_cumulative','safety_prr_cumulative'):
          e[k]=r.get(k)
        changed=True
    if changed: af.write_text(json.dumps(a,indent=2))
print('rebuilt audits')
