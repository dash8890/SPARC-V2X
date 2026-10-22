"""WiLab-native safety-only factorial calibration.

Runs no computation admission and no learner updates. Each run has its own
IPC/output directory; the resulting MainOut and bridge audit files are later
summarized into a calibration table.
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[101, 202])
    ap.add_argument("--densities", nargs="+", type=int, default=[50, 100, 150, 200])
    ap.add_argument("--generation-ms", nargs="+", type=float, default=[50, 100])
    ap.add_argument("--allocation-ms", nargs="+", type=float, default=[10, 20, 50])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--output-root", type=str, default="results/wilab_native_calibration_factorial_5epoch")
    args = ap.parse_args()
    base = ROOT / args.output_root
    base.mkdir(parents=True, exist_ok=True)
    runner = ROOT / "experiments" / "wilab_closed_loop" / "train_wilab_closed_loop.py"
    jobs=[(seed,density,gen,alloc) for seed in args.seeds for density in args.densities for gen in args.generation_ms for alloc in args.allocation_ms]
    def run_one(job):
          seed,density,gen,alloc=job
          tag=f"seed{seed}_N{density}_G{int(gen)}_A{int(alloc)}"
          outroot=base/tag; bridge=base/(tag+"_bridge")
          outroot.mkdir(parents=True, exist_ok=True); bridge.mkdir(parents=True, exist_ok=True)
          env=os.environ.copy(); env["WILAB_OUTPUT_ROOT"]=str(outroot); env["TEMP"]=str(bridge); env["TMP"]=str(bridge)
          result=base/(tag+".json")
          cmd=[PYTHON,str(runner),"--safety-only","--episodes","1","--density",str(density),"--seed",str(seed),"--generation-ms",str(gen),"--allocation-ms",str(alloc),"--controller-epochs",str(args.epochs),"--output-json",str(result)]
          print("START",tag,flush=True)
          p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
          (outroot/"runner.log").write_text(p.stdout,encoding="utf-8")
          row={"seed":seed,"density":density,"generation_ms":gen,"allocation_ms":alloc,"returncode":p.returncode}
          mainout=next(outroot.glob("episode_0000/MainOut.xls"),None)
          if mainout:
            lines=mainout.read_text(errors="ignore").splitlines()
            if len(lines)>=2:
              h=lines[-2].split("\t"); d=lines[-1].split("\t"); m=dict(zip(h,d))
              row["prr_aggregate"]=float(m.get("Packet reception ratio C-V2X","nan"))
          audit=outroot/"episode_0000"/"bridge_audit.json"
          if audit.exists():
            a=json.loads(audit.read_text()); c=[float(x["cbr"]) for x in a]
            row.update(cbr_mean=sum(c)/len(c),cbr_min=min(c),cbr_max=max(c),
                       safe_action_sets=sorted({tuple(x["safe_action_set"]) for x in a}),
                       K_values=sorted({int(x["K_t"]) for x in a}))
          print("DONE",tag,row,flush=True)
          return row
    rows=[]
    with ThreadPoolExecutor(max_workers=min(4,len(jobs))) as pool:
      futures=[pool.submit(run_one,j) for j in jobs]
      for fut in as_completed(futures):
        rows.append(fut.result())
        (base/"calibration_progress.json").write_text(json.dumps(rows,indent=2))
    fields=sorted({k for r in rows for k in r})
    with (base/"wilab_native_calibration_factorial.csv").open("w",newline="") as f:
      w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print("COMPLETE",len(rows),flush=True)

if __name__=="__main__": main()
