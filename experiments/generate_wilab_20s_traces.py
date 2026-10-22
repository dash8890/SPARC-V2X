"""Generate synthetic_highway_rho_{d}_20s.txt traces for WiLab closed-loop."""
from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "traces" / "mode2_external_validation"
OUT.mkdir(parents=True, exist_ok=True)


def write_trace(density: int, duration_s: float = 20.0, dt: float = 0.1, road_m: float = 1000.0):
    n = int(density)
    spacing = road_m / n
    path = OUT / f"synthetic_highway_rho_{n}_20s.txt"
    steps = int(round(duration_s / dt)) + 1  # include t=duration
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for k in range(steps):
            t = round(k * dt, 6)
            if t > duration_s + 1e-9:
                break
            for vid in range(1, n + 1):
                speed = 19.0 + 1.5 * math.sin(0.07 * vid)
                x = ((vid - 1) * spacing + speed * t) % road_m
                y = 0.0
                f.write(f"{t:.3f} {vid:d} {x:.3f} {y:.3f} {speed:.3f}\n")
    return path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--densities", type=int, nargs="+", required=True)
    args = ap.parse_args()
    for d in args.densities:
        p = write_trace(d)
        # quick validate
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        t0 = float(lines[0].split()[0])
        t1 = float(lines[-1].split()[0])
        n0 = sum(1 for l in lines if float(l.split()[0]) == t0)
        print({"path": str(p), "lines": len(lines), "t0": t0, "t1": t1, "n_veh": n0})


if __name__ == "__main__":
    main()
