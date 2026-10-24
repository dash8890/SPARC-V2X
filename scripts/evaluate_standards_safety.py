"""Standards-grounded safety measurement for closed-loop SC-RLGMI episodes.

Track 2 of the dual-track plan.  Computes packet-level safety metrics from the
per-epoch bridge audit without changing the simulator or the legacy metric.

Definitions (all derived from causally recorded counters):

  opportunity        one decode attempt for a periodic safety packet at a
                     receiver: ``rx_i + err_i`` in window ``i``.
  R_timely(D_max)    (safety packets decoded with latency <= D_max) /
                     (safety packet opportunities), pooled over the episode.
  PRR_window(i)      rx_i / (rx_i + err_i).
  PRR_vs_distance    correct / (correct + errors) per distance band.  WiLab's
                     `blocked` column is not used as a denominator because it
                     counts all neighbours of a half-duplex-blocked receiver,
                     not the single missed pair (neighbour-multiplicity bias).
  IR proxy           longest run of consecutive 100 ms windows containing at
                     least one failed safety decode.  True packet-level
                     inter-reception intervals are not recorded by WiLab, so
                     this is reported explicitly as a window-level proxy.

The legacy metric (P_window = Pr[PRR_100ms < 0.95 or P95 > 50 ms]) is retained
unchanged for comparability.

Usage:
    python scripts/evaluate_standards_safety.py \
        --run sc_rlgmi=sparc_performance_evaluation/results/raw/sc_rlgmi_training/eval \
        --run candidate_c=sparc_performance_evaluation/results/raw/stage_b_candidates/Candidate_C_Surgical \
        --out sparc_performance_evaluation/results/aggregates/standards_safety
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DMAX_MS: Tuple[int, ...] = (20, 50, 100)
LEGACY_PRR_FLOOR = 0.95
LEGACY_P95_CEILING_MS = 50.0
DISTANCE_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.0, 50.0),
    (50.0, 150.0),
    (150.0, 300.0),
    (300.0, 500.0),
)


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _hist_delta(cur: Sequence[float], prev: Sequence[float]) -> List[float]:
    n = max(len(cur), len(prev))
    out = []
    for i in range(n):
        c = float(cur[i]) if i < len(cur) else 0.0
        p = float(prev[i]) if i < len(prev) else 0.0
        out.append(max(0.0, c - p))
    return out


def _p95_from_hist(counts: Sequence[float], resolution_ms: float) -> Optional[float]:
    total = sum(counts)
    if total <= 0:
        return None
    cum = 0.0
    for k, c in enumerate(counts):
        cum += c
        if cum / total >= 0.95:
            return k * resolution_ms
    return (len(counts) - 1) * resolution_ms


def _episode_metrics(audit: Sequence[Dict[str, Any]], dmax_ms: Iterable[int]) -> Dict[str, Any]:
    dmax = sorted(set(int(d) for d in dmax_ms))
    valid = [e for e in audit if e.get("cbr_valid") and e.get("delay_valid")]
    if not valid:
        return {"n_windows": 0, "valid": False}

    prev_rx = prev_err = 0.0
    prev_hist: List[float] = []
    prev_dist: Dict[float, Tuple[float, float]] = {}

    opp_total = rx_total = err_total = 0.0
    timely_total = {d: 0.0 for d in dmax}
    delay_counts_total: List[float] = []

    window_prr: List[float] = []
    window_loss: List[float] = []
    window_p95: List[Optional[float]] = []
    legacy_violations = 0
    band_acc: Dict[Tuple[float, float], List[float]] = {
        b: [0.0, 0.0, 0.0] for b in DISTANCE_BANDS  # correct, errors, windows
    }

    for e in valid:
        rx = _f(e.get("safety_rx_cumulative"))
        err = _f(e.get("safety_error_cumulative"))
        d_rx = max(0.0, rx - prev_rx)
        d_err = max(0.0, err - prev_err)
        prev_rx, prev_err = rx, err

        opp = d_rx + d_err
        if opp <= 0:
            window_prr.append(1.0)
            window_loss.append(0.0)
            window_p95.append(None)
            prev_hist = [float(x) for x in (e.get("safety_delay_hist_cumulative") or [])]
            continue

        rx_total += d_rx
        err_total += d_err
        opp_total += opp

        resolution = _f(e.get("delay_resolution_ms"), 1.0) or 1.0
        hist_cur = [float(x) for x in (e.get("safety_delay_hist_cumulative") or [])]
        hist_delta = _hist_delta(hist_cur, prev_hist)
        prev_hist = hist_cur
        # Some epochs may report fewer bins than history; keep pooled length.
        if len(hist_delta) > len(delay_counts_total):
            delay_counts_total.extend([0.0] * (len(hist_delta) - len(delay_counts_total)))
        for i, c in enumerate(hist_delta):
            delay_counts_total[i] += c
        for d in dmax:
            k_max = int(math.floor(d / resolution))
            timely_total[d] += sum(hist_delta[: k_max + 1])

        p = d_rx / opp
        window_prr.append(p)
        window_loss.append(d_err / opp)
        p95 = _p95_from_hist(hist_delta, resolution)
        window_p95.append(p95)
        if p < LEGACY_PRR_FLOOR or (p95 is not None and p95 > LEGACY_P95_CEILING_MS):
            legacy_violations += 1

        # Distance-banded windowed counters (blocked decodes not exported yet).
        cur_dist = {
            float(row.get("distance", 0.0)): (
                _f(row.get("correct")), _f(row.get("errors"))
            )
            for row in (e.get("prr_vs_distance") or [])
        }
        for dist, (c, er) in cur_dist.items():
            pc, pe = prev_dist.get(dist, (0.0, 0.0))
            dc = max(0.0, c - pc)
            de = max(0.0, er - pe)
            for lo, hi in DISTANCE_BANDS:
                if lo < dist <= hi:
                    band_acc[(lo, hi)][0] += dc
                    band_acc[(lo, hi)][1] += de
                    band_acc[(lo, hi)][2] += 1.0
                    break
        prev_dist = cur_dist

    n_win = len(window_prr)
    longest_loss_run = 0
    cur_run = 0
    for loss in window_loss:
        if loss > 0:
            cur_run += 1
            longest_loss_run = max(longest_loss_run, cur_run)
        else:
            cur_run = 0

    bands = {}
    for (lo, hi), (c, er, n) in band_acc.items():
        denom = c + er
        bands[f"{int(lo)}-{int(hi)}m"] = {
            "correct": c,
            "errors": er,
            "prr": (c / denom) if denom > 0 else None,
            "observed_windows": n,
        }

    return {
        "valid": True,
        "n_windows": n_win,
        "opportunities": opp_total,
        "rx": rx_total,
        "errors": err_total,
        "prr_packet_cumulative": (rx_total / opp_total) if opp_total > 0 else None,
        "r_timely": {f"{d}ms": (timely_total[d] / opp_total) if opp_total > 0 else None for d in dmax},
        "delay_count_total": delay_counts_total,
        "mean_window_prr": (sum(window_prr) / n_win) if n_win else None,
        "worst_window_prr": min(window_prr) if window_prr else None,
        "worst_window_loss_frac": max(window_loss) if window_loss else None,
        "longest_consecutive_loss_windows": longest_loss_run,
        "legacy_violations": legacy_violations,
        "legacy_p_viol": (legacy_violations / n_win) if n_win else None,
        "prr_vs_distance": bands,
    }


def _parse_prr_xls(path: Path) -> Optional[Dict[str, Any]]:
    """Cumulative PRR vs distance from WiLab's own export (includes blocked)."""
    if not path.is_file():
        return None
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 6:
            continue
        try:
            dist, corr, err, blocked, total, prr = [float(x) for x in parts[:6]]
        except ValueError:
            continue
        rows.append((dist, corr, err, blocked, total, prr))
    if not rows:
        return None
    bands: Dict[str, Dict[str, float]] = {}
    for lo, hi in DISTANCE_BANDS:
        c = sum(r[1] for r in rows if lo < r[0] <= hi)
        e = sum(r[2] for r in rows if lo < r[0] <= hi)
        b = sum(r[3] for r in rows if lo < r[0] <= hi)
        denom = c + e + b
        bands[f"{int(lo)}-{int(hi)}m"] = {
            "correct": c,
            "errors": e,
            "blocked": b,
            "prr": (c / denom) if denom > 0 else None,
        }
    return {"bands": bands, "rows": len(rows)}


def _iter_episodes(run_root: Path) -> List[Path]:
    """Episode dirs, tolerating the double episode_XXXXX nesting."""
    found: List[Path] = []
    for p in sorted(run_root.rglob("bridge_audit.json")):
        found.append(p.parent)
    return found


def evaluate_run(label: str, run_root: Path, dmax_ms: Sequence[int]) -> Dict[str, Any]:
    episodes = []
    for ep in _iter_episodes(run_root):
        audit = json.loads((ep / "bridge_audit.json").read_text(encoding="utf-8"))
        m = _episode_metrics(audit, dmax_ms)
        seeds = sorted({e.get("sim_seed") for e in audit if e.get("sim_seed") is not None})
        if not seeds:
            # Audit epochs do not carry the simulator seed; recover it from the
            # convention used by the evaluation drivers (val_s<seed> dirs).
            for part in ep.parts:
                if part.startswith("val_s") and part[5:].isdigit():
                    seeds = [int(part[5:])]
                    break
        xls = None
        for cand in ep.glob("packet_reception_ratio_*.xls"):
            xls = _parse_prr_xls(cand)
            if xls:
                break
        m.update(
            {
                "episode_dir": ep.relative_to(ROOT).as_posix(),
                "seed": seeds[0] if len(seeds) == 1 else seeds,
                "prr_vs_distance_xls": xls,
            }
        )
        episodes.append(m)

    pooled = [e for e in episodes if e.get("valid")]
    summary: Dict[str, Any] = {"label": label, "n_episodes": len(pooled)}
    if pooled:
        opp = sum(e["opportunities"] for e in pooled)
        rx = sum(e["rx"] for e in pooled)
        summary["opportunities"] = opp
        summary["prr_packet_cumulative"] = (rx / opp) if opp else None
        summary["r_timely"] = {
            f"{d}ms": (
                sum(e["r_timely"][f"{d}ms"] * e["opportunities"] for e in pooled) / opp
                if opp
                else None
            )
            for d in dmax_ms
        }
        summary["legacy_p_viol_mean_over_episodes"] = sum(
            e["legacy_p_viol"] for e in pooled
        ) / len(pooled)
        summary["legacy_p_viol_pooled"] = (
            sum(e["legacy_violations"] for e in pooled)
            / max(1, sum(e["n_windows"] for e in pooled))
        )
        summary["worst_window_prr"] = min(e["worst_window_prr"] for e in pooled)
        summary["longest_consecutive_loss_windows"] = max(
            e["longest_consecutive_loss_windows"] for e in pooled
        )
    return {**summary, "episodes": episodes}


def _fmt(v: Optional[float], pct: bool = True, digits: int = 3) -> str:
    if v is None:
        return "-"
    return f"{v*100:.{digits}f}%" if pct else f"{v:.{digits}f}"


def render_markdown(runs: List[Dict[str, Any]]) -> str:
    lines = [
        "# Standards-grounded safety evaluation",
        "",
        "Packet-level metrics derived from the frozen closed-loop audits. "
        "`R_timely(D)` is the fraction of safety-packet decode opportunities that "
        "succeeded within `D` ms; `legacy` is the historical window metric "
        "(PRR<0.95 or P95>50 ms).",
        "",
        "| Run | Episodes | Opportunities | Packet PRR | R_timely 20 ms | R_timely 50 ms | R_timely 100 ms | Legacy P_viol | Worst window PRR | Longest loss run |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in runs:
        if not r.get("n_episodes"):
            lines.append(f"| {r['label']} | 0 | - | - | - | - | - | - | - | - |")
            continue
        rt = r["r_timely"]
        lines.append(
            "| {label} | {n} | {opp:.0f} | {prr} | {t20} | {t50} | {t100} | {legacy} | {worst} | {run} |".format(
                label=r["label"],
                n=r["n_episodes"],
                opp=r["opportunities"],
                prr=_fmt(r["prr_packet_cumulative"]),
                t20=_fmt(rt.get("20ms")),
                t50=_fmt(rt.get("50ms")),
                t100=_fmt(rt.get("100ms")),
                legacy=_fmt(r["legacy_p_viol_pooled"]),
                worst=_fmt(r["worst_window_prr"]),
                run=r["longest_consecutive_loss_windows"],
            )
        )

    lines += [
        "",
        "## Per-episode detail",
        "",
        "| Run | Seed | Windows | Packet PRR | R_timely 100 ms | Legacy violations | Worst window | Distance-band PRR (0-50/50-150/150-300/300-500 m) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for r in runs:
        for e in r.get("episodes", []):
            if not e.get("valid"):
                lines.append(f"| {r['label']} | {e.get('seed')} | 0 | - | - | - | - | - |")
                continue
            bands = e["prr_vs_distance"]
            bt = "/".join(
                _fmt(bands[b]["prr"]) if bands[b]["prr"] is not None else "-"
                for b in ("0-50m", "50-150m", "150-300m", "300-500m")
            )
            lines.append(
                "| {label} | {seed} | {n} | {prr} | {t100} | {viol} | {worst} | {bands} |".format(
                    label=r["label"],
                    seed=e.get("seed"),
                    n=e["n_windows"],
                    prr=_fmt(e["prr_packet_cumulative"]),
                    t100=_fmt(e["r_timely"]["100ms"]),
                    viol=e["legacy_violations"],
                    worst=_fmt(e["worst_window_prr"]),
                    bands=bt,
                )
            )

    lines += [
        "",
        "## Measurement caveats",
        "",
        "1. `opportunities` counts decode attempts (`NcorrectlyTxBeaconsTOT + NerrorsTOT`)."
        " Transmissions never attempted (e.g. receiver in half-duplex transmission) are not"
        " represented, so `R_timely` is bounded by WiLab's own bookkeeping.",
        "2. Distance-band PRR uses `correct / (correct + errors)`. WiLab's `blocked` column is"
        " not used as a denominator: `mainCV2XttiEnds.m:143` counts *all neighbours* of a"
        " half-duplex-blocked receiver, so the column is neighbour-multiplicity inflated.",
        "3. `longest_consecutive_loss_windows` is a window-level proxy for packet"
        " inter-reception intervals. True IR requires per-packet receive timestamps that"
        " WiLab does not export.",
        "4. Delay histograms are 1 ms bins recorded cumulatively per epoch; window values are"
        " obtained by differencing consecutive epochs.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=ROOT",
        help="Run label and directory containing episode audits (repeatable).",
    )
    ap.add_argument("--out", default="sparc_performance_evaluation/results/aggregates/standards_safety")
    ap.add_argument("--dmax", default="20,50,100")
    args = ap.parse_args()

    dmax = [int(x) for x in args.dmax.split(",") if x.strip()]
    runs = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run must be LABEL=ROOT, got {spec!r}")
        label, root = spec.split("=", 1)
        p = Path(root)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise SystemExit(f"run root does not exist: {p}")
        runs.append(evaluate_run(label, p, dmax))

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "standards_safety_metrics.json").write_text(
        json.dumps(runs, indent=2), encoding="utf-8"
    )
    md = render_markdown(runs)
    (out_dir / "STANDARDS_SAFETY_REPORT.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {out_dir/'standards_safety_metrics.json'}")


if __name__ == "__main__":
    main()
