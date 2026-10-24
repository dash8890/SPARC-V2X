"""Shared vehicle geometry for the Python task layer and WiLabV2Xsim.

WiLabV2Xsim loads the mobility trace in ``traces/mode2_external_validation``
through ``loadTrafficTrace.m``.  The trace file columns are

    time_s  vehicle_id  x_m  y_m  speed_m_s

and WiLab renumbers the vehicles sequentially from 1 in increasing input-ID
order.  The files used here already use IDs ``1..density``, so the WiLab
vehicle ID is the trace ID.

The Python task layer previously sampled unrelated synthetic positions.  That
made a helper look reachable in Python while the radio saw a different
distance (and often 0 % PRR).  ``VehicleTrace`` is the single geometry source
for both sides: the scheduler reads the same trace rows that WiLab reads.

The trace is sampled every 100 ms.  Scheduler epochs are 20 ms, so positions
and speeds are linearly interpolated between trace rows.  Over the 30--150 ms
task deadlines this is an exact representation of the trace motion (the
source trace itself is piecewise linear between samples).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class TraceSample:
    t_s: float
    x_m: float
    y_m: float
    speed_m_s: float


class VehicleTrace:
    """Piecewise-linear vehicle trajectory loaded from a WiLab trace file."""

    def __init__(self, samples: Dict[int, List[TraceSample]]):
        if not samples:
            raise ValueError("empty vehicle trace")
        self.samples = {
            int(veh): sorted(rows, key=lambda row: row.t_s)
            for veh, rows in samples.items()
        }
        self.vehicle_ids: Tuple[int, ...] = tuple(sorted(self.samples))
        self.t_min = min(rows[0].t_s for rows in self.samples.values())
        self.t_max = max(rows[-1].t_s for rows in self.samples.values())

    @classmethod
    def load(cls, path: str | Path) -> "VehicleTrace":
        path = Path(path)
        rows: Dict[int, List[TraceSample]] = {}
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 4:
                    raise ValueError(f"{path}:{line_no}: expected at least 4 columns")
                t_s = float(parts[0])
                veh = int(float(parts[1]))
                x_m = float(parts[2])
                y_m = float(parts[3])
                speed = float(parts[4]) if len(parts) >= 5 else 0.0
                rows.setdefault(veh, []).append(TraceSample(t_s, x_m, y_m, speed))
        return cls(rows)

    def _rows(self, veh_id: int) -> List[TraceSample]:
        try:
            return self.samples[int(veh_id)]
        except KeyError as exc:
            raise KeyError(f"vehicle {veh_id} is not present in the trace") from exc

    def sample(self, veh_id: int, t_s: float) -> TraceSample:
        rows = self._rows(veh_id)
        if t_s <= rows[0].t_s:
            return rows[0]
        if t_s >= rows[-1].t_s:
            return rows[-1]
        lo, hi = 0, len(rows) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if rows[mid].t_s <= t_s:
                lo = mid
            else:
                hi = mid
        a, b = rows[lo], rows[hi]
        span = b.t_s - a.t_s
        if span <= 0.0:
            return b
        w = (t_s - a.t_s) / span
        return TraceSample(
            t_s=t_s,
            x_m=a.x_m + w * (b.x_m - a.x_m),
            y_m=a.y_m + w * (b.y_m - a.y_m),
            speed_m_s=a.speed_m_s + w * (b.speed_m_s - a.speed_m_s),
        )

    def position(self, veh_id: int, t_s: float) -> Tuple[float, float]:
        row = self.sample(veh_id, t_s)
        return row.x_m, row.y_m

    def speed(self, veh_id: int, t_s: float) -> float:
        return self.sample(veh_id, t_s).speed_m_s

    def distance(self, veh_a: int, veh_b: int, t_s: float) -> float:
        xa, ya = self.position(veh_a, t_s)
        xb, yb = self.position(veh_b, t_s)
        return ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5

    def rows_at(self, veh_id: int, t_s: float) -> TraceSample:
        return self.sample(veh_id, t_s)


def resolve_trace_path(root: Path, density: int) -> Path:
    return Path(root) / "traces" / "mode2_external_validation" / f"synthetic_highway_rho_{int(density)}_20s.txt"


def vehicle_id_map(density: int, n_sources: int, n_helpers: int) -> Dict[str, Dict[int, int]]:
    """Disjoint Python-ID -> WiLab-ID map.

    Sources occupy the first ``n_sources`` trace vehicles; helpers occupy the
    next ``n_helpers`` vehicles.  The map is disjoint, so a source never sends
    to itself and a helper is never confused with a source.  ``loadTrafficTrace``
    renumbers input IDs sequentially, and the synthetic trace IDs are already
    ``1..density``.
    """
    density = int(density)
    n_sources = max(1, int(n_sources))
    n_helpers = max(0, int(n_helpers))
    if n_sources + n_helpers > density:
        raise ValueError(
            f"disjoint source/helper map needs {n_sources}+{n_helpers} vehicles, "
            f"but density is {density}"
        )
    return {
        "sources": {i: i + 1 for i in range(n_sources)},
        "helpers": {j: n_sources + j + 1 for j in range(n_helpers)},
    }
