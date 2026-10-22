"""SUMO trace loader for the Mode-2 simulator."""
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from functools import lru_cache

class TraceLoader:
    """Loads and interpolates SUMO mobility traces (e.g., FCD outputs)."""
    def __init__(self, trace_file):
        self.trace_file = trace_file
        self.df = self._load_fcd(trace_file)
        
        # Precalculate active intervals for O(1) active vehicle queries
        if not self.df.empty:
            df_reset = self.df.reset_index()
            self.min_times = df_reset.groupby('id')['time'].min()
            self.max_times = df_reset.groupby('id')['time'].max()
            # Immutable per-vehicle arrays avoid repeated Pandas slicing in
            # the 1-ms contact hot path.  Values and interpolation semantics
            # remain the same as the original dataframe implementation.
            self._arrays = {}
            for vid, group in df_reset.groupby('id', sort=False):
                g = group.sort_values('time')
                self._arrays[vid] = {
                    'time': g['time'].to_numpy(dtype=float),
                    'x': g['x'].to_numpy(dtype=float),
                    'y': g['y'].to_numpy(dtype=float),
                    'speed': g['speed'].to_numpy(dtype=float),
                    'angle': g['angle'].to_numpy(dtype=float),
                }
        else:
            self.min_times = pd.Series()
            self.max_times = pd.Series()
            self._arrays = {}

    def __deepcopy__(self, memo):
        # Loaded mobility is read-only; rollout copies only mutable simulator state.
        memo[id(self)] = self
        return self

    def _load_fcd(self, trace_file):
        """Loads the XML or CSV trace into a queryable dataframe."""
        tree = ET.parse(trace_file)
        root = tree.getroot()
        records = []
        for timestep in root.findall('timestep'):
            t = float(timestep.get('time'))
            for veh in timestep.findall('vehicle'):
                records.append({
                    'time': t,
                    'id': veh.get('id'),
                    'x': float(veh.get('x')),
                    'y': float(veh.get('y')),
                    'speed': float(veh.get('speed')),
                    'angle': float(veh.get('angle'))
                })
        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index(['id', 'time'], inplace=True)
            df.sort_index(inplace=True)
        return df

    def get_active_vehicles(self, t):
        """Expose active vehicle IDs at time t. No extrapolation beyond appearance/disappearance."""
        if self.df.empty:
            return []
        active = self.min_times[(self.min_times <= t) & (self.max_times >= t)].index.tolist()
        return active

    @lru_cache(maxsize=256)
    def get_state_at_time(self, t):
        """Returns vehicle positions at a specific time, interpolated if necessary.
        Format: dict[id] = (x, y, speed, angle)
        """
        if self.df.empty:
            return {}

        active = self.get_active_vehicles(t)
        states = {}
        for vid in active:
            arr = self._arrays[vid]
            times = arr['time']
            if t < times[0] or t > times[-1]:
                continue
            x = float(np.interp(t, times, arr['x']))
            y = float(np.interp(t, times, arr['y']))
            v = float(np.interp(t, times, arr['speed']))
            # Preserve the original shortest-angle interpolation convention.
            idx = max(0, min(len(times) - 2, int(np.searchsorted(times, t, side='right') - 1)))
            t0, t1 = times[idx], times[idx + 1]
            frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            a0, a1 = arr['angle'][idx], arr['angle'][idx + 1]
            diff = (a1 - a0 + 180) % 360 - 180
            a = float((a0 + frac * diff) % 360)
            states[vid] = (x, y, v, a)
            
        return states
