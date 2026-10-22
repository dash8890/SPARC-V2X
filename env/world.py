import math
from mobility.trace_loader import TraceLoader
from radio.propagation import PropagationModel
from compute.helpers import Helper

class ContactManager:
    """Tracks vehicle positions and computes feasible contacts."""
    def __init__(self, trace_loader: TraceLoader, propagation: PropagationModel):
        self.trace_loader = trace_loader
        self.propagation = propagation
        self.max_range = 300.0  # meters
        self._pair_intervals = {}
        
    def get_distance(self, pos1, pos2) -> float:
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

    def zone_id(self, vehicle: str, time_s: float, cell_m: float = 100.0):
        """Return a causal spatial demand zone from the SUMO position."""
        states = self.trace_loader.get_state_at_time(time_s)
        pos = states.get(vehicle)
        if pos is None:
            return None
        return (int(math.floor(pos[0] / cell_m)),
                int(math.floor(pos[1] / cell_m)))
        
    def get_feasible_helpers(self, time_s: float, uv_id: str, active_helpers: dict):
        """Returns a list of helper IDs currently in communication range."""
        states = self.trace_loader.get_state_at_time(time_s)
        if uv_id not in states:
            return []
            
        uv_pos = states[uv_id]
        feasible = []
        
        for h_id, helper in active_helpers.items():
            if h_id == uv_id:
                continue
            if h_id in states:
                dist = self.get_distance(uv_pos, states[h_id])
                if self.propagation.is_link_feasible(dist):
                    feasible.append(h_id)
                    
        return feasible

    def contact_at(self, u: str, v: str, time_s: float) -> bool:
        """Point contact query used by event-driven lifecycle replay."""
        states = self.trace_loader.get_state_at_time(time_s)
        if u not in states or v not in states:
            return False
        return self.propagation.is_link_feasible(self.get_distance(states[u], states[v]))

    def current_contact_end(self, u: str, v: str, time_s: float,
                           horizon_s: float = 10.0, step_s: float = .001) -> float:
        """Find the next contact boundary without changing simulation state."""
        t = float(time_s)
        if not self.contact_at(u, v, t):
            return t
        while t < horizon_s and self.contact_at(u, v, t + step_s):
            t += step_s
        return min(t + step_s, horizon_s)

    def next_contact_start(self, u: str, v: str, time_s: float,
                           horizon_s: float = 10.0, step_s: float = .001) -> float:
        t = float(time_s)
        while t < horizon_s and not self.contact_at(u, v, t):
            t += step_s
        return min(t, horizon_s)
