class CBRAIMDController:
    """Congestion-reactive AIMD worker control based on CBR."""
    def __init__(self, cbr_upper: float = 0.6, cbr_lower: float = 0.3):
        self.cbr_upper = cbr_upper
        self.cbr_lower = cbr_lower
        self.current_c = 0

    def reset(self):
        self.current_c = 0
        
    def select_action(self, obs, k: int, feasible_helpers: int, cbr: float, recent_success: bool) -> int | None:
        if feasible_helpers < k:
            return None
        if cbr > self.cbr_upper:
            # Multiplicative Decrease
            self.current_c = int(self.current_c / 2)
        elif not recent_success and cbr < self.cbr_lower:
            # Additive Increase
            self.current_c += 1
            
        n = k + self.current_c
        
        if n > feasible_helpers:
            n = feasible_helpers
            self.current_c = n - k
        if n < k:
            n = k
            self.current_c = 0
            
        return n
