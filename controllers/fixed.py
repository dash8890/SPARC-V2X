class FixedController:
    """Static margin redundancy controller: n = k + c."""
    def __init__(self, c: int = 0):
        self.c = c
        
    def select_action(self, obs, k: int, feasible_helpers: int) -> int | None:
        if feasible_helpers < k:
            return None
        return min(feasible_helpers, max(k, k + self.c))
