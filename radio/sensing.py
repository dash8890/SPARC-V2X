import numpy as np

class SensingHistory:
    """Maintains SB-SPS Mode-2 sensing histories and active reservations."""
    def __init__(self, num_subchannels: int, num_slots: int):
        self.num_subchannels = num_subchannels
        self.num_slots = num_slots
        self.history = np.zeros((num_slots, num_subchannels))
        
    def record_rssi(self, slot: int, subchannel: int, rssi: float):
        self.history[slot % self.num_slots, subchannel] = rssi
        
    def get_cbr(self, threshold: float = -85.0) -> float:
        return np.mean(self.history > threshold)
