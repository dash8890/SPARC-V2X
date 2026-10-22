import math

class PropagationModel:
    """Abstracts Mode-2 signal propagation, path loss, and SINR."""
    def __init__(self, tx_power_dbm=23.0, noise_floor_dbm=-95.0):
        self.tx_power_dbm = tx_power_dbm
        self.noise_floor_dbm = noise_floor_dbm
        
    def get_path_loss(self, distance: float) -> float:
        # Simple WINNER+ / Log-distance path loss abstraction
        if distance < 1.0:
            distance = 1.0
        return 40.0 + 20.0 * math.log10(distance)
        
    def get_rx_power(self, distance: float) -> float:
        pl = self.get_path_loss(distance)
        return self.tx_power_dbm - pl
        
    def is_link_feasible(self, distance: float, sinr_threshold_db=5.0) -> bool:
        rx_power = self.get_rx_power(distance)
        snr = rx_power - self.noise_floor_dbm
        return snr >= sinr_threshold_db
