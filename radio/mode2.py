import random
import math
from collections import deque

class Mode2MAC:
    """Slot-level Mode-2 radio engine abstraction."""
    def __init__(self, slot_capacity_bits=10_000_000, cbr_decay=3.0, cbr_history_window=20,
                 sps_rri_slots=5, sps_lifetime_slots=5):
        # Shared sidelink resource budget per 1 ms slot. Payloads are charged
        # in bits, so concurrent computation packets create endogenous CBR.
        self.slot_capacity_bits = float(slot_capacity_bits)
        self.cbr_decay = float(cbr_decay)
        self.cbr_history_window = int(cbr_history_window)
        self.sps_rri_slots = int(sps_rri_slots)
        self.sps_lifetime_slots = int(sps_lifetime_slots)
        self.rng = random.Random()
        self.reset()

    def reset(self, seed=None):
        self.rng.seed(seed)
        self.current_slot = 0
        self.active_tx_this_slot = set()
        self.slot_load_bits = 0.0
        self.last_cbr = 0.0
        self.cbr_history = deque(maxlen=self.cbr_history_window)
        self.reservations = {}
        self.sps_reuse_count = 0
        self.sps_create_count = 0
        self.sps_waste_count = 0
        self.sps_collision_count = 0
        self.tx_attempts = 0
        self.tx_successes = 0
        self.resource_bits = 0.0
        self.resource_bits_by_class = {}

    def get_cbr(self):
        return self.last_cbr

    def get_cbr_history_mean(self):
        return (sum(self.cbr_history) / len(self.cbr_history)) if self.cbr_history else 0.0

    @staticmethod
    def _payload_bits(packet):
        if packet is None:
            return 0.0
        # UL payloads are MB; DL payloads are kB in the coding API.
        return float(packet.payload_size) * (8_000_000 if packet.direction == "UL" else 8_000)
        
    def advance_slot(self):
        for key, state in list(self.reservations.items()):
            # A reservation commits one resource opportunity every RRI.  An
            # unused opportunity is still occupied from the system's point of
            # view and therefore contributes the same reserved payload to the
            # shared CBR accounting as an actual transmission.
            if state["opportunity_in"] <= 0:
                if not state["used_this_slot"]:
                    self.sps_waste_count += 1
                    self.slot_load_bits += state["reserved_bits"]
                state["opportunity_in"] = max(0, state["rri"] - 1)
            else:
                state["opportunity_in"] -= 1
            state["used_this_slot"] = False
            state["remaining"] -= 1
            if state["remaining"] <= 0:
                del self.reservations[key]
        self.last_cbr = min(1.0, self.slot_load_bits / self.slot_capacity_bits)
        self.cbr_history.append(self.last_cbr)
        self.current_slot += 1
        self.active_tx_this_slot.clear()
        self.slot_load_bits = 0.0
        
    def attempt_transmission(self, packet, tx_veh_id, rx_veh_id, sensing_history, current_time) -> bool:
        """
        Evaluates Mode-2 transmission success using physical constraints.
        """
        self.tx_attempts += 1
        # 1. Half-Duplex Constraint: A vehicle cannot receive if it is transmitting in the same slot.
        if rx_veh_id in self.active_tx_this_slot:
            return False
            
        self.active_tx_this_slot.add(tx_veh_id)

        # DS selects resources for this attempt only. SPS creates/reuses a
        # persistent flow reservation; both modes share the same occupancy and
        # collision probability below.
        if getattr(packet, "scheduling_mode", "DS") == "SPS":
            key = (getattr(packet, "flow_id", None) or packet.job_id,
                   packet.direction, tx_veh_id, rx_veh_id)
            if key in self.reservations:
                self.sps_reuse_count += 1
            else:
                self.reservations[key] = {
                    "remaining": self.sps_lifetime_slots,
                    "rri": self.sps_rri_slots,
                    "opportunity_in": max(0, self.sps_rri_slots - 1),
                    "reserved_bits": self._payload_bits(packet),
                    "used_this_slot": False,
                }
                self.sps_create_count += 1
            self.reservations[key]["used_this_slot"] = True
        
        # Charge this attempt to the shared slot resource budget.
        payload_bits = self._payload_bits(packet)
        self.slot_load_bits += payload_bits
        self.resource_bits += payload_bits
        cls = getattr(packet, 'traffic_class', 'unknown')
        self.resource_bits_by_class[cls] = self.resource_bits_by_class.get(cls, 0.0) + payload_bits
        cbr = min(1.0, self.slot_load_bits / self.slot_capacity_bits)
        if sensing_history:
            cbr = sensing_history.get_cbr()

        p_success = math.exp(-self.cbr_decay * cbr)
        
        if self.rng.random() > p_success:
            if getattr(packet, "scheduling_mode", "DS") == "SPS":
                self.sps_collision_count += 1
            return False
        self.tx_successes += 1
        return True

    def observed_success_rate(self):
        if self.tx_attempts <= 0:
            return None
        return self.tx_successes / self.tx_attempts

    def predicted_success_probability(self, payload_bits):
        """Causal success estimate for a prospective transmission."""
        cbr = min(1.0, (self.slot_load_bits + float(payload_bits)) /
                  max(self.slot_capacity_bits, 1.0))
        return math.exp(-self.cbr_decay * cbr)
