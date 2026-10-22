"""Minimal explicit DS/SB-SPS commitment state for the pre-RL feasibility study."""
from radio.mode2 import Mode2MAC

class CommitmentMode2MAC(Mode2MAC):
    def __init__(self, *args, rri_slots=5, reservation_lifetime=5, **kwargs):
        super().__init__(*args, **kwargs); self.rri_slots=rri_slots; self.reservation_lifetime=reservation_lifetime

    def reset(self, seed=None):
        super().reset(seed); self.reservations={}; self.wasted_reservations=0

    def advance_slot(self):
        for flow, state in list(self.reservations.items()):
            state['remaining']-=1
            if not state['used']: self.wasted_reservations+=1
            state['used']=False
            if state['remaining']<=0: del self.reservations[flow]
        super().advance_slot()

    def transmit_mode(self, packet, tx, rx, now, flow_id, mode):
        """Transmit one packet and return (success, reservation_created/reused)."""
        reused=False
        if mode=='SPS':
            if flow_id in self.reservations:
                reused=True
            else:
                self.reservations[flow_id]={'remaining':self.reservation_lifetime,'used':False}
        ok=self.attempt_transmission(packet,tx,rx,None,now)
        if flow_id in self.reservations: self.reservations[flow_id]['used']=True
        return ok,reused
