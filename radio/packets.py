from dataclasses import dataclass

@dataclass
class Packet:
    source: str
    destination: str
    job_id: int
    subtask_id: int
    direction: str       # 'UL' or 'DL'
    payload_size: float  # MB or kB
    generation_time: float
    deadline: float
    priority: int = 1
    scheduling_mode: str = "DS"
    flow_id: str | None = None
    traffic_class: str = "task"
