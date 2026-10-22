"""Effective PSSCH transport-block capacity for system-level simulation.

This is a standards-grounded abstraction, not a waveform/LDPC simulator. The
configuration is explicit so the selected MCS and resource budget are
reproducible and sensitivity-testable.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SidelinkTBSConfig:
    bandwidth_mhz: float = 10.0
    subcarrier_spacing_khz: float = 15.0
    n_prb: int = 52
    pssch_symbols: int = 10
    mcs_index: int = 10
    modulation_order: int = 4       # 16-QAM representative MCS
    target_code_rate: float = 0.48
    layers: int = 1
    overhead_fraction: float = 0.20 # DMRS/PSCCH/PSFCH and guard overhead

    def tbs_bits_per_opportunity(self) -> int:
        if self.n_prb <= 0 or self.pssch_symbols <= 0 or self.layers <= 0:
            raise ValueError("resource allocation must be positive")
        if not 0 < self.target_code_rate <= 1 or not 0 <= self.overhead_fraction < 1:
            raise ValueError("invalid code rate or overhead")
        usable_re = self.n_prb * 12 * self.pssch_symbols
        raw = usable_re * self.modulation_order * self.target_code_rate * self.layers
        return max(1, int(math.floor(raw * (1.0 - self.overhead_fraction))))


def effective_pssch_capacity(config: SidelinkTBSConfig | None = None) -> int:
    return (config or SidelinkTBSConfig()).tbs_bits_per_opportunity()
