"""NR-V2X PSSCH transport-block sizing.

This module is the single Python source for the physical transport-block size
used by the corrected payload-faithful contract.  It mirrors WiLabV2Xsim's
``findTBS_5G.m`` / ``fTBS_5G.m``, which implement 3GPP TS 38.214
Section 5.1.3.2 (transport-block size determination) for the NR sidelink
PSSCH.  The MATLAB implementation remains the simulator's authority; the
values below are pinned against it in ``tests/test_nr_sidelink.py``.

Why this exists: the previous contract assumed a fixed 1000 B (or 2000 B)
application packet was a single PHY transport block.  At 10 MHz / MCS 11 a
10-PRB subchannel carries only ~145 B, and the whole 52-PRB pool carries
~1089 B.  A 2 kB application segment therefore cannot be one TB; it must be
fragmented into TBs whose size is derived from the actual allocation.

The 300 B safety/computation TB used by the corrected configuration fits in
two 10-PRB subchannels (372 B TBS), so it is physically valid and reuses the
simulator's existing 300 B PRR model.
"""
from __future__ import annotations

import math
from typing import NamedTuple

# TS 38.214 Table 5.1.3.1-1 (MCS index table 1): (Qm, R x 1024).
MCS_TABLE: dict[int, tuple[int, int]] = {
    0: (2, 120), 1: (2, 157), 2: (2, 193), 3: (2, 251), 4: (2, 308),
    5: (2, 379), 6: (2, 449), 7: (2, 526), 8: (2, 602), 9: (2, 679),
    10: (4, 340), 11: (4, 378), 12: (4, 434), 13: (4, 490), 14: (4, 553),
    15: (4, 616), 16: (4, 658), 17: (6, 438), 18: (6, 466), 19: (6, 517),
    20: (6, 567), 21: (6, 616), 22: (6, 666), 23: (6, 719), 24: (6, 772),
    25: (6, 822), 26: (6, 873), 27: (6, 910), 28: (6, 948),
}

# TS 38.214 Table 5.1.3.2-1 (N_info' <= 3824), as stored by WiLab in
# ``tableNinfo1_5G.mat``.
TBS_TABLE: tuple[int, ...] = (
    24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144,
    152, 160, 168, 176, 184, 192, 208, 224, 240, 256, 272, 288, 304, 320,
    336, 352, 368, 384, 408, 432, 456, 480, 504, 528, 552, 576, 608, 640,
    672, 704, 736, 768, 808, 848, 888, 928, 984, 1032, 1064, 1128, 1160,
    1192, 1224, 1256, 1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672,
    1736, 1800, 1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472,
    2536, 2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496,
    3624, 3752, 3824,
)


class TbsResult(NamedTuple):
    tbs_bits: int
    n_re: int
    qm: int
    code_rate: float
    n_info: float


def _quantize_tbs(n_info: float, code_rate: float) -> int:
    """TS 38.214 Section 5.1.3.2 quantization (WiLab ``fTBS_5G``)."""
    if n_info <= 3824:
        n = max(3, int(math.floor(math.log2(n_info))) - 6)
        n_info1 = max(24, (2 ** n) * int(math.floor(n_info / (2 ** n))))
        for tbs in TBS_TABLE:
            if tbs >= n_info1:
                return tbs
        return TBS_TABLE[-1]

    n = int(math.floor(math.log2(n_info - 24))) - 5
    n_info1 = max(3840, (2 ** n) * round((n_info - 24) / (2 ** n)))
    if code_rate <= 0.25:
        c = math.ceil((n_info1 + 24) / 3816)
        return 8 * c * math.ceil((n_info1 + 24) / (8 * c)) - 24
    if n_info1 >= 8424:
        c = math.ceil((n_info1 + 24) / 8424)
        return 8 * c * math.ceil((n_info1 + 24) / (8 * c)) - 24
    return 8 * math.ceil((n_info1 + 24) / 8) - 24


def tbs(n_prb: int, mcs: int, *, n_dmrs: int = 24, sci_symbols: int = 3,
        n_rb_sci: int = 10) -> TbsResult:
    """PSSCH transport-block size for ``n_prb`` PRBs and an MCS index.

    Mirrors ``findTBS_5G.m``: 14 symbols per slot, 2 symbols for the
    sidelink shared-channel gap, SCI-1 occupying ``sci_symbols`` x
    ``n_rb_sci`` RBs, SCI-2 format 2b, and one layer.
    """
    if mcs not in MCS_TABLE:
        raise ValueError(f"unsupported MCS index {mcs}")
    if n_dmrs not in (12, 15, 18, 21, 24):
        raise ValueError(f"unsupported nDMRS {n_dmrs}")
    if sci_symbols not in (2, 3):
        raise ValueError(f"unsupported SCI symbols {sci_symbols}")
    if n_rb_sci not in (10, 12, 15, 20, 25):
        raise ValueError(f"unsupported nRB_SCI {n_rb_sci}")

    n_re1 = 12 * (14 - 2) - n_dmrs
    qm, r1024 = MCS_TABLE[mcs]
    code_rate = r1024 / 1024.0

    msc_pssch = n_re1 * n_prb
    msc_dmrs = n_dmrs * n_prb
    msc_sci2 = msc_pssch - msc_dmrs
    q_sci2 = min(
        math.ceil((48 + 24) * 1.125 / (qm * code_rate)),
        math.ceil(0.5 * msc_sci2),
    )
    n_re_sci1a = sci_symbols * n_rb_sci * 12
    n_re = n_re1 * n_prb - n_re_sci1a - q_sci2
    n_info = n_re * qm * code_rate
    return TbsResult(
        tbs_bits=_quantize_tbs(n_info, code_rate),
        n_re=int(n_re),
        qm=int(qm),
        code_rate=float(code_rate),
        n_info=float(n_info),
    )


def tbs_bytes(n_prb: int, mcs: int, **kwargs) -> int:
    return tbs(n_prb, mcs, **kwargs).tbs_bits // 8


def required_subchannels(
    payload_bytes: int,
    mcs: int,
    *,
    size_subchannel: int = 10,
    pool_prbs: int = 106,
    **kwargs,
) -> tuple[int, int, int] | None:
    """Smallest (subchannels, PRBs, TBS bits) that carries ``payload_bytes``.

    Returns ``None`` when the payload cannot fit in the pool as one TB; the
    caller must then fragment it.  This mirrors
    ``findRBsBeaconSINRmin_5G.m``.
    """
    needed_bits = int(payload_bytes) * 8
    if needed_bits <= 0:
        return (0, 0, 0)
    n_sub = 1
    while True:
        prb_hi = n_sub * size_subchannel
        if prb_hi > pool_prbs:
            return None
        result = tbs(prb_hi, mcs, **kwargs)
        if result.tbs_bits >= needed_bits:
            for prb in range((n_sub - 1) * size_subchannel + 1, prb_hi + 1):
                candidate = tbs(prb, mcs, **kwargs)
                if candidate.tbs_bits >= needed_bits:
                    return (n_sub, prb, candidate.tbs_bits)
        n_sub += 1


def fragment_count(payload_bytes: int, tb_bytes: int) -> int:
    """Number of PHY transport blocks needed for an application payload."""
    if tb_bytes <= 0:
        raise ValueError("tb_bytes must be positive")
    return max(1, math.ceil(int(payload_bytes) / int(tb_bytes)))


# TS 38.104 / WiLab ``RBperChannel.txt``: PRBs available at 15 kHz SCS.
BANDWIDTH_PRB_15KHZ: dict[int, int] = {
    5: 25,
    10: 52,
    15: 79,
    20: 106,
    25: 133,
    30: 160,
    40: 216,
    50: 270,
}


def pool_prbs_for_bandwidth(bw_mhz: int) -> int:
    """Number of PRBs in a 15 kHz-SCS carrier of ``bw_mhz`` MHz."""
    try:
        return BANDWIDTH_PRB_15KHZ[int(bw_mhz)]
    except KeyError as exc:
        raise ValueError(f"unsupported bandwidth {bw_mhz} MHz") from exc

