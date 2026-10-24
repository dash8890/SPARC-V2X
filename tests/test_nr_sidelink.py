"""Pin the 3GPP TS 38.214 PSSCH transport-block sizes.

The expected values below were produced by WiLabV2Xsim's own
``findTBS_5G.m`` (which implements TS 38.214 Section 5.1.3.2) with
``nDMRS=24``, ``SCIsymbols=3``, ``nRB_SCI=10`` and a 10-PRB subchannel.
They are the authority for the corrected payload-faithful contract.
"""
from __future__ import annotations

import pytest

from experiments.wilab_closed_loop.nr_sidelink import (
    BANDWIDTH_PRB_15KHZ,
    fragment_count,
    pool_prbs_for_bandwidth,
    required_subchannels,
    tbs,
    tbs_bytes,
)


# MATLAB findTBS_5G(prb, 24, mcs, 3, 10, 10)
MCS11_BITS = {
    10: 1160,
    20: 2976,
    30: 4736,
    40: 6400,
    50: 8192,
    60: 9992,
    70: 11784,
    80: 13576,
    90: 15368,
    100: 16896,
    106: 17928,
}
MCS7_BITS = {
    10: 808,
    20: 2024,
    30: 3240,
    40: 4480,
    50: 5632,
    52: 5888,
}


@pytest.mark.parametrize("prb,bits", MCS11_BITS.items())
def test_mcs11_tbs_matches_matlab(prb: int, bits: int):
    assert tbs(prb, 11).tbs_bits == bits
    assert tbs_bytes(prb, 11) == bits // 8


@pytest.mark.parametrize("prb,bits", MCS7_BITS.items())
def test_mcs7_tbs_matches_matlab(prb: int, bits: int):
    assert tbs(prb, 7).tbs_bits == bits


def test_bandwidth_prb_map():
    assert pool_prbs_for_bandwidth(10) == 52
    assert pool_prbs_for_bandwidth(20) == 106
    assert BANDWIDTH_PRB_15KHZ[20] == 106
    with pytest.raises(ValueError):
        pool_prbs_for_bandwidth(7)


def test_300b_tb_fits_two_subchannels():
    # 300 B = 2400 bits; 10 PRBs carry 1160 bits, 20 PRBs carry 2976 bits.
    assert required_subchannels(300, 11, pool_prbs=106) == (2, 17, 2408)


def test_2kb_segment_fits_only_in_20mhz_pool():
    # At 10 MHz (52 PRBs) a 2 kB packet cannot be a single TB.
    assert required_subchannels(2000, 11, pool_prbs=52) is None
    # At 20 MHz it needs 10 subchannels / 94 PRBs, i.e. almost the whole pool.
    assert required_subchannels(2000, 11, pool_prbs=106) == (10, 94, 16136)


def test_segment_is_fragmented_into_300b_tbs():
    # The 2 kB application segment is 7 PHY TBs, not one 2 kB TB.
    assert fragment_count(2000, 300) == 7
    assert fragment_count(10_000, 300) == 34
    assert fragment_count(80_000, 300) == 267
    assert fragment_count(0, 300) == 1
    with pytest.raises(ValueError):
        fragment_count(100, 0)
