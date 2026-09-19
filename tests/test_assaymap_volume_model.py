"""Golden test: the AssayMAP plunger volume model against captured traffic.

Every expected value here was measured from VWorks driving the real instrument
(captures in ../wireshark) and independently confirmed by the instrument's own
device profile. If these fail, either the W calibration or the µL conversion has
drifted, and any volume pyBravo commands would be wrong.

Reference points, all normalized W positions taken off the wire:

    0.184781   "Home W" in the diagnostic, i.e. 0 µL; VWorks Jog/Teach reads 0
    0.048125   one MOVE_BY step of the 0->250 µL sweep  = 20 µL
    0.012031   the MOVE_BY of the "asp 5 µL" diagnostic =  5 µL
    0.209156 -> 0.329469   a 50 µL aspirate
    0.329469 -> 0.221187   a 45 µL dispense
"""
from __future__ import annotations

import pytest

from pybravo.darwin.waxis_config import HEAD_CONFIGS, config_for_head, ul_to_mm
from pybravo.types import HeadType

AM = HeadType.HT_96_ASSAYMAP

# From the instrument profile: /Axes/W "Darwin calibration offset".
W_CALIBRATION_OFFSET_MM = 1.44375

# Measured on the wire.
W_HOME_NORMALIZED = 0.184781
STEP_20UL_NORMALIZED = 0.048125
STEP_5UL_NORMALIZED = 0.012031


@pytest.fixture
def calibration():
    return config_for_head(AM).calibration(W_CALIBRATION_OFFSET_MM)


def test_assaymap_has_w_calibration():
    assert AM in HEAD_CONFIGS, "AssayMAP must have a W-axis calibration entry"


def test_zero_volume_lands_on_the_measured_home_position(calibration):
    """0 µL must map to the position the diagnostic's "Home W" actually commands."""
    assert calibration.to_normalized(ul_to_mm(0.0, AM)) == pytest.approx(
        W_HOME_NORMALIZED, abs=1e-6
    )


@pytest.mark.parametrize(
    "volume_ul, expected_step",
    [(20.0, STEP_20UL_NORMALIZED), (5.0, STEP_5UL_NORMALIZED)],
)
def test_relative_volume_steps_match_captured_move_by(calibration, volume_ul, expected_step):
    """A volume delta must produce the same normalized step VWorks emitted."""
    home = calibration.to_normalized(ul_to_mm(0.0, AM))
    target = calibration.to_normalized(ul_to_mm(volume_ul, AM))
    assert target - home == pytest.approx(expected_step, abs=1e-5)


@pytest.mark.parametrize(
    "start_norm, end_norm, expected_ul",
    [
        (0.209156, 0.329469, 50.0),   # aspirate, from the AspDisp capture
        (0.329469, 0.221187, -45.0),  # dispense, same capture
    ],
)
def test_captured_transfers_decode_to_the_stated_volumes(
    calibration, start_norm, end_norm, expected_ul
):
    """Positions seen on the wire must decode back to the volumes requested."""
    start_mm = calibration.from_normalized(start_norm)
    end_mm = calibration.from_normalized(end_norm)
    factor = config_for_head(AM).ul_to_mm_factor
    assert (end_mm - start_mm) / factor == pytest.approx(expected_ul, abs=0.01)


def test_software_limit_matches_the_instrument_profile():
    """software_max is in engineering units (µL x factor), so it must equal the
    vendor profile's own W "Max range" of 250.01 µL -- the 250 µL syringe plus
    the same hair of headroom the instrument allows."""
    cfg = config_for_head(AM)
    assert cfg.software_max / cfg.ul_to_mm_factor == pytest.approx(250.01, abs=0.01)


def test_full_syringe_is_commandable():
    """250 µL must be reachable without tripping the software limit."""
    cfg = config_for_head(AM)
    assert ul_to_mm(250.0, AM) <= cfg.software_max


def test_full_stroke_stays_inside_the_hardware_range(calibration):
    """250 µL from home must remain within travel — this is the safety check."""
    top = calibration.to_normalized(ul_to_mm(250.0, AM))
    assert 0.0 < top < 1.0, f"250 µL lands outside the hardware range at {top}"
