"""Golden test: cartridge-seating failure detection, from captured traffic.

A force-limited press either stalls against a cartridge and stops short, or runs
to the commanded target because nothing is there. There is no protocol error
either way -- VWorks detects the failure purely from the final Z position, and so
does pybravo. A press that cannot tell those apart is the most dangerous state
this system can be in, so the numbers are pinned here.

Measured, from Cartridge_On_Off_Pos6_VW14 and TipOnError_Retry_Ignore
(normalized Z, converted with the repo's own Darwin calibration):

    approach        0.592228    98.057 mm
    press target    0.692228   123.057 mm   (25.0 mm of press travel)
    SUCCESS final   0.685313   121.328 mm   stopped 1.729 mm short
    FAILURE final   0.692220   123.055 mm   ran the full travel, nothing there
"""
from __future__ import annotations

import pytest

from pybravo.darwin import sequences
from pybravo.darwin.calibration import DEFAULT_CALIBRATION
from pybravo.protocol.errors import BravoError, ErrorType
from pybravo.types import Axis

Z_RANGE_MM = DEFAULT_CALIBRATION[Axis.Z].hardware_range  # 250.0

PRESS_TARGET = 0.692228
SUCCESS_FINAL = 0.685313
FAILURE_FINAL = 0.692220

# Tolerance must be wide enough to accept the measured 1.729 mm stall and narrow
# enough that a full-travel landing is still "exceeded". 3 mm sits between them.
TOLERANCE_MM = 3.0


def _press(monkeypatch, final_position, *, tolerance_mm=TOLERANCE_MM):
    """Run sequences.jog with the wire I/O stubbed, returning the final position."""
    monkeypatch.setattr(sequences, "force_move", lambda *a, **k: None)
    monkeypatch.setattr(sequences.time, "sleep", lambda *_: None)
    tolerance = tolerance_mm / Z_RANGE_MM
    params = sequences.JogParams(
        axis_name="Z",
        target_position=PRESS_TARGET - tolerance,
        tolerance=tolerance,
        peak_current_amps=0.6,     # "Current limits/LT 96 tips" from the profile
        velocity_mm=0.0,
        acceleration_mm=0.0,
        velocity_limit=125.0,
        acceleration_limit=375.0,
        exceed_epsilon=0.05 / Z_RANGE_MM,
    )
    return sequences.jog(
        engine=None,
        axis_address=None,
        axis_params=None,
        p=params,
        read_position=lambda engine, addr: final_position,
    )


def test_successful_seating_is_accepted(monkeypatch):
    """A press that stalls 1.729 mm short means a cartridge resisted it."""
    assert _press(monkeypatch, SUCCESS_FINAL) == pytest.approx(SUCCESS_FINAL)


def test_missing_cartridge_is_detected(monkeypatch):
    """Reaching the commanded target means nothing was there to stop the head.

    This is the whole failure detection. If it regresses, a run continues as if
    cartridges were seated when the head is empty.
    """
    with pytest.raises(BravoError) as exc:
        _press(monkeypatch, FAILURE_FINAL)
    assert exc.value.error_type == ErrorType.EXCEEDED_DEST


def test_obstruction_is_detected(monkeypatch):
    """Stopping far too early is also a failure -- something is in the way."""
    with pytest.raises(BravoError) as exc:
        _press(monkeypatch, PRESS_TARGET - (20.0 / Z_RANGE_MM))
    assert exc.value.error_type == ErrorType.UNABLE_TO_REACH_DEST


def test_tolerance_separates_the_measured_cases(monkeypatch):
    """The chosen tolerance must accept the real success and reject the real
    failure. Pinned so nobody widens it past the 1.729 mm evidence."""
    _press(monkeypatch, SUCCESS_FINAL)  # must not raise
    with pytest.raises(BravoError):
        _press(monkeypatch, FAILURE_FINAL)


def test_press_travel_matches_the_capture():
    """The commanded press is 25.0 mm below the approach height."""
    cal = DEFAULT_CALIBRATION[Axis.Z]
    travel = cal.from_normalized(PRESS_TARGET) - cal.from_normalized(0.592228)
    assert travel == pytest.approx(25.0, abs=0.01)


def test_an_early_stall_is_rejected_not_reported_as_seated(monkeypatch):
    """A press stopped well short of the seat must NOT be read as success.

    pybravo force-limits the *entire* descent from safe Z rather than doing
    VWorks' fast approach followed by a bounded 25 mm press, which raises the
    obvious worry that a collision on the way down would stall the axis and be
    reported as seated consumables.

    It would not. The acceptance window is two-sided: `farthest = target +
    tolerance` bounds it above (nothing was there), and `target - tolerance`
    bounds it below (something stopped us too early). An obstruction partway down
    lands under the lower bound and raises UNABLE_TO_REACH_DEST.

    Pinned because an earlier reading of this code claimed the opposite, and the
    difference is "collision reported as success" versus "collision reported".
    """
    early_stall = PRESS_TARGET - (40.0 / Z_RANGE_MM)   # 40 mm short of the seat
    with pytest.raises(BravoError) as exc:
        _press(monkeypatch, early_stall)
    assert exc.value.error_type == ErrorType.UNABLE_TO_REACH_DEST


@pytest.mark.parametrize("short_mm,accepted", [
    (0.0, False),    # rode to the commanded farthest point -> nothing there
    (1.729, True),   # the measured VWorks stall
    (4.0, True),     # still inside the window
    (12.0, False),   # far too early -> something else stopped the head
])
def test_acceptance_window_is_bounded_at_both_ends(monkeypatch, short_mm, accepted):
    """Only a stall near the expected seat depth counts as seated."""
    final = PRESS_TARGET - (short_mm / Z_RANGE_MM)
    if accepted:
        assert _press(monkeypatch, final) == pytest.approx(final)
    else:
        with pytest.raises(BravoError):
            _press(monkeypatch, final)
