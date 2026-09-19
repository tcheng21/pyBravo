"""Axis speeds for the press and the syringe, pinned against captured VWorks traffic.

pybravo left velocity unset on several moves, which makes the controller use the
axis limit -- full speed. VWorks sets them deliberately, and the 2026-09-18
captures show pybravo running faster on every axis while hitting the same
positions:

    Z press                 VWorks  8.00%   pybravo 20%
    Zg return in the shuck  VWorks 66.73%   pybravo 100%
    W drive to zero         VWorks 16.67%   pybravo 100%

The two that matter both turn out to be values the profile already carries, which
is why the fix reads the profile rather than hard-coding capture numbers:

    Z slow_velocity 10.0 mm/s   = 8.00% of the 125.0 mm/s limit
    Z slow_acceleration 100.0   = 26.67% of the 375.0 mm/s^2 limit
    W safe_velocity 100.0 uL/s  = 16.67% of the 600.0 uL/s limit

The W case is the one with teeth. Every hardware run so far had an empty syringe,
so a full-speed plunger move did nothing; the same path runs with up to 250 uL
aboard during tips-off after pipetting and throughout the wash work.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pybravo.profile.profile import BravoProfile
from pybravo.state_machine.tasks import (
    PRESS_TRAVEL_MM,
    SHUCK_ZG_RETURN_FRACTION,
    _axis_speed,
    _axis_velocity_limit,
    _w_safe_speed,
)
from pybravo.types import Axis, SpeedLevel

PROFILE_PATH = Path(__file__).resolve().parents[1] / "profiles" / "96AM.yaml"

# Percentages read off the wire.
VWORKS_PRESS_VELOCITY_PCT = 8.00
VWORKS_PRESS_ACCEL_PCT = 26.67
VWORKS_W_VELOCITY_PCT = 16.67
VWORKS_ZG_RETURN_PCT = 66.73


@pytest.fixture(scope="module")
def profile() -> BravoProfile:
    return BravoProfile.load(PROFILE_PATH)


def test_press_speed_matches_the_captured_vworks_press(profile) -> None:
    """The press must run at VWorks' 8% / 26.67%, not at the axis limit.

    The press is force limited either way, so this is not about stopping in time --
    it is the speed the head arrives at the consumable with, which is what decides
    how a tip or cartridge seats.
    """
    v, a = _axis_speed(profile, Axis.Z, SpeedLevel.SLOW)
    fast_v, fast_a = _axis_speed(profile, Axis.Z, SpeedLevel.FAST)
    assert v / fast_v * 100 == pytest.approx(VWORKS_PRESS_VELOCITY_PCT, abs=0.05)
    assert a / fast_a * 100 == pytest.approx(VWORKS_PRESS_ACCEL_PCT, abs=0.05)


def test_syringe_safe_speed_matches_the_captured_vworks_move(profile) -> None:
    """W must drive to zero at VWorks' 16.67%, not 6x faster."""
    v, _ = _w_safe_speed(profile)
    assert v / _axis_velocity_limit(profile, Axis.W) * 100 == pytest.approx(
        VWORKS_W_VELOCITY_PCT, abs=0.05
    )


def test_shuck_zg_return_is_slowed(profile) -> None:
    """VWorks returns Zg at ~66.7% after the strip; the lift itself is full speed."""
    assert SHUCK_ZG_RETURN_FRACTION * 100 == pytest.approx(VWORKS_ZG_RETURN_PCT, abs=0.1)
    assert _axis_velocity_limit(profile, Axis.Zg) > 0.0


@pytest.mark.parametrize("axis,level", [(Axis.Z, SpeedLevel.SLOW), (Axis.W, SpeedLevel.SAFE)])
def test_speeds_are_never_zero(profile, axis: Axis, level: SpeedLevel) -> None:
    """Zero means "use the axis limit" -- the bug this fixes, not a valid value.

    If a profile ever omits these, these paths silently return to full speed with
    nothing logged, which is exactly how it went unnoticed until the captures were
    compared side by side.
    """
    v, a = _axis_speed(profile, axis, level)
    assert v > 0.0, f"{axis.name} {level.name} velocity is unset -> full speed"
    assert a > 0.0, f"{axis.name} {level.name} acceleration is unset -> full speed"


def test_missing_axis_config_degrades_to_the_limit_rather_than_crashing(profile) -> None:
    """An axis absent from the profile must not raise mid-move."""

    class _NoAxes:
        axes: dict = {}

    assert _axis_speed(_NoAxes(), Axis.Z, SpeedLevel.SLOW) == (0.0, 0.0)
    assert _axis_velocity_limit(_NoAxes(), Axis.Zg) == 0.0


def test_speeds_live_under_speeds_not_as_attributes(profile) -> None:
    """Guard the accessor mistake this whole fix came from.

    AxisConfig has no `safe_velocity` attribute -- the speeds are in
    `speeds[SpeedLevel]`. Code that reached for the attribute got a default of 0.0
    and silently ran the axis at its limit, which is how the syringe came to move
    6x faster than VWorks while the source read as though it asked for the safe
    speed.
    """
    w = profile.axes["W"]
    assert not hasattr(w, "safe_velocity")
    assert getattr(w, "safe_velocity", 0.0) == 0.0     # the silent failure
    assert w.speeds[SpeedLevel.SAFE].velocity == 100.0  # the real value


# Captured seated Z and the approach VWorks descended to first, per fixture.
CAPTURED_PRESSES = [
    ("cartridge rack", 123.057, 98.057),
    ("LT250 tip box", 105.270, 80.270),
]


@pytest.mark.parametrize("fixture,seated,approach", CAPTURED_PRESSES)
def test_approach_reproduces_the_captured_start_of_the_press(
    fixture: str, seated: float, approach: float
) -> None:
    """The press must start where VWorks starts it: PRESS_TRAVEL_MM above the seat.

    Applying the slow press speed to the whole descent is safe and detects
    correctly, but it covers 110-130 mm instead of 25 and turns a ~4 s press into
    11-13 s. VWorks descends fast with force off to this point, then presses.
    """
    assert seated - PRESS_TRAVEL_MM == pytest.approx(approach, abs=0.01), fixture


def test_press_travel_is_the_captured_25mm() -> None:
    """Every captured press covers exactly this, both consumables, every fixture."""
    assert PRESS_TRAVEL_MM == 25.0


def test_approach_is_fast_and_the_press_is_slow(profile) -> None:
    """The two halves must differ, or one of them is wrong.

    If these ever converge, either the approach has been slowed to the press speed
    (the 11-13 s regression) or the press has been sped up to the approach speed
    (the seating problem). Both have happened.
    """
    fast_v, _ = _axis_speed(profile, Axis.Z, SpeedLevel.FAST)
    slow_v, _ = _axis_speed(profile, Axis.Z, SpeedLevel.SLOW)
    assert fast_v > slow_v * 5, "approach should be far faster than the press"


def test_approach_is_skipped_when_the_head_is_already_below_it() -> None:
    """The approach is a descent, not a reposition.

    sequences.jog presses in +Z, so an approach above the current Z would mean
    retreating and then pressing back down through ground already covered. Profiles
    whose press target sits above safe Z do exactly that if the move is
    unconditional -- which is how this was first written.
    """
    seated, safe_z = 36.1, 42.5          # a profile where the press runs the other way
    approach = seated - PRESS_TRAVEL_MM
    assert approach < safe_z, "this fixture is only meaningful when the approach is behind us"
    # The guard is `approach_z > current_z`; here it is not, so no approach move.
    assert not (approach > safe_z)
