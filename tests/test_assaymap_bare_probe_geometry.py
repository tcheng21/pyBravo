"""Golden test: bare-probe pipetting Z, solved from the Startup/Shutdown captures.

The AssayMAP head pipettes three ways -- LT250 tips, cartridges, or bare probes.
The bare case has no consumable to measure, and the probes protrude below the head
reference the teachpoints were taught against, so "bare" is not zero length.

The operator confirms the Startup and Shutdown captures pipette bare, and they do
it at the wash station, whose vendor definition we hold. That is enough to solve
for the protrusion: it is the only value that puts VWorks' own Z commands on
sensible numbers.

    retract      10.000 mm ABOVE the labware top   (= the profile approach height)
    post-expel    3.000 mm below the top
    dispense     10.000 mm below the top
    aspirate     10.800 mm below the top = 9.000 mm from the well bottom

Independent corroboration: the resulting tip_delta of 38.2 is exactly the
DISPOSABLE_TIP_LENGTH the vendor records for both fixtures used bare (Tip Wash
Station, 96AM Receiver Plate) -- a field that played no part in the fit.

This previously read 0.0, which is impossible rather than merely imprecise; the
last test here is the one that would have caught it.
"""
from __future__ import annotations

import pytest

from pybravo.deck.labware import Labware
from pybravo.deck.teachpoints import Teachpoints
from pybravo.state_machine.tasks import BARE_PROBE_LENGTH_MM, _build_liquid_z_geometry
from pybravo.types import Axis, HeadType

TEACH_TIP_MM = 55.5
WASH_LOCATION = 1
WASH_TEACH_Z = 105.35      # profile teachpoint 1
WASH_THICKNESS = 49.50     # catalog height_mm, from the vendor file
WASH_WELL_DEPTH = 19.80    # catalog well_depth_mm

# Head Z as VWorks commanded it, and where that puts the probe relative to the
# labware top (negative = above it).
CAPTURED = [
    ("retract", 84.050, -10.000),
    ("dispense to waste", 96.850, 2.800),
    ("post-expel", 97.050, 3.000),
    ("dispense", 104.050, 10.000),
    ("aspirate", 104.850, 10.800),
]


@pytest.fixture
def teachpoints() -> Teachpoints:
    tp = Teachpoints()
    tp.set_teachpoint(WASH_LOCATION, Axis.Z, WASH_TEACH_Z)
    return tp


@pytest.fixture
def wash_station() -> Labware:
    return Labware(
        id="lw-fba7222815b4",
        name="96AM Tip Wash Station",
        height=WASH_THICKNESS,
        width=85.48,
        length=127.76,
        metadata={"well_depth_mm": WASH_WELL_DEPTH, "base_class": "Tip Wash Station"},
    )


def _geometry(teachpoints: Teachpoints, labware: Labware, distance_from_bottom: float):
    return _build_liquid_z_geometry(
        teachpoints=teachpoints,
        location=WASH_LOCATION,
        labware=labware,
        head_type=HeadType.HT_96_ASSAYMAP,
        teach_tip_length_mm=TEACH_TIP_MM,
        attached_tip_length_mm=None,
        tips_on_head=False,
        distance_from_bottom_mm=distance_from_bottom,
    )


def test_bare_delta_is_the_vendor_bare_fixture_tip_length(teachpoints, wash_station) -> None:
    """tip_delta must come out 38.2 -- the vendor's DISPOSABLE_TIP_LENGTH for this fixture."""
    geom = _geometry(teachpoints, wash_station, 0.0)
    assert geom.tip_delta_mm == pytest.approx(TEACH_TIP_MM - BARE_PROBE_LENGTH_MM)
    assert geom.tip_delta_mm == pytest.approx(38.2, abs=1e-6)


def test_aspirate_reproduces_the_captured_head_z(teachpoints, wash_station) -> None:
    """9.0 mm from the well bottom must command exactly the Z VWorks sent."""
    geom = _geometry(teachpoints, wash_station, 9.0)
    assert geom.target_head_z == pytest.approx(104.850, abs=0.01)


def test_dispense_reproduces_the_captured_head_z(teachpoints, wash_station) -> None:
    """9.8 mm from the bottom -- i.e. 10.0 mm below the labware top."""
    geom = _geometry(teachpoints, wash_station, 9.8)
    assert geom.target_head_z == pytest.approx(104.050, abs=0.01)


@pytest.mark.parametrize("label,head_z,depth_below_top", CAPTURED)
def test_every_captured_move_sits_where_expected(
    teachpoints, wash_station, label: str, head_z: float, depth_below_top: float
) -> None:
    """Each captured Z must place the probe at the stated depth below the labware top.

    This is what pins the protrusion: one value has to satisfy all five at once.
    """
    geom = _geometry(teachpoints, wash_station, 0.0)
    probe_z = head_z - geom.tip_delta_mm
    assert probe_z - geom.top_plane_tip_z == pytest.approx(depth_below_top, abs=0.01), label


def test_bare_probes_are_never_treated_as_zero_length(teachpoints, wash_station) -> None:
    """The regression that matters.

    With attached=0.0 the delta is the full teach length, which places the probe
    ABOVE the labware top at the moment of aspirating -- physically impossible --
    and commands the head 17.3 mm lower than VWorks for the same requested depth,
    driving the probes through the bottom of a 19.8 mm well.
    """
    geom = _geometry(teachpoints, wash_station, 9.0)
    assert geom.tip_delta_mm != pytest.approx(TEACH_TIP_MM)

    naive_head_z = geom.target_tip_z + TEACH_TIP_MM
    assert naive_head_z - geom.target_head_z == pytest.approx(BARE_PROBE_LENGTH_MM, abs=1e-6)
    # And that naive Z would put the probe well past the bottom of the well.
    assert (naive_head_z - geom.tip_delta_mm) > geom.well_bottom_tip_z
