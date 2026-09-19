"""AssayMAP ejects cartridges with the gripper; standard heads eject with W.

The safety-critical assertion is the negative one: with an AssayMAP head the W
axis must never be commanded during Tips Off. W on that head is a syringe whose
position is liquid state, so moving it would silently change the volume held.
"""
from __future__ import annotations

import asyncio

import pytest

from pybravo.deck.labware import Labware
from pybravo.deck.teachpoints import Teachpoints
from pybravo.head_mode import TipSelection, normalize_head_mode
from pybravo.profile.profile import BravoProfile
from pybravo.state_machine.tasks import TipsOffTask, TipsOnTask
from pybravo.types import Axis, HeadType


class RecordingController:
    """Records every commanded move without touching hardware."""

    def __init__(self) -> None:
        self.moves: list[tuple[str, float]] = []
        self.positions = {Axis.G: 0.0, Axis.Zg: -20.0, Axis.Z: 0.0, Axis.W: 0.0}

    def move(self, moves, wait=True):  # noqa: ARG002 - mirrors the controller API
        for m in moves:
            self.moves.append((m.axis.name, float(m.position)))
            self.positions[m.axis] = float(m.position)

    def get_position(self, axis):
        return self.positions[axis]

    def __getattr__(self, name):  # tolerate unrelated controller calls
        def _noop(*a, **k):
            return None
        return _noop

    @property
    def axes_moved(self) -> set[str]:
        return {axis for axis, _ in self.moves}


def _build_task(head_type: HeadType, consumable: str = "am_cartridge_60ul") -> tuple[TipsOffTask, RecordingController]:
    profile = BravoProfile.default()
    profile.head.head_type = head_type
    teachpoints = Teachpoints()
    teachpoints.set_default_teachpoints(head_type)
    labware = Labware(
        id="tipbox", name="tip box", height=49.9, width=85.48, length=127.76,
        metadata={"tip_definition_id": consumable},
    )
    mode = normalize_head_mode(head_type, "all_barrels", "back_left")
    ctrl = RecordingController()
    task = TipsOffTask(
        ctrl,
        teachpoints,
        profile,
        labware,
        mode,
        None,
        1,
        attached_tip_length_mm=0.0,
    )
    return task, ctrl


def test_assaymap_tips_off_never_moves_w():
    """The hazard: AssayMAP Tips Off must not command W at all."""
    task, ctrl = _build_task(HeadType.HT_96_ASSAYMAP)
    asyncio.run(task._eject_tips())
    assert "W" not in ctrl.axes_moved, (
        f"W was commanded during AssayMAP Tips Off: {ctrl.moves}"
    )


def test_assaymap_tips_off_shucks_with_gripper():
    """Gripper opens by the profile's G motion, gripper Z rises by its Zg motion,
    and both return to where they started."""
    task, ctrl = _build_task(HeadType.HT_96_ASSAYMAP)
    g0, zg0 = ctrl.get_position(Axis.G), ctrl.get_position(Axis.Zg)
    g_delta = task._profile.safety.cartridge_shuck_g_mm
    zg_delta = task._profile.safety.cartridge_shuck_zg_mm

    asyncio.run(task._eject_tips())

    gripper_moves = [m for m in ctrl.moves if m[0] in ("G", "Zg")]
    assert gripper_moves == [
        ("G", g0 + g_delta),
        ("Zg", zg0 + zg_delta),
        ("Zg", zg0),
        ("G", g0),
    ]


@pytest.mark.parametrize("head_type", [HeadType.HT_96_D_70, HeadType.HT_384_D_70])
def test_standard_heads_still_eject_with_w(head_type):
    """Regression: heads that already worked must be unchanged — they still
    drive W and never use the gripper shuck."""
    task, ctrl = _build_task(head_type)
    asyncio.run(task._eject_tips())
    assert "W" in ctrl.axes_moved
    w_targets = [pos for axis, pos in ctrl.moves if axis == "W"]
    assert w_targets[-1] == 0.0, "W should be reset to 0 after ejection"
    assert "G" not in ctrl.axes_moved and "Zg" not in ctrl.axes_moved


def _build_tips_on_task(head_type: HeadType, consumable: str = "am_cartridge_60ul") -> tuple[TipsOnTask, RecordingController]:
    profile = BravoProfile.default()
    profile.head.head_type = head_type
    teachpoints = Teachpoints()
    teachpoints.set_default_teachpoints(head_type)
    labware = Labware(
        id="tipbox", name="tip box", height=49.9, width=85.48, length=127.76,
        metadata={"tip_definition_id": consumable},
    )
    mode = normalize_head_mode(head_type, "all_barrels", "back_left")
    ctrl = RecordingController()
    selection = TipSelection(
        location=1, row=0, col=0,
        row_count=mode.row_count, column_count=mode.column_count,
    )
    task = TipsOnTask(
        ctrl, teachpoints, profile, labware, mode, selection, 1, tip_length_mm=0.0
    )
    return task, ctrl


def test_assaymap_tips_on_does_not_zero_the_syringe():
    """Seating cartridges must not expel what the syringe is holding.

    The captured instrument issues no W command during cartridge seating. On this
    head W is liquid state, so the generic "zero the plunger before tips on" step
    would dump the syringe contents onto the deck.
    """
    task, ctrl = _build_tips_on_task(HeadType.HT_96_ASSAYMAP)
    ctrl.positions[Axis.W] = 50.0          # syringe holding 50 uL
    asyncio.run(task._ensure_w_zero())
    assert "W" not in ctrl.axes_moved, f"W was commanded: {ctrl.moves}"
    assert ctrl.positions[Axis.W] == 50.0


@pytest.mark.parametrize("head_type", [HeadType.HT_96_D_70, HeadType.HT_384_D_70])
def test_standard_heads_still_zero_w_before_tips_on(head_type):
    """Regression: disposable-tip heads keep the existing behaviour."""
    task, ctrl = _build_tips_on_task(head_type)
    ctrl.positions[Axis.W] = 50.0
    asyncio.run(task._ensure_w_zero())
    assert "W" in ctrl.axes_moved
    assert ctrl.positions[Axis.W] == 0.0


def test_assaymap_press_current_matches_the_captured_force():
    """Cartridge seating must use the long-tip current, not the short-tip one.

    The instrument pressed at 66.67% force. The LT table's 0.6 A at 96 channels
    maps to 67%; the ST table's 0.3 A maps to 38% and would leave cartridges
    unseated. AssayMAP is in neither LT set by default, so it fell through to ST.
    """
    from pybravo.darwin.sequences import _z_axis_force_percent

    task, _ = _build_tips_on_task(HeadType.HT_96_ASSAYMAP)
    task._profile.current_limits = {"LT": {"96 tips": 0.6}, "ST": {"96 tips": 0.3}}
    amps = task._tip_press_current()
    assert amps == pytest.approx(0.6)
    assert _z_axis_force_percent(amps) == pytest.approx(67.0, abs=0.5)


@pytest.mark.parametrize("head_type", [HeadType.HT_96_D_70, HeadType.HT_384_D_70])
def test_short_tip_heads_keep_the_st_current(head_type):
    """Regression: heads that used ST must be unchanged."""
    task, _ = _build_tips_on_task(head_type)
    task._profile.current_limits = {"LT": {"96 tips": 0.6}, "ST": {"96 tips": 0.3}}
    assert task._tip_press_current() == pytest.approx(0.3)


def test_shuck_refuses_without_a_gripper():
    """A gripperless machine must refuse rather than stall mid-ejection."""
    task, ctrl = _build_task(HeadType.HT_96_ASSAYMAP)
    task._profile.axes.pop("G", None)
    with pytest.raises(RuntimeError, match="needs the gripper"):
        asyncio.run(task._shuck_with_gripper())
    assert ctrl.moves == [], "nothing should have been commanded"


# --- cartridges vs pipette tips on the same head ----------------------------
# Vendor: "The w-axis is not engaged during cartridge mounting or removal from
# the head so that fluid can be held in the syringes and probes. For the Tips On
# and Tips Off tasks, the Bravo w-axis goes to the zero position to empty any
# fluid contained in the syringes."

def test_mounting_pipette_tips_does_empty_the_syringes():
    """The same head DOES zero W for pipette tips — only cartridges are spared."""
    task, ctrl = _build_tips_on_task(HeadType.HT_96_ASSAYMAP, consumable="am_lt250_teach")
    ctrl.positions[Axis.W] = 50.0
    asyncio.run(task._ensure_w_zero())
    assert "W" in ctrl.axes_moved, "pipette tips should empty the syringes first"
    assert ctrl.positions[Axis.W] == 0.0


def test_removing_pipette_tips_empties_then_shucks():
    """Tips off: empty the syringes, then strip with the gripper."""
    task, ctrl = _build_task(HeadType.HT_96_ASSAYMAP, consumable="am_lt250_teach")
    ctrl.positions[Axis.W] = 50.0
    asyncio.run(task._eject_tips())
    assert ("W", 0.0) in ctrl.moves, "syringes should be emptied before tip removal"
    assert "G" in ctrl.axes_moved and "Zg" in ctrl.axes_moved, "stripper plate still used"


def test_removing_cartridges_still_never_touches_w():
    """Cartridges keep their fluid — this is the capability the head exists for."""
    task, ctrl = _build_task(HeadType.HT_96_ASSAYMAP, consumable="am_cartridge_60ul")
    ctrl.positions[Axis.W] = 50.0
    asyncio.run(task._eject_tips())
    assert "W" not in ctrl.axes_moved, f"W was commanded: {ctrl.moves}"
    assert ctrl.positions[Axis.W] == 50.0
