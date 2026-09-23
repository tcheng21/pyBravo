"""The LT250 tip stops containing liquid at 140 uL, not at its 250 uL.

The vendor's guidance for the AssayMAP head: to aspirate more than 140 uL, use
the bare probes rather than disposable tips, because above that "the excess
aspirated liquid will enter the syringes". The tip holds 250; the head only
keeps 140 of it out of the plumbing.

That outcome is contamination and a wash, not a crash, so this **warns and lets
the operator continue** rather than refusing. Two things are worth pinning:

  * the warning fires **before the head moves** -- a question you can only
    answer after the probes are in the wells is not much of a question; and
  * a **bare** head is never warned. `active_tip_id()` falls back to the taught
    tip when nothing is mounted, and this head is taught with the very LT250 tip
    that carries the limit -- so the naive check would warn about a tip that is
    not there, while the operator was doing the exact thing the guidance asks
    for.
"""
from __future__ import annotations

import asyncio

import pytest

from pybravo.bravo import Bravo
from pybravo.controllers.base import AxisMoveInfo
from pybravo.controllers.simulation import SimulationController
from pybravo.deck.labware import Labware, LabwareDefinition
from pybravo.deck.teachpoints import Teachpoints
from pybravo.profile.profile import BravoProfile
from pybravo.state_machine.tasks import _tip_overflow_prompt
from pybravo.tips import get_tip_definition
from pybravo.types import HeadType

LIMIT_UL = 140.0
TEACH_TIP = "am_lt250_teach"
CARTRIDGE = "am_cartridge_60ul"
PLATE_LOCATION = 5


# ---------------------------------------------------------------- the data

def test_the_limit_is_on_the_tip_definition() -> None:
    """Through the real loader, not the YAML -- the catalog has dropped rows
    silently before (F9), and a test that reads the file would not notice."""
    tip = get_tip_definition(HeadType.HT_96_ASSAYMAP, TEACH_TIP)
    assert tip is not None, f"{TEACH_TIP} is missing from the AssayMAP tip options"
    assert tip.overflow_ul == pytest.approx(LIMIT_UL)
    # Separate from capacity on purpose: the tip really does hold 250.
    assert tip.capacity_ul == pytest.approx(250.0)


def test_cartridges_carry_no_limit() -> None:
    """Cartridges are packed beds, not tips; the 140 figure is not theirs."""
    tip = get_tip_definition(HeadType.HT_96_ASSAYMAP, CARTRIDGE)
    assert tip is not None
    assert tip.overflow_ul is None


# ------------------------------------------------------------- the decision

def _prompt(liquid_ul: float, overflow_ul: float | None = LIMIT_UL):
    return _tip_overflow_prompt(
        overflow_ul=overflow_ul, liquid_ul=liquid_ul, location=5, operation="Aspirate"
    )


def test_over_the_limit_prompts() -> None:
    prompt = _prompt(200.0)
    assert prompt is not None
    assert prompt["kind"] == "tip_overflow"
    assert "200.0" in prompt["message"] and "140.0" in prompt["message"]


def test_the_choices_are_ignore_and_abort() -> None:
    """No "retry": re-running the same step cannot change the volume, so it
    would only be a second way to spell "ignore"."""
    assert _prompt(200.0)["choices"] == ["ignore", "abort"]


def test_at_the_limit_is_not_over_it() -> None:
    assert _prompt(LIMIT_UL) is None


def test_under_the_limit_is_silent() -> None:
    assert _prompt(139.9) is None


def test_a_tip_with_no_limit_never_prompts() -> None:
    """Every other tip in the catalog, and every other head."""
    assert _prompt(10_000.0, overflow_ul=None) is None


# --------------------------------------------------- what counts as liquid

class _Recorder(SimulationController):
    """A simulation controller that remembers whether it was asked to move."""

    def __init__(self) -> None:
        super().__init__()
        self.move_calls: list[list[AxisMoveInfo]] = []

    def move(self, moves, wait: bool = True):
        self.move_calls.append(list(moves))
        return super().move(moves, wait)


def _assaymap_bravo(*, tips_on: bool = True, tip_id: str = TEACH_TIP) -> Bravo:
    profile = BravoProfile.default()
    profile.head.head_type = HeadType.HT_96_ASSAYMAP
    profile.resolve_tips_for_head()
    profile.safety.z_safe_position = 0.0
    tp = Teachpoints()
    tp.set_default_teachpoints(profile.head.head_type)
    profile.teachpoints = tp

    bravo = Bravo(profile=profile)
    controller = _Recorder()
    controller.open_tcp("simulation")
    bravo._controller = controller
    bravo._teachpoints = tp

    bravo.deck.set_single(PLATE_LOCATION, Labware.from_definition(LabwareDefinition(
        id="plate-96-overflow-test",
        name="96 Test Plate",
        kind="sbs_plate",
        base_class="microplate",
        wells=96, rows=8, cols=12,
        spacing_x_mm=9.0, spacing_y_mm=9.0,
        height_mm=14.4, stack_height_mm=8.6, gripper_offset_mm=0.5,
    )))

    if tips_on:
        bravo._tips_on_head = True
        bravo._tip_labware_name = "96 V11 LT250 Tip Box 19477.002"
        bravo._tip_definition_id = tip_id
        bravo._attached_tip_length_mm = 55.5
        bravo._tips_on_head_mode = bravo._head_mode
        bravo._tips_on_head_selection = None
    return bravo


def test_a_bare_head_is_not_warned_about_a_tip_it_is_not_wearing() -> None:
    """The trap this whole gate exists to avoid.

    Both assertions matter together: the taught-tip fallback really does name
    the LT250 tip on a bare head, and the limit must still come back None --
    because bare probes are the vendor's own answer to needing more than 140 uL.
    """
    bravo = _assaymap_bravo(tips_on=False)
    assert bravo.active_tip_id() == TEACH_TIP
    assert bravo.mounted_tip_overflow_ul() is None


def test_mounted_lt250_tips_report_the_limit() -> None:
    assert _assaymap_bravo().mounted_tip_overflow_ul() == pytest.approx(LIMIT_UL)


def test_mounted_cartridges_report_no_limit() -> None:
    assert _assaymap_bravo(tip_id=CARTRIDGE).mounted_tip_overflow_ul() is None


# --------------------------------------------------------- end to end

async def _wait_for_prompt(bravo: Bravo, timeout: float = 5.0) -> dict:
    engine = bravo._engine
    for _ in range(int(timeout / 0.01)):
        if engine._awaiting_error_action:
            payload = engine._current_task.status_payload()
            prompt = payload.get("operator_prompt")
            assert prompt is not None, f"task failed without a prompt: {payload}"
            return prompt
        await asyncio.sleep(0.01)
    raise AssertionError("no operator prompt appeared")


@pytest.mark.asyncio
async def test_the_warning_arrives_before_the_head_moves() -> None:
    """The point of making this the first step rather than a check at the well."""
    bravo = _assaymap_bravo()
    running = asyncio.create_task(bravo.aspirate(PLATE_LOCATION, volume=200.0))
    prompt = await _wait_for_prompt(bravo)

    assert prompt["kind"] == "tip_overflow"
    assert bravo.controller.move_calls == [], (
        "the head moved before the operator was asked"
    )

    bravo._engine.abort()
    await running


@pytest.mark.asyncio
async def test_abort_leaves_the_head_where_it_was() -> None:
    bravo = _assaymap_bravo()
    running = asyncio.create_task(bravo.aspirate(PLATE_LOCATION, volume=200.0))
    await _wait_for_prompt(bravo)
    bravo._engine.abort()
    await running
    assert bravo.controller.move_calls == []


@pytest.mark.asyncio
async def test_ignore_goes_ahead_and_aspirates() -> None:
    """Warning, not a block: the operator can say "I meant that"."""
    bravo = _assaymap_bravo()
    running = asyncio.create_task(bravo.aspirate(PLATE_LOCATION, volume=200.0))
    await _wait_for_prompt(bravo)
    bravo._engine.ignore()
    await running
    assert bravo.controller.move_calls, "ignore should have let the aspirate run"


@pytest.mark.asyncio
async def test_a_volume_under_the_limit_is_never_interrupted() -> None:
    bravo = _assaymap_bravo()
    await asyncio.wait_for(bravo.aspirate(PLATE_LOCATION, volume=100.0), timeout=5.0)
    assert bravo.controller.move_calls


@pytest.mark.asyncio
async def test_bare_probes_may_draw_the_full_stroke_uninterrupted() -> None:
    """250 uL bare is a supported operation, and must not be second-guessed."""
    bravo = _assaymap_bravo(tips_on=False)
    await asyncio.wait_for(bravo.aspirate(PLATE_LOCATION, volume=250.0), timeout=5.0)
    assert bravo.controller.move_calls


@pytest.mark.asyncio
async def test_mixing_is_checked_too() -> None:
    """Mix aspirates on every cycle, so it has exactly the same exposure."""
    bravo = _assaymap_bravo()
    running = asyncio.create_task(bravo.mix(PLATE_LOCATION, volume=200.0, mix_cycles=2))
    prompt = await _wait_for_prompt(bravo)
    assert prompt["kind"] == "tip_overflow"
    assert bravo.controller.move_calls == []
    bravo._engine.abort()
    await running


@pytest.mark.asyncio
async def test_a_post_aspirate_gap_pushes_the_liquid_but_a_pre_aspirate_gap_does_not() -> None:
    """Which volume gets compared, pinned.

    A pre-aspirate gap is drawn first, so it rides above the liquid and cannot
    push it toward the syringe. A post-aspirate gap is drawn afterwards, below
    the liquid, and does. Comparing raw plunger travel instead would warn on the
    first case -- a false alarm on the standard air-gap technique.
    """
    quiet = _assaymap_bravo()
    await asyncio.wait_for(
        quiet.aspirate(PLATE_LOCATION, volume=135.0, pre_aspirate=20.0),
        timeout=5.0,
    )

    loud = _assaymap_bravo()
    running = asyncio.create_task(
        loud.aspirate(PLATE_LOCATION, volume=135.0, post_aspirate=20.0)
    )
    prompt = await _wait_for_prompt(loud)
    assert prompt["volume_ul"] == pytest.approx(155.0)
    loud._engine.abort()
    await running
