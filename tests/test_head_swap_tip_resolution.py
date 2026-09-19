"""Swapping heads must re-resolve the tip, because tip length moves the Z origin.

A Bravo's head is physically swapped, so one install has to keep working across
all head types. Tip geometry is head-specific and feeds:

    deck_surface_Z = teachpoint_Z + teach_tip_length_mm

Carrying the previous head's tip across a swap therefore displaces every
subsequent Z by the difference in tip length. Between an AssayMAP teach tip
(55.5 mm) and a 384ST tip (26.1 mm) that is 29.4 mm — a crash, not a rounding
error.
"""
from __future__ import annotations

import pytest

from pybravo.profile.profile import BravoProfile
from pybravo.tips import get_default_tip_id_for_head, get_tip_length_mm
from pybravo.types import HeadType

SWAP_SEQUENCE = [
    HeadType.HT_96_ASSAYMAP,
    HeadType.HT_96_D_200,
    HeadType.HT_384_D_70,
    HeadType.HT_96_D_70,
    HeadType.HT_96_ASSAYMAP,
]


@pytest.mark.parametrize("head_type", SWAP_SEQUENCE)
def test_resolve_tips_matches_the_head(head_type):
    profile = BravoProfile.default()
    profile.head.head_type = head_type
    profile.resolve_tips_for_head()

    expected_tip = get_default_tip_id_for_head(head_type)
    assert profile.head.teach_tip_id == expected_tip
    assert profile.head.default_tip_id == expected_tip
    assert profile.head.teach_tip_length_mm == get_tip_length_mm(head_type, expected_tip)


def test_no_tip_survives_a_head_swap():
    """Walk the whole swap sequence; the tip must change with the head."""
    profile = BravoProfile.default()
    seen = []
    for head_type in SWAP_SEQUENCE:
        profile.head.head_type = head_type
        profile.resolve_tips_for_head()
        seen.append((head_type, profile.head.teach_tip_id, profile.head.teach_tip_length_mm))

    for head_type, tip_id, length in seen:
        assert tip_id == get_default_tip_id_for_head(head_type), (
            f"{head_type.name} kept a tip belonging to another head: {tip_id}"
        )
        assert length == get_tip_length_mm(head_type, tip_id)

    # Round trip: back on AssayMAP we must be back on the AssayMAP teach tip,
    # not whatever the intervening heads left behind.
    assert seen[0][1:] == seen[-1][1:]


def test_assaymap_teach_tip_is_the_lt250_not_the_cartridge():
    """The head is taught with a 250 µL LT tip and operated with cartridges.

    The instrument profile records "Default tip = 250". Resolving to the 60 µL
    cartridge instead would put teach_tip_length_mm at 29.2 rather than 55.5 —
    26.3 mm of Z error.
    """
    tip_id = get_default_tip_id_for_head(HeadType.HT_96_ASSAYMAP)
    assert tip_id == "am_lt250_teach"
    assert get_tip_length_mm(HeadType.HT_96_ASSAYMAP, tip_id) == 55.5


def test_assaymap_to_384_does_not_keep_the_long_teach_tip():
    """The specific 29.4 mm regression this file exists for."""
    profile = BravoProfile.default()
    profile.head.head_type = HeadType.HT_96_ASSAYMAP
    profile.resolve_tips_for_head()
    assert profile.head.teach_tip_length_mm == 55.5

    profile.head.head_type = HeadType.HT_384_D_70
    profile.resolve_tips_for_head()
    assert profile.head.teach_tip_length_mm == 26.1, (
        "384ST head kept the AssayMAP teach tip length — every Z would be "
        "29.4 mm too deep"
    )
