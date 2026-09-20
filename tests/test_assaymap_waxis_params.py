"""Golden test: the W-axis parameter table pybravo writes for an AssayMAP head
must match, word for word, what the vendor software wrote to the real instrument.

Fixture: tests/fixtures/assaymap_waxis_params.json, extracted from a capture of
the vendor software initializing this machine. These 57 values are the plunger's PID and
motion tuning -- if they drift, the syringe's force and settling behaviour change
on real hardware, and that is not something a unit test elsewhere would catch.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from pybravo.darwin.waxis_params import (
    WAXIS_PARAM_TABLE,
    WAxisParamSet,
    apply_waxis_parameters,
    param_set_for_head,
)
from pybravo.protocol.gemini.enums import ParamDBs
from pybravo.types import HeadType

FIXTURE = Path(__file__).parent / "fixtures" / "assaymap_waxis_params.json"
AM = HeadType.HT_96_ASSAYMAP


class RecordingParameterAccess:
    """Captures writes as (param_id, raw_uint32) exactly as they'd hit the wire."""

    def __init__(self) -> None:
        self.writes: list[tuple[int, int]] = []
        self.applied = 0

    def write_uint(self, param_id, value, timeout_ms=5000):  # noqa: ARG002
        self.writes.append((int(param_id), int(value) & 0xFFFFFFFF))

    def write_float(self, param_id, value, timeout_ms=5000):  # noqa: ARG002
        raw = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        self.writes.append((int(param_id), raw))

    def apply(self, timeout_ms=10_000):  # noqa: ARG002
        self.applied += 1


@pytest.fixture(scope="module")
def expected():
    data = json.loads(FIXTURE.read_text())
    return [(w["param"], int(w["raw"], 16)) for w in data["writes"]]


def test_assaymap_resolves_to_the_am_parameter_set():
    assert param_set_for_head(AM) is WAxisParamSet.AM


def test_waxis_writes_match_the_captured_frames(expected):
    """Every parameter, in order, with the same raw word the vendor software sent."""
    rec = RecordingParameterAccess()
    assert apply_waxis_parameters(rec, AM) is True

    assert len(rec.writes) == len(expected), (
        f"wrote {len(rec.writes)} parameters, capture had {len(expected)}"
    )
    names = {int(p): p.name for p in ParamDBs}
    mismatches = [
        f"{names.get(got_id, got_id)} (id {got_id}): "
        f"ours 0x{got_raw:08X} vs measured 0x{exp_raw:08X}"
        for (got_id, got_raw), (exp_id, exp_raw) in zip(rec.writes, expected)
        if (got_id, got_raw) != (exp_id, exp_raw)
    ]
    assert not mismatches, "W parameter mismatch:\n  " + "\n  ".join(mismatches)


def test_parameters_are_committed_once(expected):
    """the vendor software follows each table with a single PARAM_DB_APPLY."""
    rec = RecordingParameterAccess()
    apply_waxis_parameters(rec, AM)
    assert rec.applied == 1


def test_i2t_time_is_written_as_an_unsigned_int(expected):
    """I2T_TIME is declared UInt32 by the firmware; float-encoding it lands at a
    huge magnitude and the firmware NAKs with OUT_OF_RANGE."""
    rec = RecordingParameterAccess()
    apply_waxis_parameters(rec, AM)
    raw = dict(rec.writes)[int(ParamDBs.I2T_TIME)]
    assert raw == 2000, f"I2T_TIME should be the plain integer 2000, got 0x{raw:08X}"


def test_other_heads_do_not_get_the_am_table():
    """Regression: branching by head type must not leak AssayMAP tuning."""
    am = RecordingParameterAccess()
    apply_waxis_parameters(am, AM)
    for other in (HeadType.HT_96_D_70, HeadType.HT_384_D_70, HeadType.HT_96_D_200):
        rec = RecordingParameterAccess()
        apply_waxis_parameters(rec, other)
        assert rec.writes != am.writes, f"{other.name} received the AssayMAP table"


def test_every_table_entry_is_covered_by_the_capture(expected):
    """The capture should exercise the whole table -- no entry silently untested."""
    assert {p for p, _ in expected} == {int(e.param) for e in WAXIS_PARAM_TABLE}
