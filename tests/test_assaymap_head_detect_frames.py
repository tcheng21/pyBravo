"""Golden test: the smart-head detection exchange pybravo emits must match what
the vendor software sends.

Fixture: tests/fixtures/assaymap_head_detect.json — the exchange for an
AssayMAP 96AM head, which identifies itself with type byte 14.

The existing tests in tests/darwin/test_head_detect.py cover the *behaviour* of
detection. This covers the *wire*: right subcommands, right order, right EEPROM
address encoding. Getting the address wrong would read some other byte of the
head's EEPROM and silently identify the wrong head.

The msg_id nibble of packet byte 2 is a rotating correlation counter, so the
comparison is on the semantic fields rather than raw bytes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pybravo.darwin.controller import DarwinController
from pybravo.protocol.gemini.engine import GeminiEngine
from pybravo.protocol.gemini.enums import CommandTypes, DarwinMasterNodeSubCommands
from pybravo.types import HeadType
from tests.fakes.gemini_fake import FakeGeminiServer

FIXTURE = Path(__file__).parent / "fixtures" / "assaymap_head_detect.json"
DETECTION_SUBCOMMANDS = {
    int(DarwinMasterNodeSubCommands.SMART_INIT),
    int(DarwinMasterNodeSubCommands.SMART_RD_EEPROM),
    int(DarwinMasterNodeSubCommands.SMART_RD_EEPROM_VAL),
}


@pytest.fixture(scope="module")
def golden():
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def fake():
    s = FakeGeminiServer()
    s.start()
    try:
        yield s
    finally:
        s.stop()


@pytest.fixture
def controller(fake, golden):
    # Seed the fake with the byte the real AssayMAP head reported.
    fake.storage[
        (1, 0, DarwinMasterNodeSubCommands.SMART_RD_EEPROM_VAL)
    ] = golden["head_type_byte"]
    engine = GeminiEngine("127.0.0.1", port=fake.port)
    ctrl = DarwinController(engine=engine)
    ctrl.open_tcp("127.0.0.1")
    try:
        yield ctrl, fake
    finally:
        ctrl.close()


def _detection_packets(fake):
    return [p for p in fake.received_packets if p.sub_command in DETECTION_SUBCOMMANDS]


def test_detection_emits_the_captured_exchange(controller, golden):
    """Same subcommands, same order, same values as VWorks sent."""
    ctrl, fake = controller
    assert ctrl.detect_head_type() == HeadType.HT_96_ASSAYMAP

    sent = _detection_packets(fake)
    expected = golden["sent"]
    assert len(sent) == len(expected), (
        f"emitted {len(sent)} detection packets, capture had {len(expected)}"
    )
    for got, exp in zip(sent, expected):
        assert got.src.node_id == exp["src_node"]
        assert got.src.dev_id == exp["src_dev"]
        assert got.dest.node_id == exp["dest_node"]
        assert got.dest.dev_id == exp["dest_dev"]
        assert CommandTypes(got.cmd_type).name == exp["cmd_type"]
        assert got.sub_command == exp["sub_command_value"]
        assert got.cmd_val == exp["cmd_val"]


def test_eeprom_address_is_one_byte_at_offset_one(controller, golden):
    """The read is encoded (offset << 8) | length, i.e. 0x0101.

    Pinned separately because it is the single value that decides *which* byte of
    the head's EEPROM is read. A wrong address still returns a byte, so the
    failure would be a confidently wrong head identification.
    """
    ctrl, fake = controller
    ctrl.detect_head_type()

    reads = [
        p for p in fake.received_packets
        if p.sub_command == int(DarwinMasterNodeSubCommands.SMART_RD_EEPROM)
    ]
    assert reads, "no SMART_RD_EEPROM packet was sent"
    assert reads[0].cmd_val == 0x0101
    # and it agrees with the capture
    captured = next(
        s for s in golden["sent"] if s["sub_command"] == "SMART_RD_EEPROM"
    )
    assert reads[0].cmd_val == captured["cmd_val"]


def test_detection_order_matches_the_instrument(controller, golden):
    """SMART_INIT, then the EEPROM address, then the value read."""
    ctrl, fake = controller
    ctrl.detect_head_type()
    order = [
        DarwinMasterNodeSubCommands(p.sub_command).name
        for p in _detection_packets(fake)
    ]
    assert order == [s["sub_command"] for s in golden["sent"]]
    assert order == ["SMART_INIT", "SMART_RD_EEPROM", "SMART_RD_EEPROM_VAL"]


def test_adc_path_is_not_used(controller):
    """VWorks never reads STUPID_HEAD_COUNTS when a smart head answers, and
    neither should we -- the EEPROM is authoritative."""
    ctrl, fake = controller
    ctrl.detect_head_type()
    adc = [
        p for p in fake.received_packets
        if p.sub_command == int(DarwinMasterNodeSubCommands.STUPID_HEAD_COUNTS)
    ]
    assert adc == [], "detection should not fall back to the ADC head-count read"
