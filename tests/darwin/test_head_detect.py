"""Tests for head-type auto-detection via smart-head EEPROM."""

from __future__ import annotations

import pytest

from pybravo.darwin.controller import DarwinController
from pybravo.protocol.gemini.engine import GeminiEngine
from pybravo.protocol.gemini.enums import (
    CommandNAKTypes,
    CommandTypes,
    DarwinMasterNodeSubCommands,
)
from pybravo.protocol.gemini.packet import InstructionAddress
from pybravo.types import HeadType
from tests.fakes.gemini_fake import FakeGeminiServer


@pytest.fixture
def fake():
    s = FakeGeminiServer()
    s.start()
    try:
        yield s
    finally:
        s.stop()


@pytest.fixture
def controller(fake):
    engine = GeminiEngine("127.0.0.1", port=fake.port)
    ctrl = DarwinController(engine=engine)
    ctrl.open_tcp("127.0.0.1")
    try:
        yield ctrl, fake
    finally:
        ctrl.close()


# --- detect_smart_head --------------------------------------------------------


def test_detect_smart_head_true_when_init_succeeds(controller):
    ctrl, _ = controller
    # Fake's default set_handler returns SETCMD_RESP (success) → smart head present
    assert ctrl.detect_smart_head() is True


def test_detect_smart_head_false_when_nak_unsuccessful(controller):
    ctrl, fake = controller
    master = InstructionAddress(1, 0)
    fake.seed_nak(
        master,
        DarwinMasterNodeSubCommands.SMART_INIT,
        CommandNAKTypes.UNSUCCESSFUL_OPERATION,
    )
    assert ctrl.detect_smart_head() is False


def test_detect_smart_head_reraises_other_naks(controller):
    ctrl, fake = controller
    master = InstructionAddress(1, 0)
    fake.seed_nak(
        master,
        DarwinMasterNodeSubCommands.SMART_INIT,
        CommandNAKTypes.INVALID_SUBCMD,
    )
    from pybravo.protocol.gemini.errors import NAKError
    with pytest.raises(NAKError):
        ctrl.detect_smart_head()


# --- read_smart_head_type -----------------------------------------------------


def test_read_smart_head_type_sends_eeprom_read_and_returns_byte(controller):
    ctrl, fake = controller
    master = InstructionAddress(1, 0)

    # Seed the GET response for SMART_RD_EEPROM_VAL
    fake.storage[(1, 0, DarwinMasterNodeSubCommands.SMART_RD_EEPROM_VAL)] = 7

    head_byte = ctrl.read_smart_head_type()
    assert head_byte == 7

    # Verify SMART_RD_EEPROM was issued with (offset=1 << 8) | length=1 = 0x0101
    rd_eeprom = [
        p for p in fake.received_packets
        if p.dest == master
        and p.sub_command == DarwinMasterNodeSubCommands.SMART_RD_EEPROM
        and p.cmd_type == CommandTypes.SETCMD
    ]
    assert rd_eeprom and rd_eeprom[-1].cmd_val == 0x0101


# --- detect_head_type (vendor byte → HeadType, verified entries only) --------


def test_detect_head_type_maps_confirmed_vendor_bytes(controller):
    """Vendor head-type bytes confirmed against real hardware translate."""
    ctrl, fake = controller
    for vendor_byte, expected in (
        (1, HeadType.HT_384_D_70),      # 384ST 70µL Series III, observed on bench
        (3, HeadType.HT_96_D_200),      # 96LT profile export
        (14, HeadType.HT_96_ASSAYMAP),  # AssayMAP head and profile
    ):
        fake.storage[(1, 0, DarwinMasterNodeSubCommands.SMART_RD_EEPROM_VAL)] = vendor_byte
        assert ctrl.detect_head_type() == expected


def test_detect_head_type_never_guesses_unknown_bytes(controller):
    """An unconfirmed byte must be HT_UNKNOWN, never HeadType(byte).

    The two numbering schemes are unrelated, so coercing the byte would be
    confidently wrong — vendor 1 is a 384ST, but HeadType(1) is HT_8_F_50.
    """
    ctrl, fake = controller
    fake.storage[(1, 0, DarwinMasterNodeSubCommands.SMART_RD_EEPROM_VAL)] = 99
    assert ctrl.detect_head_type() == HeadType.HT_UNKNOWN


def test_detect_head_type_unknown_without_smart_head(controller):
    """No smart head attached → HT_UNKNOWN rather than a stale/garbage byte."""
    ctrl, fake = controller
    fake.seed_nak(
        InstructionAddress(1, 0),
        DarwinMasterNodeSubCommands.SMART_INIT,
        CommandNAKTypes.UNSUCCESSFUL_OPERATION,
    )
    assert ctrl.detect_head_type() == HeadType.HT_UNKNOWN


# --- read_head_identification (raw data) --------------------------------------


def test_read_head_identification_smart_head_present(controller):
    ctrl, fake = controller
    fake.storage[(1, 0, DarwinMasterNodeSubCommands.SMART_RD_EEPROM_VAL)] = 1
    fake.storage[(1, 0, DarwinMasterNodeSubCommands.STUPID_HEAD_COUNTS)] = 1803

    ident = ctrl.read_head_identification()
    assert ident["has_smart_head"] is True
    assert ident["eeprom_byte"] == 1
    assert ident["adc_counts"] == 1803


def test_read_head_identification_no_smart_head(controller):
    ctrl, fake = controller
    master = InstructionAddress(1, 0)
    fake.seed_nak(
        master,
        DarwinMasterNodeSubCommands.SMART_INIT,
        CommandNAKTypes.UNSUCCESSFUL_OPERATION,
    )
    fake.storage[(1, 0, DarwinMasterNodeSubCommands.STUPID_HEAD_COUNTS)] = 4095

    ident = ctrl.read_head_identification()
    assert ident["has_smart_head"] is False
    assert ident["eeprom_byte"] is None  # skipped — no smart head to read from
    assert ident["adc_counts"] == 4095
