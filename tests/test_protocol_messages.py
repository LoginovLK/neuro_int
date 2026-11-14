import pytest
from neuroint_protocol import CommandMsg, HelloMsg, StatusMsg, parse_message
from pydantic import ValidationError


def test_hello_roundtrip() -> None:
    raw = HelloMsg(client="edge-test", capture_fps=60).model_dump_json()
    msg = parse_message(raw)
    assert isinstance(msg, HelloMsg)
    assert msg.client == "edge-test"


def test_command_requires_action() -> None:
    with pytest.raises(ValidationError):
        CommandMsg(keys=["alt"])


def test_unknown_type_rejected() -> None:
    with pytest.raises(ValueError, match="unknown message type"):
        parse_message('{"type": "nope"}')


def test_non_object_rejected() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        parse_message("[1,2,3]")


def test_status_defaults() -> None:
    msg = parse_message('{"type": "status"}')
    assert isinstance(msg, StatusMsg)
    assert msg.paused is False
