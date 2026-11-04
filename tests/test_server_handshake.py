"""The wire v2 handshake through the transport-free server core."""

from __future__ import annotations

import json

from field_agent.agent import FieldAgent
from field_agent.models import ChatResult
from field_agent.server import ServerCore

SID = "handshake-test"


class _Chat:
    def complete(self, messages, tools=None, on_first_token=None, deadline_s=None):
        return ChatResult(text="OK.", tool_calls=[])


def _line(kind, seq, payload):
    return json.dumps({"v": "1", "kind": kind, "seq": seq, "session_id": SID,
                       "t_mono_ms": 1.0, "t_wall_ms": None, "payload": payload})


def _core(**kw):
    return ServerCore(lambda: FieldAgent(_Chat(), model_label="m"), **kw)


def test_capabilities_opens_the_v2_gate_without_spending_a_seq():
    core = _core()
    conn = core.new_connection()
    replies, close = conn.feed_line(
        _line("session_start", 0, {"walk_id": "w", "frame": "site_local"}))
    assert close is None
    assert [r["kind"] for r in replies] == ["capabilities"]
    assert replies[0]["payload"]["wire"] == 2
    assert core.sessions[SID].next_agent_seq == 0
    replies, close = conn.feed_line(
        _line("anchor", 1, {"anchor_id": "a1", "action": "placed", "position_m": [0, 0, 0]}))
    assert close is None
    assert core.sessions[SID].agent.world.anchors


def test_v2_kind_before_capabilities_is_rejected():
    core = _core(wire_v2=False)
    conn = core.new_connection()
    conn.feed_line(_line("session_start", 0, {"walk_id": "w", "frame": "site_local"}))
    replies, close = conn.feed_line(
        _line("anchor", 1, {"anchor_id": "a1", "action": "placed", "position_m": [0, 0, 0]}))
    assert close is not None
    assert replies[0]["payload"]["code"] == "wire_v2_not_open"
