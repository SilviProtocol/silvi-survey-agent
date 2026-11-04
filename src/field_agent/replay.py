"""Replay a phone link journal as a timed wire feed.

The journal (`field-agent.jsonl`) records what the phone queued and
sent, including wire v2 messages it held back while a link was still
v1 (they sit there with seq -1). Replaying gives every phone-to-agent
data message a fresh seq in journal order, so a journal drives the
agent exactly as the live link would have with the gate open.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

from field_agent.contracts import validate_wire

_PHONE_KINDS = {
    "pose",
    "plant_sighting",
    "plant_snapshot",
    "coverage",
    "transcript",
    "clip_ref",
    "anchor",
    "mode",
    "evidence_photo",
    "plant_view",
}


def iter_journal_replay(journal_path: Path) -> Iterator[dict]:
    lines = [
        json.loads(line)
        for line in Path(journal_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    start = next((d for d in lines if d.get("type") == "link_start"), {})
    session_id = str(start.get("session_id") or "journal-replay")
    seq = 0

    def emit(kind: str, t_mono_ms: float, payload: dict) -> dict:
        nonlocal seq
        msg = {
            "v": "1",
            "kind": kind,
            "seq": seq,
            "session_id": session_id,
            "t_mono_ms": float(t_mono_ms),
            "t_wall_ms": None,
            "payload": payload,
        }
        seq += 1
        validate_wire(msg)
        return msg

    yield emit(
        "session_start",
        float(start.get("t_mono_ms") or 0.0),
        {
            "walk_id": str(start.get("walk_id") or "journal-walk"),
            "frame": "site_local",
            "device": start.get("device"),
            "app_commit": start.get("app_commit"),
        },
    )
    last_mono = 0.0
    for d in lines:
        last_mono = float(d.get("t_mono_ms") or last_mono)
        kind = d.get("kind")
        if kind in _PHONE_KINDS and isinstance(d.get("payload"), dict):
            yield emit(str(kind), last_mono, dict(d["payload"]))
        elif d.get("type") == "transcript" and str(d.get("text") or "").strip():
            yield emit(
                "transcript",
                last_mono,
                {"text": str(d["text"]), "is_final": True, "speaker": "walker"},
            )
    yield emit("session_end", last_mono, {"reason": "journal_replay_complete"})


def run_replay(messages: Iterator[dict], speed: float, consume) -> dict:
    """Drive messages into consume(msg), pacing at speed x real time (0 = max)."""
    prev_mono: float | None = None
    counts: dict[str, int] = {}
    total = 0
    for msg in messages:
        if speed and speed > 0 and prev_mono is not None:
            delay = (msg["t_mono_ms"] - prev_mono) / 1000.0 / speed
            if delay > 0:
                time.sleep(delay)
        prev_mono = msg["t_mono_ms"]
        consume(msg)
        counts[msg["kind"]] = counts.get(msg["kind"], 0) + 1
        total += 1
    return {"total": total, "by_kind": counts}
