"""Recorded model responses for no-key runs and tests.

These are canned transcripts of the shape a real model returns (text and/or
tool calls), selected by matching the walker's turn. They are a test double,
not model output, and are labelled as the ``mock`` model everywhere they run.
``__POSE__`` inside recorded tool arguments is replaced with the current
phone position parsed from the agent's own context summary, mirroring what
the system prompt instructs a live model to do.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from field_agent.models import ChatResult

DEFAULT_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "recorded-model-responses.json"
_POSE_RE = re.compile(r"phone position \(site-local m\) \[([-\d.]+), ([-\d.]+), ([-\d.]+)\]")


def _pose_from(messages: list[dict]) -> list[float]:
    for message in messages:
        match = _POSE_RE.search(str(message.get("content") or ""))
        if match:
            return [float(match.group(1)), float(match.group(2)), float(match.group(3))]
    return [0.0, 0.0, 0.0]


_TARGET_RE = re.compile(r"(?:tree|plant)[\s-]?(\d+)", re.IGNORECASE)


def _substitute(value, pose: list[float], target: str = "tree-1"):
    if value == "__TARGET__":
        return target
    if value == "__POSE__":
        return pose
    if value == "__POSE_TRI__":
        # A small triangle around the current position, for polygon fixtures.
        x, y, z = pose
        return [[x, y, z], [x + 2.0, y, z], [x, y, z - 2.0]]
    if isinstance(value, dict):
        return {k: _substitute(v, pose, target) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, pose, target) for v in value]
    return value


def recorded_handler(path: Path | None = None):
    fixture = json.loads((Path(path) if path else DEFAULT_FIXTURE).read_text(encoding="utf-8"))
    entries = fixture["responses"]
    fallback = fixture["fallback"]

    def handler(messages: list[dict], tools=None) -> ChatResult:
        user_texts = [str(m.get("content") or "") for m in messages if m.get("role") == "user"]
        if user_texts and user_texts[-1].startswith("Tool: "):
            raise RuntimeError("recorded handler does not repair tool calls")
        turn = user_texts[-1].lower() if user_texts else ""
        pose = _pose_from(messages)
        for entry in entries:
            if entry["match"].lower() in turn:
                selected = entry
                break
        else:
            selected = fallback
        match = _TARGET_RE.search(turn)
        target = f"tree-{match.group(1)}" if match else "tree-1"
        return ChatResult(
            text=selected.get("text", ""),
            tool_calls=_substitute(copy.deepcopy(selected.get("tool_calls", [])), pose, target),
            input_tokens=0,
            output_tokens=0,
            usage_estimated=True,
            model_id="recorded-fixture",
        )

    return handler
