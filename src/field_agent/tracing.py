"""Latency tracing for agent turns.

Turns aim to feel immediate.
5 s (p50),
measured per model. A turn trace therefore records, on one monotonic clock:
when speech ended, when the model call started, when the first reply (text or
first accepted tool call) arrived, and when the turn finished.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field


@dataclass
class TurnTrace:
    turn_id: str
    model: str
    t_speech_end: float
    t_model_start: float | None = None
    t_first_reply: float | None = None
    t_done: float | None = None
    notes: dict = field(default_factory=dict)

    @property
    def speech_end_to_reply_ms(self) -> float | None:
        if self.t_first_reply is None:
            return None
        return (self.t_first_reply - self.t_speech_end) * 1000.0

    @property
    def total_ms(self) -> float | None:
        if self.t_done is None:
            return None
        return (self.t_done - self.t_speech_end) * 1000.0

    def as_dict(self) -> dict:
        out = {
            "turn_id": self.turn_id,
            "model": self.model,
            "speech_end_to_reply_ms": (
                round(self.speech_end_to_reply_ms, 1)
                if self.speech_end_to_reply_ms is not None
                else None
            ),
            "total_ms": round(self.total_ms, 1) if self.total_ms is not None else None,
        }
        if self.notes:
            out["notes"] = self.notes
        return out


class LatencyTracer:
    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self.traces: list[TurnTrace] = []
        # Local tool work inside a turn (species naming, outlining),
        # timed on the machine that ran it: (kind, ms) in call order.
        self.tool_samples: list[tuple[str, float]] = []

    def record_tool(self, kind: str, latency_ms: float) -> None:
        self.tool_samples.append((kind, float(latency_ms)))

    def now(self) -> float:
        return self._clock()

    def start_turn(self, turn_id: str, model: str, *, at: float | None = None) -> TurnTrace:
        trace = TurnTrace(turn_id=turn_id, model=model, t_speech_end=self.now() if at is None else at)
        self.traces.append(trace)
        return trace

    def summary(self) -> dict[str, dict]:
        """Per-model turn latency, plus local tool latency when any ran."""
        by_model: dict[str, list[float]] = {}
        for trace in self.traces:
            value = trace.speech_end_to_reply_ms
            if value is not None:
                by_model.setdefault(trace.model, []).append(value)
        out = {
            model: {
                "turns": len(values),
                "p50_ms": round(statistics.median(values), 1),
                "max_ms": round(max(values), 1),
                "under_1500ms_p50": statistics.median(values) < 1500.0,
            }
            for model, values in sorted(by_model.items())
        }
        if self.tool_samples:
            by_kind: dict[str, list[float]] = {}
            for kind, ms in self.tool_samples:
                by_kind.setdefault(kind, []).append(ms)
            out["tools"] = {
                kind: {
                    "calls": len(values),
                    "p50_ms": round(statistics.median(values), 1),
                    "max_ms": round(max(values), 1),
                }
                for kind, values in sorted(by_kind.items())
            }
        return out
