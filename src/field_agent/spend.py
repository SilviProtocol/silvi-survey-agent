"""Spend guard: append-only usage log, per-provider totals, credit cap.

Every model and transcription call records one line in ``data/spend.ndjson``
(git-ignored). Calls are refused once estimated spend for a provider reaches
the configured fraction of its credit (80% in config/providers.json). Cost
estimates use the dated prices in that config; provider invoices win.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from field_agent.providers import ProviderRegistry


class SpendCapExceeded(RuntimeError):
    """The provider is at or over its configured spend fraction."""


class SpendGuard:
    def __init__(self, registry: ProviderRegistry, log_path: str | Path | None = None):
        self._registry = registry
        self._path = registry.spend_log_path(log_path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def records(self) -> list[dict]:
        if not self._path.exists():
            return []
        out: list[dict] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def total_usd(self, provider: str) -> float:
        return sum(
            float(r.get("est_cost_usd", 0.0)) for r in self.records() if r.get("provider") == provider
        )

    def totals(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for name in self._registry.provider_names():
            info = self._registry.provider(name)
            spent = self.total_usd(name)
            fraction = spent / info.credit_usd if info.credit_usd else 0.0
            result[name] = {
                "spent_usd": round(spent, 6),
                "credit_usd": info.credit_usd,
                "fraction_used": round(fraction, 6),
                "blocked": fraction >= self._registry.guard_fraction(),
            }
        return result

    def check(self, provider: str) -> None:
        info = self._registry.provider(provider)
        spent = self.total_usd(provider)
        limit = self._registry.guard_fraction() * info.credit_usd
        if spent >= limit:
            raise SpendCapExceeded(
                f"{info.display_name}: estimated spend ${spent:.4f} is at or over "
                f"{self._registry.guard_fraction():.0%} of its ${info.credit_usd:.2f} credit; "
                "refusing further calls"
            )

    def record(
        self,
        *,
        provider: str,
        model_id: str,
        kind: str,
        est_cost_usd: float,
        input_tokens: int = 0,
        cached_input_tokens: int = 0,
        output_tokens: int = 0,
        audio_seconds: float = 0.0,
        latency_ms: float | None = None,
        note: str | None = None,
    ) -> dict:
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "provider": provider,
            "model": model_id,
            "kind": kind,
            "input_tokens": int(input_tokens),
            "cached_input_tokens": int(cached_input_tokens),
            "output_tokens": int(output_tokens),
            "audio_seconds": float(audio_seconds),
            "est_cost_usd": round(float(est_cost_usd), 8),
            "latency_ms": latency_ms,
            "note": note,
        }
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
        return entry

    def record_chat(
        self,
        *,
        selector: str,
        input_tokens: int,
        cached_input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float | None = None,
    ) -> dict:
        model = self._registry.model(selector)
        cost = self._registry.estimate_cost(
            model,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
        )
        return self.record(
            provider=model.provider,
            model_id=model.model_id,
            kind="chat",
            est_cost_usd=cost,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

    def record_stt(self, *, audio_seconds: float, latency_ms: float | None = None) -> dict:
        model_id = self._registry.stt()["model_id"]
        return self.record(
            provider="meta",
            model_id=model_id,
            kind="stt",
            est_cost_usd=self._registry.estimate_stt_cost(audio_seconds),
            audio_seconds=audio_seconds,
            latency_ms=latency_ms,
        )
