"""Provider registry: config, credentials (presence only), model lookup.

Keys come only from the process environment or a local ``.env`` file.
No key value is ever printed, logged, or written to
any file by this module. Models are sent text and transcripts only; nothing
here loads or transmits imagery.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "providers.json"


def _load_dotenv(path: Path) -> dict[str, str]:
    """Parse a minimal KEY=value .env file. Values stay in memory only."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


@dataclass(frozen=True)
class ModelInfo:
    selector: str
    provider: str
    model_id: str
    role: str
    input_per_million: float
    cached_input_per_million: float | None
    output_per_million: float
    price_source: str
    price_verified: str
    tool_choice: str | None = None
    reasoning_effort: str | None = None
    unavailable: str | None = None
    fallback: str | None = None


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    display_name: str
    env_var: str
    credit_usd: float
    base_url: str
    key_present: bool


class ProviderRegistry:
    """Reads config/providers.json and answers key/model/price questions."""

    def __init__(
        self,
        config: dict,
        *,
        environ: dict[str, str] | None = None,
        dotenv: Path | None = None,
    ):
        self._config = config
        env = dict(os.environ if environ is None else environ)
        dotenv_path = REPO_ROOT / ".env" if dotenv is None else dotenv
        for key, value in _load_dotenv(dotenv_path).items():
            env.setdefault(key, value)
        self._env = env

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        *,
        environ: dict[str, str] | None = None,
        dotenv: Path | None = None,
    ) -> ProviderRegistry:
        config_path = Path(path) if path is not None else DEFAULT_CONFIG
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(config, environ=environ, dotenv=dotenv)

    @property
    def config(self) -> dict:
        return self._config

    def provider_names(self) -> list[str]:
        return list(self._config["providers"].keys())

    def _key(self, provider: str) -> str | None:
        env_var = self._config["providers"][provider]["env_var"]
        value = self._env.get(env_var, "").strip()
        return value or None

    def key_present(self, provider: str) -> bool:
        return self._key(provider) is not None

    def require_key(self, provider: str) -> str:
        key = self._key(provider)
        if key is None:
            env_var = self._config["providers"][provider]["env_var"]
            raise MissingKeyError(f"{env_var} is not set (add it to .env; never commit .env)")
        return key

    def provider(self, name: str) -> ProviderInfo:
        raw = self._config["providers"][name]
        return ProviderInfo(
            name=name,
            display_name=raw["display_name"],
            env_var=raw["env_var"],
            credit_usd=float(raw["credit_usd"]),
            base_url=raw["base_url"],
            key_present=self.key_present(name),
        )

    def model(self, selector: str) -> ModelInfo:
        for provider, raw in self._config["providers"].items():
            models = raw.get("models") or {}
            if selector not in models:
                continue
            spec = models[selector]
            prices = spec["prices"]
            env_name = selector.upper().replace("-", "_")
            override = self._env.get(f"FIELD_AGENT_MODEL_{env_name}")
            effort = self._env.get(f"FIELD_AGENT_EFFORT_{env_name}") or spec.get(
                "reasoning_effort"
            )
            return ModelInfo(
                selector=selector,
                provider=provider,
                model_id=override or spec["model_id"],
                reasoning_effort=str(effort) if effort else None,
                unavailable=spec.get("unavailable"),
                fallback=spec.get("fallback"),
                role=spec.get("role", ""),
                input_per_million=float(prices["input_per_million_tokens_usd"]),
                cached_input_per_million=(
                    float(prices["cached_input_per_million_tokens_usd"])
                    if prices.get("cached_input_per_million_tokens_usd") is not None
                    else None
                ),
                output_per_million=float(prices["output_per_million_tokens_usd"]),
                price_source=prices["source"],
                price_verified=prices["verified"],
                tool_choice=spec.get("tool_choice"),
            )
        raise KeyError(f"unknown model selector: {selector!r}")

    def selectable_models(self) -> list[str]:
        return list(self._config["agent"]["selectable_models"])

    def default_model(self) -> str:
        return self._config["agent"]["default_model"]

    def stt(self) -> dict:
        return dict(self._config["providers"]["meta"]["stt"])

    def guard_fraction(self) -> float:
        return float(self._config["guard"]["refuse_at_fraction_of_credit"])

    def spend_log_path(self, override: str | Path | None = None) -> Path:
        if override is not None:
            return Path(override)
        return REPO_ROOT / self._config["guard"]["spend_log"]

    def estimate_cost(
        self,
        model: ModelInfo,
        *,
        input_tokens: int,
        cached_input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> float:
        uncached = max(0, input_tokens - cached_input_tokens)
        cached_rate = (
            model.cached_input_per_million
            if model.cached_input_per_million is not None
            else model.input_per_million
        )
        return (
            uncached * model.input_per_million
            + cached_input_tokens * cached_rate
            + output_tokens * model.output_per_million
        ) / 1_000_000

    def estimate_stt_cost(self, audio_seconds: float) -> float:
        per_hour = float(self.stt()["prices"]["per_audio_hour_usd"])
        return max(0.0, audio_seconds) / 3600.0 * per_hour


class MissingKeyError(RuntimeError):
    """A required provider key is absent. The message never carries a value."""
