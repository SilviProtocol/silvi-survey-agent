"""`field-agent doctor`: key presence, one minimal live call, spend state.

Never prints a key. A live check sends one short text prompt per available
provider and is logged through the spend guard like any other call. Speech
to text shares the Meta key; its live check needs real audio, so doctor
reports it as pending rather than faking a result.
"""

from __future__ import annotations

from field_agent.models import OpenAICompatibleChatClient
from field_agent.providers import MissingKeyError, ProviderRegistry
from field_agent.spend import SpendGuard


def run_doctor(
    registry: ProviderRegistry,
    guard: SpendGuard,
    *,
    live: bool = True,
    transport=None,
) -> dict:
    providers: dict[str, dict] = {}
    selector_for: dict[str, str] = {}
    for selector in registry.selectable_models():
        model = registry.model(selector)
        if not model.unavailable:
            selector_for.setdefault(model.provider, selector)

    for name in registry.provider_names():
        info = registry.provider(name)
        spent = guard.totals()[name]
        entry: dict = {
            "provider": name,
            "display_name": info.display_name,
            "env_var": info.env_var,
            "key_present": info.key_present,
            "credit_usd": info.credit_usd,
            "spent_usd": spent["spent_usd"],
            "fraction_used": spent["fraction_used"],
            "blocked_by_guard": spent["blocked"],
            "live_check": "pending: no key" if not info.key_present else "pending",
            "latency_ms": None,
        }
        providers[name] = entry
        if not info.key_present:
            continue
        selector = selector_for.get(name)
        if selector is None:
            entry["live_check"] = "skipped: no text model configured"
            continue
        if not live:
            entry["live_check"] = "skipped: --no-live"
            continue
        entry["model"] = registry.model(selector).model_id
        try:
            client = OpenAICompatibleChatClient(registry, selector, guard, transport=transport)
            result = client.complete(
                [{"role": "user", "content": "Reply with exactly: ok"}], max_tokens=16
            )
            entry["live_check"] = "ok"
            entry["latency_ms"] = result.latency_ms
        except MissingKeyError as exc:
            entry["live_check"] = f"pending: {exc}"
        except Exception as exc:  # noqa: BLE001 - report provider failures, do not crash doctor
            entry["live_check"] = f"error: {exc}"

    stt = registry.stt()
    return {
        "providers": providers,
        "stt": {
            "model": stt["model_id"],
            "endpoint": stt["realtime_url"],
            "key_env_var": registry.provider("meta").env_var,
            "key_present": registry.key_present("meta"),
            "live_check": "pending: needs a short recorded test phrase (see report)",
        },
        "spend_log": str(guard.path),
    }


def format_doctor(report: dict) -> str:
    lines = []
    for entry in report["providers"].values():
        key = "key present" if entry["key_present"] else "key MISSING"
        latency = f", latency {entry['latency_ms']} ms" if entry["latency_ms"] is not None else ""
        model = f" [{entry['model']}]" if entry.get("model") else ""
        lines.append(
            f"{entry['display_name']}: {key} ({entry['env_var']}), "
            f"credit ${entry['credit_usd']:.0f}, spent ${entry['spent_usd']:.4f} "
            f"({entry['fraction_used']:.2%}), guard {'BLOCKING' if entry['blocked_by_guard'] else 'open'}, "
            f"live text check: {entry['live_check']}{model}{latency}"
        )
    stt = report["stt"]
    lines.append(
        f"Speech to text ({stt['model']}): key {'present' if stt['key_present'] else 'MISSING'} "
        f"({stt['key_env_var']}), live check: {stt['live_check']}"
    )
    lines.append(f"Spend log: {report['spend_log']}")
    return "\n".join(lines)
