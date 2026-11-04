"""Command line: serve the live agent, check providers, run the demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _registry_and_guard(args):
    from field_agent.providers import ProviderRegistry
    from field_agent.spend import SpendGuard

    registry = ProviderRegistry.load(Path(args.config) if getattr(args, "config", None) else None)
    return registry, SpendGuard(registry)


def _make_client(args):
    from field_agent.models import make_chat_client
    from field_agent.scripted import recorded_handler

    registry, guard = _registry_and_guard(args)
    mock = getattr(args, "model", None) == "mock"
    selector = registry.default_model() if mock else args.model
    client = make_chat_client(
        registry, selector, guard, mock=mock, mock_handler=recorded_handler()
    )
    return client, guard, ("mock" if mock else selector), registry


def _species_plug(args):
    from field_agent.species.plug import make_species_plug

    plug = make_species_plug()
    ready, reason = plug.warm_up()
    print("species plug: " + ("ready" if ready else f"off, {reason}"), file=sys.stderr)
    return plug


def _cmd_serve(args) -> int:
    from field_agent.agent import FieldAgent
    from field_agent.server import ServerCore, run_server

    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.allow_lan:
        print(
            f"refusing non-loopback bind {args.host} without --allow-lan "
            "(no auth or TLS; see docs/PHONE-CLIENT.md)",
            file=sys.stderr,
        )
        return 2
    client, guard, label, registry = _make_client(args)
    plug = _species_plug(args)
    stt_factory = None
    try:
        from field_agent.voice import make_meta_stt_factory

        stt_factory = make_meta_stt_factory(registry, spend_guard=guard)
    except Exception:  # noqa: BLE001 - no voice extra: phones keep on-device speech
        stt_factory = None
    print(
        "server speech: " + ("on (Muse Voice Transcribe)" if stt_factory else "off"),
        file=sys.stderr,
    )
    core = ServerCore(
        lambda: FieldAgent(client, model_label=label, species_plug=plug),
        stt_factory=stt_factory,
    )
    run_server(args.host, args.port, core)
    return 0


def _cmd_doctor(args) -> int:
    from field_agent.doctor import run_doctor

    registry, guard = _registry_and_guard(args)
    report = run_doctor(registry, guard, live=args.live)
    print(json.dumps(report, indent=2))
    return 0


def _cmd_demo(args) -> int:
    """Replay a wire journal through the agent on recorded mock replies."""
    from field_agent.agent import FieldAgent
    from field_agent.models import MockChatClient
    from field_agent.replay import iter_journal_replay, run_replay
    from field_agent.scripted import recorded_handler
    from field_agent.species.plug import OpenFallbackPlug

    journal = Path(args.journal) if args.journal else ROOT / "fixtures" / "synthetic-journal.jsonl"
    plug = OpenFallbackPlug(
        scorer=lambda photo: [
            {"name": "Acer rubrum", "rank": "species", "p": 0.9, "common_name": "red maple"}
        ]
    )
    agent = FieldAgent(
        MockChatClient(handler=recorded_handler()), model_label="mock", species_plug=plug
    )
    summary = run_replay(iter_journal_replay(journal), 0, agent)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "records.json").write_text(
        json.dumps(agent.survey.records, indent=2) + "\n", encoding="utf-8"
    )
    speech = [m["payload"]["text"] for m in agent.outputs if m["kind"] == "speech"]
    print(json.dumps({
        "journal": str(journal),
        "messages": summary,
        "records": len(agent.survey.records),
        "records_file": str(out_dir / "records.json"),
        "speech": speech,
    }, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="field-agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the live agent WebSocket server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--allow-lan", action="store_true")
    serve.add_argument("--model", default="spark")
    serve.add_argument("--config", default=None, help="providers config JSON")
    serve.set_defaults(func=_cmd_serve)

    doctor = sub.add_parser("doctor", help="check provider keys and spend state")
    doctor.add_argument("--config", default=None, help="providers config JSON")
    doctor.add_argument("--live", action="store_true",
                        help="also make one small live call per provider (spends a little)")
    doctor.set_defaults(func=_cmd_doctor)

    demo = sub.add_parser("demo", help="replay a synthetic journal on mock replies")
    demo.add_argument("--journal", default=None, help="wire journal (default: bundled synthetic)")
    demo.add_argument("--out", default="data/demo")
    demo.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
