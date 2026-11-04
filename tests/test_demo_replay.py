"""The synthetic journal replays through the agent on mock replies."""

from __future__ import annotations

import json
from pathlib import Path

from field_agent.agent import FieldAgent
from field_agent.contracts import validate_observation
from field_agent.models import MockChatClient
from field_agent.replay import iter_journal_replay, run_replay
from field_agent.scripted import recorded_handler
from field_agent.species.plug import OpenFallbackPlug

ROOT = Path(__file__).resolve().parents[1]


def _run_demo():
    plug = OpenFallbackPlug(
        scorer=lambda photo: [
            {"name": "Acer rubrum", "rank": "species", "p": 0.9, "common_name": "red maple"}
        ]
    )
    agent = FieldAgent(
        MockChatClient(handler=recorded_handler()), model_label="mock", species_plug=plug
    )
    summary = run_replay(
        iter_journal_replay(ROOT / "fixtures" / "synthetic-journal.jsonl"), 0, agent
    )
    return agent, summary


def test_journal_reaches_the_agent_including_v2_kinds():
    agent, summary = _run_demo()
    assert summary["by_kind"]["anchor"] == 1
    assert summary["by_kind"]["evidence_photo"] == 1
    assert summary["by_kind"]["transcript"] == 3
    assert len(agent.world.anchors) == 1


def test_demo_makes_a_valid_record_and_the_plug_name_lands_on_it():
    agent, _summary = _run_demo()
    assert len(agent.survey.records) == 1
    record = agent.survey.records[0]
    validate_observation(record)
    assert record["identification"]["taxon"] == "Acer rubrum"
    assert record["identification"]["revisable"] is True
    speech = [m["payload"]["text"] for m in agent.outputs if m["kind"] == "speech"]
    assert any("confirmed" in s for s in speech)


def test_cli_demo_writes_records(tmp_path, capsys):
    from field_agent.cli import main

    assert main(["demo", "--out", str(tmp_path)]) == 0
    records = json.loads((tmp_path / "records.json").read_text())
    assert len(records) == 1
    validate_observation(records[0])
