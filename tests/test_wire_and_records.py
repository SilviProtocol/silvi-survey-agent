"""Wire tolerance, record schema, and plain world status rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from field_agent.contracts import validate_observation, validate_wire
from field_agent.world import SurveyWorld

ROOT = Path(__file__).resolve().parents[1]


def _snap(**extra):
    payload = {
        "plant_id": 5, "status": "confirmed", "position_m": [1.0, 0.0, 2.0],
        "sigma_m": 0.3, "n_views": 4,
    }
    payload.update(extra)
    return {"v": "1", "kind": "plant_snapshot", "seq": 1, "session_id": "s",
            "t_mono_ms": 1.0, "payload": payload}


def test_phone_fields_the_schema_dropped_are_accepted_and_ignored():
    # a phone still sending older fields must not be rejected
    validate_wire(_snap(baseline_m=1.2, something_new=True))
    validate_wire({
        "v": "1", "kind": "plant_sighting", "seq": 2, "session_id": "s", "t_mono_ms": 2.0,
        "payload": {"keyframe_seq": 1, "observation_id": "o1", "label": "tree",
                    "score": 0.9, "confirmed": True, "track_id": 7, "depth_m": 3.1},
    })
    validate_wire({
        "v": "1", "kind": "anchor", "seq": 3, "session_id": "s", "t_mono_ms": 3.0,
        "payload": {"anchor_id": "a1", "action": "placed", "anchor_type": "base",
                    "position_m": [0, 0, 0]},
    })


def test_agent_kinds_stay_strict():
    with pytest.raises(ValidationError):
        validate_wire({
            "v": "1", "kind": "tree_status", "seq": 0, "session_id": "s", "t_mono_ms": 0.0,
            "payload": {"trees": [{"plant_id": 1, "number": 1, "retired_field": 0.5,
                                    "what_ring": 0.5, "short_name": "oak"}]},
        })


def test_sample_records_validate():
    for name in ("sample-observation.json", "sample-bearing-only.json"):
        validate_observation(json.loads((ROOT / "fixtures" / name).read_text()))


def test_needs_and_completion_are_plain_status():
    world = SurveyWorld()
    world.add_snapshot({"plant_id": 5, "status": "tentative", "position_m": [0, 0, 0],
                        "sigma_m": 0.9, "n_views": 1})
    assert world.needs(5) == ["unconfirmed", "no_species"]
    assert world.is_complete(5) is False
    world.add_snapshot({"plant_id": 5, "status": "confirmed", "position_m": [0, 0, 0],
                        "sigma_m": 0.9, "n_views": 1})
    assert world.is_complete(5) is True
    assert world.needs(5) == ["no_species"]
    world.set_species(5, {"taxon": "Acer rubrum", "confidence": 0.9})
    assert world.needs(5) == []
    row = world.status_row(5)
    assert "retired_field" not in row
    assert row["what_ring"] == 0.9
