"""The species plug and its fallback: each photo stands alone."""

from __future__ import annotations

import logging

from field_agent.species import bioclip
from field_agent.species.plug import OpenFallbackPlug, make_species_plug


def test_second_photo_replaces_the_first_no_averaging():
    scores = iter([0.9, 0.3])
    plug = OpenFallbackPlug(
        scorer=lambda photo: [{"name": "Acer rubrum", "rank": "species", "p": next(scores)}]
    )
    first = plug.add_evidence(survey_id="w", plant_id=1, photo=b"a", kind_hint="whole")
    assert first.confidence == 0.9 and first.done is True
    second = plug.add_evidence(survey_id="w", plant_id=1, photo=b"b", kind_hint="bark")
    assert second.confidence == 0.3  # replaced, not averaged to 0.6
    assert second.done is False
    assert plug.state(survey_id="w", plant_id=1).confidence == 0.3
    # kinds already had are not asked for again; plain wants, no numbers attached
    assert second.next_evidence == ["leaf", "flower_or_fruit"]


def test_done_is_one_plain_threshold_on_one_photo():
    plug = OpenFallbackPlug(scorer=lambda photo: [{"name": "X y", "p": 0.79}])
    result = plug.add_evidence(survey_id="w", plant_id=2, photo=b"a", kind_hint="leaf")
    assert result.done is False
    plug = OpenFallbackPlug(scorer=lambda photo: [{"name": "X y", "p": 0.8}])
    result = plug.add_evidence(survey_id="w", plant_id=2, photo=b"a", kind_hint="leaf")
    assert result.done is True and result.next_evidence == []


def test_http_reply_extra_keys_are_ignored(monkeypatch):
    import io
    import json

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(request, timeout=None):
        return Resp(json.dumps({
            "candidates": [{"name": "Quercus alba", "rank": "species", "p": 0.7}],
            "confidence": 0.7, "done": False, "seen_kind": "bark",
            "next_evidence": [{"want": "leaf", "gain": 0.4}, {"want": "whole", "gain": 0.2}],
        }).encode())

    plug = make_species_plug(environ={"SPECIES_SERVICE_URL": "http://127.0.0.1:9"})
    plug._opener = opener
    result = plug.add_evidence(survey_id="w", plant_id=3, photo=b"x", kind_hint="bark")
    assert result.next_evidence == ["leaf", "whole"]


def test_no_weights_gives_empty_and_one_logged_reason(monkeypatch, caplog):
    monkeypatch.delenv("BIOCLIP2_WEIGHTS", raising=False)
    bioclip._reset_for_tests()
    try:
        with caplog.at_level(logging.WARNING, logger="field_agent.species"):
            assert bioclip.score_photo(b"jpg") == []
            assert bioclip.score_photo(b"jpg") == []
        off = [r for r in caplog.records if "species: off" in r.getMessage()]
        assert len(off) == 1
        ready, reason = bioclip.bioclip_status()
        assert ready is False and "BIOCLIP2_WEIGHTS" in reason
    finally:
        bioclip._reset_for_tests()
