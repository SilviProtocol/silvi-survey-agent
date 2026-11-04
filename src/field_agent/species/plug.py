"""The species plug: where species answers come from.

One operation — "add evidence" — plus a read of a plant's current
state. Silvi runs a closed species service behind this plug; point
`SPECIES_SERVICE_URL` at one and answers come from there. Without a
service, a plain local fallback answers instead: each photo stands
alone against a list of names, the state read returns the latest
photo's answer, and what to ask for next is a fixed order that
skips kinds already had.

Photos travel to the plug only. They never enter a model's messages.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass, field

from field_agent.species.bioclip import bioclip_status, score_photo

WANTS = ("whole", "bark", "leaf", "flower_or_fruit", "base", "side")
_FALLBACK_ORDER = ("whole", "bark", "leaf", "flower_or_fruit")
DONE_CONFIDENCE = 0.8


@dataclass
class Candidate:
    name: str
    rank: str = "species"
    p: float = 0.0
    common_name: str | None = None


@dataclass
class EvidenceResult:
    candidates: list[Candidate] = field(default_factory=list)
    confidence: float = 0.0
    done: bool = False
    next_evidence: list[str] = field(default_factory=list)  # wants, best first
    seen_kind: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> EvidenceResult:
        wants = []
        for item in data.get("next_evidence") or []:
            want = item.get("want") if isinstance(item, dict) else item
            if want in WANTS:
                wants.append(str(want))
        return cls(
            candidates=[
                Candidate(
                    name=str(c.get("name") or ""),
                    rank=str(c.get("rank") or "species"),
                    p=float(c.get("p") or 0.0),
                    common_name=c.get("common_name"),
                )
                for c in (data.get("candidates") or [])[:5]
            ],
            confidence=float(data.get("confidence") or 0.0),
            done=bool(data.get("done")),
            next_evidence=wants,
            seen_kind=data.get("seen_kind"),
        )


class HttpSpeciesPlug:
    """The plug over HTTP, for a service running elsewhere.

    - ``POST {base}/evidence`` with the add-evidence fields (photo as
      base64 in ``photo_b64``) returns the result as JSON.
    - ``GET {base}/plants/{survey_id}/{plant_id}`` returns the same
      shape for the plant's current state, or 404 for "nothing yet".
    Extra keys in a reply are ignored.
    """

    def __init__(self, base_url: str, *, opener=None, timeout_s: float = 20.0):
        self._base = base_url.rstrip("/")
        self._opener = opener or urllib.request.urlopen
        self._timeout = timeout_s

    def add_evidence(self, *, survey_id, plant_id, photo, kind_hint, request_id=None,
                     captured_at=None, distance_m=None, lat=None, lon=None):
        body = json.dumps({
            "survey_id": survey_id,
            "plant_id": int(plant_id),
            "photo_b64": base64.b64encode(photo).decode("ascii"),
            "kind_hint": kind_hint,
            "request_id": request_id,
            "captured_at": captured_at,
            "distance_m": distance_m,
            "lat": lat,
            "lon": lon,
        }).encode()
        request = urllib.request.Request(
            f"{self._base}/evidence", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with self._opener(request, timeout=self._timeout) as response:
            return EvidenceResult.from_dict(json.loads(response.read()))

    def state(self, *, survey_id, plant_id):
        try:
            with self._opener(
                f"{self._base}/plants/{survey_id}/{int(plant_id)}", timeout=self._timeout
            ) as response:
                return EvidenceResult.from_dict(json.loads(response.read()))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise

    def warm_up(self) -> tuple[bool, str]:
        return True, "service over HTTP"


class OpenFallbackPlug:
    """The plain local fallback: every photo judged on its own."""

    def __init__(self, scorer=None):
        self._scorer = scorer
        self._latest: dict[tuple[str, int], EvidenceResult] = {}
        self._kinds: dict[tuple[str, int], set] = {}

    def warm_up(self) -> tuple[bool, str]:
        if self._scorer is not None:
            return True, "injected scorer"
        return bioclip_status()

    def _score(self, photo: bytes) -> list[dict]:
        if self._scorer is not None:
            return list(self._scorer(photo) or [])
        return score_photo(photo)

    def add_evidence(self, *, survey_id, plant_id, photo, kind_hint, request_id=None,
                     captured_at=None, distance_m=None, lat=None, lon=None):
        key = (str(survey_id), int(plant_id))
        kinds = self._kinds.setdefault(key, set())
        kinds.add(kind_hint)
        ranked = [
            Candidate(
                name=str(c.get("name") or ""),
                rank=str(c.get("rank") or "species"),
                p=float(c.get("p") or 0.0),
                common_name=c.get("common_name"),
            )
            for c in self._score(photo)
            if c.get("name")
        ][:5]
        confidence = ranked[0].p if ranked else 0.0
        done = confidence >= DONE_CONFIDENCE
        result = EvidenceResult(
            candidates=ranked,
            confidence=confidence,
            done=done,
            next_evidence=[] if done else [w for w in _FALLBACK_ORDER if w not in kinds],
            seen_kind=kind_hint,
        )
        self._latest[key] = result
        return result

    def state(self, *, survey_id, plant_id):
        return self._latest.get((str(survey_id), int(plant_id)))


def make_species_plug(*, environ: dict | None = None, scorer=None):
    """HTTP when ``SPECIES_SERVICE_URL`` is set, else the open fallback."""
    url = (environ if environ is not None else os.environ).get("SPECIES_SERVICE_URL")
    if url:
        return HttpSpeciesPlug(str(url))
    return OpenFallbackPlug(scorer=scorer)
