"""Field agent core: wire feed + transcript turns -> checked wire replies.

The agent consumes the same wire messages as the stub agent (replay or live),
keeps a small text summary of survey state, and on each final transcript asks
the configured model (text only) what to do. Model tool calls pass through
the checker in tools.py before they become validated wire outputs.

Structured intents (tap / crosshair in later packages) enter through
``handle_intent`` and produce identical wire outputs without any model call,
so no agent action depends on voice.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import UTC, datetime
from typing import ClassVar

from field_agent.contracts import validate_observation, validate_wire
from field_agent.coverage import CoverageGrid
from field_agent.gap_planner import plan_gaps
from field_agent.revisit import guidance
from field_agent.survey import SurveyState
from field_agent.tools import check_tool_call, check_tool_calls, tool_definitions
from field_agent.tracing import LatencyTracer
from field_agent.world import SurveyWorld, parse_tree_id, tree_label

SYSTEM_PROMPT = """You are Spark, Silvi's field survey assistant, out on a planting site with the surveyor, talking while they walk. Every reply is read aloud, so write for ears: one or two sentences, about twenty words, the answer first. Offer at most one thing to do next. Only go longer when the surveyor asks for more.
Each turn you get a packet — NOW, FOCUS (the tree in the crosshair, if any), AROUND YOU, NEAREST, NEEDS WORK, SURVEY and LOGGED — and tool results as you call them. Read both before answering. All the measuring is done in code for you; take distances, directions and counts from the packet or the query tools (scene, tree, find_trees, route_to, area_report), never work them out yourself, and never make up a species, a distance, a count or a tree number. For "how many have I logged", read the LOGGED line only; never add packet lines together, and never count rows. Tree IDs in tools and the packet look like T45; out loud say "tree 45". Directions are clock and metres: "about 3 metres at 2 o'clock".
The packet talks in shorthand; translate it before speaking. "Tentative" or very few views means "not confirmed yet" or "I've only glimpsed it". A rough position is just that; don't quote its margin. Leave view counts, degrees, sigma and coverage percentages unsaid unless asked.
"This one", "here" and "it" mean the focus tree. With no focus, ask the surveyor to point the crosshair at the tree they mean; never answer about some other tree or borrow another tree's notes. If an ID was merged into a newer one, use the new number and mention the merge once; a retired ID is not a plant.
"What is this?" wants the species before anything else. If the focus tree has no species yet, get one in this same turn with identify_species, then answer with plain certainty first — "probably a tulip tree, tree 39, 2 metres ahead". Genus only when the result's rank is genus. If you still can't name it, the surveyor still hears which tree and where first — "That's tree 36, 3 metres ahead. I can't tell the species yet." — plus its location need in a few words, only if it has one. Never end a turn on a promise ("I'll get…", "let me check…"): answer from what is in front of you, or say plainly that you don't know.
Changing records takes an ask, in this sentence, out loud: note, mark, record, start, close, name, correct and the like. A question never changes anything, and something said turns ago is never a standing instruction to write. The write tools are add_note, correct_record, record_growth, boundary, create_site, create_boundary, create_plot, create_planting, create_observation, add_corner and close_polygon; they are refused in code when this utterance did not ask, and you will be told so. When the surveyor does ask, act without fuss: "planting here" is a planting at the position given, "what have I missed?" is report_gaps, a stated correction is correct_record, and "start/pause/resume/close the boundary" is the boundary tool, with area_report after a close.
"Closest" means physically closest, confirmed or not; if the nearest is not confirmed yet, say so in the same breath. "What needs work?" gets the nearest one fixable thing and what to do about it, not a tour of everything.
For each tree there is a species answer building on the phone's screen. Species certainty goes in ordinary words with a rough number — "probably a tulip tree, about 60%" — and a name can change as photos arrive; when it does, say it changed. To build that answer, ask for one photo at a time with request_evidence: the first item in that tree's next_evidence, on the nearest tree that needs it, unless the surveyor is asking about a different one. When a tree is confirmed and named, say so in a few words and move on."""

_SURVEY_TOOLS = {
    "scene", "tree", "find_trees", "route_to", "area_report",
    "add_note", "boundary", "request_evidence",
}

_TOOL_TO_WIRE = {
    "say": "speech",
    "mark_waypoint": "waypoint",
    "mark_polygon": "polygon",
    "report_gap": "gap",
}

_DIRECT_WIRE_TOOLS = set(_TOOL_TO_WIRE)

_FALLBACK = "I could not place that. Say it again, or mark the spot with the crosshair."


# Tools that change a record. In a spoken turn they run only when the
# current utterance asks for a change , and only the tool
# the utterance's own words call for ; taps through
# handle_intent are explicit by construction and are not gated here.
_WRITE_TOOLS = {
    "add_note",
    "correct_record",
    "record_growth",
    "boundary",
    "create_site",
    "create_boundary",
    "create_plot",
    "create_planting",
    "create_observation",
    "add_corner",
    "close_polygon",
}

# Which record-changing tools a way of asking allows. A word match is
# enough — this is a guard against the wrong write, not an understander.
_WRITE_RULES: list[tuple[re.Pattern, frozenset]] = [
    (
        re.compile(
            r"\b(note|noted|remember|says|looks|dead|sick|damaged|flag|spots|browse)\b",
            re.IGNORECASE,
        ),
        frozenset({"add_note", "correct_record"}),
    ),
    (re.compile(r"\bcorner\b", re.IGNORECASE), frozenset({"add_corner"})),
    (
        re.compile(
            r"(it's a\b|it is a\b|that's a\b|that is a\b|this is a\b|"
            r"\bcorrect\b|\bactually\b|not a (plant|tree)\b|\bnot a\b|"
            r"two plants|same (tree )?as|\bi planted\b|\bname it\b|\bcall it\b)",
            re.IGNORECASE,
        ),
        frozenset({"correct_record"}),
    ),
    (
        re.compile(
            r"\b(height|tall|metres?|meters?|trunk|centimetres?|centimeters?|"
            r"diameter|dbh|growth|grown)\b",
            re.IGNORECASE,
        ),
        frozenset({"record_growth"}),
    ),
    (
        re.compile(r"\bboundary\b", re.IGNORECASE),
        frozenset({"boundary", "create_boundary"}),
    ),
    (
        re.compile(r"\b(close|closing)\b[^.]*\b(plot|polygon|boundary)\b", re.IGNORECASE),
        frozenset({"boundary", "close_polygon"}),
    ),
    (
        re.compile(r"\bplanting\b|\bplanted (here|one|a)\b", re.IGNORECASE),
        frozenset({"create_planting"}),
    ),
    (
        re.compile(
            r"start the survey|this is the [^.]*\b(plot|site)\b|new (site|plot)\b",
            re.IGNORECASE,
        ),
        frozenset({"create_site", "create_plot"}),
    ),
    (
        re.compile(r"\b(create|make) (a |an )?(observation|record)\b", re.IGNORECASE),
        frozenset({"create_observation"}),
    ),
]

_GENERIC_ASK_RE = re.compile(r"\b(mark|record|log|save|add|set)\b", re.IGNORECASE)
_STRONG_ASK_RE = re.compile(
    r"\b(note|mark|record|start|close|add|set|correct|remember|pause|resume|"
    r"create|planting|log|save)\b|\b(name|call) it\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r"^\s*(what|which|where|how many|how much|how|did i|is this|is it|"
    r"have i|are there|can you tell|who|when)\b",
    re.IGNORECASE,
)
def _normalise_spoken(text: str) -> str:
    """Speech-recogniser spelling, smoothed for matching only.

    Curly apostrophes become straight ones and everything lower-cases,
    so "this one’s", "THIS ONE'S" and "this ones" all meet the same
    patterns. The surveyor's original words are never altered.
    """
    return (text or "").replace("’", "'").replace("‘", "'").lower()


_DEICTIC_RE = re.compile(
    r"\b(this one's|that one's|this one is|this one|this tree|that tree|"
    r"this plant|that one|it's|this is|that's|that|this|it|here)\b"
)
_CLAIM_RE = re.compile(
    r"\b(closed|noted|recorded|marked|started|planted|saved|logged|added|done)\b",
    re.IGNORECASE,
)

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_NAMED_TREE_RE = re.compile(
    r"(?i:\b(?:tree|plant)\s*-?\s*(\d+|[a-z]+(?:[\s-]+[a-z]+)?))|\bT-?(\d+)\b"
)


def _named_tree_id(text: str) -> int | None:
    """The tree number an utterance names, digits or words, if any."""
    for match in _NAMED_TREE_RE.finditer(text or ""):
        if match.group(2):
            return int(match.group(2))
        token = (match.group(1) or "").lower().replace("-", " ")
        if token.isdigit():
            return int(token)
        parts = token.split()
        if parts and parts[0] in _NUMBER_WORDS:
            # "tree one has…" is one followed by a verb, not a compound.
            if len(parts) == 2 and parts[1] in _NUMBER_WORDS:
                tens, ones = _NUMBER_WORDS[parts[0]], _NUMBER_WORDS[parts[1]]
                if tens >= 20 and 0 < ones < 10:
                    return tens + ones
            return _NUMBER_WORDS[parts[0]]
    return None


def _utterance_is_question(text: str) -> bool:
    low = (text or "").strip().lower()
    return low.endswith("?") or bool(_QUESTION_RE.match(low))


# Question openings that are asking, never instructing — "did I close
# the boundary?" mentions a write without asking for one.
_HARD_QUESTION_RE = re.compile(
    r"^\s*(what|which|where|how many|how much|did i|is this|is it|have i|"
    r"are there|can you tell|who|when)\b",
    re.IGNORECASE,
)


def allowed_write_tools(text: str) -> set[str]:
    """Record-changing tools this utterance's own words allow.

    A question with no instruction verb allows none, whatever else it
    mentions, and the hard question openings above allow none even
    then; a bare ask verb with no subject words allows only a note,
    the least destructive write.
    """
    words = _normalise_spoken(text)
    if _HARD_QUESTION_RE.match(words.strip()):
        return set()
    if _utterance_is_question(words) and not _STRONG_ASK_RE.search(words):
        return set()
    allowed: set[str] = set()
    for pattern, tools in _WRITE_RULES:
        if pattern.search(words):
            allowed |= tools
    if not allowed and _GENERIC_ASK_RE.search(words):
        allowed = {"add_note"}
    return allowed

_SPECIES_QUESTION_RE = re.compile(
    r"\b(what is this|what's this|what is it|what species|which species|"
    r"what tree is|identify|species is)\b",
    re.IGNORECASE,
)

_EXPLICIT_ID_RE = re.compile(r"\b(?:tree|plant|t)\s*-?\s*\d+\b", re.IGNORECASE)

_PLAIN_NEEDS = {
    "unconfirmed": "not confirmed yet",
    "few_views": "needs a few more views",
    "one_side": "only seen from one side — walk round to the far side",
    "position_loose": "its position is still rough",
    "no_species": "no species name yet",
}


def utterance_requests_change(text: str) -> bool:
    """True when this utterance itself asks for a record to change."""
    return bool(allowed_write_tools(text))


def _spoken_label(label) -> str:
    """'T45' / 'tree-45' said the way ears want it: 'tree 45'."""
    text = str(label)
    match = re.fullmatch(r"(?:tree|plant|t)[-\s]*(\d+)", text.strip(), re.IGNORECASE)
    return f"tree {match.group(1)}" if match else text.replace("-", " ")


_COUNT_QUESTION_RE = re.compile(
    r"how many[^.?!]*(logged|recorded|counted|got|so far|have i|i have|"
    r"inside|boundary|plot|area)|\b(logged|got) so far\b",
    re.IGNORECASE,
)
_SCOPED_COUNT_RE = re.compile(r"\b(boundary|plot|inside|area)\b", re.IGNORECASE)

_PROMISE_RE = re.compile(
    r"\b(i'll|i will|i'm going to|i am going to|let me)\s+"
    r"(get|check|look|find|fetch|see|work out|figure)\b",
    re.IGNORECASE,
)

_LOCATION_NEED_CLAUSES = {
    "one_side": "It is only seen from one side — walk round to the far side.",
    "position_loose": "Its position is still rough.",
    "few_views": "It needs a few more views.",
    "unconfirmed": "It is not confirmed yet.",
}


def _where_phrase(row: dict) -> str:
    """'about 3 metres ahead' / 'about 6 metres at 2 o'clock', or ''."""
    dist = row.get("dist_m")
    if dist is None:
        return ""
    if row.get("where") == "ahead":
        return f"about {float(dist):.0f} metres ahead"
    return f"about {float(dist):.0f} metres at {row.get('clock')} o'clock"


def _location_need_clause(row: dict) -> str:
    for need in row.get("needs") or []:
        if need in _LOCATION_NEED_CLAUSES:
            return _LOCATION_NEED_CLAUSES[need]
    return ""


def _certainty_phrase(taxon, confidence) -> str:
    if confidence is None:
        return str(taxon)
    conf = float(confidence)
    if conf >= 0.85:
        return f"{taxon}, fairly sure"
    if conf >= 0.55:
        return f"probably {taxon}"
    return f"might be {taxon}"


def _a_or_an(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


_WANT_WORDS = {
    "bark": "bark",
    "leaf": "leaf",
    "whole": "whole tree",
    "flower_or_fruit": "flower or fruit",
    "base": "base",
    "side": "other side",
}


def _want_words(want: str) -> str:
    return _WANT_WORDS.get(want, want)


def _want_of(item) -> str:
    """One next-evidence entry: a plain want string, or a dict naming one."""
    if isinstance(item, dict):
        return str(item.get("want") or "whole")
    return str(item)


def _identification_from(species: dict) -> dict:
    """A record identification from a plug answer, in schema shape."""
    out = {
        "taxon": species["taxon"],
        "confidence": float(species.get("confidence") or 0.0),
        "basis": "agent",
        "identified_by": None,
        "revisable": True,
        "taxon_rank": species.get("rank") or "species",
    }
    candidates = [
        {"name": c.get("name"), "p": float(c.get("p") or 0.0)}
        for c in (species.get("candidates") or [])
        if c.get("name")
    ]
    if candidates:
        out["candidates"] = candidates
    return out


def _parse_json_object(text: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class FieldAgent:
    def __init__(
        self,
        client,
        *,
        model_label: str,
        tracer: LatencyTracer | None = None,
        species_plug=None,
        turn_deadline_s: float | None = None,
    ):
        # Per-call model deadline; FIELD_AGENT_TURN_DEADLINE_S overrides
        # the 12 s default without a code change on the day.
        env_deadline = None
        try:
            env_deadline = float(os.environ.get("FIELD_AGENT_TURN_DEADLINE_S") or "")
        except ValueError:
            env_deadline = None
        self.turn_deadline_s = (
            float(turn_deadline_s) if turn_deadline_s is not None
            else env_deadline if env_deadline else 12.0
        )
        # The species plug: HTTP service when configured,
        # else the open fallback. Evidence photos go here, never to
        # the model.
        if species_plug is None:
            from field_agent.species.plug import make_species_plug

            species_plug = make_species_plug()
        self.species_plug = species_plug
        self._evidence_requests: dict[str, tuple[int, str]] = {}
        self._evidence_counter = 0
        self._plant_views: dict[int, bytes] = {}
        self._last_status_sent: dict[int, dict] = {}
        self._last_status_at: dict[int, float] = {}
        self._last_status_batch_at = 0.0
        self._last_geo: dict | None = None
        self._client = client
        self.model_label = model_label
        self.tracer = tracer or LatencyTracer()
        self.outputs: list[dict] = []
        self.dropped: list[dict] = []
        self.survey = SurveyState()
        self.world = SurveyWorld()
        self.coverage = CoverageGrid()
        self._history: list[dict] = []
        self._session_id = "agent"
        self._out_seq = 0
        self._last_mono = 0.0
        self._last_wall_ms: float | None = None
        self._pose: list[float] | None = None
        self._quat: list[float] = [0.0, 0.0, 0.0, 1.0]
        self._counts: dict[str, int] = {}
        self._plants: dict[int, dict] = {}
        self._turn_n = 0
        self.promotion: dict | None = None
        # plant_id -> local_id, from promotion backing (this session's
        # session_end, or load_stored's backed_by from the earlier visit).
        self._plant_records: dict[int, str] = {}
        # Records seeded by load_stored for a revisit, and the last plant
        # the walker was guided to (normalized request id). Growth for the
        # guided plant must update that identity's record, never mint one.
        self._stored_local_ids: list[str] = []
        self._guided_request: str | None = None
        self._last_turn_text: str | None = None
        # Set by the server once it has sent capabilities; v2 outputs
        # (tree_status, evidence_request, view_request) never travel
        # to a v1 phone.
        self.wire_v2 = False
        self._focus_override: int | None = None

    @property
    def next_output_seq(self) -> int:
        """Seq the next emitted wire output will carry (agent direction)."""
        return self._out_seq

    def _now_iso(self) -> str:
        if self._last_wall_ms:
            return datetime.fromtimestamp(self._last_wall_ms / 1000.0, tz=UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # -- wire feed in -------------------------------------------------------
    def __call__(self, message: dict) -> None:
        validate_wire(message)
        self._session_id = message["session_id"]
        self._last_mono = float(message["t_mono_ms"])
        if message.get("t_wall_ms"):
            self._last_wall_ms = float(message["t_wall_ms"])
        kind = message["kind"]
        self._counts[kind] = self._counts.get(kind, 0) + 1
        payload = message["payload"]
        if kind == "session_start":
            self.survey.walk_id = str(payload.get("walk_id") or self._session_id)
        elif kind == "pose":
            self._pose = [float(x) for x in payload["position_m"]]
            self._quat = [float(x) for x in payload["quaternion_xyzw"]]
            self.coverage.add_pose(self._pose)
            self.world.add_pose(payload, t_mono_ms=self._last_mono)
            if payload.get("geo"):
                self._last_geo = dict(payload["geo"])
        elif kind == "plant_sighting":
            self.world.add_sighting(payload)
        elif kind == "coverage":
            self.world.add_coverage(payload)
        elif kind == "plant_snapshot":
            self._plants[int(payload["plant_id"])] = payload
            self.world.add_snapshot(payload)
            if payload.get("position_m"):
                self.coverage.add_point(payload["position_m"])
            _state, live_id = self.world.resolve(int(payload["plant_id"]))
            self._emit_tree_status([live_id])
        elif kind == "anchor":
            self.world.add_anchor(payload, t_mono_ms=self._last_mono)
        elif kind == "mode":
            self.world.mode = str(payload.get("mode") or "") or None
        elif kind == "evidence_photo":
            self._on_evidence_photo(payload)
        elif kind == "plant_view":
            self._on_plant_view(payload)
        elif kind == "session_end":
            # Snapshots are end-state; promote/merge mid-walk records now.
            self.promotion = self.survey.promote_from_snapshots(self._plants.values())
            for local_id, plant_id in (self.promotion.get("backed_by") or {}).items():
                self._plant_records[int(plant_id)] = str(local_id)
            self._copy_plug_species_to_records()
        elif kind == "transcript" and payload.get("is_final") and (payload.get("text") or "").strip():
            if payload.get("focus") is not None:
                self._focus_override = int(payload["focus"])
            self.handle_turn(payload["text"].strip())

    def handle_server_transcript(self, utterance_id: str, text: str, *, focus=None) -> None:
        """One finished server-side transcription.

        Exactly one final per utterance, sent as agent data even when
        empty; a final with words then runs the normal turn on them,
        because the phone does not send its own transcript for it.
        """
        self._emit(
            "transcript",
            {"utterance_id": str(utterance_id), "text": str(text or ""), "final": True},
        )
        cleaned = str(text or "").strip()
        if cleaned:
            if focus is not None:
                self._focus_override = int(focus)
            self.handle_turn(cleaned)

    # -- evidence --------------------------------------------------
    def spoken_number(self, live_id: int) -> int:
        """The number to say out loud: the phone's screen number."""
        return self.world.display_of(live_id) or int(live_id)

    def request_evidence(self, live_id: int, want: str) -> dict:
        """Ask the phone to show one wanted view of one plant."""
        self._evidence_counter += 1
        request_id = f"ev-{self._evidence_counter}"
        self._evidence_requests[request_id] = (int(live_id), str(want))
        if self.wire_v2:
            self._emit(
                "evidence_request",
                {"request_id": request_id, "plant_id": int(live_id), "want": str(want)},
            )
        return {"request_id": request_id, "plant_id": int(live_id), "want": str(want)}

    def request_view(self, live_id: int, max_px: int = 768) -> None:
        if self.wire_v2:
            self._emit("view_request", {"plant_id": int(live_id), "max_px": int(max_px)})

    def _on_plant_view(self, payload: dict) -> None:
        import base64
        import binascii

        try:
            self._plant_views[int(payload["plant_id"])] = base64.b64decode(
                payload.get("jpeg") or "", validate=True
            )
        except (binascii.Error, ValueError, TypeError):
            return

    def _emit_tree_status(self, live_ids: list[int]) -> None:
        """Changed ring rows only, at most 2 batches/s and 1/tree/s."""
        if not self.wire_v2:
            return
        now = time.monotonic()
        if now - self._last_status_batch_at < 0.5:
            return
        rows = []
        for live_id in live_ids:
            if now - self._last_status_at.get(live_id, 0.0) < 1.0:
                continue
            row = self.world.status_row(live_id)
            if self._last_status_sent.get(live_id) == row:
                continue
            rows.append(row)
        if not rows:
            return
        self._emit("tree_status", {"trees": rows})
        self._last_status_batch_at = now
        for row in rows:
            self._last_status_sent[row["plant_id"]] = row
            self._last_status_at[row["plant_id"]] = now

    def _on_evidence_photo(self, payload: dict) -> None:
        """One evidence photo: to the plug, then rings and words.

        The pixels never touch a model message. What comes back is
        numbers and names, and those are what the surveyor hears.
        """
        import base64
        import binascii

        try:
            photo = base64.b64decode(payload.get("jpeg") or "", validate=True)
        except (binascii.Error, ValueError, TypeError):
            return
        _state, live_id = self.world.resolve(int(payload["plant_id"]))
        request_id = payload.get("request_id")
        asked: str | None = None
        if request_id and request_id in self._evidence_requests:
            asked = self._evidence_requests.pop(str(request_id))[1]
        else:
            for rid, (pid, want) in list(self._evidence_requests.items()):
                if pid == live_id:
                    asked = want
                    request_id = rid
                    del self._evidence_requests[rid]
                    break
        distance = payload.get("distance_m")
        kind_hint = asked or "whole"  # a far shot with no request reads as a whole view too
        geo = self._last_geo or {}
        try:
            result = self.species_plug.add_evidence(
                survey_id=self.survey.walk_id or self._session_id,
                plant_id=int(live_id),
                photo=photo,
                kind_hint=kind_hint,
                request_id=str(request_id) if request_id else None,
                captured_at=self._now_iso(),
                distance_m=float(distance) if distance is not None else None,
                lat=geo.get("lat"),
                lon=geo.get("lon"),
            )
        except Exception:  # noqa: BLE001 - a dead plug must not kill the walk
            return
        top = result.candidates[0] if result.candidates else None
        self.world.set_species(
            live_id,
            {
                "taxon": top.name if top else None,
                "common_name": top.common_name if top else None,
                "rank": top.rank if top else "species",
                "confidence": result.confidence,
                "done": result.done,
                "next_evidence": [dict(n) for n in result.next_evidence],
                "candidates": [c.__dict__ for c in result.candidates],
            },
        )
        number = self.spoken_number(live_id)
        words: list[str] = []
        seen = result.seen_kind
        if asked and seen and seen != asked:
            words.append(f"That looks like {_a_or_an(seen)} {seen}, I asked for {asked}.")
        name = (top.common_name or top.name) if top else None
        both_full = result.done and (self.world.snapshots.get(live_id) or {}).get("status") == "confirmed"
        if result.done:
            certainty = _certainty_phrase(name, result.confidence) if name else "species settled"
            done_word = "done" if both_full else "species done"
            words.append(f"Tree {number} is {done_word}: {certainty}.")
        else:
            if name:
                pct = round(result.confidence * 100)
                words.append(f"Tree {number}: {_certainty_phrase(name, result.confidence)}, about {pct}%.")
            if result.next_evidence:
                nxt = _want_of(result.next_evidence[0])
                self.request_evidence(live_id, nxt)
                words.append(f"Show me the {_want_words(nxt)} of {number}.")
        if words:
            self._emit("speech", {"text": " ".join(words)})
        self._emit_tree_status([live_id])

    def stt_keywords(self) -> list[str]:
        """Speech biasing words: the survey basics plus species in play."""
        words = ["Silvi", "waypoint", "crosshair", "seedling", "canopy", "survey", "boundary"]
        for result in self.world.species.values():
            name = (result or {}).get("taxon")
            if name and str(name) not in words:
                words.append(str(name))
        return words[:40]

    def context_summary(self) -> str:
        packet = self.world.packet(t_mono_ms=self._last_mono, t_wall_ms=self._last_wall_ms)
        return f"{packet}\nRECORDS: {self.survey.summary()}."

    # -- turns ---------------------------------------------------------------
    def _asks_about_unfocused(self, text: str, focus_at_turn) -> bool:
        """A 'this one?' question with nothing in the crosshair."""
        low = (text or "").strip().lower()
        if not low or _EXPLICIT_ID_RE.search(text):
            return False
        if focus_at_turn is not None:
            return False
        deictic = _DEICTIC_RE.search(_normalise_spoken(text))
        question = low.endswith("?") or low.startswith(
            ("what", "is ", "are ", "how", "does", "do ", "can you tell")
        )
        return bool(deictic and question)

    def _species_prepass_note(self, text: str, focus_at_turn=None) -> str | None:
        """For 'what is this?': name the focus tree's species in this turn.

        Runs the identifier before the model is asked anything, so the
        answer can lead with the species instead of reporting that one
        is missing. Returns a system note for the model, or None when
        the question was not about the focus tree's species.
        """
        if not _SPECIES_QUESTION_RE.search(_normalise_spoken(text)):
            return None
        focus = focus_at_turn if focus_at_turn is not None else self.world.focus_id()
        if focus is None:
            return None
        row = self.world.tree_row(focus)
        if row is None:
            return None
        spoken = _spoken_label(row["id"])
        where = _where_phrase(row)
        place = f"{spoken}, {where}" if where else spoken
        existing = row.get("species")
        if existing and existing.get("taxon"):
            return (
                f"The focus tree already has a species result: "
                f"{_certainty_phrase(existing['taxon'], existing.get('confidence'))} "
                f"(rank {existing.get('rank') or 'species'}). "
                f"Lead with that in plain certainty, then say {place}."
            )
        # No species either way below: the surveyor still hears which
        # tree and where, then the honest can't-tell, then a location
        # need in a few words only when there is one.
        need_clause = _location_need_clause(row)
        cant_tell = (
            f"Answer in this shape: 'That's {place}. "
            "I can't tell the species yet.'"
            + (f" Then add, in a few words: {need_clause}" if need_clause else "")
        )
        try:
            result = self._survey_identify_species({"id": row["id"]})
        except Exception:  # noqa: BLE001 - a sick species service still gets an answer
            return "The species check failed just now. " + cant_tell
        species = result.get("species")
        if species and species.get("taxon"):
            return (
                f"Species check just ran for {spoken}: "
                f"{_certainty_phrase(species['taxon'], species.get('confidence'))} "
                f"(rank {species.get('rank') or 'species'}). "
                f"Lead with the species in plain certainty, then say {place}."
            )
        return (
            f"No species answer for {spoken} yet. " + cant_tell
            + " Then ask for a photo of it."
        )

    def _named_tree_note(self, text: str) -> tuple[str | None, str | None, str | None]:
        """Resolve a named tree before the model sees the question.

        Returns (system note, ready-made answer, kind). Merged IDs get
        the live tree's route in the packet for this turn, so the reply
        says where it is now instead of promising to fetch it. Retired
        and unknown numbers get the same treatment. Write utterances
        skip this — their targets are settled in _write_target.
        """
        named = _named_tree_id(text)
        if named is None or allowed_write_tools(text):
            return None, None, None
        state, live = self.world.resolve(named)
        if state == "merged":
            row = self.world.tree_row(live)
            if row is None:
                return None, None, None
            where = _where_phrase(row)
            answer = (
                f"Tree {named} is now tree {live}, {where}."
                if where
                else f"Tree {named} is now tree {live}."
            )
            route = self.world.route_to(live)
            note = (
                f"Tree {named} was merged into tree {live}. Route to it now: "
                f"{json.dumps({k: route.get(k) for k in ('id', 'dist_m', 'clock', 'where')})}. "
                f"Answer in this shape: '{answer}' Do not promise to fetch anything."
            )
            return note, answer, "merged"
        if state == "retired":
            answer = f"Tree {named} was removed by the phone — it is not a plant."
            return answer, answer, "retired"
        if state == "unknown":
            answer = f"I don't know a tree {named} here."
            return answer, answer, "unknown"
        return None, None, None

    def _counts_note(self, text: str) -> tuple[str | None, str | None, int | None]:
        """Logged-count questions answered in code, like merged IDs.

        "So far" means the whole walk; only a question naming the
        boundary, plot or area gets the scoped totals, and says so.
        Returns (system note, ready-made answer, expected confirmed) —
        the answer replaces any spoken count that disagrees with it.
        """
        if not _COUNT_QUESTION_RE.search(_normalise_spoken(text)):
            return None, None, None
        report = self.world.area_report()
        scoped = bool(_SCOPED_COUNT_RE.search(text or "")) and (
            report.get("scope") == "closed boundary"
        )
        if scoped:
            confirmed = int(report["trees_confirmed"])
            not_yet = int(report["trees_tentative"])
            answer = (
                f"Inside the boundary: {confirmed} confirmed, "
                f"{not_yet} not yet confirmed."
            )
            note = (
                f"Counts inside the closed boundary, computed now: {answer} "
                "Use exactly these numbers."
            )
            return note, answer, confirmed
        totals = self.world.survey_totals()
        confirmed = int(totals["trees_confirmed"])
        answer = (
            f"{confirmed} confirmed, "
            f"{int(totals['trees_not_yet_confirmed'])} not yet confirmed so far."
        )
        note = (
            f"Whole-walk logged counts, computed now: {answer} "
            "Use exactly these numbers. The packet's SURVEY line is "
            f"scoped ({report.get('scope')}) — do not read counts from it "
            "for this question, and never add packet lines together."
        )
        return note, answer, confirmed

    def _closest_note(self, text: str) -> tuple[str | None, str | None, int | None]:
        """'Closest' answered in code from straight-line distance.

        Every live tree counts, confirmed or not, nearest first by the
        same ground-plane distance the packet's NEAREST line uses —
        the model picking from AROUND YOU (ahead first) named farther
        trees in live runs, so the answer is computed here and any
        spoken tree number that disagrees is replaced by it.
        """
        if not re.search(r"\b(closest|nearest)\b", _normalise_spoken(text)):
            return None, None, None
        rows = [
            row
            for row in (self.world.tree_row(i) for i in self.world.live_ids())
            if row is not None and row.get("dist_m") is not None
        ]
        if not rows:
            return None, None, None
        best = min(rows, key=lambda r: (float(r["dist_m"]), str(r["id"])))
        number = int(str(best["id"]).lstrip("T"))
        where = _where_phrase(best)
        answer = f"Tree {number}, {where}" if where else f"Tree {number}"
        if best.get("status") != "confirmed":
            answer += ", not confirmed yet"
        answer += "."
        note = (
            f"Closest live tree, computed now: {answer} "
            "Use exactly this tree and distance; do not pick from AROUND YOU."
        )
        return note, answer, number

    @staticmethod
    def _counts_disagree(spoken: str, expected_confirmed: int) -> bool:
        numbers = re.findall(r"\d+", spoken or "")
        return not numbers or int(numbers[0]) != expected_confirmed

    def handle_turn(self, text: str, *, speech_end_at: float | None = None) -> list[dict]:
        self._turn_n += 1
        self._last_turn_text = text
        trace = self.tracer.start_turn(
            f"turn-{self._turn_n}", self.model_label, at=speech_end_at
        )
        # One focus rule for the whole turn: focus_id() as the
        # utterance arrives, captured before any model round. The
        # no-focus check, the species prepass and every write target
        # all read this one value, never a fresh lookup of their own.
        # Wire v2: the phone can name the crosshair plant for this
        # utterance (transcript.focus / voice_start.focus). When it
        # does, that plant — merges resolved — is the turn's focus;
        # otherwise the computed crosshair stands.
        wire_focus = getattr(self, "_focus_override", None)
        self._focus_override = None
        focus_at_turn = self.world.focus_id()
        if wire_focus is not None:
            state, live_id = self.world.resolve(int(wire_focus))
            if state in ("live", "merged") and live_id in self.world.snapshots:
                focus_at_turn = live_id
        if self._asks_about_unfocused(text, focus_at_turn):
            # Nothing is in the crosshair, so there is no honest answer
            # from another tree's notes. Ask for the crosshair instead,
            # without spending a model call on it.
            trace.t_model_start = time.perf_counter()
            trace.t_first_reply = trace.t_model_start
            trace.notes.update(no_focus=True)
            spoken_text = "Point the crosshair at the tree you mean, then ask me again."
            emitted = [self._emit("speech", {"text": spoken_text})]
            self._history.append({"role": "user", "content": text})
            self._history.append({"role": "assistant", "content": spoken_text})
            trace.t_done = time.perf_counter()
            return emitted
        species_note = self._species_prepass_note(text, focus_at_turn)
        named_note, named_answer, named_kind = self._named_tree_note(text)
        if named_kind:
            trace.notes.update(named_tree=named_kind)
        counts_note, counts_answer, counts_expected = self._counts_note(text)
        if counts_answer:
            trace.notes.update(counts_answer=counts_answer)
        closest_note, closest_answer, closest_number = self._closest_note(text)
        if closest_answer:
            trace.notes.update(closest_answer=closest_answer)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": self.context_summary()},
            *([{"role": "system", "content": species_note}] if species_note else []),
            *([{"role": "system", "content": named_note}] if named_note else []),
            *([{"role": "system", "content": counts_note}] if counts_note else []),
            *([{"role": "system", "content": closest_note}] if closest_note else []),
            *self._history[-6:],
            {"role": "user", "content": text},
        ]
        trace.t_model_start = time.perf_counter()

        def _first_token(_chunk: str) -> None:
            if trace.t_first_reply is None:
                trace.t_first_reply = time.perf_counter()

        emitted: list[dict] = []
        call_latencies: list[float] = []
        deadline_hit = [False]
        holding_done = [False]
        model_rounds = [0]
        fallbacks_before = len(getattr(self._client, "fallback_events", []) or [])

        def _call_once(msgs, *, with_tools: bool, stream_ok: bool):
            tools = tool_definitions() if with_tools else None
            stream = getattr(self._client, "complete_stream", None) if stream_ok else None
            if stream is not None:
                out = stream(msgs, tools=tools, on_token=_first_token)
                trace.notes.update(
                    streamed=True, first_token_ms=getattr(out, "first_token_ms", None)
                )
                return out
            return self._client.complete(msgs, tools=tools)

        def _attempt(msgs, *, with_tools: bool, stream_ok: bool):
            started = time.perf_counter()
            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(
                _call_once, msgs, with_tools=with_tools, stream_ok=stream_ok
            )
            try:
                return future.result(timeout=self.turn_deadline_s)
            except FutureTimeoutError as exc:
                raise TimeoutError(
                    f"model call passed the {self.turn_deadline_s:.0f} s deadline"
                ) from exc
            finally:
                call_latencies.append(round((time.perf_counter() - started) * 1000.0, 1))
                pool.shutdown(wait=False)

        def call_model(msgs, *, with_tools: bool):
            """One model round, deadline-bound. None means: apologise.

            A missed deadline speaks one holding line, retries once
            without streaming, and gives up audibly rather than leaving
            the surveyor in silence. A provider failure of any kind —
            HTTP error, connection break, broken stream — takes the
            same road: the client's configured fallback has its one
            hop inside the attempt, and if the attempt still fails the
            turn apologises. Nothing provider-shaped ever escapes
            handle_turn.
            """
            model_rounds[0] += 1
            try:
                return _attempt(msgs, with_tools=with_tools, stream_ok=True)
            except TimeoutError:
                deadline_hit[0] = True
                if not holding_done[0]:
                    holding_done[0] = True
                    emitted.append(self._emit("speech", {"text": "One moment."}))
                try:
                    return _attempt(msgs, with_tools=with_tools, stream_ok=False)
                except Exception as exc:  # noqa: BLE001 - spoken, not raised
                    trace.notes.update(provider_failure=str(exc))
                    return None
            except Exception as exc:  # noqa: BLE001 - spoken, not raised
                trace.notes.update(provider_failure=str(exc))
                return None

        def repair(name, args, error, schema):
            try:
                fixed = self._client.complete(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Return only a JSON object: valid arguments for the tool. "
                                "No prose, no code fences."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Tool: {name}\nSchema: {json.dumps(schema)}\n"
                                f"Bad arguments: {json.dumps(args, default=str)}\nError: {error}"
                            ),
                        },
                    ]
                )
            except Exception as exc:  # noqa: BLE001 - a failed repair drops the call
                trace.notes.update(provider_failure=str(exc))
                return None
            return _parse_json_object(fixed.text)

        # Tool loop: at most two model rounds. Round one may use tools;
        # round two never does. A successful write is confirmed in code
        # from its tool result and needs no second round unless the
        # utterance also asked a question; a query-only or blocked first
        # round gets exactly one tool-free answering round.
        final_text = ""
        total_calls = total_accepted = total_repaired = 0
        allowed_writes = allowed_write_tools(text)
        successes: list[str] = []
        confirmations: list[str] = []
        write_targets: list[dict] = []
        write_attempted = False
        no_focus_block = False
        for round_n in range(2):
            result = call_model(messages, with_tools=round_n == 0)
            if result is None:
                final_text = "Sorry, I lost that — say it again?"
                break
            if getattr(result, "reasoning_only", False):
                # All thought, nothing said: an empty reply. The
                # reasoning itself is never carried into speech.
                trace.notes.update(reasoning_only=True)
            if trace.t_first_reply is None:
                trace.t_first_reply = time.perf_counter()
            if round_n == 1:
                final_text = result.text or ""
                break
            accepted, dropped = check_tool_calls(result.tool_calls, repair=repair)
            self.dropped.extend(dropped)
            total_calls += len(result.tool_calls)
            total_accepted += len(accepted)
            total_repaired += sum(1 for a in accepted if a["repaired"])
            if not accepted:
                if self._code_note_on_focus(
                    text, focus_at_turn, allowed_writes, successes,
                    confirmations, write_targets,
                ):
                    write_attempted = True
                    emitted.append(self._emit("speech", {"text": confirmations[0]}))
                    final_text = ""
                else:
                    final_text = result.text or ""
                break
            survey_flags = [self._is_survey_call(c["name"], c["arguments"]) for c in accepted]
            blocked_any = False
            say_texts: list[str] = []
            tool_messages: list[dict] = []
            assistant_calls = []
            for call, is_survey in zip(accepted, survey_flags):
                name = call["name"]
                args = dict(call["arguments"])
                result_dict: dict = {"ok": True}
                wires: list[dict] = []
                if name == "say":
                    # Staged, not spoken yet: a claim in here is checked
                    # against what this turn actually did, below.
                    say_texts.append(str(args.get("text") or ""))
                elif name in _WRITE_TOOLS:
                    write_attempted = True
                    if name not in allowed_writes:
                        blocked_any = True
                        reason = (
                            f"blocked: this utterance does not ask for {name}. "
                            f"Its words allow {sorted(allowed_writes) or 'no record change'}. "
                            "Records change only with the tool the surveyor's "
                            "own words call for; answer without writing."
                        )
                        self.dropped.append(
                            {"id": call.get("id"), "name": name, "reason": reason}
                        )
                        write_targets.append(
                            {"tool": name, "target": None, "reason": "blocked_wrong_tool"}
                        )
                        result_dict = {"error": reason}
                    else:
                        args, target, why = self._write_target(
                            name, args, text, focus_at_turn
                        )
                        write_targets.append(
                            {"tool": name, "target": target, "reason": why}
                        )
                        if why == "blocked_no_focus":
                            no_focus_block = True
                            result_dict = {
                                "error": "blocked: no tree in the crosshair to write on"
                            }
                        elif is_survey:
                            result_dict, wires = self._execute_survey(name, args)
                            if "error" not in result_dict:
                                successes.append(name)
                                conf = self._confirmation_text(name, args, result_dict)
                                if conf:
                                    confirmations.append(conf)
                        else:
                            wires = self._execute(name, args)
                            result_dict = {
                                "ok": True,
                                "emitted": [w["kind"] for w in wires],
                            }
                            spoken_bits = [
                                str(w["payload"].get("text") or "")
                                for w in wires
                                if w["kind"] == "speech"
                            ]
                            failed = any(
                                bit.startswith(
                                    ("I do not", "I could not", "That correction did not")
                                )
                                for bit in spoken_bits
                            )
                            if wires and not failed:
                                successes.append(name)
                elif is_survey:
                    result_dict, wires = self._execute_survey(name, args)
                else:
                    wires = self._execute(name, args)
                    result_dict = {"ok": True, "emitted": [w["kind"] for w in wires]}
                emitted.extend(wires)
                call_id = str(call.get("id") or f"call-{round_n}-{len(tool_messages)}")
                assistant_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result_dict, default=str),
                    }
                )
            if not successes and not no_focus_block and self._code_note_on_focus(
                text, focus_at_turn, allowed_writes, successes,
                confirmations, write_targets,
            ):
                write_attempted = True
            if confirmations:
                # The confirmation is the code's sentence, built from the
                # tool result; any say text from the model is replaced.
                emitted.append(self._emit("speech", {"text": confirmations[0]}))
            elif say_texts:
                replaced_claim = False
                for say_text in say_texts:
                    if closest_answer and str(closest_number) not in re.findall(
                        r"\d+", say_text
                    ):
                        trace.notes.update(closest_corrected=True)
                        if not replaced_claim:
                            emitted.append(
                                self._emit("speech", {"text": closest_answer})
                            )
                            replaced_claim = True
                    elif counts_answer and self._counts_disagree(
                        say_text, counts_expected
                    ):
                        trace.notes.update(counts_corrected=True)
                        if not replaced_claim:
                            emitted.append(
                                self._emit("speech", {"text": counts_answer})
                            )
                            replaced_claim = True
                    elif _PROMISE_RE.search(say_text):
                        # A promise is not an answer; the ready-made
                        # one (or an honest don't-know) goes out instead.
                        trace.notes.update(promise_replaced=True)
                        if not replaced_claim:
                            emitted.append(
                                self._emit(
                                    "speech", {"text": named_answer or "I don't know."}
                                )
                            )
                            replaced_claim = True
                    elif _CLAIM_RE.search(say_text) and write_attempted and not successes:
                        if not replaced_claim:
                            emitted.append(
                                self._emit("speech", {"text": "I didn't change anything."})
                            )
                            replaced_claim = True
                    elif say_text.strip():
                        emitted.append(self._emit("speech", {"text": say_text.strip()}))
            if no_focus_block:
                final_text = "Point the crosshair at the tree you mean, then ask me again."
                break
            if successes and not _utterance_is_question(text):
                final_text = ""
                break
            if not any(survey_flags) and not blocked_any and not successes:
                final_text = result.text or ""
                break
            messages = [
                *messages,
                {"role": "assistant", "content": result.text or "", "tool_calls": assistant_calls},
                *tool_messages,
            ]
        fallbacks_after = len(getattr(self._client, "fallback_events", []) or [])
        trace.notes.update(
            tool_calls=total_calls,
            accepted=total_accepted,
            dropped=len(self.dropped),
            repaired=total_repaired,
            write_allowed=bool(allowed_writes),
            allowed_writes=sorted(allowed_writes),
            write_targets=write_targets,
            confirmations=confirmations,
            model_rounds=model_rounds[0],
            call_latencies_ms=call_latencies,
            deadline_hit=deadline_hit[0],
            fallback_events=(getattr(self._client, "fallback_events", []) or [])[
                fallbacks_before:fallbacks_after
            ],
        )

        if (
            closest_answer
            and final_text
            and str(closest_number) not in re.findall(r"\d+", final_text)
        ):
            trace.notes.update(closest_corrected=True)
            final_text = closest_answer
        if counts_answer and final_text and self._counts_disagree(
            final_text, counts_expected
        ):
            # Counts are read from the code's numbers or not at all.
            trace.notes.update(counts_corrected=True)
            final_text = counts_answer
        if final_text and _PROMISE_RE.search(final_text):
            # The last round has no tools left to keep a promise with.
            trace.notes.update(promise_replaced=True)
            final_text = named_answer or "I don't know."
        if (
            final_text
            and _CLAIM_RE.search(final_text)
            and write_attempted
            and not successes
        ):
            # Never speak an action this turn did not perform.
            final_text = "I didn't change anything."
        spoken = [m for m in emitted if m["kind"] == "speech"]
        if final_text and final_text.strip():
            said = [m["payload"].get("text", "") for m in spoken]
            if not spoken:
                emitted.insert(0, self._emit("speech", {"text": final_text.strip()}))
                spoken = emitted[:1]
            elif (confirmations or holding_done[0]) and final_text.strip() not in said:
                # Confirmation (or the holding line) first, then the
                # actual answer the model eventually produced.
                emitted.append(self._emit("speech", {"text": final_text.strip()}))
                spoken = [m for m in emitted if m["kind"] == "speech"]
        if not emitted:
            emitted.append(self._emit("speech", {"text": _FALLBACK}))
            spoken = emitted

        self._history.append({"role": "user", "content": text})
        self._history.append(
            {"role": "assistant", "content": spoken[0]["payload"]["text"] if spoken else ""}
        )
        trace.t_done = time.perf_counter()
        return emitted

    _TARGET_ARG: ClassVar[dict[str, str]] = {
        "add_note": "id",
        "correct_record": "local_id",
        "record_growth": "local_id",
    }

    def _write_target(self, name, args, text, focus_at_turn):
        """Decide, in code, which tree a write lands on.

        A named tree in the utterance wins (merges resolved for survey
        ids). A deictic utterance ("this one") goes to the focus tree
        captured when the turn began — the model's own ID is ignored,
        which is the whole point — or is blocked when there is no
        focus. Anything else keeps the model's id and says so.
        Returns (args, target_label, reason).
        """
        key = self._TARGET_ARG.get(name)
        if key is None:
            return args, None, "untargeted"
        named = _named_tree_id(text)
        if named is not None:
            if name == "add_note":
                _state, live = self.world.resolve(named)
                args[key] = tree_label(live)
                return args, args[key], "named"
            args[key] = f"tree-{named}"
            return args, args[key], "named"
        if _DEICTIC_RE.search(_normalise_spoken(text)):
            if focus_at_turn is None:
                return args, None, "blocked_no_focus"
            if name == "add_note":
                args[key] = tree_label(focus_at_turn)
                return args, args[key], "focus"
            mapped = self._plant_records.get(int(focus_at_turn))
            if mapped:
                args[key] = mapped
                return args, mapped, "focus"
            # No stored record backs the focus tree yet: name it by its
            # walk number, which the record-side tools resolve through
            # the snapshot. The model's stale id never gets through.
            args[key] = f"tree-{int(focus_at_turn)}"
            return args, args[key], "focus"
        return args, args.get(key), "model"

    def _code_note_on_focus(
        self, text, focus_at_turn, allowed_writes, successes, confirmations, write_targets
    ) -> bool:
        """Note the surveyor's own words on the focus tree, in code.

        Fires only when the utterance explicitly asked for a note about
        "this one", no write has landed this turn, and a focus tree was
        captured as the utterance arrived. In live runs the model
        sometimes answered such turns by asking for the crosshair —
        above all on contracted forms — and the words were lost.
        """
        if successes or focus_at_turn is None:
            return False
        if "add_note" not in allowed_writes or _named_tree_id(text) is not None:
            return False
        normalised = _normalise_spoken(text)
        if not _DEICTIC_RE.search(normalised):
            return False
        if not re.search(r"\b(note|remember|flag)\b", normalised):
            return False
        note_text = re.sub(
            r"[\s,.;]*\bnote that\b.*$", "", text.strip(), flags=re.IGNORECASE
        ).strip(" ,.;") or text.strip()
        note_args = {"id": tree_label(focus_at_turn), "text": note_text}
        note_result, _wires = self._execute_survey("add_note", note_args)
        if "error" in note_result:
            return False
        successes.append("add_note")
        write_targets.append(
            {"tool": "add_note", "target": note_args["id"], "reason": "focus_code_note"}
        )
        conf = self._confirmation_text("add_note", note_args, note_result)
        if conf:
            confirmations.append(conf)
        return True

    @staticmethod
    def _confirmation_text(name: str, args: dict, result: dict) -> str | None:
        """The spoken receipt for a write, built from its tool result."""
        if "error" in result:
            return None
        if name == "add_note":
            note = " ".join(str(args.get("text") or "").split()[:5])
            where = _spoken_label(result.get("id") or args.get("id") or "")
            return f"Noted on {where}: {note}." if note else f"Noted on {where}."
        if name == "boundary":
            state = result.get("boundary_state")
            action = str(args.get("action") or "")
            if state == "closed":
                inside = result.get("trees_inside") or []
                return (
                    f"Boundary closed. {result.get('confirmed_inside')} confirmed "
                    f"of {len(inside)} trees inside."
                )
            if action == "start" and state == "recording":
                return "Boundary recording started."
            if action == "pause" and state == "paused":
                return "Boundary paused."
            if action == "resume" and state == "recording":
                return "Boundary recording again."
            return None
        return None


    # -- tool execution (shared by voice turns and tap intents) -------------
    @staticmethod
    def _is_survey_call(name: str, args: dict) -> bool:
        if name in _SURVEY_TOOLS:
            return True
        if name == "identify_species":
            return "id" in args and "local_id" not in args
        if name == "mark_waypoint":
            return "id" in args
        return False

    def _execute(self, name: str, args: dict) -> list[dict]:
        if self._is_survey_call(name, args):
            _result, wires = self._execute_survey(name, args)
            if wires:
                return wires
            return [self._emit("speech", {"text": self._survey_text(name, _result)})]
        if name in _DIRECT_WIRE_TOOLS:
            return [self._emit(_TOOL_TO_WIRE[name], args)]
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise ValueError(f"unknown tool: {name}")
        return handler(args)

    @staticmethod
    def _spoken_tree(row: dict) -> str:
        parts = [_spoken_label(row["id"]).capitalize()]
        species = row.get("species")
        if species and species.get("taxon"):
            parts.append(_certainty_phrase(species["taxon"], species.get("confidence")))
        if row.get("dist_m") is not None:
            parts.append(f"about {float(row['dist_m']):.0f} metres at {row['clock']} o'clock")
        if row.get("status") != "confirmed":
            parts.append("not confirmed yet")
        needs = [n for n in row.get("needs") or [] if n not in ("unconfirmed", "no_species")]
        if needs:
            parts.append(_PLAIN_NEEDS.get(needs[0], needs[0]))
        return ", ".join(parts) + "."

    @classmethod
    def _survey_text(cls, name: str, result: dict) -> str:
        if "error" in result:
            return str(result["error"])
        if name == "tree" and "id" in result:
            return cls._spoken_tree(result)
        if name == "area_report":
            return (
                f"{result.get('trees_confirmed')} confirmed, "
                f"{result.get('confirmed_complete')} complete, "
                f"{result.get('trees_tentative')} not confirmed yet in {result.get('scope')}."
            )
        return json.dumps(result, default=str)[:300]

    def _resolve_survey_tree(self, raw) -> tuple[dict | None, dict | None]:
        """-> (row, None) for a live tree, or (None, result_dict) to return."""
        plant_id = parse_tree_id(raw)
        if plant_id is None:
            return None, {"error": f"could not read a tree id from {raw!r}"}
        state, live = self.world.resolve(plant_id)
        label = tree_label(plant_id)
        if state == "unknown":
            return None, {"error": f"no tree called {label}"}
        if state == "retired":
            return None, {
                "id": label,
                "status": "retired",
                "message": f"{label} was removed by the phone (not a plant)",
            }
        row = self.world.tree_row(live)
        if row is None:
            return None, {"error": f"no tree called {label}"}
        if state == "merged":
            row = {**row, "requested": label, "merged_note": f"{label} was merged into {tree_label(live)}"}
        return row, None

    @staticmethod
    def _slim_row(row: dict) -> dict:
        return {k: v for k, v in row.items() if k not in ("position_m", "angle_deg")}

    def _execute_survey(self, name: str, args: dict) -> tuple[dict, list[dict]]:
        """Run one survey query/action tool. Never raises for a bad lookup:
        failures come back as {"error": ...} for the model."""
        try:
            if name == "scene":
                radius = float(args.get("radius_m") or 12.0)
                rows = [self._slim_row(r) for r in self.world.scene_rows(radius)]
                focus = self.world.focus_id()
                return {
                    "focus": tree_label(focus) if focus is not None else None,
                    "trees": rows,
                }, []
            if name == "tree":
                row, problem = self._resolve_survey_tree(args.get("id"))
                return (problem if row is None else self._slim_row(row)), []
            if name == "request_evidence":
                row, problem = self._resolve_survey_tree(args.get("id"))
                if row is None:
                    return problem, []
                live_id = parse_tree_id(row["id"])
                want = str(args.get("want") or "whole")
                made = self.request_evidence(int(live_id), want)
                return {
                    "ok": True,
                    "id": row["id"],
                    "number": self.spoken_number(int(live_id)),
                    "want": want,
                    "request_id": made["request_id"],
                }, []
            if name == "find_trees":
                limit = int(args.get("limit") or 5)
                found = []
                for live_id in self.world.live_ids():
                    row = self.world.tree_row(live_id)
                    if row is None:
                        continue
                    if args.get("need") and args["need"] not in row["needs"]:
                        continue
                    if args.get("status") and row["status"] != args["status"]:
                        continue
                    if args.get("species"):
                        taxon = ((row.get("species") or {}).get("taxon") or "").lower()
                        if str(args["species"]).lower() not in taxon:
                            continue
                    found.append(row)
                found.sort(key=lambda r: (r["dist_m"] is None, r["dist_m"] or 0.0))
                return {"trees": [self._slim_row(r) for r in found[:limit]]}, []
            if name == "route_to":
                row, problem = self._resolve_survey_tree(args.get("id"))
                if row is None:
                    return problem, []
                result = self.world.route_to(parse_tree_id(row["id"]))
                if "merged_note" in row:
                    result = {**result, "merged_note": row["merged_note"], "requested": row["requested"]}
                return result, []
            if name == "area_report":
                return self.world.area_report(), []
            if name == "add_note":
                row, problem = self._resolve_survey_tree(args.get("id"))
                if row is None:
                    return problem, []
                live = parse_tree_id(row["id"])
                self.world.add_note(live, str(args["text"]))
                result = {"ok": True, "id": row["id"], "note": str(args["text"])}
                if "merged_note" in row:
                    result["merged_note"] = row["merged_note"]
                return result, []
            if name == "boundary":
                result = self.world.boundary_action(str(args["action"]))
                wires: list[dict] = []
                if result.get("boundary_state") == "closed" and result.get("vertices_m"):
                    wires.append(
                        self._emit(
                            "polygon",
                            {"vertices_m": result["vertices_m"], "label": "walked boundary"},
                        )
                    )
                    result = {k: v for k, v in result.items() if k != "vertices_m"}
                return result, wires
            if name == "mark_waypoint":
                row, problem = self._resolve_survey_tree(args.get("id"))
                if row is None:
                    return problem, []
                snap = self.world.snapshots[parse_tree_id(row["id"])]
                position = snap.get("position_m")
                if not position:
                    return {"error": f"{row['id']} has no stored position yet"}, []
                wire = self._emit(
                    "waypoint", {"position_m": [float(c) for c in position], "label": row["id"]}
                )
                return {"ok": True, "id": row["id"], "dist_m": row["dist_m"], "clock": row["clock"]}, [wire]
            if name == "identify_species":
                return self._survey_identify_species(args), []
        except Exception as exc:  # noqa: BLE001 - a failed lookup is data for the model
            return {"error": f"{name} failed: {exc}"}, []
        return {"error": f"unknown survey tool {name}"}, []

    def _plug_species_for(self, live_id: int) -> dict | None:
        """The plug's current answer for a plant, mirrored into the world."""
        existing = self.world.species.get(live_id)
        if existing and existing.get("taxon"):
            return existing
        state = self.species_plug.state(
            survey_id=self.survey.walk_id or self._session_id, plant_id=int(live_id)
        )
        if state is None or not state.candidates:
            return None
        top = state.candidates[0]
        species = {
            "taxon": top.name,
            "common_name": top.common_name,
            "rank": top.rank,
            "confidence": state.confidence,
            "done": state.done,
            "next_evidence": list(state.next_evidence),
            "candidates": [c.__dict__ for c in state.candidates],
        }
        self.world.set_species(live_id, species)
        return species

    def _copy_plug_species_to_records(self) -> None:
        """At session end, put each backed plant's plug answer on its record."""
        for plant_id, local_id in list(self._plant_records.items()):
            state = self.world.species.get(plant_id)
            if not state or not state.get("taxon"):
                continue
            if self.survey.find_record(local_id) is None:
                continue
            try:
                self.survey.attach_identification(local_id, _identification_from(state))
            except Exception:  # noqa: BLE001, S112 - a bad name must not lose the walk
                continue

    def _survey_identify_species(self, args: dict) -> dict:
        row, problem = self._resolve_survey_tree(args.get("id"))
        if row is None:
            return problem
        live = parse_tree_id(row["id"])
        species = self._plug_species_for(live)
        if species is None:
            return {
                "id": row["id"],
                "species": None,
                "note": "I can't tell the species yet; a photo of it would help",
            }
        return {"id": row["id"], "species": species}

    def _tool_create_site(self, args: dict) -> list[dict]:
        site = self.survey.create_site(str(args["name"]))
        return [self._emit("speech", {"text": f"Site {site['name']} created."})]

    def _tool_create_boundary(self, args: dict) -> list[dict]:
        entity = self.survey.create_boundary(str(args["label"]), args["vertices_m"])
        return [self._emit("polygon", {"vertices_m": entity["vertices_m"], "label": entity["label"]})]

    def _tool_create_plot(self, args: dict) -> list[dict]:
        entity = self.survey.create_plot(str(args["label"]), args["vertices_m"])
        return [self._emit("polygon", {"vertices_m": entity["vertices_m"], "label": entity["label"]})]

    def _tool_create_planting(self, args: dict) -> list[dict]:
        position = args["position_m"]
        record = self.survey.create_planting(
            position,
            str(args["label"]),
            observed_at=self._now_iso(),
            taxon=args.get("taxon"),
            sigma_m=args.get("sigma_m"),
        )
        self.coverage.add_point(position)
        return [
            self._emit(
                "waypoint",
                {"position_m": record["position"]["xyz_m"], "label": str(args["label"])},
            )
        ]

    def _tool_create_observation(self, args: dict) -> list[dict]:
        position = args["position_m"]
        record = self.survey.create_observation(
            position,
            str(args["label"]),
            observed_at=self._now_iso(),
            sigma_m=args.get("sigma_m"),
            establishment=str(args.get("establishment") or "unknown"),
        )
        self.coverage.add_point(position)
        return [
            self._emit(
                "waypoint",
                {"position_m": record["position"]["xyz_m"], "label": str(args["label"])},
            )
        ]

    def _tool_add_corner(self, args: dict) -> list[dict]:
        count = self.survey.add_corner(args["position_m"])
        return [
            self._emit(
                "waypoint",
                {"position_m": [float(c) for c in args["position_m"]], "label": f"corner {count}"},
            )
        ]

    def _tool_close_polygon(self, args: dict) -> list[dict]:
        try:
            entity = self.survey.close_polygon(
                str(args["label"]), kind=str(args.get("kind") or "plot")
            )
        except ValueError as exc:
            return [self._emit("speech", {"text": f"Cannot close the polygon yet: {exc}."})]
        return [self._emit("polygon", {"vertices_m": entity["vertices_m"], "label": entity["label"]})]

    def _stored_target(self, local_id: str):
        record = self.survey.find_record(local_id)
        if record and record["position"]["xyz_m"]:
            return record["identity"]["local_id"], record["position"]["xyz_m"]
        wanted = str(local_id).strip().lower()
        for plant_id, snap in self._plants.items():
            if wanted in (str(plant_id), f"plant-{plant_id}", f"tree-{plant_id}") and snap.get(
                "position_m"
            ):
                return f"plant {plant_id}", snap["position_m"]
        return None

    def _tool_guide_to_plant(self, args: dict) -> list[dict]:
        target = self._stored_target(str(args["local_id"]))
        if target is None or self._pose is None:
            return [
                self._emit(
                    "speech",
                    {"text": "I do not have a stored position for that plant yet."},
                )
            ]
        label, position = target
        self._guided_request = str(args["local_id"]).strip().lower()
        guide = guidance(self._pose, self._quat, position)
        return [
            self._emit("waypoint", {"position_m": [float(c) for c in position], "label": label}),
            self._emit("speech", {"text": guide.spoken(label)}),
        ]

    def load_stored(
        self, *, site: dict | None = None, records=(), plants=(), backed_by=None
    ) -> None:
        """Seed a revisit session with stored records/plants (second visit).

        ``plants`` are wire plant_snapshot payloads already expressed in the
        *current* walk's frame by the caller; matching across walks is
        the caller's evidence problem, never assumed here. ``backed_by`` is the
        earlier visit's promotion map (local_id -> plant_id): it links a
        stored record to the plant identity that backs it, so growth said
        against the plant ("tree 202") lands on that record.
        """
        if site is not None:
            self.survey.site = dict(site)
        for record in records:
            stored = copy.deepcopy(record)
            validate_observation(stored)
            self.survey.records.append(stored)
            self._stored_local_ids.append(stored["identity"]["local_id"])
            # Keep new-record ids from colliding with loaded ones.
            tail = stored["identity"]["local_id"].rsplit("-", 1)[-1]
            if tail.isdigit():
                self.survey._seq = max(self.survey._seq, int(tail))
        for snap in plants:
            self._plants[int(snap["plant_id"])] = snap
            self.world.add_snapshot(snap)
        for local_id, plant_id in (backed_by or {}).items():
            self._plant_records[int(plant_id)] = str(local_id)

    def _plant_id_of(local_id: str) -> int | None:
        match = re.fullmatch(r"(?:plant-|tree-)?(\d+)", str(local_id).strip().lower())
        return int(match.group(1)) if match else None

    def _record_for(self, local_id: str) -> dict | None:
        """Record named directly, or via its backing plant's id."""
        record = self.survey.find_record(local_id)
        if record is not None:
            return record
        plant_id = self._plant_id_of(local_id)
        mapped = self._plant_records.get(plant_id) if plant_id is not None else None
        return self.survey.find_record(mapped) if mapped else None

    def _tool_record_growth(self, args: dict) -> list[dict]:
        local_id = str(args["local_id"])
        record = self._record_for(local_id)
        # No identity inference here: growth lands on the record named
        # directly or through its explicit backing map. Being guided to
        # a plant, or having exactly one stored record, does not make
        # that plant this record.
        if record is not None:
            local_id = record["identity"]["local_id"]
        if record is None:
            target = self._stored_target(local_id)
            if target is None:
                return [
                    self._emit(
                        "speech",
                        {"text": "I do not have a stored plant by that name yet."},
                    )
                ]
            label, position = target
            record = self.survey.create_observation(
                [float(c) for c in position],
                f"{label} revisit",
                observed_at=self._now_iso(),
            )
            local_id = record["identity"]["local_id"]
        record = self.survey.record_growth(
            local_id, height_m=args.get("height_m"), dbh_cm=args.get("dbh_cm")
        )
        m = record["measurements"]
        return [
            self._emit(
                "speech",
                {
                    "text": (
                        f"Growth recorded for {record['identity']['local_id']}: "
                        f"height {m['height_m']} m, trunk {m['dbh_cm']} cm."
                    )
                },
            )
        ]

    def _tool_identify_species(self, args: dict) -> list[dict]:
        local_id = str(args["local_id"])
        target = self._stored_target(local_id)
        record = self.survey.find_record(local_id)
        if record is None and target is None:
            return [
                self._emit(
                    "speech",
                    {"text": "I do not have a stored plant by that name yet."},
                )
            ]
        spoken = _spoken_label(local_id)
        plant_id = self._plant_id_of(local_id)
        species = self._plug_species_for(plant_id) if plant_id is not None else None
        if species is None or not species.get("taxon"):
            return [
                self._emit(
                    "speech",
                    {"text": f"That's {spoken}. I can't tell the species yet "
                             "— a photo of it would help."},
                )
            ]
        if record is None:
            label, position = target  # type: ignore[misc]
            record = self.survey.create_observation(
                [float(c) for c in position],
                f"{label} species",
                observed_at=self._now_iso(),
            )
        record = self.survey.attach_identification(
            record["identity"]["local_id"], _identification_from(species)
        )
        spoken = _spoken_label(record["identity"]["local_id"])
        phrase = _certainty_phrase(species["taxon"], species.get("confidence"))
        rank = species.get("rank") or "species"
        text = f"{phrase.capitalize()} — {spoken}." if rank == "species" else f"{phrase.capitalize()} ({rank} only) — {spoken}."
        return [self._emit("speech", {"text": text})]

    def _tool_correct_record(self, args: dict) -> list[dict]:
        from field_agent.corrections import make_correction

        local_id = str(args["local_id"])
        record = self._record_for(local_id)
        if record is None:
            target = self._stored_target(local_id)
            if target is None:
                return [
                    self._emit(
                        "speech",
                        {"text": "I do not have a stored plant by that name yet."},
                    )
                ]
            label, position = target
            record = self.survey.create_observation(
                [float(c) for c in position],
                f"{label} record",
                observed_at=self._now_iso(),
            )
        source = str(args.get("source") or "voice")
        try:
            entry = make_correction(
                self.survey.next_correction_id(),
                record_ref=record["identity"]["local_id"],
                operation=str(args["operation"]),
                value=args.get("value"),
                other_ref=args.get("other_local_id"),
                who=str(args.get("who") or "walker"),
                source=source,
                at=self._now_iso(),
                transcript=self._last_turn_text if source == "voice" else None,
            )
            stored = self.survey.apply_correction(entry)
        except ValueError as exc:
            return [self._emit("speech", {"text": f"That correction did not land: {exc}."})]
        spoken = {
            "exclude": f"Ruled out: {stored['record_ref']} stays in the history, marked not a plant.",
            "split": (
                f"Split noted: {stored['record_ref']} now has two child records "
                f"({', '.join(stored.get('child_refs') or [])}), both awaiting review."
            ),
            "merge_hint": f"Noted: {stored['record_ref']} may be the same tree as {stored.get('other_ref')}; linked for review.",
            "mark_planted": f"Marked: a person planted {stored['record_ref']}.",
            "set_species": f"Named by a person: {stored['record_ref']} is {stored.get('value')}; the model's suggestion stays on file.",
            "note": f"Note added to {stored['record_ref']}.",
        }[stored["operation"]]
        return [self._emit("speech", {"text": spoken})]

    def _tool_report_gaps(self, args: dict) -> list[dict]:
        boundary = None
        if self.survey.boundaries:
            boundary = [(v[0], v[2]) for v in self.survey.boundaries[-1]["vertices_m"]]
        gaps = plan_gaps(self.coverage, boundary)
        if not gaps:
            return [self._emit("speech", {"text": "No unsurveyed gaps in the walked area."})]
        emitted = [
            self._emit(
                "gap",
                {
                    "position_m": gap.center_xyz_m(self._pose[1] if self._pose else 0.0),
                    "radius_m": max(gap.radius_m, 0.5),
                    "reason": f"unseen ground: {gap.n_cells} cells, {gap.area_m2:.0f} m2",
                },
            )
            for gap in gaps[:3]
        ]
        emitted.append(
            self._emit(
                "speech",
                {"text": f"{len(gaps)} unsurveyed gap{'s' if len(gaps) != 1 else ''}; largest marked."},
            )
        )
        return emitted

    # -- structured intents (tap / crosshair path) ---------------------------
    def handle_intent(self, tool_name: str, arguments: dict) -> dict:
        """Apply a tool without voice or a model. Raises ValueError if invalid.

        Returns the first wire message emitted (survey tools may emit more;
        all are in ``self.outputs``).
        """
        accepted, reason, _ = check_tool_call(tool_name, arguments)
        if accepted is None:
            raise ValueError(f"intent {tool_name} rejected: {reason}")
        emitted = self._execute(tool_name, accepted)
        return emitted[0]

    def _emit(self, kind: str, payload: dict) -> dict:
        message = {
            "v": "1",
            "kind": kind,
            "seq": self._out_seq,
            "session_id": self._session_id,
            "t_mono_ms": self._last_mono,
            "t_wall_ms": None,
            "payload": payload,
        }
        validate_wire(message)
        self._out_seq += 1
        self.outputs.append(message)
        return message
