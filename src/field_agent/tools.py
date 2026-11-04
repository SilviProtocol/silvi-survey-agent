"""Agent tools and the tool-call checker.

Tools map one-to-one onto agent -> phone wire kinds, and every tool can be
invoked from a structured intent (tap / crosshair in later packages) exactly
as from a voice turn: voice is a layer on top, not the only way in.

Spark only supports ``tool_choice: "auto"`` and its generated arguments are
not schema-guaranteed (Meta docs, read 2026-10-28), so every model tool call
is validated here. A bad call gets one repair attempt (deterministic coercion
first, then optionally one model repair); a call that still fails is dropped
with its reason, never executed.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable

from jsonschema import Draft202012Validator, ValidationError

_VEC3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}

TOOL_SCHEMAS: dict[str, dict] = {
    "say": {
        "description": "Speak a short sentence to the walker.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["text"],
            "properties": {"text": {"type": "string", "minLength": 1}},
        },
    },
    "mark_waypoint": {
        "description": "Mark a named point in site-local metres, e.g. a planting spot; or, with id, mark a survey tree's position as a waypoint.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "anyOf": [{"required": ["position_m", "label"]}, {"required": ["id"]}],
            "properties": {
                "position_m": _VEC3,
                "label": {"type": "string"},
                "id": {"type": "string", "minLength": 1},
            },
        },
    },
    "mark_polygon": {
        "description": "Mark a named area from at least three site-local points.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["vertices_m", "label"],
            "properties": {
                "vertices_m": {"type": "array", "items": _VEC3, "minItems": 3},
                "label": {"type": "string"},
            },
        },
    },
    "report_gap": {
        "description": "Report unsurveyed ground around a site-local point.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["position_m", "radius_m", "reason"],
            "properties": {
                "position_m": _VEC3,
                "radius_m": {"type": "number", "exclusiveMinimum": 0},
                "reason": {"type": "string"},
            },
        },
    },
    # -- survey tools ---------------------------------------------------
    "create_site": {
        "description": "Start a named survey site.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name"],
            "properties": {"name": {"type": "string", "minLength": 1}},
        },
    },
    "create_boundary": {
        "description": "Create a site boundary polygon from at least three site-local points.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["label", "vertices_m"],
            "properties": {
                "label": {"type": "string"},
                "vertices_m": {"type": "array", "items": _VEC3, "minItems": 3},
            },
        },
    },
    "create_plot": {
        "description": "Create a plot polygon from at least three site-local points.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["label", "vertices_m"],
            "properties": {
                "label": {"type": "string"},
                "vertices_m": {"type": "array", "items": _VEC3, "minItems": 3},
            },
        },
    },
    "create_planting": {
        "description": "Record a planting at a site-local point (usually the current position).",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["position_m", "label"],
            "properties": {
                "position_m": _VEC3,
                "label": {"type": "string"},
                "taxon": {"type": ["string", "null"]},
                "sigma_m": {"type": ["number", "null"], "minimum": 0},
            },
        },
    },
    "create_observation": {
        "description": "Record an observation at a site-local point (crosshair or current position).",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["position_m", "label"],
            "properties": {
                "position_m": _VEC3,
                "label": {"type": "string"},
                "sigma_m": {"type": ["number", "null"], "minimum": 0},
                "establishment": {"enum": ["planted", "wild", "unknown"]},
            },
        },
    },
    "add_corner": {
        "description": "Add one crosshair ground hit as the next polygon corner.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["position_m"],
            "properties": {"position_m": _VEC3},
        },
    },
    "close_polygon": {
        "description": "Close the accumulated corners into a boundary or plot polygon.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["label"],
            "properties": {
                "label": {"type": "string"},
                "kind": {"enum": ["boundary", "plot"]},
            },
        },
    },
    "identify_species": {
        "description": "Identify the species of a tree. Use id for a survey tree (e.g. 'T14'), or local_id for a stored record (e.g. 'tree-7'). Tentative survey trees answer 'not enough views yet'. The answer is revisable and may be at genus rank.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "anyOf": [{"required": ["id"]}, {"required": ["local_id"]}],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "local_id": {"type": "string", "minLength": 1},
            },
        },
    },
    # -- survey-brain query tools (results come back to the model) -----------
    "scene": {
        "description": "List survey trees within a radius of the walker (default 12 m), ahead first then nearest, with distance, clock direction, views and needs. Use this instead of guessing what is around.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"radius_m": {"type": "number", "exclusiveMinimum": 0}},
        },
    },
    "tree": {
        "description": "Look up one survey tree by id ('T14', 'tree 14' or '14'): status, distance, clock, views, species, needs and notes. Merged ids answer with the live tree they became.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id"],
            "properties": {"id": {"type": "string", "minLength": 1}},
        },
    },
    "find_trees": {
        "description": "Find survey trees matching a need, status and/or species words, physically nearest first. With no status filter, unconfirmed trees count too — that is what 'closest tree?' means. Use for 'what needs work near me?' style questions.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "need": {"enum": ["unconfirmed", "few_views", "one_side", "position_loose", "no_species"]},
                "status": {"enum": ["tentative", "confirmed"]},
                "species": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
        },
    },
    "route_to": {
        "description": "Distance and clock direction to a survey tree, plus where to stand (2.5 m beyond the tree, opposite the mean view direction) to see its not-yet-seen side when it still needs one.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id"],
            "properties": {"id": {"type": "string", "minLength": 1}},
        },
    },
    "area_report": {
        "description": "Survey totals for the closed boundary if there is one, else the whole walk: confirmed/complete/tentative counts, confirmed needs by kind, coverage, and the 3 nearest incomplete confirmed trees.",
        "parameters": {"type": "object", "additionalProperties": False, "properties": {}},
    },
    "add_note": {
        "description": "Attach the surveyor's own words as a note (evidence) to a survey tree by id. Only when this utterance asks for the note to be kept (note, record, remember and the like); never off the back of a question or an earlier turn.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "text"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "text": {"type": "string", "minLength": 1},
            },
        },
    },
    "request_evidence": {
        "description": "Ask the surveyor for one more view of a survey tree by id (shown on their screen until the next shot of that tree). Pick the first item from the tree's next_evidence unless the surveyor asks for something else; one ask at a time.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "want"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "want": {"enum": ["bark", "leaf", "whole", "flower_or_fruit", "base", "side"]},
            },
        },
    },
    "boundary": {
        "description": "Walked boundary control: start recording the walked loop, pause, resume, or close it into a simplified polygon and report what is inside.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["action"],
            "properties": {"action": {"enum": ["start", "pause", "resume", "close"]}},
        },
    },
    "guide_to_plant": {
        "description": "Guide the walker back to a stored plant by its local id, e.g. tree-7.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["local_id"],
            "properties": {"local_id": {"type": "string", "minLength": 1}},
        },
    },
    "record_growth": {
        "description": "Record a stored plant's growth measurements on a revisit (height and/or trunk diameter).",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["local_id"],
            "properties": {
                "local_id": {"type": "string", "minLength": 1},
                "height_m": {"type": ["number", "null"], "minimum": 0},
                "dbh_cm": {"type": ["number", "null"], "minimum": 0},
            },
        },
    },
    "report_gaps": {
        "description": "Compute unsurveyed gaps from the coverage grid and report the largest.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
    },
    "correct_record": {
        "description": (
            "Record a person's correction to a stored plant record. Corrections are kept "
            "as their own dated entries; earlier model output is never rewritten. "
            "exclude = it is not a plant; split = it is really two plants; "
            "merge_hint = it is the same tree as other_local_id; mark_planted = a person "
            "planted it; set_species = the person names the species in value; "
            "note = free words in value."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["local_id", "operation"],
            "properties": {
                "local_id": {"type": "string", "minLength": 1},
                "operation": {
                    "enum": [
                        "exclude",
                        "split",
                        "merge_hint",
                        "mark_planted",
                        "set_species",
                        "note",
                    ]
                },
                "value": {"type": ["string", "null"]},
                "other_local_id": {"type": ["string", "null"]},
                "who": {"enum": ["walker", "reviewer"]},
                "source": {"enum": ["voice", "tap"]},
            },
        },
    },
}

_VALIDATORS = {
    name: Draft202012Validator(spec["parameters"]) for name, spec in TOOL_SCHEMAS.items()
}

RepairFn = Callable[[str, object, str, dict], dict | None]


def tool_definitions() -> list[dict]:
    """OpenAI-style function tool definitions for the chat APIs."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for name, spec in TOOL_SCHEMAS.items()
    ]


def _coerce(value, schema: dict):
    """Small deterministic fixes: numeric strings to numbers, recursively."""
    stype = schema.get("type")
    if stype == "number" and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    if stype == "integer" and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    if stype == "array" and isinstance(value, list):
        return [_coerce(item, schema.get("items") or {}) for item in value]
    if stype == "object" and isinstance(value, dict):
        props = schema.get("properties") or {}
        out = {}
        for key, item in value.items():
            if schema.get("additionalProperties") is False and key not in props:
                continue  # drop unknown properties rather than fail the turn
            out[key] = _coerce(item, props.get(key) or {})
        return out
    return value


def parse_arguments(arguments) -> dict:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise TypeError("tool arguments must be a JSON object")
        return parsed
    raise TypeError(f"tool arguments must be an object or JSON string, got {type(arguments)}")


def validate_arguments(name: str, arguments: dict) -> None:
    if name not in _VALIDATORS:
        raise ValidationError(f"unknown tool: {name}")
    _VALIDATORS[name].validate(arguments)


def check_tool_call(
    name: str, arguments, *, repair: RepairFn | None = None
) -> tuple[dict | None, str | None, bool]:
    """Return (accepted_args | None, drop_reason | None, was_repaired)."""
    schema = (TOOL_SCHEMAS.get(name) or {}).get("parameters", {})
    try:
        parsed = parse_arguments(arguments)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        parsed, first_error = None, str(exc)
    else:
        first_error = None
        try:
            validate_arguments(name, parsed)
            return parsed, None, False
        except ValidationError as exc:
            first_error = exc.message

    # One repair path only: deterministic coercion, then one model repair.
    if parsed is not None and name in TOOL_SCHEMAS:
        coerced = _coerce(copy.deepcopy(parsed), schema)
        try:
            validate_arguments(name, coerced)
            return coerced, None, True
        except ValidationError:
            parsed = coerced

    if repair is not None and name in TOOL_SCHEMAS:
        try:
            fixed = repair(name, parsed if parsed is not None else arguments, first_error or "", schema)
        except Exception as exc:  # noqa: BLE001 - repair failed; drop, do not retry
            return None, f"repair failed: {exc}", False
        if fixed is not None:
            try:
                validate_arguments(name, fixed)
                return fixed, None, True
            except ValidationError as exc:
                return None, f"still invalid after one repair: {exc.message}", False

    return None, first_error or "invalid arguments", False


def check_tool_calls(
    calls: list[dict], *, repair: RepairFn | None = None
) -> tuple[list[dict], list[dict]]:
    """Validate model tool calls. Returns (accepted, dropped with reasons)."""
    accepted: list[dict] = []
    dropped: list[dict] = []
    for call in calls:
        name = str(call.get("name", ""))
        args, reason, repaired = check_tool_call(name, call.get("arguments"), repair=repair)
        if args is not None:
            accepted.append(
                {"id": call.get("id"), "name": name, "arguments": args, "repaired": repaired}
            )
        else:
            dropped.append({"id": call.get("id"), "name": name, "reason": reason})
    return accepted, dropped
