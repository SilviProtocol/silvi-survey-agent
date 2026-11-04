"""Schema loading and validation. JSON Schema files are the source of truth."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


@cache
def load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


@cache
def _validator(name: str) -> Draft202012Validator:
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_wire(message: dict) -> None:
    _validator("wire.schema.json").validate(message)


def validate_observation(record: dict) -> None:
    _validator("observation.schema.json").validate(record)


def wire_kinds() -> list[str]:
    return list(load_schema("wire.schema.json")["properties"]["kind"]["enum"])
