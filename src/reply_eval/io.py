"""Input loading with actionable schema errors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class InputError(ValueError):
    """Raised when an evaluation input is missing or malformed."""


def _read_json_array(path: Path) -> list[Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(f"{path}: file not found") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise InputError(f"{path}: top-level JSON must be an array")
    return payload


def _validate_records(
    path: Path, payload: list[Any], required_fields: tuple[str, ...]
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise InputError(f"{path}: record {index} must be an object")
        record: dict[str, str] = {}
        for field in required_fields:
            value = raw.get(field)
            if not isinstance(value, str) or not value.strip():
                raise InputError(
                    f"{path}: record {index} field '{field}' must be a non-empty string"
                )
            record[field] = value
        case_id = record["id"]
        if case_id in seen:
            raise InputError(f"{path}: duplicate id: {case_id}")
        seen.add(case_id)
        records.append(record)
    return records


def load_cases(path: Path) -> list[dict[str, str]]:
    return _validate_records(
        path,
        _read_json_array(path),
        ("id", "user_question", "auto_reply"),
    )


def load_human_references(
    path: Path, expected_ids: set[str]
) -> list[dict[str, str]]:
    records = _validate_records(
        path,
        _read_json_array(path),
        ("id", "human_reference", "annotator_notes"),
    )
    actual_ids = {record["id"] for record in records}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise InputError(
            f"{path}: ID mismatch; missing={missing}, unexpected={extra}"
        )
    return records
