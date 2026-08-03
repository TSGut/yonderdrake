"""Collective Firedrake checkpoint helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

_FORMAT_VERSION = 1
_METADATA_KEY = "yonderdrake_metadata"
_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


def stepper_metadata(
    *,
    kind: str,
    operator_kinds: Sequence[str],
    parameters: Sequence[float],
    representations: Sequence[Mapping[str, Any]],
    formulation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the common time-stepper checkpoint header."""
    return {
        "version": 1,
        "kind": kind,
        "operator_kinds": tuple(operator_kinds),
        "parameters": tuple(parameters),
        "representations": list(representations),
        "formulation": dict(formulation),
    }


def validate_stepper_metadata(
    state: Mapping[str, Any],
    *,
    kind: str,
    operator_kinds: Sequence[str],
    parameters: Sequence[float],
) -> list[Any]:
    """Validate a common time-stepper checkpoint header."""
    if state.get("version") != 1 or state.get("kind") != kind:
        raise ValueError("unsupported checkpoint version")
    if tuple(state.get("operator_kinds", ())) != tuple(operator_kinds):
        raise ValueError("checkpoint time-memory operators do not match")
    if tuple(state.get("parameters", ())) != tuple(parameters):
        raise ValueError("checkpoint time-memory parameters do not match")
    representations = state.get("representations")
    if (
        not isinstance(representations, list)
        or len(representations) != len(parameters)
    ):
        raise ValueError(
            "checkpoint representation count does not match the stepper"
        )
    return representations


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or _NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(
            "checkpoint name must start with a letter and contain only "
            "letters, digits, and underscores"
        )
    return name


def _group_path(name: str) -> str:
    return f"/yonderdrake/checkpoints/{name}"


def _function_name(name: str, field: str) -> str:
    return f"yonderdrake_{name}_{field}"


def save_checkpoint_file(
    checkpoint: Any,
    *,
    name: str,
    metadata: Mapping[str, Any],
    fields: Mapping[str, Any],
) -> None:
    """Save metadata and distributed fields to a CheckpointFile."""
    name = _validate_name(name)
    field_names = tuple(fields)
    if not field_names or any(
        _NAME_PATTERN.fullmatch(field) is None for field in field_names
    ):
        raise ValueError("checkpoint field names are invalid")
    path = _group_path(name)
    checkpoint.require_group(path)
    if checkpoint.has_attr(path, _METADATA_KEY):
        raise ValueError(f"checkpoint {name!r} already exists")
    record = {
        "format_version": _FORMAT_VERSION,
        "fields": field_names,
        "state": dict(metadata),
    }
    try:
        encoded = json.dumps(
            record,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint metadata is not JSON-compatible") from error
    for field, function in fields.items():
        checkpoint.save_function(
            function,
            name=_function_name(name, field),
        )
    checkpoint.set_attr_byte_string(path, _METADATA_KEY, encoded)


def inspect_checkpoint_file(
    checkpoint: Any,
    *,
    name: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return validated metadata and field names."""
    name = _validate_name(name)
    path = _group_path(name)
    try:
        encoded = checkpoint.get_attr_byte_string(path, _METADATA_KEY)
    except (KeyError, RuntimeError) as error:
        raise ValueError(f"checkpoint {name!r} was not found") from error
    try:
        record = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint metadata is invalid") from error
    if not isinstance(record, dict) or record.get("format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported CheckpointFile checkpoint version")
    fields = record.get("fields")
    if (
        not isinstance(fields, list)
        or not fields
        or any(
            not isinstance(field, str)
            or _NAME_PATTERN.fullmatch(field) is None
            for field in fields
        )
    ):
        raise ValueError("checkpoint fields are invalid")
    state = record.get("state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint state metadata is invalid")
    return state, tuple(fields)


def load_checkpoint_file(
    checkpoint: Any,
    *,
    name: str,
    mesh: Any,
    expected_fields: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load metadata and distributed fields from a CheckpointFile."""
    state, fields = inspect_checkpoint_file(checkpoint, name=name)
    if fields != tuple(expected_fields):
        raise ValueError("checkpoint fields do not match the stepper")
    loaded = {
        field: checkpoint.load_function(
            mesh,
            _function_name(name, field),
        )
        for field in expected_fields
    }
    return state, loaded
