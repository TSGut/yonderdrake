"""Checkpoint schema validation."""

from __future__ import annotations

import json

import pytest

from yonderdrake.time.checkpointing import (
    inspect_checkpoint_file,
    load_checkpoint_file,
    save_checkpoint_file,
)


class MemoryCheckpoint:
    def __init__(self) -> None:
        self.attributes: dict[tuple[str, str], str] = {}
        self.functions: dict[str, object] = {}

    def require_group(self, path: str) -> None:
        pass

    def has_attr(self, path: str, key: str) -> bool:
        return (path, key) in self.attributes

    def save_function(self, function: object, *, name: str) -> None:
        self.functions[name] = function

    def set_attr_byte_string(self, path: str, key: str, value: str) -> None:
        self.attributes[path, key] = value

    def get_attr_byte_string(self, path: str, key: str) -> str:
        return self.attributes[path, key]

    def load_function(self, mesh: object, name: str) -> object:
        assert mesh is not None
        return self.functions[name]


def test_checkpoint_schema_round_trip() -> None:
    checkpoint = MemoryCheckpoint()
    fields = {"u": object(), "mode_0": object()}
    save_checkpoint_file(
        checkpoint,
        name="restart_2",
        metadata={"time": 0.4},
        fields=fields,
    )

    state, names = inspect_checkpoint_file(checkpoint, name="restart_2")
    loaded_state, loaded = load_checkpoint_file(
        checkpoint,
        name="restart_2",
        mesh=object(),
        expected_fields=("u", "mode_0"),
    )

    assert state == loaded_state == {"time": 0.4}
    assert names == ("u", "mode_0")
    assert loaded == fields


@pytest.mark.parametrize("name", [None, "", "1state", "two-states"])
def test_checkpoint_name_is_validated(name) -> None:
    with pytest.raises(ValueError, match="checkpoint name"):
        inspect_checkpoint_file(MemoryCheckpoint(), name=name)


@pytest.mark.parametrize("fields", [{}, {"bad-name": object()}])
def test_checkpoint_field_names_are_validated(fields) -> None:
    with pytest.raises(ValueError, match="field names"):
        save_checkpoint_file(
            MemoryCheckpoint(),
            name="state",
            metadata={},
            fields=fields,
        )


@pytest.mark.parametrize("metadata", [{"bad": object()}, {"bad": float("nan")}])
def test_checkpoint_metadata_must_be_json_compatible(metadata) -> None:
    with pytest.raises(ValueError, match="JSON-compatible"):
        save_checkpoint_file(
            MemoryCheckpoint(),
            name="state",
            metadata=metadata,
            fields={"u": object()},
        )


def checkpoint_with_record(record: object) -> MemoryCheckpoint:
    checkpoint = MemoryCheckpoint()
    checkpoint.attributes[
        "/yonderdrake/checkpoints/state",
        "yonderdrake_metadata",
    ] = record if isinstance(record, str) else json.dumps(record)
    return checkpoint


def test_missing_and_malformed_checkpoint_metadata_are_rejected() -> None:
    with pytest.raises(ValueError, match="was not found"):
        inspect_checkpoint_file(MemoryCheckpoint(), name="state")
    with pytest.raises(ValueError, match="metadata is invalid"):
        inspect_checkpoint_file(checkpoint_with_record("{"), name="state")


@pytest.mark.parametrize(
    "record,error",
    [
        ([], "unsupported"),
        ({"format_version": 2}, "unsupported"),
        ({"format_version": 1, "fields": []}, "fields"),
        (
            {"format_version": 1, "fields": ["bad-name"], "state": {}},
            "fields",
        ),
        (
            {"format_version": 1, "fields": ["u"], "state": []},
            "state metadata",
        ),
    ],
)
def test_checkpoint_record_schema_is_validated(record, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        inspect_checkpoint_file(checkpoint_with_record(record), name="state")


def test_checkpoint_field_layout_must_match_stepper() -> None:
    checkpoint = checkpoint_with_record(
        {"format_version": 1, "fields": ["u"], "state": {}}
    )
    with pytest.raises(ValueError, match="fields do not match"):
        load_checkpoint_file(
            checkpoint,
            name="state",
            mesh=object(),
            expected_fields=("physical",),
        )
