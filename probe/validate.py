# schema validation
# SMRK-GUEGAP Probe — validate.py
# JSON schema validation helpers (v1)
#
# Usage:
#   from validate import validate_input_json, validate_output_json
#
# CLI quick check:
#   python -m probe.validate input schema/input_schema.json
#   python -m probe.validate output schema/output_schema.json

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import jsonschema
    from jsonschema import Draft202012Validator
except Exception as e:
    raise RuntimeError(
        "jsonschema is required. Install with: pip install jsonschema"
    ) from e


@dataclass
class ValidationResult:
    ok: bool
    errors: str = ""


def _repo_root() -> str:
    # .../probe/validate.py -> repo root is parent of probe/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_schema(schema_path: str) -> Dict[str, Any]:
    """
    schema_path can be:
      - absolute path
      - relative to repo root (recommended): schema/input_schema.json
    """
    if not os.path.isabs(schema_path):
        schema_path = os.path.join(_repo_root(), schema_path)
    return _load_json(schema_path)


def _format_errors(errors) -> str:
    """
    Make jsonschema errors readable and stable.
    """
    lines = []
    for err in errors:
        loc = "$"
        if err.absolute_path:
            loc += "".join([f"[{repr(p)}]" if isinstance(p, int) else f".{p}" for p in err.absolute_path])
        msg = err.message
        lines.append(f"{loc}: {msg}")
    return "\n".join(lines)


def validate_data(data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
    """
    Validate data dict against a JSON Schema dict (Draft 2020-12).
    Returns ValidationResult with aggregated error messages.
    """
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
    if errors:
        return ValidationResult(ok=False, errors=_format_errors(errors))
    return ValidationResult(ok=True)


def validate_input_json(
    input_data: Dict[str, Any],
    schema_path: str = "schema/input_schema.json",
) -> ValidationResult:
    schema = _load_schema(schema_path)
    return validate_data(input_data, schema)


def validate_output_json(
    output_data: Dict[str, Any],
    schema_path: str = "schema/output_schema.json",
) -> ValidationResult:
    schema = _load_schema(schema_path)
    return validate_data(output_data, schema)


def validate_file(
    json_path: str,
    schema_path: str,
) -> ValidationResult:
    """
    Validate a JSON file on disk against a schema path.
    """
    if not os.path.isabs(json_path):
        json_path = os.path.join(_repo_root(), json_path)
    data = _load_json(json_path)
    schema = _load_schema(schema_path)
    return validate_data(data, schema)


def _main(argv: list[str]) -> int:
    """
    CLI:
      python -m probe.validate <input|output> <json_path> [schema_path]
    Examples:
      python -m probe.validate input examples/example_input.json
      python -m probe.validate output examples/example_output.json
      python -m probe.validate input examples/example_input.json schema/input_schema.json
    """
    if len(argv) < 3:
        print(
            "Usage: python -m probe.validate <input|output> <json_path> [schema_path]\n"
            "Example: python -m probe.validate input examples/example_input.json"
        )
        return 2

    kind = argv[1].strip().lower()
    json_path = argv[2].strip()
    schema_path = argv[3].strip() if len(argv) >= 4 else (
        "schema/input_schema.json" if kind == "input" else "schema/output_schema.json"
    )

    res = validate_file(json_path, schema_path)
    if res.ok:
        print("OK")
        return 0
    print("VALIDATION FAILED")
    print(res.errors)
    return 1


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
