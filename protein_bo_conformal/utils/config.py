"""Configuration loading, merging, and serialization helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def _strip_comments(line: str) -> str:
    """Remove whole-line comments while leaving scalar content untouched."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return ""
    return line.rstrip("\n")


def _parse_scalar(raw: str) -> Any:
    """Parse a scalar YAML-like value using only the standard library."""
    value = raw.strip()
    if value == "":
        return ""

    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None

    if value.startswith(('"', "{", "[")) or value.startswith("'"):
        for parser in (json.loads,):
            try:
                return parser(value)
            except json.JSONDecodeError:
                pass
        if value[0] == value[-1] and value[0] in {"'", '"'}:
            return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def _prepare_lines(text: str) -> list[tuple[int, str]]:
    """Convert raw text into indentation-aware logical lines."""
    prepared: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        cleaned = _strip_comments(raw_line)
        if not cleaned:
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        prepared.append((indent, cleaned.strip()))
    return prepared


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    """Recursively parse a small YAML subset based on indentation."""
    if index >= len(lines):
        return {}, index

    current_indent, current_text = lines[index]
    if current_indent != indent:
        raise ValueError(f"Unexpected indentation near '{current_text}'.")

    if current_text.startswith("- "):
        items: list[Any] = []
        while index < len(lines):
            line_indent, line_text = lines[index]
            if line_indent < indent:
                break
            if line_indent != indent or not line_text.startswith("- "):
                raise ValueError(f"Malformed list item near '{line_text}'.")

            item_body = line_text[2:].strip()
            index += 1
            if item_body:
                items.append(_parse_scalar(item_body))
            else:
                nested, index = _parse_block(lines, index, indent + 2)
                items.append(nested)
        return items, index

    mapping: dict[str, Any] = {}
    while index < len(lines):
        line_indent, line_text = lines[index]
        if line_indent < indent:
            break
        if line_indent != indent:
            raise ValueError(f"Unexpected indentation near '{line_text}'.")
        if ":" not in line_text:
            raise ValueError(f"Expected 'key: value' near '{line_text}'.")

        key, raw_value = line_text.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1

        if raw_value:
            mapping[key] = _parse_scalar(raw_value)
            continue

        if index >= len(lines) or lines[index][0] <= indent:
            mapping[key] = {}
            continue

        nested, index = _parse_block(lines, index, indent + 2)
        mapping[key] = nested

    return mapping, index


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a small, dependency-free YAML subset."""
    text = path.read_text(encoding="utf-8")
    lines = _prepare_lines(text)
    if not lines:
        return {}
    data, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ValueError(f"Failed to parse all config lines in {path}.")
    if not isinstance(data, dict):
        raise ValueError(f"Top-level config in {path} must be a mapping.")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override values into a copy of base."""
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def stable_config_hash(config: dict[str, Any]) -> str:
    """Compute a stable hash from a normalized config dictionary."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dump_scalar(value: Any) -> str:
    """Serialize a scalar to YAML-safe text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if value == "" or any(ch in value for ch in ':#[]{}"\n\t'):
            return json.dumps(value)
        return value
    return json.dumps(value)


def _dump_yaml_lines(value: Any, indent: int = 0) -> list[str]:
    """Serialize nested dict/list structures to a simple YAML format."""
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_dump_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_dump_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_dump_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_dump_scalar(item)}")
        return lines
    return [f"{prefix}{_dump_scalar(value)}"]


def dump_yaml(value: dict[str, Any]) -> str:
    """Dump a dictionary as a YAML-like string."""
    return "\n".join(_dump_yaml_lines(value)) + "\n"


@dataclass(frozen=True)
class ConfigNode:
    """Lightweight wrapper that enables attribute access to nested config dicts."""

    _data: dict[str, Any]

    def __getattr__(self, item: str) -> Any:
        try:
            value = self._data[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return self._wrap(value)

    def __getitem__(self, item: str) -> Any:
        return self._wrap(self._data[item])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep Python dictionary representation."""
        return self._unwrap(self._data)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    @classmethod
    def _unwrap(cls, value: Any) -> Any:
        if isinstance(value, ConfigNode):
            return cls._unwrap(value._data)
        if isinstance(value, dict):
            return {key: cls._unwrap(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._unwrap(item) for item in value]
        return value


def load_config(
    base_path: Path,
    override_path: Path | None = None,
    default_layer_paths: list[Path] | tuple[Path, ...] | None = None,
) -> ConfigNode:
    """Load layered config files and return a merged config node."""
    merged = load_yaml(base_path)

    for layer_path in default_layer_paths or ():
        if not layer_path.exists():
            continue
        merged = _deep_merge(merged, load_yaml(layer_path))

    if override_path is not None:
        override = load_yaml(override_path)
        merged = _deep_merge(merged, override)
    return ConfigNode(merged)
