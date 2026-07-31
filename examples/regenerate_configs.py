#!/usr/bin/env python3
"""Rewrite every example's ``*_config.yaml`` from its schema.

The config files are generated, not hand-maintained: each one carries its
settings' help text, units, allowed values and prerequisites as comments, so
they can only say what the schema says. Run this after editing any schema::

    python examples/regenerate_configs.py

Values already present in a config file are preserved, so regenerating picks up
new settings and refreshed documentation without discarding local edits.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "examples"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ImageLynx.parsers import Schema, dump_config, load_config  # noqa: E402

#: config file (relative to the repo root) -> module holding its ``SCHEMA``.
CONFIGS: dict[str, str] = {
    "examples/simple_network_config.yaml": "simple_network_schema",
}


def _schema_for(module_name: str) -> Schema:
    module = __import__(module_name)
    return module.SCHEMA


def regenerate(root: Path = ROOT) -> list[Path]:
    written: list[Path] = []
    for relative_path, module_name in CONFIGS.items():
        schema = _schema_for(module_name)
        path = root / relative_path
        existing = load_config(path, schema) if path.is_file() else {}
        written.append(dump_config(path, schema, values=existing))
    return written


if __name__ == "__main__":
    for path in regenerate():
        print(f"wrote {path.relative_to(ROOT)}")
