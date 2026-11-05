"""Read a run's exports back against the scored tree's data dictionary.

``python _read_back.py <run folder>``, with ``PYTHONPATH`` holding the scored
tree's ``src`` (for its dictionary) and this scorer's ``scripts``. Writes
``readback.json`` into the folder.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


def main(folder: str) -> int:
    from haemolynx.pipeline.export_dictionary import FILES, undocumented

    from release_review.exports import read_back

    described = {entry.name for entry in FILES if entry.columns}

    def not_described(entry: str, names):
        # An entry with no columns of its own describes them by reference
        # (the statistics glossary, another file's): nothing to match here.
        return undocumented(entry, names) if entry in described else []

    found = read_back(Path(folder), [entry.name for entry in FILES], not_described)
    (Path(folder) / "readback.json").write_text(
        json.dumps([asdict(each) for each in found], indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
