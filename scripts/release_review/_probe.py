"""Facts only the package itself can give, written as JSON.

``python _probe.py <out.json>``, run by the quick tier in a subprocess whose
``PYTHONPATH`` starts with the scored tree's ``src``, so the facts are that
tree's, not the editable install's. Imports nothing from the scorer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path: str) -> None:
    import haemolynx
    from haemolynx.gui.stage_help import topics_for_tabs
    from haemolynx.gui.tabs import tab_titles
    from haemolynx.pipeline.schema import SCHEMA

    settings = list(SCHEMA)
    topics = topics_for_tabs(tab_titles())
    facts = {
        "haemolynx_file": Path(haemolynx.__file__).as_posix(),
        "version": getattr(haemolynx, "__version__", None),
        "settings": len(settings),
        "settings_without_help": [each.name for each in settings if not str(each.help or "").strip()],
        "help_pages": len(topics),
        "help_pages_written": sum(bool(topic.body.strip()) for topic in topics),
    }
    Path(path).write_text(json.dumps(facts, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
