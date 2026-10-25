#!/usr/bin/env python3
"""A marked grid as one interactive page: each case drawn from the
segmentation (the slab the sheets show, or one plane at a time), its layers
toggled, zoomed and panned, marked by button or key, with a note. Marks are
saved with the page when it is published as an artifact with the ``db``
capability (``marks/<case id>``: ``mark``, ``note``, ``rule``,
``updatedAt``), else in the browser, and "Copy marks" gives them as text.

The data is ``phase1_rules_grid.py --page-data``'s JSON; the page is
``grid_page_template.html`` with it filled in.

    python scripts/thick_junction_review/grid_page.py \\
        --data outputs/thick_junction_review/grid_P/page.json \\
        --out outputs/thick_junction_review/grid_P/grid_P.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "grid_page_template.html"
PLACEHOLDER = "/*GRID_DATA*/"


def build_page(data: dict) -> str:
    """The template with *data* in its JSON script block (``</`` escaped,
    so nothing in the data can close the block)."""
    text = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise ValueError(f"{TEMPLATE} must hold {PLACEHOLDER} exactly once")
    return template.replace(PLACEHOLDER, text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="phase1_rules_grid.py --page-data's JSON")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    page = build_page(data)
    Path(args.out).write_text(page, encoding="utf-8")
    cases = sum(len(sheet["cases"]) for sheet in data["sheets"])
    print(f"{cases} cases on {len(data['sheets'])} sheets -> {args.out} ({len(page) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
