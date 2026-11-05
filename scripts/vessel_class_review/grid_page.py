"""A vessel-class marked grid as one interactive page (grid_page_template.html):
a card per case (its four-panel image), a mark per case by button or key, an
optional note. Published as an artifact with the ``db`` capability, marks
land in collection ``marks``, doc id = case number (``mark``, ``note``,
``updatedAt``); elsewhere they stay in the browser and "Copy marks" gives
them as text.

    python scripts/vessel_class_review/grid_page.py SZ1
"""
from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/vessel_class_review"
TEMPLATE = Path(__file__).resolve().parent / "grid_page_template.html"

PANELS = ("<div>Each row: the whole image's centrelines seen down z, with a yellow box round the "
          "window; then the segmentation, dextran and claudin-5 in that window, projected through "
          "the vessel's own thinnest axis over the depth it spans.</div>")

SPECS = {
    "AV1": {
        "title": "Grid AV1 Arteriole or Venule",
        "question": "Is the red vessel an arteriole or a venule?",
        "marks": [
            {"key": "A", "label": "Arteriole", "hint": "A", "token": "m0"},
            {"key": "V", "label": "Venule", "hint": "V", "token": "m1"},
            {"key": "?", "label": "Can't tell", "hint": "?", "token": "unk",
             "explain": "Can't tell (a capillary counts here too)"},
        ],
        "tokens": ("#b42318", "#fbe4e1", "#1d4ed8", "#e1e8fb"),
        "tokens_dark": ("#f07468", "#3a1c19", "#7aa2ff", "#18233d"),
        "about": ("Stack A region, 30 vessels at least 8.4 µm wide (wall to wall), shuffled."),
        "blue": "the rest of its tree (vessels ≥ 8.4 µm joined to it)",
    },
    "SZ1": {
        "title": "Grid SZ1 Small Vessel or Capillary",
        "question": "Is the red vessel a small vessel or a capillary?",
        "marks": [
            {"key": "S", "label": "Small vessel", "hint": "S", "token": "m0",
             "explain": "Small vessel (an arteriole or venule branch, above the capillary bed)"},
            {"key": "C", "label": "Capillary", "hint": "C", "token": "m1"},
            {"key": "?", "label": "Can't tell", "hint": "?", "token": "unk"},
        ],
        "tokens": ("#9a3412", "#fde8dc", "#0e7490", "#dbf1f6"),
        "tokens_dark": ("#fb923c", "#3b2214", "#67d3e8", "#12303a"),
        "about": ("Stack A region, 30 vessels 5-15 µm wide (wall to wall), three in each 1 µm band, "
                  "none in your large masks, shuffled. Your marks set where the run tells small "
                  "vessels from capillaries."),
        "blue": "vessels in your large arteriole and vein masks",
    },
}
# SZ2 asks SZ1's question of fresh vessels: the check of the 6-15 um rule.
SPECS["SZ2"] = {
    **SPECS["SZ1"],
    "title": "Grid SZ2 Small Vessel or Capillary",
    "about": ("Stack A region, 30 fresh vessels 4-18 µm wide (wall to wall), none in your large "
              "masks or in grid SZ1, shuffled. Your marks check the small-vessel rule (6-15 µm, grown "
              "from the large vessels) on vessels it was not set on."),
}
# SZR shows twenty SZ1 and SZ2 cases again: how often the marks repeat.
SPECS["SZR"] = {
    **SPECS["SZ1"],
    "title": "Grid SZR Small Vessel or Capillary Again",
    "about": ("Stack A region, 20 vessels from grids SZ1 and SZ2 shown again under new numbers, "
              "shuffled. Mark each as you see it now; your earlier marks are not shown. Your marks "
              "say how often a mark repeats."),
}


def build(grid: str) -> Path:
    spec = SPECS[grid]
    folder = OUT / grid
    with (folder / f"{grid}_key.csv").open(encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["id"]))
    cases = [{"id": int(r["id"]), "img": f"cases/{grid}_{int(r['id']):02d}.png",
              "voxel": r["mid_voxel_zyx"], "plane": r["projected_through"]} for r in rows]
    marks = [{k: m[k] for k in ("key", "label", "hint", "token")} for m in spec["marks"]]
    light = spec["tokens"]
    dark = spec["tokens_dark"]
    key_items = "".join(
        f'\n    <li><span class="chip" style="--c: var(--{m["token"]}); --cs: var(--{m["token"]}-soft)">'
        f'{html.escape(m["key"])}</span> {html.escape(m.get("explain", m["label"]))}</li>'
        for m in spec["marks"]) + "\n  "
    legend = (
        f'\n    <div>{html.escape(spec["about"])}</div>\n    {PANELS}\n'
        '    <div><span class="swatch" style="background: var(--sw-vessel)"></span>the vessel\n'
        f'      <span class="swatch" style="background: var(--sw-blue); margin-left: 14px"></span>{html.escape(spec["blue"])}\n'
        '      <span class="swatch" style="background: var(--sw-other); margin-left: 14px"></span>other centrelines '
        '(white over the segmentation)</div>'
    )
    keys_hint = " ".join(f"<kbd>{html.escape(m['hint'])}</kbd>" for m in spec["marks"])
    fills = {
        "__TITLE__": html.escape(spec["title"]),
        "__MARK_TOKENS_DARK__": (f"--m0: {dark[0]}; --m0-soft: {dark[1]}; "
                                 f"--m1: {dark[2]}; --m1-soft: {dark[3]};"),
        "__MARK_TOKENS__": (f"  --m0: {light[0]};\n  --m0-soft: {light[1]};\n"
                            f"  --m1: {light[2]};\n  --m1-soft: {light[3]};"),
        "__GRID__": f"Grid {grid}",
        "__N__": str(len(cases)),
        "__QUESTION__": html.escape(spec["question"]),
        "__KEY_ITEMS__": key_items,
        "__LEGEND__": legend,
        "__KEYS_HINT__": keys_hint,
        "__CASES__": json.dumps(cases, separators=(",", ":")),
        "__MARKS__": json.dumps(marks, separators=(",", ":")),
        "__LOCAL_KEY__": f"haemolynx-grid-{grid.lower()}-marks",
    }
    page = TEMPLATE.read_text(encoding="utf-8")
    for placeholder, value in fills.items():
        if placeholder not in page:
            raise ValueError(f"{TEMPLATE.name} has no {placeholder}")
        page = page.replace(placeholder, value)
    out = folder / f"{grid}_page.html"
    out.write_text(page, encoding="utf-8")
    return out


if __name__ == "__main__":
    for name in sys.argv[1:] or ["SZ1"]:
        print(f"wrote {build(name)}")
