"""A 2D map of how a network is connected, drawn from its connectivity CSV.

The CSV is the one :func:`haemolynx.graph.write_connectivity_csv` writes (the
"Export connectivity CSV" button on the panel's Export tab), and the map reads
nothing else, so any exported file can be drawn again later. The map is a
schematic, not the vessels' shape.

The page does its own drawing: :func:`write_connectivity_map` puts the CSV's
text, a little config and ``connectivity_map.js`` (beside this module) into
one HTML file, and the script lays the map out in the browser. That is what
lets the page's "Load CSV..." button (or dropping a CSV on the page) draw a
different CSV without Python. ``connectivity_map.js`` holds the two views
("Vessels from an inlet" and "By branch order"), the bolder inlet and outlet
vessels, and the hovering; its header says how each works.
"""
from __future__ import annotations

import csv
import html
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from haemolynx.graph.branch_order import BRANCH_ORDER_CATEGORY_SEQUENCE

from ._helpers import _BRANCH_ORDER_SORT_GROUPS

__all__ = [
    "VIEWS",
    "connectivity_map_html",
    "read_connectivity_csv",
    "write_connectivity_map",
]

#: The views the page's "View" dropdown switches between.
VIEWS = ("inlet", "branch_order")


def read_connectivity_csv(path: Path | str) -> list[dict[str, str]]:
    """The rows of a connectivity CSV, as written."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def map_script() -> str:
    """The page's script, ``connectivity_map.js``."""
    return files("haemolynx.visualization").joinpath("connectivity_map.js").read_text(
        encoding="utf-8"
    )


def map_config() -> dict[str, Any]:
    """What the script takes from Python, so the two cannot drift apart.

    The branch-order sort groups are ``visualization._helpers``' table (which
    ``graph.branch_order`` defines the order of); Ven and Large_Ven are the
    groups that count up from the outlets.
    """
    from plotly.colors import qualitative

    return {
        "orderGroups": dict(_BRANCH_ORDER_SORT_GROUPS),
        "venousGroups": [
            BRANCH_ORDER_CATEGORY_SEQUENCE.index("venule"),
            BRANCH_ORDER_CATEGORY_SEQUENCE.index("large_venule"),
        ],
        "palette": list(qualitative.Dark24 + qualitative.Light24),
    }


def _script_json(value: Any) -> str:
    """*value* as JSON that cannot close the ``<script>`` element it sits in."""
    return json.dumps(value).replace("</", "<\\/")


def connectivity_map_html(csv_text: str, *, name: str = "", view: str = "inlet") -> str:
    """The map page for a connectivity CSV's text.

    *view* is the one shown first: ``"inlet"`` (columns count vessels from an
    inlet) or ``"branch_order"`` (one column per branch order).
    """
    from plotly.offline import get_plotlyjs_version

    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, not {view!r}")
    data = {"csv": csv_text, "name": name, "view": view, "config": map_config()}
    title = html.escape(f"{name or 'Network connectivity'} — 2D connectivity map")
    # Escaped outside the f-string: a backslash inside an f-string's braces is
    # Python 3.12+ only, and this package supports 3.9.
    script = map_script().replace("</script", "<\\/script")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js" charset="utf-8"></script>
</head>
<body>
<script id="hl-data" type="application/json">{_script_json(data)}</script>
<script>
{script}
</script>
<script>
HaemoLynxConnectivityMap.start(
  Plotly, document, JSON.parse(document.getElementById("hl-data").textContent)
);
</script>
</body>
</html>
"""


def write_connectivity_map(
    csv_path: Path | str, html_path: Path | str | None = None, *, view: str = "inlet"
) -> Path:
    """Draw the connectivity CSV at *csv_path* to an HTML page and return its path.

    *html_path* defaults to ``<csv name>_map.html`` beside the CSV.
    """
    source = Path(csv_path)
    target = Path(html_path) if html_path is not None else source.with_name(f"{source.stem}_map.html")
    text = source.read_text(encoding="utf-8")
    target.write_text(connectivity_map_html(text, name=source.stem, view=view), encoding="utf-8")
    return target
