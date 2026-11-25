"""The visual review grid (``visual.8a``): every saved figure and panel view,
numbered and shuffled, for an independent tester to mark GOOD or POOR.

``python -m release_review.visual_grid <out folder> <figure folder>...``
(from ``scripts/``) gathers the PNG figures under each folder (a run's plot
folder, a folder of panel screenshots), shuffles them with a fixed seed so an
ID gives away nothing, and writes ``grid.html`` (each picture large, under
its ID, with the question in the header), ``key.csv`` (ID to file, kept away
from the tester) and ``marks.csv`` (an empty mark per ID). Following
``CLAUDE.md`` § Marked example grids, the key stays beside the grid under
``outputs/visual_review/`` (git-ignored), and the tester's marks are recorded
in ``human_checks.yaml`` with their name; no mark is ever filled in here.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import random
import sys
from pathlib import Path
from typing import Sequence

QUESTION = (
    "For each picture: is it GOOD (clear, correctly labelled, units on every axis and colour bar, "
    "readable at this size) or POOR? Mark G or P against its number, and say why for any P."
)
#: About what a marked grid holds (CLAUDE.md § Marked example grids).
GRID_SIZE = 30
PICTURE_SUFFIXES = (".png", ".jpg", ".jpeg")


def gather(folders: Sequence[Path]) -> list[Path]:
    found = []
    for folder in folders:
        found += sorted(path for path in Path(folder).rglob("*") if path.suffix.lower() in PICTURE_SUFFIXES)
    return found


def numbered(pictures: Sequence[Path], *, seed: int = 0, limit: int = GRID_SIZE) -> list[tuple[int, Path]]:
    """Up to *limit* pictures, shuffled by *seed*, numbered from 1. With more
    pictures than the limit, a seeded draw keeps the grid to a markable size
    and leaves the rest for the next grid."""
    chosen = list(pictures)
    generator = random.Random(seed)
    generator.shuffle(chosen)
    return list(enumerate(chosen[:limit], start=1))


def page(items: Sequence[tuple[int, Path]], title: str) -> str:
    cells = []
    for number, path in items:
        data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        kind = "png" if path.suffix.lower() == ".png" else "jpeg"
        cells.append(
            f'<figure><figcaption>{number}</figcaption>'
            f'<img src="data:image/{kind};base64,{data}" alt="picture {number}"></figure>'
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>"
        "body{font-family:sans-serif;margin:16px;background:#fff;color:#111}"
        "header{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ccc}"
        "figure{margin:24px 0}figcaption{font-size:28px;font-weight:bold}"
        "img{max-width:100%;border:1px solid #ddd}</style></head><body>"
        f"<header><h1>{html.escape(title)}</h1><p>{html.escape(QUESTION)}</p></header>"
        + "".join(cells) + "</body></html>"
    )


def write(out: Path, items: Sequence[tuple[int, Path]], title: str = "Visual review grid") -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid.html").write_text(page(items, title), encoding="utf-8")
    with (out / "key.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "file"])
        writer.writerows((number, Path(path).as_posix()) for number, path in items)
    with (out / "marks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "mark (G or P)", "why"])
        writer.writerows((number, "", "") for number, _path in items)
    return out / "grid.html"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the visual review grid (visual.8a).")
    parser.add_argument("out", type=Path)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    items = numbered(gather(arguments.folders), seed=arguments.seed)
    print(write(arguments.out, items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
