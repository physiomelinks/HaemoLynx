#!/usr/bin/env python3
"""Cross-check HaemoLynx's morphometric statistics against independent tools
on the same segmented mask (release review check ``stats.8a``).

HaemoLynx builds its graph with the run's own stages and writes its statistics
CSV; skan summarises the Lee skeleton of the mask; maskel (MIT) runs in an
interpreter of its own when ``--maskel-python`` names one; VesselVio (GPL-3)
is never run from here -- run it on the ``<name>.tif`` this writes (its app,
or ``VVTerminal.py`` with "save segment results" on) and pass its results
folder to ``--vesselvio-results``. The table compares total centreline length,
segment, branch-point and end-point counts, and the per-segment length,
tortuosity and diameter distributions; ``comparison.md`` also says how each
tool defines each quantity.

This is not part of the test suite (``tests/test_stats_cross_check.py`` pins
the synthetic case with skan); E14.5 takes about ten minutes a side.

Usage
-----
The synthetic dense capillary bed (``tests/dense_capillary_fixtures.py``),
whose design gives the right answers::

    python scripts/stats_cross_check.py --dense-capillary-bed --out outputs/stats_cross_check/bed

A real mask, with HaemoLynx's settings from a config (an ilastik "Simple
Segmentation" export is labelled 1 = vessel)::

    python scripts/stats_cross_check.py --mask "<stem>.tiff" --mask-value 1 \\
        --voxel-size-xyz 0.98 0.98 1.0 --config config.yaml \\
        --maskel-python <env>/python --vesselvio-results <VesselVio results folder> \\
        --out outputs/stats_cross_check/e14
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stats_cross_check import haemolynx_side, maskel_side, skan_side, vesselvio_files  # noqa: E402
from stats_cross_check.measures import comparison_table, summary, write_measurements  # noqa: E402

BED_NAME = "dense_capillary_bed"


def _commit(source_commit: str | None) -> str:
    """Which HaemoLynx the run measured: this checkout's commit (flagged when
    ``src`` has changes), or *source_commit* for a ``git archive`` snapshot of
    one put first on ``PYTHONPATH``."""
    import haemolynx

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()

    package = Path(haemolynx.__file__).resolve().parent
    if (REPO / "src").resolve() not in package.parents:
        return f"{source_commit or 'an unnamed snapshot'} (src from {package.parent})"
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", "src")
    return git("rev-parse", "--short", "HEAD") + (" with src modified" if dirty else "")


def _load_mask(path: Path, value: int | None) -> np.ndarray:
    if path.suffix == ".npy":
        data = np.load(path)
    else:
        import tifffile

        data = tifffile.imread(path)
    return data == value if value is not None else data > 0


def _bed() -> tuple[np.ndarray, tuple[float, float, float], dict]:
    sys.path.insert(0, str(REPO / "tests"))
    from dense_capillary_fixtures import dense_capillary_bed, design_segments, design_topology

    from stats_cross_check.measures import Measurements

    bed = dense_capillary_bed()
    truths = {}
    for label, as_segmented in (("design", False), ("as segmented", True)):
        segments = design_segments(as_segmented=as_segmented)
        topology = design_topology(segments)
        lengths = np.array([s.length_um for s in segments])
        truths[label] = summary(Measurements(
            tool=label, total_length_um=float(lengths.sum()), segments=len(segments),
            branch_points=topology["branch_points"], end_points=topology["end_points"],
            segment_lengths_um=lengths, tortuosity=np.ones(len(segments)),
            diameters_um=np.array([s.diameter_um for s in segments]),
        ))
    voxel_xyz = tuple(float(v) for v in bed.voxel_size_zyx[::-1])
    return bed.mask, voxel_xyz, truths


def _definitions(sides) -> str:
    quantities = list(next(iter(sides)).definitions)
    lines = []
    for quantity in quantities:
        lines.append(f"**{quantity}**\n")
        for m in sides:
            if quantity in m.definitions:
                lines.append(f"- {m.tool}: {m.definitions[quantity]}")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mask", type=Path, help="segmented mask (TIFF or .npy), axes (z, y, x)")
    source.add_argument("--dense-capillary-bed", action="store_true", help="the synthetic bed, with its design")
    parser.add_argument("--mask-value", type=int, help="the label that is vessel (default: anything above 0)")
    parser.add_argument("--voxel-size-xyz", nargs=3, type=float, help="microns; required with --mask")
    parser.add_argument("--config", type=Path, help="HaemoLynx config for the run's settings (default: the schema's)")
    parser.add_argument("--maskel-python", type=Path, help="interpreter with maskel installed")
    parser.add_argument("--vesselvio-results", type=Path, help="VesselVio results folder for <name>.tif")
    parser.add_argument("--name", help="dataset name (default: the mask's stem)")
    parser.add_argument("--source-commit", help="the commit of a src snapshot put first on PYTHONPATH, for the report")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    truths = None
    if args.dense_capillary_bed:
        mask, voxel_xyz, truths = _bed()
        name = args.name or BED_NAME
    else:
        if args.voxel_size_xyz is None:
            parser.error("--voxel-size-xyz is required with --mask")
        mask, voxel_xyz = _load_mask(args.mask, args.mask_value), tuple(args.voxel_size_xyz)
        name = args.name or args.mask.stem.replace(" ", "_")
    voxel_zyx = voxel_xyz[::-1]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # One binary mask that every tool reads.
    import tifffile

    mask_tif, mask_npy = out / f"{name}.tif", out / f"{name}.npy"
    tifffile.imwrite(mask_tif, (mask.astype(np.uint8) * 255))
    np.save(mask_npy, mask)

    times, sides = {}, []
    started = time.perf_counter()
    haemolynx, build, csv_path = haemolynx_side.run(mask_tif, voxel_xyz, out / "haemolynx", args.config)
    times["HaemoLynx"] = time.perf_counter() - started
    sides.append(haemolynx)

    started = time.perf_counter()
    raw_skeleton = skan_side.lee_skeleton(mask)
    skan = skan_side.run(mask, voxel_zyx, skeleton=raw_skeleton)
    times["skan"] = time.perf_counter() - started
    sides.append(skan)
    ledger = haemolynx_side.ledger(build, skan)
    reach = {
        "the mask's Lee skeleton (skan's)": haemolynx_side.skeleton_in_haemolynx_lumen(build, raw_skeleton),
        "HaemoLynx's cleaned skeleton": haemolynx_side.skeleton_in_haemolynx_lumen(build, build.skeleton),
    }

    if args.maskel_python is not None:
        for cleanup in (False, True):
            started = time.perf_counter()
            side = maskel_side.run(args.maskel_python, mask_npy, voxel_zyx,
                                   out / f"maskel{'_cleanup' if cleanup else ''}.json", cleanup=cleanup)
            times[side.tool] = time.perf_counter() - started
            sides.append(side)

    if args.vesselvio_results is not None:
        sides.append(vesselvio_files.load(args.vesselvio_results, name))

    write_measurements(sides, out / "measurements.json")
    report = [
        f"# Statistics cross-check: {name}",
        "",
        f"HaemoLynx at `{_commit(args.source_commit)}`; mask {mask.shape} (z, y, x) at {voxel_xyz} um (x, y, z), "
        f"{int(mask.sum()):,} vessel voxels. HaemoLynx statistics CSV: `{csv_path.name}`.",
        "",
        "Bracketed: the difference from HaemoLynx.",
        "",
        comparison_table(sides, truth=truths["design"] if truths else None),
        "",
    ]
    if truths:
        report += [
            "The design is the bed as drawn: 10 lattice lines face to face, two parallel vessels and two "
            "sprouts, every centreline straight. As segmented, the two one-voxel dropouts cut two vessels: "
            f"{truths['as segmented']['Segments']} segments, {truths['as segmented']['End points (degree 1)']} "
            f"end points, {truths['as segmented']['Total centreline length (um)']:.1f} um.",
            "",
        ]
    extras = {m.tool: m.extra for m in sides if m.extra}
    if extras:
        report += ["Also reported: " + "; ".join(f"{tool}: {extra}" for tool, extra in extras.items()), ""]
    report += ["Seconds per tool: " + ", ".join(f"{k} {v:.0f}" for k, v in times.items()), "",
               "## From a plain skeleton to HaemoLynx's statistics", "",
               "Each row is what one of HaemoLynx's definitions changes; brackets, the change from the row above.",
               "", haemolynx_side.ledger_table(ledger), "",
               "Where the skeleton HaemoLynx's graph leaves out lies: in the lumen of a vessel it drew "
               "(within the lumen radius plus a voxel of its centreline: the same vessel again, a strand "
               "or spur or junction knot the clean-up removes by definition) or outside every one (a "
               "vessel the graph does not draw; a flattened lumen's far side counts here too):", "",
               *(f"- {label}: {r['share in a drawn vessel']:.1%} of {r['skeleton voxels']:,} voxels in a drawn vessel"
                 for label, r in reach.items()), "",
               "## Definitions", "", _definitions(sides)]
    (out / "ledger.json").write_text(json.dumps({"ledger": ledger, "skeleton in a drawn vessel": reach}, indent=1), encoding="utf-8")
    (out / "comparison.md").write_text("\n".join(report), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({m.tool: summary(m) for m in sides}, indent=1), encoding="utf-8")
    print("\n".join(report[:9]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
