#!/usr/bin/env python3
"""Rebuild a run's graph step by step and measure what each step leaves in a
segmented vessel that is not a vessel: two edges in one lumen, loops inside
one lumen, dead ends by kind (``graph.diagnose_lumen_artefacts``).

The working tree, and each ``--ref`` (a ``git archive`` of it), builds the
graph from the run's saved skeleton in a process of its own, with its own code
and the graph-building settings the config gives (``pipeline.stages.
graph_build_arguments``; a setting an older version lacks is left out and
named in the report). Every graph is then measured by this checkout, so the
versions are compared with one ruler. When the config uses thickness-gated
skeletonisation, each side's final graph is split at the thick-vessel region
as ``build_network`` saves it (the region found once, by this checkout, and
shared); ``--no-thick-split`` measures the graph before that split.

This is not part of the test suite: a full E14.5 build takes about seven
minutes a side.

Usage
-----
The working tree, on the run ``config.yaml`` describes::

    python scripts/graph_artefacts.py --config config.yaml \\
        --skeleton "outputs/<stem>_skeleton.npy"

Against a ref, with a table for every topology step::

    python scripts/graph_artefacts.py --config config.yaml \\
        --skeleton "outputs/<stem>_skeleton.npy" --ref 5fab427 --steps

The mask is the config's ``input_path`` unless ``--mask`` names another; an
ilastik "Simple Segmentation" export is labelled 1 = vessel, so pass
``--mask-value 1`` for one rather than relying on the binarisation.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_artefacts.report import comparison_table, measure, steps_table  # noqa: E402

WORKING_TREE = "working tree"


def _export_ref(ref: str, destination: Path) -> Path:
    """``src`` of *ref*, written under *destination*; returns the src path."""
    archive = subprocess.run(["git", "archive", ref, "src"], cwd=REPO, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    return destination / "src"


def _side_dir(out: Path, name: str) -> Path:
    return out / ("working_tree" if name == WORKING_TREE else "ref_" + name.replace("/", "_"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="the run's settings file")
    parser.add_argument("--skeleton", required=True, help="the run's saved <stem>_skeleton.npy")
    parser.add_argument("--mask", help="the segmentation (default: the config's input_path)")
    parser.add_argument("--mask-value", type=int, help="the label that is vessel (default: binarise as the loaders do)")
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, help="default: the config's voxel_size_override_xyz")
    parser.add_argument("--ref", action="append", default=[], help="a git ref to build as well (repeatable)")
    parser.add_argument("--out", default=str(REPO / "outputs" / "graph_artefacts"))
    parser.add_argument("--steps", action="store_true", help="also measure the graph after every topology step")
    parser.add_argument("--no-thick-split", action="store_true",
                        help="measure the graph before the thick-vessel split build_network ends with")
    parser.add_argument("--thick-vessel-mask",
                        help="a .npy thick-vessel region to split at (default: found from the mask "
                             "and the config, as a resumed run does)")
    args = parser.parse_args(argv)

    import numpy as np
    import tifffile

    from haemolynx import io as hio
    from haemolynx.io.load import _to_binary_volume_for_skeletonization
    from haemolynx.pipeline import default_schema, resolve_settings
    from haemolynx.pipeline.stages import graph_build_arguments, thick_vessel_mask_from_image
    from haemolynx.preprocessing import MaskSupport

    settings = resolve_settings(schema=default_schema(), config_path=args.config)
    if args.voxel_size_zyx:
        voxel_size = tuple(args.voxel_size_zyx)
    elif settings.get("voxel_size_override_xyz"):
        voxel_size = hio.voxel_size_zyx_from_xyz(tuple(settings["voxel_size_override_xyz"]))
    else:
        parser.error("no voxel size: pass --voxel-size-zyx")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = hio.apply_axis_order(tifffile.imread(args.mask or settings["input_path"]), settings["image_axis_order"])
    mask = raw == args.mask_value if args.mask_value is not None else _to_binary_volume_for_skeletonization(raw)
    mask_path = out / "mask.npy"
    np.save(mask_path, np.asarray(mask, dtype=bool))
    thick_path = None
    if args.thick_vessel_mask and not args.no_thick_split:
        thick_path = Path(args.thick_vessel_mask)
    elif not args.no_thick_split and settings.get("use_thick_vessel_skeletonisation"):
        print("Finding the thick-vessel region ...", flush=True)
        thick = thick_vessel_mask_from_image(
            settings, np.asarray(mask, dtype=bool), hio.voxel_size_xyz_from_zyx(voxel_size)
        )
        if thick is not None:
            thick_path = out / "thick_vessel_mask.npy"
            np.save(thick_path, np.asarray(thick, dtype=bool))
    smoothing = None
    if settings["smooth_centrelines"]:
        smoothing = {
            "method": settings["centreline_smoothing_method"],
            "iterations": int(settings["centreline_smoothing_iterations"]),
            "max_deviation": float(settings["centreline_max_deviation"]),
        }

    sides = {}
    # Refs first, so a two-side table's change column reads ref -> working tree.
    for name in [*args.ref, WORKING_TREE]:
        side = _side_dir(out, name)
        side.mkdir(parents=True, exist_ok=True)
        src = REPO / "src" if name == WORKING_TREE else _export_ref(name, side / "code")
        job = {
            "src": str(src),
            "out": str(side),
            "skeleton": str(Path(args.skeleton).resolve()),
            "mask": str(mask_path.resolve()),
            "voxel_size_zyx": list(voxel_size),
            "build": graph_build_arguments(settings),
            "smoothing": smoothing,
            "thick_vessel_mask": None if thick_path is None else str(thick_path.resolve()),
        }
        (side / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
        print(f"Building {name} ...", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / "graph_artefacts" / "build_side.py"),
                        str(side / "job.json")], check=True)
        sides[name] = json.loads((side / "side.json").read_text(encoding="utf-8"))

    import pickle

    support = MaskSupport(np.load(mask_path), voxel_size)
    multiple = float(settings.get("min_stub_length_radius_multiple") or 3.0)
    finals, lines = {}, ["# Graph artefacts", "", f"Config `{args.config}`, skeleton `{args.skeleton}`.", ""]
    for name, info in sides.items():
        with open(_side_dir(out, name) / "final.pkl", "rb") as f:
            finals[name] = measure(pickle.load(f), support, stub_radius_multiple=multiple, coverage=True)
    saved_as = "after smoothing and the thick-vessel split" if thick_path else "after smoothing"
    lines += [f"## Final graph ({saved_as})", "", comparison_table(finals), ""]
    for name, info in sides.items():
        lines.append(f"- **{name}**: built in {info['seconds']:.0f} s"
                     + (f"; settings it does not take: {', '.join(info['arguments_not_taken'])}"
                        if info["arguments_not_taken"] else ""))
    if args.steps:
        for name, info in sides.items():
            rows = []
            for label in info["steps"]:
                with open(_side_dir(out, name) / "steps" / f"{label}.pkl", "rb") as f:
                    rows.append((label, measure(pickle.load(f), support, stub_radius_multiple=multiple)))
            rows.append(("final (smoothed)", finals[name]))
            lines += ["", f"## Step by step: {name}", "", steps_table(rows)]
    report = "\n".join(lines) + "\n"
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
