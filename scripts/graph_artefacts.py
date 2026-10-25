#!/usr/bin/env python3
"""Rebuild a run's graph step by step and measure what each step leaves in a
segmented vessel that is not a vessel: two edges in one lumen, loops inside
one lumen, dead ends by kind (``graph.diagnose_lumen_artefacts``), and what
shape each thick-vessel bridge has (``graph.diagnose_thick_vessel_bridges``).

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

Where each side's graph reaches its skeleton, by place (the surface slab,
beside the large-vessel masks, the thick-vessel region), why each piece it
still loses is left by the skeleton re-trace, and the centreline plain Lee
would draw that the thick gate drops (``PLAN_surface_coverage.md``, Phase 0);
with ``--without-retrace``, each side's graph without the re-trace beside it::

    python scripts/graph_artefacts.py --config config.yaml \\
        --skeleton "outputs/<stem>_skeleton.npy" --by-place --without-retrace

What the run's raw image costs a build (``PLAN_raw_confirmation.md``,
decision D6), each side built again with the config's raw image loaded and
calibrated after the graph as ``build_network`` does, against the build
without it -- time, peak resident memory, and the raw image's reads::

    python scripts/graph_artefacts.py --config config.yaml \\
        --skeleton "outputs/<stem>_skeleton.npy" --raw-cost
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

from graph_artefacts.dead_end_census import (  # noqa: E402
    census,
    census_markdown,
    collapse_changes,
    made_by_step,
)
from graph_artefacts.report import (  # noqa: E402
    bridge_shapes, bridge_shapes_table, comparison_table, measure, raw_cost_table, steps_table,
)

WORKING_TREE = "working tree"

#: The settings ``preprocessing.raw_evidence_from_settings`` reads, handed to
#: a side built with the raw image.
RAW_SETTINGS = (
    "raw_image_path", "raw_image_channel", "raw_image_stain", "raw_image_downsample", "image_axis_order",
    "input_path", "use_memmap_loading", "memmap_directory",
)


def _export_ref(ref: str, destination: Path) -> Path:
    """``src`` of *ref*, written under *destination*; returns the src path."""
    archive = subprocess.run(["git", "archive", ref, "src"], cwd=REPO, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    return destination / "src"


def _side_dir(out: Path, name: str) -> Path:
    return out / ("working_tree" if name == WORKING_TREE else "ref_" + name.replace("/", "_"))


def _large_vessel_masks(settings, mask, voxel_size_zyx):
    """The union of the run's large-vessel masks on the segmentation's grid,
    read as the thick-vessel restriction reads them; None without them."""
    if not settings.get("use_large_vessel_masks"):
        return None
    import numpy as np

    from haemolynx import io as hio
    from haemolynx.pipeline.stages import thick_vessel_restriction_mask

    large = thick_vessel_restriction_mask(
        mask, dict(settings) | {"skeleton_thick_vessel_restrict_to_mask": "large", "use_memmap_loading": False},
        hio.voxel_size_xyz_from_zyx(voxel_size_zyx),
    )
    return None if large is None or not large.any() else np.asarray(large, dtype=bool)


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
    parser.add_argument("--raw-cost", action="store_true",
                        help="also build each side with the config's raw image loaded and calibrated, as "
                             "build_network does, and report what it costs")
    parser.add_argument("--by-place", action="store_true",
                        help="also report each side's reach of its skeleton by place, why the pieces it loses "
                             "are left, and the centreline the thick gate drops")
    parser.add_argument("--surface-depth-um", type=float, default=40.0,
                        help="--by-place: how deep the surface slab runs from the first slice (default 40)")
    parser.add_argument("--near-large-um", type=float, default=10.0,
                        help="--by-place: within this of the large-vessel masks is beside them (default 10)")
    parser.add_argument("--without-retrace", action="store_true",
                        help="also measure each side's graph without the skeleton re-trace, as a side of its own")
    parser.add_argument("--thick-vessel-mask",
                        help="a .npy thick-vessel region to split at (default: found from the mask "
                             "and the config, as a resumed run does)")
    args = parser.parse_args(argv)

    import numpy as np
    import tifffile

    from haemolynx import io as hio
    from haemolynx.io.load import _to_binary_volume_for_skeletonization
    from haemolynx.pipeline import default_schema, resolve_settings
    from haemolynx.pipeline.stages import (
        graph_build_arguments, thick_junction_bridge_arguments, thick_vessel_mask_from_image,
        thick_vessel_region_cache,
    )
    from haemolynx.preprocessing import MaskSupport

    settings = resolve_settings(schema=default_schema(), config_path=args.config)
    if args.raw_cost and not settings.get("raw_image_path"):
        parser.error("--raw-cost needs a raw image: the config sets no raw_image_path")
    if args.voxel_size_zyx:
        voxel_size = tuple(args.voxel_size_zyx)
    elif settings.get("voxel_size_override_xyz"):
        voxel_size = hio.voxel_size_zyx_from_xyz(tuple(settings["voxel_size_override_xyz"]))
    else:
        parser.error("no voxel size: pass --voxel-size-zyx")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mask_file = args.mask or settings["input_path"]
    raw = hio.apply_axis_order(
        tifffile.imread(mask_file),
        hio.axis_order_for_file(hio.axis_order_for_run(settings), mask_file),
    )
    mask = raw == args.mask_value if args.mask_value is not None else _to_binary_volume_for_skeletonization(raw)
    mask_path = out / "mask.npy"
    np.save(mask_path, np.asarray(mask, dtype=bool))
    thick_path = None
    if args.thick_vessel_mask and not args.no_thick_split:
        thick_path = Path(args.thick_vessel_mask)
    elif not args.no_thick_split and settings.get("use_thick_vessel_skeletonisation"):
        print("Finding the thick-vessel region (or reading the run's cached one) ...", flush=True)
        thick = thick_vessel_mask_from_image(
            settings, np.asarray(mask, dtype=bool), hio.voxel_size_xyz_from_zyx(voxel_size),
            cache=thick_vessel_region_cache(settings),
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

    # The skeleton re-trace, as build_network runs it: inert while short
    # loops are kept for no hole.
    keep_hole = float(graph_build_arguments(settings).get("short_loop_keep_hole_over_um") or 0.0)
    retrace = {"keep_hole_over_um": keep_hole} if settings.get("retrace_lost_skeleton") and keep_hole > 0 else None

    sides, raw_costs = {}, {}
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
            "split": thick_junction_bridge_arguments(settings),
            "retrace": retrace,
        }
        (side / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
        print(f"Building {name} ...", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / "graph_artefacts" / "build_side.py"),
                        str(side / "job.json")], check=True)
        sides[name] = json.loads((side / "side.json").read_text(encoding="utf-8"))
        if args.raw_cost:
            # The same build in a fresh process, the raw image read after it.
            with_raw = side / "with_raw"
            with_raw.mkdir(parents=True, exist_ok=True)
            raw_job = job | {"out": str(with_raw), "keep_steps": False,
                             "raw": {key: str(value) if isinstance(value, Path) else value
                                     for key, value in settings.items() if key in RAW_SETTINGS}}
            (with_raw / "job.json").write_text(json.dumps(raw_job, indent=1), encoding="utf-8")
            print(f"Building {name} with the raw image ...", flush=True)
            subprocess.run([sys.executable, str(Path(__file__).parent / "graph_artefacts" / "build_side.py"),
                            str(with_raw / "job.json")], check=True)
            raw_costs[name] = {"without": sides[name],
                               "with": json.loads((with_raw / "side.json").read_text(encoding="utf-8"))}

    import pickle

    support = MaskSupport(np.load(mask_path), voxel_size)
    multiple = float(settings.get("min_stub_length_radius_multiple") or 3.0)
    finals, lines = {}, ["# Graph artefacts", "", f"Config `{args.config}`, skeleton `{args.skeleton}`.", ""]

    def saved_graphs():
        """``(column, path)`` of every graph measured: each side's saved
        graph, after its graph without the re-trace when asked for."""
        for name in sides:
            without = _side_dir(out, name) / "without_retrace.pkl"
            if args.without_retrace and without.exists():
                yield f"{name} without re-trace", without
            yield name, _side_dir(out, name) / "final.pkl"

    for column, path in saved_graphs():
        with open(path, "rb") as f:
            finals[column] = measure(pickle.load(f), support, stub_radius_multiple=multiple, coverage=True)
    saved_as = "after smoothing and the thick-vessel split" if thick_path else "after smoothing"
    lines += [f"## Final graph ({saved_as})", "", comparison_table(finals), ""]
    for name, info in sides.items():
        lines.append(f"- **{name}**: built in {info['seconds']:.0f} s"
                     + (f" (the skeleton re-trace {info['retrace_seconds']:.0f} s of it)"
                        if info.get("retrace_seconds") is not None else "")
                     + (f"; settings it does not take: {', '.join(info['arguments_not_taken'])}"
                        if info["arguments_not_taken"] else ""))
    if raw_costs:
        lines += ["", "## The raw image's cost", "",
                  f"`{settings['raw_image_path']}`, channel {settings.get('raw_image_channel')}, read after the "
                  "graph as build_network reads it. No rule reads it while the graph is built, so its windows "
                  "stay 0 until one does (PLAN_raw_confirmation.md).", "", raw_cost_table(raw_costs)]

    # What shape each thick-vessel bridge has (PLAN_thick_vessel_bridges.md,
    # Phase 0), every side against the one region, by this checkout.
    if thick_path is not None:
        import csv

        thick = np.load(thick_path)
        shapes = {}
        for name in sides:
            with open(_side_dir(out, name) / "final.pkl", "rb") as f:
                shapes[name] = bridge_shapes(pickle.load(f), thick, support.mask, voxel_size)
        lines += ["", "## Thick-vessel bridges by shape", "", bridge_shapes_table(shapes)]
        rows = [{"side": name, **row} for name, report in shapes.items() for row in report["bridges"]]
        if rows:
            with open(out / "bridges.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            lines.append(f"Every bridge and its measures: `{out / 'bridges.csv'}`.")

    # Where each graph reaches its skeleton, by place, and what the thick
    # gate drops (PLAN_surface_coverage.md, Phase 0), by this checkout.
    if args.by_place:
        import csv

        from graph_artefacts.by_place import by_place_section

        skeleton = np.load(args.skeleton)
        large = _large_vessel_masks(settings, support.mask, voxel_size)
        thick = np.load(thick_path) if thick_path is not None else None
        graphs = {}
        for column, path in saved_graphs():
            with open(path, "rb") as f:
                graphs[column] = pickle.load(f)
        section, pieces, dropped = by_place_section(
            graphs, skeleton, support, large=large, thick=thick, settings=settings,
            surface_depth_um=args.surface_depth_um, near_large_um=args.near_large_um,
        )
        lines += ["", section]
        for name, rows in (("lost_pieces.csv", pieces), ("dropped_by_thick_gate.csv", dropped)):
            if rows:
                with open(out / name, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
                    writer.writeheader()
                    writer.writerows(rows)
                lines.append(f"Every row: `{out / name}`.")

    # What each dead end is (PLAN_dead_ends.md, Phase 0), every side by this
    # checkout's one definition.
    from haemolynx.graph.dead_ends import image_extent_um

    extent = image_extent_um(support.mask.shape, support.voxel_size_zyx)
    censuses = {}
    for name in sides:
        with open(_side_dir(out, name) / "final.pkl", "rb") as f:
            censuses[name] = census(pickle.load(f), support=support, extent_um=extent,
                                    stub_radius_multiple=multiple)
    lines += ["", "## Dead ends", "", census_markdown(censuses)]
    rows = []
    for name, info in sides.items():
        for tip, end in censuses[name]["ends"].items():
            rows.append({"side": name, **end.measures()})
    if rows:
        import csv

        keys = list(dict.fromkeys(k for row in rows for k in row))
        with open(out / "dead_ends.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        lines.append(f"Every dead end and its measures: `{out / 'dead_ends.csv'}`.")
    if args.steps:
        for name, info in sides.items():
            rows = []
            for label in info["steps"]:
                with open(_side_dir(out, name) / "steps" / f"{label}.pkl", "rb") as f:
                    rows.append((label, measure(pickle.load(f), support, stub_radius_multiple=multiple)))
            rows.append(("final (smoothed)", finals[name]))
            lines += ["", f"## Step by step: {name}", "", steps_table(rows)]
            with open(_side_dir(out, name) / "final.pkl", "rb") as f:
                final_graph = pickle.load(f)

            def step_graphs(name=name, info=info):
                for label in info["steps"]:
                    with open(_side_dir(out, name) / "steps" / f"{label}.pkl", "rb") as f:
                        yield label, pickle.load(f)

            made = made_by_step(final_graph, step_graphs())
            lines += ["", f"Where the final dead ends of {name} first appear:", "",
                      "| step | dead ends |", "|---|---|"]
            lines += [f"| {label} | {count} |" for label, count in made.items()]
            # What cluster collapse did to the dead ends it was handed
            # (PLAN_dead_ends.md, inconsistency 8): it has no mask, face or
            # direction rule, and runs before the prune.
            labels = list(info["steps"])
            if "collapse_node_clusters" in labels and labels.index("collapse_node_clusters") > 0:
                handed = labels[labels.index("collapse_node_clusters") - 1]
                graphs = {}
                for label in (handed, "collapse_node_clusters"):
                    with open(_side_dir(out, name) / "steps" / f"{label}.pkl", "rb") as f:
                        graphs[label] = pickle.load(f)
                changes = collapse_changes(graphs[handed], graphs["collapse_node_clusters"])
                lines += ["", f"What cluster collapse did to the dead ends of {name} it was handed:", "",
                          "| kept | joined | removed |", "|---|---|---|",
                          f"| {changes['kept']} | {changes['joined']} | {changes['removed']} |"]
    report = "\n".join(lines) + "\n"
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
