#!/usr/bin/env python3
"""One run to ``build_network``, with everything a connection could be made
in kept for ``connection_steps.py`` to trace: the thickness-gated skeleton
before and after the thin-arm join, the skeleton graph building is handed,
the segmentation it reads, the thick region and its tree, the graph after
every topology step and after every step of ``stages.finish_network_graph``,
and the graph the split was handed.

    python scripts/thick_bridge_review/capture_steps.py --config config.yaml \\
        --input <segmented image> --out <run folder> [--set name=<json>]

From a worktree, run it with ``PYTHONPATH=src`` (the venv's haemolynx is the
main checkout's). An E14.5 capture holds about 5 GB at its peak.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np


def save_packed(path: Path, volume: np.ndarray) -> None:
    """A boolean volume as packed bits with its shape (``connection_steps._unpacked``)."""
    volume = np.asarray(volume, dtype=bool)
    np.savez_compressed(path, bits=np.packbits(volume.ravel()), shape=np.array(volume.shape))


@contextmanager
def capturing(out: Path) -> Iterator[Any]:
    """While open, every step's graph and the skeleton's layers are written
    into *out*; yields the ``on_stage_output`` callback to hand the run. The
    functions it wraps are put back on leaving."""
    from haemolynx import graph
    from haemolynx.pipeline import stages
    from haemolynx.preprocessing import thick_vessels

    (out / "steps").mkdir(parents=True, exist_ok=True)
    counter = [0]
    in_finish = [False]
    restore: list[tuple[Any, str, Any]] = []

    def snap(label: str, G) -> None:
        counter[0] += 1
        with open(out / "steps" / f"{counter[0]:02d}_{label}.pkl", "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    def patch(owner, name: str, replacement) -> None:
        restore.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    real_join = thick_vessels._join_thin_arms_to_fat_ridge

    def join(skeleton, *args, **kwargs):
        save_packed(out / "skel_before_join.npz", skeleton)
        result = real_join(skeleton, *args, **kwargs)
        save_packed(out / "skel_after_join.npz", result)
        return result

    patch(thick_vessels, "_join_thin_arms_to_fat_ridge", join)

    def after(owner, name: str, label: str, *, returns_graph: bool) -> None:
        real = getattr(owner, name)

        def wrapped(G, *args, **kwargs):
            result = real(G, *args, **kwargs)
            if in_finish[0]:
                snap(label, result if returns_graph else G)
            return result

        patch(owner, name, wrapped)

    # finish_network_graph's steps, read through the modules it calls them on.
    after(graph, "smooth_graph_centrelines", "finish_smoothing", returns_graph=False)
    after(graph.assemble, "consolidate_lumen", "finish_consolidate_lumen", returns_graph=True)
    after(graph.skeleton_retrace, "retrace_lost_skeleton", "finish_retrace", returns_graph=False)
    after(graph, "smooth_straight_steps", "finish_smooth_straight_steps", returns_graph=False)

    real_split = graph.insert_thick_vessel_junction_nodes

    def split(G, *args, **kwargs):
        with open(out / "split_call.pkl", "wb") as f:
            pickle.dump({"G": G}, f, protocol=pickle.HIGHEST_PROTOCOL)
        result = real_split(G, *args, **kwargs)
        snap("finish_thick_split", result)
        return result

    patch(graph, "insert_thick_vessel_junction_nodes", split)
    real_finish = stages.finish_network_graph

    def finish(*args, **kwargs):
        in_finish[0] = True
        try:
            return real_finish(*args, **kwargs)
        finally:
            in_finish[0] = False

    patch(stages, "finish_network_graph", finish)

    def on_output(stage: str, output) -> None:
        if stage.startswith(stages.TOPOLOGY_STEP):
            snap(stage[len(stages.TOPOLOGY_STEP):], output)
        elif stage == "skeletonise":
            save_packed(out / "skeleton.npz", output.skeleton)
            if output.image is not None:
                # The segmentation graph building reads, as it reads it.
                from haemolynx.io.load import _to_binary_volume_for_skeletonization

                save_packed(out / "mask.npz", _to_binary_volume_for_skeletonization(output.image))
            if output.thick_vessel_mask is not None:
                save_packed(out / "thick.npz", output.thick_vessel_mask)
            tree = getattr(output, "thick_vessel_centreline", None)
            if tree is not None:
                np.save(out / "tree.npy", np.asarray(tree))
            (out / "voxel_size_zyx.json").write_text(json.dumps([float(v) for v in output.voxel_size_zyx]))
        elif stage == "build_network":
            with open(out / "graph_built.pkl", "wb") as f:
                pickle.dump(output.graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        yield on_output
    finally:
        for owner, name, real in reversed(restore):
            setattr(owner, name, real)


#: A capture builds the graph and stops: nothing drawn or solved.
CAPTURE_OVERRIDES: Mapping[str, Any] = {
    "visualize_results": False,
    "save_step_artifacts": False,
    "do_equiv_resistance_calculation": False,
    "do_skeletonize": True,
    "do_graph_building": True,
}


def capture_run(settings: dict, schema, out: Path) -> dict[str, Any]:
    """Run *settings* to ``build_network`` with :func:`capturing` into *out*."""
    from haemolynx.pipeline import stages

    started = time.perf_counter()
    with capturing(out) as on_output:
        stages.run_pipeline_stages(settings, schema, on_stage_output=on_output, stop_after="build_network")
    return {"seconds": time.perf_counter() - started}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="the segmented image")
    parser.add_argument("--out", required=True)
    parser.add_argument("--set", action="append", default=[], help="name=<json value>, repeatable")
    args = parser.parse_args(argv)
    import haemolynx
    from haemolynx.pipeline import default_schema, resolve_settings

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "run.log", mode="w", encoding="utf-8"), logging.StreamHandler()],
    )
    overrides = {
        "input_path": args.input, "vtk_output_prefix": str(out / "run"), "plot_dir": str(out / "plots"),
        **CAPTURE_OVERRIDES,
    }
    for item in args.set:
        name, value = item.split("=", 1)
        overrides[name] = json.loads(value)
    schema = default_schema()
    settings = resolve_settings(schema=schema, config_path=args.config, overrides=overrides)
    result = capture_run(settings, schema, out)
    result["haemolynx"] = haemolynx.__file__
    (out / "capture.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
