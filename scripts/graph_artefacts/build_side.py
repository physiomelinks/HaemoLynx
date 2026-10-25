"""Build one side of a graph-artefact comparison: ``python build_side.py <job.json>``.

The job names the ``src`` directory to import ``haemolynx`` from, so an older
version (a ``git archive`` of a ref) builds with its own code. It writes the
graph after every topology step (``steps/<label>.pkl``), the graph as
``build_network`` saves it (``final.pkl``: after smoothing, the clean-up run
again on the smoothed curves, and -- when the job names a thick-vessel region
-- the thick-vessel split, with the job's ``split`` arguments), the graph before that split
(``before_thick_split.pkl``) and ``side.json`` (timings, the process's peak
resident memory, and the build arguments this version does not take). A job
naming a ``raw`` image (the run's ``raw_image_*`` settings) also loads and
calibrates it after the graph, as ``build_network`` does, and ``side.json``
says what that cost (:func:`raw_cost`); ``keep_steps: false`` times the steps
without saving them.
"""
from __future__ import annotations

import inspect
import json
import os
import pickle
import sys
import time
from typing import Any, Callable


def accepted_arguments(func: Callable, kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """``(the kwargs *func* takes, the names it does not)``: an older version
    lacks the newer settings, and is built without them."""
    parameters = inspect.signature(func).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(kwargs), []
    taken = {name: value for name, value in kwargs.items() if name in parameters}
    return taken, sorted(set(kwargs) - set(taken))


#: The build arguments the end-of-build clean-up reads too.
_CLEANUP_ARGUMENTS = (
    "min_stub_length",
    "min_stub_length_radius_multiple",
    "bridge_require_mask_support",
    "bridge_max_background_gap_um",
    "bridge_min_mask_fraction",
    "stub_face_protection",
    "remove_short_loops_under_um",
    "short_loop_keep_hole_over_um",
)


def peak_rss_mb() -> float | None:
    """This process's peak resident memory so far, in MB, or None where the
    version's ``pipeline.provenance`` cannot read it."""
    try:
        from haemolynx.pipeline.provenance import peak_rss_bytes
    except ImportError:
        return None
    peak = peak_rss_bytes()
    return None if peak is None else peak / 2**20


def raw_cost(raw_settings: dict[str, Any], mask, voxel_size) -> dict[str, Any]:
    """What the raw image costs a build, paid as ``build_network`` pays it:
    the channel loaded and calibrated round *mask*
    (``preprocessing.raw_evidence_from_settings``), the RAM it holds, and the
    windows the build's rules read of it -- none while no rule reads it.
    Empty for a version without the raw-image reader."""
    try:
        from haemolynx.preprocessing import raw_evidence
    except ImportError:
        return {}
    started = time.perf_counter()
    evidence = raw_evidence.raw_evidence_from_settings(raw_settings, mask, voxel_size)
    cost: dict[str, Any] = {"seconds": time.perf_counter() - started, "stain": None}
    if evidence is not None:
        held = getattr(raw_evidence, "resident_bytes", None)
        cost.update(
            stain=evidence.calibration.stain,
            held_mb=None if held is None else held(evidence.raw) / 2**20,
            queries=evidence.stats.queries,
            voxels_read=evidence.stats.voxels_read,
            read_seconds=evidence.stats.seconds,
        )
    return cost


def main(job_path: str) -> None:
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    sys.path.insert(0, job["src"])
    import numpy as np

    from haemolynx import graph
    from haemolynx.graph import assemble

    out = job["out"]
    os.makedirs(os.path.join(out, "steps"), exist_ok=True)
    voxel_size = tuple(float(v) for v in job["voxel_size_zyx"])
    skeleton = np.load(job["skeleton"])
    mask = np.load(job["mask"])
    radius_at = assemble.mask_radius_sampler(mask, voxel_size, 1.0)
    build, dropped = accepted_arguments(assemble.build_graph_from_skeleton, job["build"])
    step_seconds: dict[str, float] = {}
    last = [time.perf_counter()]

    keep_steps = job.get("keep_steps", True)

    def snapshot(G, label):
        now = time.perf_counter()
        step_seconds[label] = now - last[0]
        last[0] = now
        if keep_steps:
            with open(os.path.join(out, "steps", f"{label}.pkl"), "wb") as f:
                pickle.dump(G, f)

    started = time.perf_counter()
    G = assemble.build_graph_from_skeleton(
        skeleton, voxel_size=voxel_size, stub_radius_at=radius_at,
        segmentation_mask=mask, step_callback=snapshot, **build,
    )
    smoothing = job.get("smoothing")
    # As build_network does, in the versions that have it: the clean-up
    # again after smoothing and, where the version does so, after the split.
    cleanup = None
    if hasattr(assemble, "consolidate_lumen"):
        cleanup = assemble.lumen_cleanup(
            mask, voxel_size, image_shape=skeleton.shape, stub_radius_at=radius_at,
            **accepted_arguments(
                assemble.lumen_cleanup,
                {name: job["build"][name] for name in _CLEANUP_ARGUMENTS if name in job["build"]},
            )[0],
        )
    if smoothing:
        graph.smooth_graph_centrelines(
            G, skeleton, voxel_size_zyx=voxel_size, method=smoothing["method"],
            iterations=smoothing["iterations"], max_deviation=smoothing["max_deviation"],
            radius_at=radius_at,
        )
        if cleanup is not None:
            G = assemble.consolidate_lumen(G, cleanup)
    thick_split = False
    if job.get("thick_vessel_mask"):
        # As build_network does last, before it saves the graph.
        with open(os.path.join(out, "before_thick_split.pkl"), "wb") as f:
            pickle.dump(G, f)
        # The segmentation too, in the versions whose split reads it.
        split = graph.insert_thick_vessel_junction_nodes
        extra = {"segmentation": mask} if "segmentation" in inspect.signature(split).parameters else {}
        # The run's own choices (``thick_junction_bridge_arguments``), those
        # this version takes.
        extra.update(accepted_arguments(split, job.get("split") or {})[0])
        G = split(G, np.load(job["thick_vessel_mask"]), voxel_size_zyx=voxel_size, **extra)
        thick_split = True
        # A version whose clean-up knows bridges runs it again after the split.
        if cleanup is not None and hasattr(graph.thick_vessel_junctions, "remove_dangling_bridges"):
            G = assemble.consolidate_lumen(G, cleanup)
    # As build_network does, in the versions that have it: the chords the
    # clean-ups drew after smoothing, smoothed.
    if smoothing and hasattr(graph, "smooth_straight_steps"):
        graph.smooth_straight_steps(
            G, skeleton, voxel_size_zyx=voxel_size, method=smoothing["method"],
            iterations=smoothing["iterations"], max_deviation=smoothing["max_deviation"],
            radius_at=radius_at,
        )
    with open(os.path.join(out, "final.pkl"), "wb") as f:
        pickle.dump(G, f)
    side = {
        "seconds": time.perf_counter() - started,
        "step_seconds": step_seconds,
        "steps": list(step_seconds),
        "arguments_not_taken": dropped,
        "thick_split": thick_split,
    }
    if job.get("raw"):
        # As build_network does: after the graph, with the mask still held.
        side["raw"] = raw_cost(job["raw"], mask, voxel_size)
        side["seconds"] = time.perf_counter() - started
    side["peak_rss_mb"] = peak_rss_mb()
    with open(os.path.join(out, "side.json"), "w", encoding="utf-8") as f:
        json.dump(side, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
