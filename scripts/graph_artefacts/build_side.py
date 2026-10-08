"""Build one side of a graph-artefact comparison: ``python build_side.py <job.json>``.

The job names the ``src`` directory to import ``haemolynx`` from, so an older
version (a ``git archive`` of a ref) builds with its own code. It writes the
graph after every topology step (``steps/<label>.pkl``), the final graph after
smoothing (``final.pkl``) and ``side.json`` (timings, and the build arguments
this version does not take).
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
)


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

    def snapshot(G, label):
        now = time.perf_counter()
        step_seconds[label] = now - last[0]
        last[0] = now
        with open(os.path.join(out, "steps", f"{label}.pkl"), "wb") as f:
            pickle.dump(G, f)

    started = time.perf_counter()
    G = assemble.build_graph_from_skeleton(
        skeleton, voxel_size=voxel_size, stub_radius_at=radius_at,
        segmentation_mask=mask, step_callback=snapshot, **build,
    )
    smoothing = job.get("smoothing")
    if smoothing:
        graph.smooth_graph_centrelines(
            G, skeleton, voxel_size_zyx=voxel_size, method=smoothing["method"],
            iterations=smoothing["iterations"], max_deviation=smoothing["max_deviation"],
            radius_at=radius_at,
        )
        # As build_network does, in the versions that have it.
        if hasattr(assemble, "consolidate_lumen"):
            cleanup = assemble.lumen_cleanup(
                mask, voxel_size, image_shape=skeleton.shape, stub_radius_at=radius_at,
                **{name: job["build"][name] for name in _CLEANUP_ARGUMENTS if name in job["build"]},
            )
            G = assemble.consolidate_lumen(G, cleanup)
    with open(os.path.join(out, "final.pkl"), "wb") as f:
        pickle.dump(G, f)
    with open(os.path.join(out, "side.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seconds": time.perf_counter() - started,
            "step_seconds": step_seconds,
            "steps": list(step_seconds),
            "arguments_not_taken": dropped,
        }, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
