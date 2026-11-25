"""maskel's side (Wittmann et al. 2026, MIT): run in its own interpreter.

maskel thins the mask with its own Lee 1994 implementation, optionally
collapses triangle junctions and prunes short spurs, then summarises the
skeleton with skan -- so its lengths and chords are skan's definitions on
maskel's skeleton, and its radius is the mask's distance transform at the
skeleton, as skan's side reads it.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .measures import Measurements, tortuosity_of

RUNNER = Path(__file__).resolve().with_name("maskel_runner.py")

DEFINITIONS = {
    "skeleton": "maskel's own Lee 1994 thinning, per connected object, nothing cleaned (maskel's defaults)",
    "segment": "a skan path on maskel's skeleton",
    "length": "skan `branch-distance` on maskel's skeleton",
    "tortuosity": "`branch-distance` over `euclidean-distance`",
    "diameter": "maskel `mean_diameter`: twice the mean distance transform (to the nearest background voxel's centre) over the path",
    "branch point": "maskel `num_bifurcations`: nodes of degree >= 3 in skan's branch graph, summed over objects",
    "end point": "maskel `num_endpoints`, summed over objects",
}


#: What ``--cleanup`` changes.
CLEANUP_SKELETON = (
    "maskel's own Lee 1994 thinning, per connected object, then its optional clean-ups at their "
    "defaults: triangle junctions collapsed, and one pass removing end branches under 10 um"
)


def measurements(data: dict, *, tool: str = "maskel") -> Measurements:
    """What :mod:`maskel_runner` wrote, as a :class:`Measurements`."""
    definitions = dict(DEFINITIONS)
    if data.get("config", {}).get("prune_spurs"):
        definitions["skeleton"] = CLEANUP_SKELETON
    branches = data["branches"]
    lengths = np.asarray([b["branch-distance"] for b in branches], dtype=float)
    chords = np.asarray([b["euclidean-distance"] for b in branches], dtype=float)
    diameters = np.asarray(
        [np.nan if b["mean_diameter"] is None else b["mean_diameter"] for b in branches], dtype=float
    )
    tortuosity, loops = tortuosity_of(lengths, chords)
    summaries = data["summary"]
    return Measurements(
        tool=tool,
        total_length_um=float(sum(s["total_length"] for s in summaries)),
        segments=int(sum(s["num_edges"] for s in summaries)),
        branch_points=int(sum(s["num_bifurcations"] for s in summaries)),
        end_points=int(sum(s["num_endpoints"] for s in summaries)),
        segment_lengths_um=lengths,
        tortuosity=tortuosity,
        diameters_um=diameters,
        loops=loops,
        definitions=definitions,
        extra={"Objects": len(summaries), "maskel version": data.get("maskel_version")},
    )


def run(maskel_python: Path, mask_npy: Path, voxel_size_zyx, out_json: Path, *, cleanup: bool = False) -> Measurements:
    command = [str(maskel_python), "-I", str(RUNNER), str(mask_npy), str(out_json), "--spacing-zyx",
               *(str(float(v)) for v in voxel_size_zyx)]
    if cleanup:
        command.append("--cleanup")
    subprocess.run(command, check=True)
    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    return measurements(data, tool="maskel (cleanup)" if cleanup else "maskel")
