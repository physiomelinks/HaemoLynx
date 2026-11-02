"""Phase 0 of PLAN_vessel_classes_from_segmentation.md: E14.5 solved, with its
real large masks' labels on the vessels.

Resumes the release-review E14.5 run (commit 75e856bf, 2026-10-25: large masks
on, mask-stump inlets/outlets, endothelial diameters from the raw image) from
its saved skeleton and graph, runs Boundaries, Diameters, Haemodynamics and the
solve, and pickles the solved graph. Its `large_mask_vessel_type` tags are the
truth the published rules are scored against (score_rules.py).

    python scripts/vessel_class_review/solve_e14p5.py
"""
from __future__ import annotations

import pickle
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE_RUN = REPO / "outputs/release_review/2026-10-25_75e856bf/e14p5"
OUT = REPO / "outputs/vessel_class_review"
RUN_DIR = OUT / "e14p5_run"
SOLVED = OUT / "e14p5_solved.pkl"

#: The saved build: skeleton, graph, thick region and voxel size.
COPIED_SUFFIXES = (
    "_skeleton.npy",
    "_graph.pkl",
    "_thick_vessel_region.npy",
    "_thick_vessel_region.json",
    "_voxel_size.json",
)


def main() -> None:
    sys.path.insert(0, str(REPO / "src"))
    from haemolynx.parsers.config import load_config
    from haemolynx.pipeline.schema import SCHEMA
    from haemolynx.pipeline.settings import resolve_settings
    from haemolynx.pipeline.stages import run_pipeline_stages

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    settings_file = next(SOURCE_RUN.glob("*_run_settings.yaml"))
    for suffix in COPIED_SUFFIXES:
        source = next(SOURCE_RUN.glob(f"*{suffix}"))
        target = RUN_DIR / source.name
        if not target.exists():
            shutil.copy2(source, target)

    # As scripts/release_review/_run_dataset.py: the outputs in RUN_DIR, and
    # the plots beside them.
    values = load_config(settings_file, SCHEMA)
    values.update({
        "output_dir": str(RUN_DIR),
        "plot_dir": str(RUN_DIR / "plots"),
        "do_skeletonize": False,
        "do_graph_building": False,
        "statistics": False,
        "perturbations": [],
        "show_plots_in_ide": False,
        "interactive_plots": False,
        "save_step_artifacts": False,
    })
    settings = resolve_settings(values, schema=SCHEMA, config_path=settings_file)
    started = time.perf_counter()
    G = run_pipeline_stages(settings, SCHEMA, stop_after="solve")
    print(f"solved in {time.perf_counter() - started:.0f} s: "
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)
    with SOLVED.open("wb") as f:
        pickle.dump(G, f)
    print(f"wrote {SOLVED}", flush=True)


if __name__ == "__main__":
    main()
