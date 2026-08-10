"""Helpers for writing pipeline graph artifacts."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np

from .plot import visualize_edges_and_nodes

logger = logging.getLogger(__name__)


def save_graph_snapshot(
    graph: nx.MultiGraph,
    image: np.ndarray,
    output_dir: Path,
    plot_dir: Path,
    image_stem: str,
    step_name: str,
    projection: np.ndarray | None = None,
    extra_plot_names: Sequence[str] = (),
) -> None:
    """Persist graph + PNG snapshot for a named pipeline step.

    ``projection`` is an already-computed Z-projection of *image*; the pipeline
    passes one because it draws the same volume once per topology step.
    ``extra_plot_names`` names further PNGs in *plot_dir* that get the same
    figure, so a second filename costs a file write rather than a second render.
    """
    safe_step = step_name.strip().replace(" ", "_")
    graph_snapshot_path = output_dir / f"{image_stem}_graph_after_{safe_step}.pkl"
    with graph_snapshot_path.open("wb") as handle:
        pickle.dump(graph, handle)
    logger.info(f"Saved graph after '{step_name}': {graph_snapshot_path}")

    plot_snapshot_path = plot_dir / f"graph_after_{safe_step}.png"
    extra_paths = [plot_dir / f"{name}.png" for name in extra_plot_names]
    visualize_edges_and_nodes(
        image,
        graph,
        label_nodes=True,
        save_path=plot_snapshot_path,
        projection=projection,
        extra_save_paths=extra_paths,
    )
    for extra_path in extra_paths:
        logger.info(f"Saved graph plot after '{step_name}': {extra_path}")
    logger.info(f"Saved graph plot after '{step_name}': {plot_snapshot_path}")
