"""Helpers for writing pipeline graph artifacts."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

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
) -> None:
    """Persist graph + PNG snapshot for a named pipeline step."""
    safe_step = step_name.strip().replace(" ", "_")
    graph_snapshot_path = output_dir / f"{image_stem}_graph_after_{safe_step}.pkl"
    with graph_snapshot_path.open("wb") as handle:
        pickle.dump(graph, handle)
    logger.info(f"Saved graph after '{step_name}': {graph_snapshot_path}")

    plot_snapshot_path = plot_dir / f"graph_after_{safe_step}.png"
    visualize_edges_and_nodes(
        image,
        graph,
        label_nodes=True,
        save_path=plot_snapshot_path,
    )
    logger.info(f"Saved graph plot after '{step_name}': {plot_snapshot_path}")
