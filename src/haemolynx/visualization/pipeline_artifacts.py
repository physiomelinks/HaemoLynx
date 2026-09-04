"""Helpers for writing pipeline graph artifacts."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np

from .large_vessel_assignment import visualize_3d_plotly_large_vessel_assignment
from .plot import visualize_3d_plotly, visualize_edges_and_nodes

logger = logging.getLogger(__name__)


def selected_vessel_masks_for_html(
    *,
    use_large_vessel_masks: bool,
    use_small_vessel_masks: bool,
    large_arteriole_mask: np.ndarray | None = None,
    large_venule_mask: np.ndarray | None = None,
    small_arteriole_mask: np.ndarray | None = None,
    small_venule_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray | None]:
    """Return the mask arrays the final-graph HTML should overlay.

    Flags come from the run (``use_large_vessel_masks``,
    ``use_small_vessel_masks_for_boundary_assignment``). Arrays are the
    volumes the pipeline already loaded — full stacks, not a Z-cropped
    viewer copy. A disabled role is ``None`` even if an array was passed.
    """
    large_on = bool(use_large_vessel_masks)
    small_on = bool(use_small_vessel_masks)
    large_pair = (
        large_arteriole_mask is not None and large_venule_mask is not None
    )
    small_pair = (
        small_arteriole_mask is not None and small_venule_mask is not None
    )
    return {
        "large_arteriole_mask": large_arteriole_mask if large_on and large_pair else None,
        "large_venule_mask": large_venule_mask if large_on and large_pair else None,
        "small_arteriole_mask": small_arteriole_mask if small_on and small_pair else None,
        "small_venule_mask": small_venule_mask if small_on and small_pair else None,
    }


def write_final_graph_3d_html(
    graph: nx.Graph,
    *,
    save_html_path: str | Path | None,
    title: str = "Final Graph (Interactive 3D)",
    use_large_vessel_masks: bool = False,
    use_small_vessel_masks: bool = False,
    large_arteriole_mask: np.ndarray | None = None,
    large_venule_mask: np.ndarray | None = None,
    small_arteriole_mask: np.ndarray | None = None,
    small_venule_mask: np.ndarray | None = None,
    input_nodes: Sequence[Any] | None = None,
    output_nodes: Sequence[Any] | None = None,
    arteriole_boundary_nodes: Sequence[Any] | None = None,
    venule_boundary_nodes: Sequence[Any] | None = None,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    volume_downsample_stride: int = 1,
    show: bool = False,
):
    """Write ``final_graph_3d.html`` with pipeline volume overlay when selected.

    When large and/or small vessel masks are selected and loaded, this calls
    :func:`visualize_3d_plotly_large_vessel_assignment` so colours, opacity
    and downsample stride match the pipeline assignment HTML. When neither
    mask type is selected, it falls back to :func:`visualize_3d_plotly`.
    """
    masks = selected_vessel_masks_for_html(
        use_large_vessel_masks=use_large_vessel_masks,
        use_small_vessel_masks=use_small_vessel_masks,
        large_arteriole_mask=large_arteriole_mask,
        large_venule_mask=large_venule_mask,
        small_arteriole_mask=small_arteriole_mask,
        small_venule_mask=small_venule_mask,
    )
    has_volume_overlay = (
        masks["large_arteriole_mask"] is not None
        or masks["small_arteriole_mask"] is not None
    )
    save_path = None if save_html_path is None else str(save_html_path)
    if has_volume_overlay:
        return visualize_3d_plotly_large_vessel_assignment(
            graph,
            large_arteriole_mask=masks["large_arteriole_mask"],
            large_venule_mask=masks["large_venule_mask"],
            small_arteriole_mask=masks["small_arteriole_mask"],
            small_venule_mask=masks["small_venule_mask"],
            input_nodes=list(input_nodes or []),
            output_nodes=list(output_nodes or []),
            arteriole_boundary_nodes=list(arteriole_boundary_nodes or []),
            venule_boundary_nodes=list(venule_boundary_nodes or []),
            voxel_size_zyx=tuple(float(v) for v in voxel_size_zyx),
            volume_downsample_stride=int(volume_downsample_stride),
            title=title,
            save_html_path=save_path,
            show=show,
        )
    return visualize_3d_plotly(
        graph,
        title=title,
        save_html_path=save_path,
        show=show,
    )


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
