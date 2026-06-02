"""Inline plot display helpers for the pipeline tutorial notebook."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from ImageLynx import visualization


def in_jupyter() -> bool:
    """Return True when running inside an IPython/Jupyter kernel."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        kernel = getattr(shell, "kernel", None)
        return kernel is not None
    except ImportError:
        return False


def show_saved_plot(
    path: Path | str,
    *,
    title: str | None = None,
    width: int | None = 900,
) -> bool:
    """Display a saved PNG in a Jupyter notebook. Returns True if displayed."""
    plot_path = Path(path)
    if not plot_path.is_file():
        print(f"(plot not found: {plot_path})")
        return False
    if not in_jupyter():
        print(f"Saved plot: {plot_path}")
        return False

    from IPython.display import Image, Markdown, display

    if title:
        display(Markdown(f"**{title}**"))
    display(Image(filename=str(plot_path), width=width))
    return True


def show_stage_plots(
    stage_title: str,
    plot_paths: list[Path | str],
    *,
    enabled: bool = True,
    width: int | None = 900,
) -> None:
    """Show a titled group of saved figures inline (notebook only)."""
    if not enabled:
        return
    if not in_jupyter():
        return

    from IPython.display import Markdown, display

    display(Markdown(f"### {stage_title}"))
    for plot_path in plot_paths:
        show_saved_plot(plot_path, width=width)


class GraphBuildPlotter:
    """Save (and optionally show) graph overlays after each topology step."""

    def __init__(
        self,
        image: np.ndarray,
        plot_dir: Path,
        *,
        show_inline: bool = False,
        label_nodes: bool = False,
        subdir: str = "graph_steps",
    ) -> None:
        self.image = image
        self.plot_dir = Path(plot_dir) / subdir
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.show_inline = show_inline
        self.label_nodes = label_nodes
        self.saved: list[tuple[str, Path]] = []

    def __call__(self, graph_obj: nx.MultiGraph, label: str) -> None:
        plot_name = label
        if label == "smart_multigraph_degree2_removal_pass1":
            plot_name = "smart_multigraph_degree2_removal"
        plot_path = self.plot_dir / f"{plot_name}.png"
        visualization.visualize_edges_and_nodes(
            self.image,
            graph_obj,
            label_nodes=self.label_nodes,
            save_path=plot_path,
            show=False,
        )
        self.saved.append((label, plot_path))
        if self.show_inline:
            show_saved_plot(plot_path, title=label)

    def display_all(
        self,
        stage_title: str,
        *,
        enabled: bool = True,
        width: int | None = 900,
    ) -> None:
        """Show every step plot saved during graph construction."""
        if not self.saved:
            return
        show_stage_plots(
            stage_title,
            [path for _label, path in self.saved],
            enabled=enabled,
            width=width,
        )

    def plot_paths(self) -> list[Path]:
        return [path for _label, path in self.saved]


def make_graph_build_step_callback(
    image: np.ndarray,
    plot_dir: Path,
    *,
    show_inline: bool = False,
    label_nodes: bool = False,
) -> GraphBuildPlotter:
    """Compatibility wrapper returning a :class:`GraphBuildPlotter` instance."""
    return GraphBuildPlotter(
        image,
        plot_dir,
        show_inline=show_inline,
        label_nodes=label_nodes,
    )
