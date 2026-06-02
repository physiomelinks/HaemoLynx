"""Inline plot display helpers for the pipeline tutorial notebook."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from ImageLynx import visualization

# Milestone topology steps shown in the tutorial (full pipeline still runs all 11).
TUTORIAL_GRAPH_STEPS_TO_DISPLAY: frozenset[str] = frozenset(
    {
        "build_graph_segment_skan_stitched_loops",
        "optimise_graph_topology_fixed",
        "collapse_node_clusters",
        "prune_vascular_stubs",
        "reconnect_orphan_and_dangling_nodes",
    }
)

TUTORIAL_GRAPH_STEP_TITLES: dict[str, str] = {
    "build_graph_segment_skan_stitched_loops": "Initial graph (skan + loop stitching)",
    "optimise_graph_topology_fixed": "After topology optimisation",
    "collapse_node_clusters": "After node-cluster collapse",
    "prune_vascular_stubs": "After stub pruning",
    "reconnect_orphan_and_dangling_nodes": "After orphan / dangling reconnection",
}


def in_jupyter() -> bool:
    """Return True when running inside an IPython frontend (notebook, VS Code, Cursor, etc.)."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        if shell_name == "TerminalInteractiveShell":
            return False
        if shell_name in {
            "ZMQInteractiveShell",
            "GoogleColabShell",
            "ColabInteractiveShell",
        }:
            return True
        if getattr(shell, "kernel", None) is not None:
            return True
        config = getattr(shell, "config", None)
        if config is not None and "IPKernelApp" in config:
            return True
        return hasattr(shell, "user_ns")
    except ImportError:
        return False


def _resolve_plot_path(path: Path | str) -> Path:
    """Resolve plot path (handles relative PLOT_DIR from any notebook cwd)."""
    return Path(path).expanduser().resolve()


def show_saved_plot(
    path: Path | str,
    *,
    title: str | None = None,
    width: int | None = 900,
) -> bool:
    """Display a saved PNG inline when an IPython frontend is active."""
    plot_path = _resolve_plot_path(path)
    if not plot_path.is_file():
        print(f"(plot not found: {plot_path})")
        return False
    if not in_jupyter():
        print(f"Saved plot: {plot_path}")
        return False

    try:
        from IPython.display import Image, Markdown, display
    except ImportError:
        print(f"Saved plot: {plot_path}")
        return False

    if title:
        display(Markdown(f"**{title}**"))
    display(Image(filename=str(plot_path), width=width, embed=True))
    return True


def show_stage_plots(
    stage_title: str,
    plot_paths: list[Path | str],
    *,
    enabled: bool = True,
    width: int | None = 900,
) -> None:
    """Show a titled group of saved figures inline (notebook / IPython only)."""
    if not enabled:
        return
    if not in_jupyter():
        print(
            f"{stage_title}: inline plots skipped (not in an IPython notebook kernel). "
            f"Plots are saved under the tutorial plot directory."
        )
        return

    from IPython.display import Markdown, display

    display(Markdown(f"### {stage_title}"))
    shown = 0
    for plot_path in plot_paths:
        if show_saved_plot(plot_path, width=width):
            shown += 1
    if shown == 0:
        print(f"{stage_title}: no plot files were found to display.")


class GraphBuildPlotter:
    """Save (and optionally show) graph overlays after selected topology steps."""

    def __init__(
        self,
        image: np.ndarray,
        plot_dir: Path,
        *,
        show_inline: bool = False,
        label_nodes: bool = False,
        subdir: str = "graph_steps",
        steps_to_display: frozenset[str] | None = TUTORIAL_GRAPH_STEPS_TO_DISPLAY,
    ) -> None:
        self.image = image
        self.plot_dir = Path(plot_dir) / subdir
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.show_inline = show_inline
        self.label_nodes = label_nodes
        self.steps_to_display = steps_to_display
        self.saved: list[tuple[str, Path]] = []

    def __call__(self, graph_obj: nx.MultiGraph, label: str) -> None:
        if self.steps_to_display is not None and label not in self.steps_to_display:
            return
        plot_path = self.plot_dir / f"{label}.png"
        visualization.visualize_edges_and_nodes(
            self.image,
            graph_obj,
            label_nodes=self.label_nodes,
            save_path=plot_path,
            show=False,
        )
        self.saved.append((label, plot_path))
        if self.show_inline:
            title = TUTORIAL_GRAPH_STEP_TITLES.get(label, label)
            show_saved_plot(plot_path, title=title)

    def display_all(
        self,
        stage_title: str,
        *,
        enabled: bool = True,
        width: int | None = 900,
    ) -> None:
        """Show milestone step plots saved during graph construction."""
        if not self.saved:
            return
        if not enabled:
            return
        if not in_jupyter():
            print(
                f"{stage_title}: inline plots skipped (not in an IPython notebook kernel)."
            )
            return

        from IPython.display import Markdown, display

        display(Markdown(f"### {stage_title}"))
        for label, plot_path in self.saved:
            title = TUTORIAL_GRAPH_STEP_TITLES.get(label, label)
            show_saved_plot(plot_path, title=title, width=width)

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
