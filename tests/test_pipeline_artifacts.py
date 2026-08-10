"""Tests for visualization.pipeline_artifacts.save_graph_snapshot.

The pipeline calls this after each topology step, and the resulting pickles are
what anyone debugging a bad graph goes back to. The snapshot therefore has to be
a faithful, point-in-time copy of the graph, and its filename has to identify
which step produced it — two snapshots that collide overwrite each other and the
step that actually broke the topology becomes invisible.
"""
from __future__ import annotations

import pickle

import networkx as nx
import numpy as np
import pytest

from haemolynx.visualization import save_graph_snapshot


@pytest.fixture
def snapshot_dirs(tmp_path):
    output_dir = tmp_path / "outputs"
    plot_dir = tmp_path / "plots"
    output_dir.mkdir()
    plot_dir.mkdir()
    return output_dir, plot_dir


@pytest.fixture
def snapshot_graph():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 1.0, 2.0]))
    G.add_node(1, pos=np.array([8.0, 1.0, 2.0]))
    G.add_node(2, pos=np.array([8.0, 6.0, 2.0]))
    G.add_edge(0, 1, length=8.0, branch_order="BO1",
               voxels=[(0.0, 1.0, 2.0), (4.0, 1.0, 2.0), (8.0, 1.0, 2.0)])
    G.add_edge(1, 2, length=5.0, branch_order="BO2",
               voxels=[(8.0, 1.0, 2.0), (8.0, 6.0, 2.0)])
    G.graph["voxel_size"] = (2.0, 0.5, 0.4)
    return G


@pytest.fixture
def snapshot_image():
    return np.zeros((10, 10, 10), dtype=np.uint8)


@pytest.mark.plotting
def test_snapshot_writes_both_a_pickle_and_a_plot(
    snapshot_dirs, snapshot_graph, snapshot_image
):
    output_dir, plot_dir = snapshot_dirs
    save_graph_snapshot(
        snapshot_graph, snapshot_image, output_dir, plot_dir, "sample", "prune_stubs"
    )

    pickle_path = output_dir / "sample_graph_after_prune_stubs.pkl"
    plot_path = plot_dir / "graph_after_prune_stubs.png"
    assert pickle_path.is_file()
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0


@pytest.mark.plotting
def test_snapshot_filenames_are_derived_from_the_step_name(
    snapshot_dirs, snapshot_graph, snapshot_image
):
    """Steps that share a filename overwrite each other, hiding the one that broke."""
    output_dir, plot_dir = snapshot_dirs
    for step in ("collapse_clusters", "prune_stubs"):
        save_graph_snapshot(
            snapshot_graph, snapshot_image, output_dir, plot_dir, "sample", step
        )

    assert {path.name for path in output_dir.glob("*.pkl")} == {
        "sample_graph_after_collapse_clusters.pkl",
        "sample_graph_after_prune_stubs.pkl",
    }
    assert {path.name for path in plot_dir.glob("*.png")} == {
        "graph_after_collapse_clusters.png",
        "graph_after_prune_stubs.png",
    }


@pytest.mark.plotting
def test_spaces_in_a_step_name_become_underscores_in_both_filenames(
    snapshot_dirs, snapshot_graph, snapshot_image
):
    """Step labels are prose; spaces in paths break downstream globbing and shell use."""
    output_dir, plot_dir = snapshot_dirs
    save_graph_snapshot(
        snapshot_graph, snapshot_image, output_dir, plot_dir, "sample", "  degree 2 removal  "
    )

    assert (output_dir / "sample_graph_after_degree_2_removal.pkl").is_file()
    assert (plot_dir / "graph_after_degree_2_removal.png").is_file()
    assert not any(" " in path.name for path in output_dir.iterdir())
    assert not any(" " in path.name for path in plot_dir.iterdir())


@pytest.mark.plotting
def test_the_pickle_round_trips_topology_and_edge_attributes(
    snapshot_dirs, snapshot_graph, snapshot_image
):
    """A snapshot that loses `length` or `pos` is useless for diagnosing a bad graph."""
    output_dir, plot_dir = snapshot_dirs
    save_graph_snapshot(
        snapshot_graph, snapshot_image, output_dir, plot_dir, "sample", "build"
    )

    with (output_dir / "sample_graph_after_build.pkl").open("rb") as handle:
        restored = pickle.load(handle)

    assert isinstance(restored, nx.MultiGraph)
    assert restored.number_of_nodes() == 3
    assert restored.number_of_edges() == 2
    assert restored.graph["voxel_size"] == (2.0, 0.5, 0.4)
    assert np.allclose(restored.nodes[2]["pos"], [8.0, 6.0, 2.0])
    lengths = sorted(float(data["length"]) for _u, _v, data in restored.edges(data=True))
    assert lengths == pytest.approx([5.0, 8.0])
    assert restored.edges[0, 1, 0]["branch_order"] == "BO1"


@pytest.mark.plotting
def test_the_snapshot_captures_the_graph_as_it_was_at_the_call(
    snapshot_dirs, snapshot_graph, snapshot_image
):
    """Later steps mutate the same object; a snapshot must not follow those edits."""
    output_dir, plot_dir = snapshot_dirs
    save_graph_snapshot(
        snapshot_graph, snapshot_image, output_dir, plot_dir, "sample", "before"
    )

    snapshot_graph.remove_node(2)
    snapshot_graph.add_node(99, pos=np.array([0.0, 0.0, 0.0]))

    with (output_dir / "sample_graph_after_before.pkl").open("rb") as handle:
        restored = pickle.load(handle)

    assert restored.number_of_nodes() == 3
    assert 2 in restored.nodes
    assert 99 not in restored.nodes


@pytest.mark.plotting
def test_snapshots_of_different_images_do_not_collide_in_the_output_dir(
    snapshot_dirs, snapshot_graph, snapshot_image
):
    """Runs over several stacks share an output dir; the stem keeps them apart."""
    output_dir, plot_dir = snapshot_dirs
    for stem in ("stack_a", "stack_b"):
        save_graph_snapshot(
            snapshot_graph, snapshot_image, output_dir, plot_dir, stem, "build"
        )

    assert (output_dir / "stack_a_graph_after_build.pkl").is_file()
    assert (output_dir / "stack_b_graph_after_build.pkl").is_file()


@pytest.mark.plotting
def test_extra_plot_names_get_the_same_figure_without_redrawing(
    snapshot_dirs, snapshot_graph, snapshot_image, monkeypatch
):
    """A second filename must cost a file write, not a second render.

    The pipeline wants the same overlay under two names. Drawing it twice was
    the single most expensive thing a large run did, and the two files came out
    byte-identical anyway.
    """
    from haemolynx.visualization import pipeline_artifacts

    renders = []
    real_render = pipeline_artifacts.visualize_edges_and_nodes

    def counting_render(*args, **kwargs):
        renders.append(kwargs.get("save_path"))
        return real_render(*args, **kwargs)

    monkeypatch.setattr(pipeline_artifacts, "visualize_edges_and_nodes", counting_render)

    output_dir, plot_dir = snapshot_dirs
    save_graph_snapshot(
        snapshot_graph,
        snapshot_image,
        output_dir,
        plot_dir,
        "sample",
        "smart_multigraph_degree2_removal_pass1",
        extra_plot_names=("smart_multigraph_degree2_removal",),
    )

    assert len(renders) == 1, "the overlay was drawn more than once"
    canonical = plot_dir / "graph_after_smart_multigraph_degree2_removal_pass1.png"
    alias = plot_dir / "smart_multigraph_degree2_removal.png"
    assert canonical.is_file() and alias.is_file()
    assert canonical.read_bytes() == alias.read_bytes()


@pytest.mark.plotting
def test_a_supplied_projection_is_used_instead_of_reprojecting(
    snapshot_dirs, snapshot_graph, snapshot_image, monkeypatch
):
    """Projecting reads the whole stack; the pipeline passes one in for reuse."""
    from haemolynx.visualization import plot as plot_mod

    calls = []
    real_projection = plot_mod.overlay_z_projection

    def counting_projection(image):
        calls.append(image.shape)
        return real_projection(image)

    monkeypatch.setattr(plot_mod, "overlay_z_projection", counting_projection)

    output_dir, plot_dir = snapshot_dirs
    projection = real_projection(snapshot_image)
    for step in ("build", "prune", "collapse"):
        save_graph_snapshot(
            snapshot_graph,
            snapshot_image,
            output_dir,
            plot_dir,
            "sample",
            step,
            projection=projection,
        )

    assert calls == [], "the image was re-projected despite a projection being supplied"
    assert len(list(plot_dir.glob("*.png"))) == 3
