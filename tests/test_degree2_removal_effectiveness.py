"""Degree-2 removal effectiveness must not deteriorate.

`build_graph_from_skeleton` runs `smart_multigraph_degree2_removal` four times.
A vascular graph should end up with no degree-2 nodes at all: such a node is a
mid-vessel point that carries no topological information and inflates node
counts, resistance-matrix size and every per-node statistic.

Every committed fixture currently reaches exactly zero, so these tests pin that
outcome rather than an upper bound. If a future change leaves any behind, this
fails with the count and the diagnostic reason.
"""
from pathlib import Path

import pytest

from haemolynx import graph, io, preprocessing

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "tests" / "data"

SKELETON_KWARGS = dict(
    min_branch_length=3,
    max_bridge_distance=2,
    component_connectivity=3,
    min_component_fraction=0.01,
    closing_radius=1,
    bridge_gap_size=1,
)
GRAPH_KWARGS = dict(
    voxel_size=(1.0, 1.0, 1.0),
    graph_reconnect_threshold=10.0,
    final_orphan_reconnect_threshold=3.0,
    cluster_collapse_distance=5.0,
    min_stub_length=3.0,
)

# fixture filename -> (expected nodes, expected edges)
FIXTURES = {
    "seven_vessel_noisy_3d.tif": (11, 10),
    "bundled_vessels_8_to_2.h5": (10, 9),
}


def _build(path: Path):
    if path.suffix == ".h5":
        _image, skeleton, *_rest = io.load_and_skeletonize_3d_h5(str(path))
    else:
        _image, skeleton, *_rest = io.load_and_skeletonize_3d_tif(str(path))
    skeleton = preprocessing.preprocess_skeleton_for_graph(skeleton, **SKELETON_KWARGS)
    return graph.build_graph_from_skeleton(skeleton, **GRAPH_KWARGS)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("filename", sorted(FIXTURES))
def test_no_degree2_nodes_remain_after_graph_build(filename):
    pytest.importorskip("skan")
    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip(f"fixture not available: {filename}")

    G = _build(path)
    remaining = [n for n in G.nodes() if G.degree[n] == 2]

    if remaining:
        report = graph.diagnose_degree2_nodes(G, max_degree=4)
        pytest.fail(
            f"{filename}: {len(remaining)} degree-2 node(s) survived graph building "
            f"(was 0). Reasons: {report['reason_counts']}"
        )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("filename", sorted(FIXTURES))
def test_graph_build_topology_is_stable(filename):
    """Node/edge counts alongside the degree-2 check, so a regression is attributable."""
    pytest.importorskip("skan")
    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip(f"fixture not available: {filename}")

    expected_nodes, expected_edges = FIXTURES[filename]
    G = _build(path)
    assert (G.number_of_nodes(), G.number_of_edges()) == (expected_nodes, expected_edges)


@pytest.mark.slow
@pytest.mark.integration
def test_degree2_diagnostics_agree_with_the_graph():
    """The diagnostic used to justify these numbers must itself be correct."""
    pytest.importorskip("skan")
    path = DATA_DIR / "seven_vessel_noisy_3d.tif"
    if not path.exists():
        pytest.skip("fixture not available")

    G = _build(path)
    report = graph.diagnose_degree2_nodes(G, max_degree=4)
    counted = sum(1 for n in G.nodes() if G.degree[n] == 2)
    assert report["total_degree2"] == counted
    assert sum(report["reason_counts"].values()) == counted
