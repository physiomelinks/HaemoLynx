"""The shipped defaults must find inlets and outlets in an unfamiliar image.

They used to name six coordinates from one brain stack and an empty list of
outlet coordinates, so every run on any other image ended at

    ValueError: No inlets or outlet nodes found from manual input coordinates.

before a single vessel was solved. These tests pin the two properties that
stops: the schema's own defaults select non-empty, disjoint inlets and outlet
sets, and a configuration that genuinely cannot be satisfied says which setting
to change rather than that nothing was found.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from haemolynx import graph
from haemolynx.pipeline import default_schema, stages

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_TIFF = REPO_ROOT / "tests" / "data" / "seven_vessel_noisy_3d.tif"

IMAGE_SHAPE = (48, 48, 48)


def _defaults(**overrides) -> dict:
    """The schema's defaults, as a run that overrides nothing would see them."""
    settings = default_schema().defaults()
    settings.update(overrides)
    return settings


def _branching_network() -> nx.MultiGraph:
    """A Y-shaped network whose terminals stop short of the image border."""
    positions = {
        0: (24.0, 8.0, 24.0),
        1: (24.0, 20.0, 24.0),
        2: (24.0, 32.0, 12.0),
        3: (24.0, 32.0, 36.0),
        4: (24.0, 41.0, 8.0),
        5: (24.0, 41.0, 40.0),
    }
    G = nx.MultiGraph()
    for node_id, position in positions.items():
        G.add_node(node_id, pos=np.asarray(position, dtype=float))
    for u, v in ((0, 1), (1, 2), (1, 3), (2, 4), (3, 5)):
        G.add_edge(u, v, length=1.0)
    return G


def _select_both_roles(G, settings, image_shape=IMAGE_SHAPE):
    inlets = graph.select_boundary_nodes_for_role(G, image_shape, settings, "inlet")
    outlets = graph.select_boundary_nodes_for_role(
        G, image_shape, settings, "outlet", exclude_nodes=inlets
    )
    return inlets, outlets


# --- the defaults on their own ---------------------------------------------


def test_the_defaults_select_inlets_and_outlets_from_a_graph_alone():
    inlets, outlets = _select_both_roles(_branching_network(), _defaults())

    assert inlets, "the default settings found no inlet"
    assert outlets, "the default settings found no outlet"
    assert set(inlets).isdisjoint(outlets)


def test_the_defaults_name_no_dataset_of_their_own():
    """A default coordinate or box describes one image and misleads on the rest."""
    defaults = _defaults()

    assert defaults["inlet_node_selection_method"] == "edge_percent"
    assert defaults["outlet_node_selection_method"] == "edge_percent"
    for name in (
        "inlet_node_coordinates",
        "outlet_node_coordinates",
        "inlet_node_volumes",
        "outlet_node_volumes",
    ):
        assert defaults[name] == [], f"{name} defaults to values from one dataset"


def test_the_boundary_axis_setting_reaches_the_selector():
    """A stack whose flow runs along z has to be able to say so."""
    G = _branching_network()

    along_y = _select_both_roles(G, _defaults(boundary_axis=1))
    along_z = _select_both_roles(G, _defaults(boundary_axis=0))

    assert along_y == ([0], [4, 5])
    # Every node sits at z=24, so no band along z can separate two of them.
    assert along_z[1] == []


def test_the_band_width_settings_reach_the_selector():
    G = _branching_network()
    G.add_node(6, pos=np.asarray((24.0, 26.0, 24.0)))
    G.add_edge(1, 6, length=1.0)

    narrow = _select_both_roles(G, _defaults())
    wide = _select_both_roles(G, _defaults(boundary_first_percent=60.0))

    assert narrow[0] == [0]
    assert wide[0] == [0, 6]


# --- what an unsatisfiable configuration says ------------------------------


def test_an_empty_coordinate_list_names_the_setting_to_fix():
    settings = _defaults(
        outlet_node_selection_method="coordinates", outlet_node_coordinates=[]
    )

    with pytest.raises(ValueError, match="outlet_node_coordinates"):
        _select_both_roles(_branching_network(), settings)


def test_an_empty_volume_list_names_the_setting_to_fix():
    settings = _defaults(
        inlet_node_selection_method="volume", inlet_node_volumes=[]
    )

    with pytest.raises(ValueError, match="inlet_node_volumes"):
        _select_both_roles(_branching_network(), settings)


def test_distance_from_inlet_nodes_that_do_not_exist_yet_names_the_method():
    settings = _defaults(inlet_node_selection_method="degree_1_from_inlet")

    with pytest.raises(ValueError, match="inlet_node_selection_method"):
        _select_both_roles(_branching_network(), settings)


def test_a_stage_that_finds_no_outlets_names_both_methods(tmp_path):
    """`all_degree_1` for both roles leaves no terminal for the outlets.

    The inlets take every terminal and the outlet call excludes them, so the
    run ends with nothing to solve between. The message has to say which two
    settings did that.
    """
    settings = _defaults(
        inlet_node_selection_method="all_degree_1",
        outlet_node_selection_method="all_degree_1",
        plot_dir=tmp_path,
    )
    network = _network_for_stages(_branching_network(), tmp_path)

    with pytest.raises(ValueError) as failure:
        stages.assign_boundaries(settings, network)

    message = str(failure.value)
    assert "inlet_node_selection_method" in message
    assert "outlet_node_selection_method" in message
    assert "'all_degree_1'" in message


def test_a_network_with_no_span_to_split_names_both_methods(tmp_path):
    """Two terminals at the same position along the axis cannot be told apart."""
    G = nx.MultiGraph()
    for node_id, position in {
        0: (2.0, 4.0, 2.0),
        1: (2.0, 4.0, 4.0),
        2: (2.0, 4.0, 6.0),
    }.items():
        G.add_node(node_id, pos=np.asarray(position, dtype=float))
    G.add_edge(0, 1, length=1.0)
    G.add_edge(1, 2, length=1.0)

    settings = _defaults(plot_dir=tmp_path)
    network = _network_for_stages(G, tmp_path)

    with pytest.raises(ValueError, match="boundary_axis"):
        stages.assign_boundaries(settings, network)


def _network_for_stages(G: nx.MultiGraph, output_dir: Path) -> stages.VesselNetwork:
    """Wrap a hand-built graph in what ``assign_boundaries`` reads."""
    image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    volume = stages.SkeletonisedVolume(
        image=image,
        skeleton=image.astype(bool),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=output_dir,
    )
    return stages.VesselNetwork(graph=G, volume=volume)


# --- the defaults on a real image ------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_the_defaults_find_inlets_and_outlets_on_the_fixture_image(tmp_path):
    """The whole point: an image the settings have never seen still runs.

    Nothing here is dataset-specific but the input path, so this is what a user
    dropping an image into the napari panel and pressing Run gets.
    """
    pytest.importorskip("skan")
    input_tiff = tmp_path / "seven_vessel_noisy_3d.tif"
    shutil.copy(FIXTURE_TIFF, input_tiff)

    schema = default_schema()
    settings = _defaults(
        input_path=input_tiff,
        # Every artefact the stages write is placed beside the VTK prefix.
        vtk_output_prefix=tmp_path / "outlets" / "network",
        base_plot_dir=tmp_path / "plots",
        plot_dir=tmp_path / "plots",
        verbose_logging=False,
    )

    segmented = stages.segment(settings)
    volume = stages.skeletonise(settings, segmented)
    network = stages.build_network(settings, volume, schema)
    boundaries = stages.assign_boundaries(settings, network)

    assert boundaries.inlet_nodes, "no inlets found with the shipped defaults"
    assert boundaries.outlet_nodes, "no outlet found with the shipped defaults"
    assert set(boundaries.inlet_nodes).isdisjoint(boundaries.outlet_nodes)
    assert boundaries.resistance_node_pair is not None
    assert boundaries.resistance_node_pair[0] != boundaries.resistance_node_pair[1]
