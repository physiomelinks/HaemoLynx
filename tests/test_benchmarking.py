import numpy as np
import pytest
import networkx as nx
import ImageLynx.statistics.benchmarking as bench

def create_straight_vessel_graph(length=10, radius=3.0):
    """Helper fixture to create a perfectly straight 1D graph."""
    G = nx.MultiGraph()
    pts = [(z, 10, 10) for z in range(length)]
    G.add_edge(0, 1, key=0, pts=pts, assigned_diameter_um=radius*2.0, fwhm_diameter_um=radius*2.0)
    return G

def test_volumetric_reconstruction_perfect_match():
    """Core metric: volumetric redilation."""
    G = create_straight_vessel_graph(length=20, radius=3.0)
    shape = (20, 20, 20)
    voxel_size = (1.0, 1.0, 1.0)
    
    # We use the module's own redilation to create a mathematically perfect ground-truth binary mask
    perfect_mask = bench.redilate_skeleton_to_volume(G, shape, voxel_size)
    
    # The reconstruction benchmark should therefore evaluate to exactly 1.0
    results = bench.evaluate_volumetric_reconstruction(G, perfect_mask, voxel_size)
    
    assert results["dice_coefficient"] == 1.0, f"Expected 1.0 DSC, got {results['dice_coefficient']}"
    assert results["jaccard_index"] == 1.0, f"Expected 1.0 Jaccard, got {results['jaccard_index']}"
    assert results["reconstructed_volume_fraction"] == 1.0

def test_centerline_centricity_perfect_alignment():
    """Core metric: distance-transform centricity."""
    G = create_straight_vessel_graph(length=20, radius=4.0)
    shape = (20, 20, 20)
    voxel_size = (1.0, 1.0, 1.0)
    
    perfect_mask = bench.redilate_skeleton_to_volume(G, shape, voxel_size)
    results = bench.evaluate_centerline_centricity(G, perfect_mask, voxel_size)

    # On the axis the voxelised tube's EDT reads sqrt(17) = 4.12 um, 0.12 um above the radius.
    centred = results["mean_centricity_error_um"]
    assert centred < 0.5, f"Centricity error too high: {centred}"

    # The same centreline moved one voxel off the axis reads the larger error (0.84 um).
    shifted = nx.MultiGraph()
    shifted.add_edge(0, 1, key=0, pts=[(z, 11, 10) for z in range(20)],
                     assigned_diameter_um=8.0, fwhm_diameter_um=8.0)
    off_axis = bench.evaluate_centerline_centricity(shifted, perfect_mask, voxel_size)
    assert off_axis["mean_centricity_error_um"] > centred + 0.5

def test_topological_preservation_components_and_loops():
    """Core metric: Euler/Betti topology."""
    # 1. Straight Tube: 1 component, 0 holes
    G_line = create_straight_vessel_graph(length=10)
    mask_line = bench.redilate_skeleton_to_volume(G_line, (20, 20, 20), (1.0, 1.0, 1.0))
    res_line = bench.evaluate_topological_preservation(G_line, mask_line)
    
    assert res_line["binary_mask_connected_components"] == 1
    assert res_line["graph_connected_components"] == 1
    assert res_line["graph_fundamental_loops"] == 0
    
    # 2. Torus/Cycle: 1 component, 1 hole
    G_loop = nx.MultiGraph()
    # A simple triangle loop, but with densely packed points and a large radius so they physically overlap!
    pts_1 = [(x, x, 10) for x in range(5, 10)]
    pts_2 = [(10, y, 10) for y in range(10, 5, -1)]
    pts_3 = [(x, 5, 10) for x in range(10, 5, -1)]
    
    G_loop.add_edge(0, 1, key=0, pts=pts_1, assigned_diameter_um=4.0)
    G_loop.add_edge(1, 2, key=0, pts=pts_2, assigned_diameter_um=4.0)
    G_loop.add_edge(2, 0, key=0, pts=pts_3, assigned_diameter_um=4.0)
    
    mask_loop = bench.redilate_skeleton_to_volume(G_loop, (20, 20, 20), (1.0, 1.0, 1.0))
    res_loop = bench.evaluate_topological_preservation(G_loop, mask_loop)
    
    assert res_loop["graph_fundamental_loops"] == 1
    assert res_loop["binary_mask_connected_components"] == 1

def test_completeness_and_overpruning_detection():
    """Core metric: orphaned-volume detection."""
    G_full = create_straight_vessel_graph(length=20, radius=2.0)
    # Generate the ground truth representing the entire biological vessel
    mask_full = bench.redilate_skeleton_to_volume(G_full, (20, 20, 20), (1.0, 1.0, 1.0))
    
    # Generate a heavily "pruned" skeleton that stops halfway through the vessel
    G_pruned = nx.MultiGraph()
    pts_pruned = [(z, 10, 10) for z in range(10)] # Only z=0 to 9
    G_pruned.add_edge(0, 1, key=0, pts=pts_pruned, assigned_diameter_um=4.0)
    
    # Any tissue voxel further than 3.0um from the skeleton is considered orphaned
    results = bench.evaluate_completeness_and_overpruning(
        G_pruned, mask_full, (1.0, 1.0, 1.0), max_capillary_radius_um=3.0
    )
    
    # Because exactly half of the straight cylinder was pruned away, 
    # but the terminal sphere stretches forward, it will be around 40%.
    orphaned_fraction = results["orphaned_volume_fraction"]
    assert 0.30 < orphaned_fraction < 0.60, f"Expected ~0.40-0.50, got {orphaned_fraction}"

def test_empty_or_missing_keys():
    """Degradation: missing pts or diameter keys."""
    G = nx.MultiGraph()
    # Missing 'pts', using 'voxels'. Missing 'fwhm_diameter_um', relying on 'assigned_diameter_um'
    G.add_edge(0, 1, key=0, voxels=[(5, 5, 5)], assigned_diameter_um=4.0)
    
    mask = bench.redilate_skeleton_to_volume(G, (10, 10, 10), (1.0, 1.0, 1.0))
    
    res_vol = bench.evaluate_volumetric_reconstruction(G, mask, (1.0, 1.0, 1.0))
    assert res_vol["dice_coefficient"] == 1.0
    
    res_comp = bench.evaluate_completeness_and_overpruning(G, mask, (1.0, 1.0, 1.0))
    assert res_comp["orphaned_volume_fraction"] == 0.0

def test_empty_binary_mask():
    """Degradation: total anoxia / empty volumes."""
    G = nx.MultiGraph()
    mask = np.zeros((10, 10, 10), dtype=bool)
    voxel_size = (1.0, 1.0, 1.0)
    
    # Dice and Jaccard of 0.0 are the worst score, so returning them is honest.
    res_vol = bench.evaluate_volumetric_reconstruction(G, mask, voxel_size)
    assert res_vol["dice_coefficient"] == 0.0
    assert res_vol["jaccard_index"] == 0.0

    # Centricity and orphaned volume of 0.0 would read as a perfect score, so they raise.
    with pytest.raises(ValueError, match="no centreline samples"):
        bench.evaluate_centerline_centricity(G, mask, voxel_size)

    with pytest.raises(ValueError, match="mask is empty"):
        bench.evaluate_completeness_and_overpruning(G, mask, voxel_size)

def test_advanced_preprocessing_metrics():
    """Validates the Euler Characteristic and Surface-Area-to-Volume Ratio (Compactness)."""
    # 1. Perfect Solid Cube
    solid_mask = np.zeros((10, 10, 10), dtype=bool)
    solid_mask[2:8, 2:8, 2:8] = True
    
    # 2. Swiss Cheese Cube (Hollow center)
    hollow_mask = solid_mask.copy()
    hollow_mask[4:6, 4:6, 4:6] = False
    
    euler_solid = bench.evaluate_preprocessing_euler_characteristic(solid_mask)
    euler_hollow = bench.evaluate_preprocessing_euler_characteristic(hollow_mask)
    
    # A solid cube has 1 component, 0 tunnels, 0 cavities -> Euler = 1
    assert euler_solid == 1
    # A hollow cube has 1 component, 0 tunnels, 1 cavity -> Euler = 1 - 0 + 1 = 2
    assert euler_hollow == 2
    
    compact_solid = bench.evaluate_preprocessing_compactness(solid_mask)
    compact_hollow = bench.evaluate_preprocessing_compactness(hollow_mask)
    # The hollow mask has MORE surface area (inner + outer) for LESS volume, so the ratio spikes
    assert compact_hollow > compact_solid

def test_advanced_topological_health_metrics():
    """Validates Terminal Node Ratio, Degree-3 Dominance, and Edge Length Variance."""
    G = nx.MultiGraph()
    
    # Create a healthy biological Y-bifurcation
    # 1 inlet (node 0) -> splits at node 1 -> 2 outlets (nodes 2, 3)
    G.add_node(0, pos=(0,0,0)); G.add_node(1, pos=(0,0,10))
    G.add_node(2, pos=(5,0,15)); G.add_node(3, pos=(-5,0,15))
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(1, 3)
    
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[0,0,0] = True # Dummy
    
    res = bench.evaluate_topological_preservation(G, mask)
    
    # Nodes: 0,1,2,3. Degrees: 0(1), 1(3), 2(1), 3(1).
    # Terminal nodes (Degree 1): nodes 0, 2, 3. Total 3. Ratio = 3/4 = 0.75
    assert res["terminal_node_ratio"] == 0.75
    
    # Bifurcation ratio: Only one branch node (node 1, degree 3). Ratio = 1 / 1 = 1.0
    assert res["degree3_bifurcation_ratio"] == 1.0
    
    # Now create an unnatural "Super-Hub" (Degree-4 X-intersection)
    G.add_node(4, pos=(0,5,15))
    G.add_edge(1, 4) # Node 1 is now Degree-4
    
    res_bad = bench.evaluate_topological_preservation(G, mask)
    assert res_bad["degree3_bifurcation_ratio"] == 0.0 # No degree 3 nodes anymore!

POINT_METRICS = [bench.evaluate_centerline_centricity, bench.evaluate_completeness_and_overpruning]


def _tube_and_mask():
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, pts=[(z, 10, 10) for z in range(20)], assigned_diameter_um=8.0)
    return G, bench.redilate_skeleton_to_volume(G, (20, 20, 20), (1.0, 1.0, 1.0))


def test_centricity_measures_against_the_assigned_diameter():
    """The frozen edt_radius path writes assigned_diameter_um, never fwhm_diameter_um."""
    G, mask = _tube_and_mask()

    results = bench.evaluate_centerline_centricity(G, mask, (1.0, 1.0, 1.0))

    # On the axis the voxelised tube's EDT reads sqrt(17) = 4.12 um against a 4 um radius:
    # an error of 0.12 um.
    assert 0.0 < results["mean_centricity_error_um"] < 0.5


@pytest.mark.parametrize("diameter", [None, 0.0, -2.0], ids=["missing", "zero", "negative"])
def test_centricity_raises_on_an_edge_without_a_usable_diameter(diameter):
    G, mask = _tube_and_mask()
    attrs = {"fwhm_diameter_um": 8.0}  # present but not what centricity reads
    if diameter is not None:
        attrs["assigned_diameter_um"] = diameter
    G.add_edge(1, 2, key=0, pts=[(z, 10, 10) for z in range(5)], **attrs)

    with pytest.raises(ValueError, match=r"\(1, 2, 0\).*assigned_diameter_um"):
        bench.evaluate_centerline_centricity(G, mask, (1.0, 1.0, 1.0))


@pytest.mark.parametrize("metric", POINT_METRICS, ids=lambda f: f.__name__)
def test_point_metrics_raise_on_an_edge_with_no_points(metric):
    G, mask = _tube_and_mask()
    G.add_edge(1, 2, key=0, pts=[], voxels=[], assigned_diameter_um=8.0)

    with pytest.raises(ValueError, match=r"\(1, 2, 0\).*no centreline points"):
        metric(G, mask, (1.0, 1.0, 1.0))


@pytest.mark.parametrize("metric", POINT_METRICS, ids=lambda f: f.__name__)
def test_point_metrics_raise_on_a_point_outside_the_volume(metric):
    G, mask = _tube_and_mask()
    G.add_edge(1, 2, key=0, pts=[(19, 10, 10), (25, 10, 10)], assigned_diameter_um=8.0)

    with pytest.raises(ValueError, match=r"\(1, 2, 0\).*outside the volume"):
        metric(G, mask, (1.0, 1.0, 1.0))


def test_run_all_benchmarks_lets_a_metric_error_reach_the_caller():
    """The suite used to log each metric's error and carry on, dropping that metric's key."""
    G = nx.MultiGraph()
    mask = np.zeros((10, 10, 10), dtype=bool)

    with pytest.raises(ValueError, match="no centreline samples"):
        bench.run_all_benchmarks(G, mask, (1.0, 1.0, 1.0))


def test_run_all_benchmarks_runs_only_the_named_metrics():
    """The tuner names the two metrics its loss reads; its graphs carry only an EDT diameter."""
    G, mask = _tube_and_mask()
    edge = G.edges[0, 1, 0]
    edge["edt_diameter_um"] = edge.pop("assigned_diameter_um")  # centricity would raise on this graph

    results = bench.run_all_benchmarks(G, mask, (1.0, 1.0, 1.0), metrics=("volumetric", "completeness"))

    assert set(results) == {"volumetric", "completeness"}


def test_run_all_benchmarks_runs_all_four_by_default():
    G, mask = _tube_and_mask()

    results = bench.run_all_benchmarks(G, mask, (1.0, 1.0, 1.0))

    assert set(results) == {"volumetric", "centricity", "topology", "completeness"}


def test_run_all_benchmarks_raises_on_an_unknown_metric_name():
    G, mask = _tube_and_mask()

    with pytest.raises(ValueError, match="'dice'"):
        bench.run_all_benchmarks(G, mask, (1.0, 1.0, 1.0), metrics=("volumetric", "dice"))


# --- Redilation reads a measured diameter or raises ----------------------------------------

def _single_edge(**attrs):
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, pts=[(z, 10, 10) for z in range(20)], **attrs)
    return G


@pytest.mark.parametrize("diameter", [None, 0.0, -2.0], ids=["missing", "zero", "negative"])
def test_redilation_raises_on_an_edge_without_a_usable_diameter(diameter):
    """It gave such an edge 5 um, so the tuner's Dice was measured on made-up vessels."""
    G = _single_edge() if diameter is None else _single_edge(edt_diameter_um=diameter)

    with pytest.raises(ValueError, match=r"\(0, 1, 0\).*diameter"):
        bench.redilate_skeleton_to_volume(G, (20, 20, 20), (1.0, 1.0, 1.0))


@pytest.mark.parametrize("key", ["assigned_diameter_um", "edt_diameter_um", "fwhm_diameter_um"])
def test_redilation_uses_whichever_measured_diameter_the_edge_carries(key):
    """Tuner graphs carry only edt_diameter_um; a solved graph carries assigned_diameter_um."""
    volume = bench.redilate_skeleton_to_volume(_single_edge(**{key: 4.0}), (20, 20, 20), (1.0, 1.0, 1.0))

    # A 2 um radius about the line y = x = 10: the cross-section is the 13 voxels within 2.
    assert volume[10].sum() == 13


def test_redilation_prefers_the_assigned_diameter_over_the_measurements():
    G = _single_edge(assigned_diameter_um=2.0, edt_diameter_um=8.0, fwhm_diameter_um=8.0)

    volume = bench.redilate_skeleton_to_volume(G, (20, 20, 20), (1.0, 1.0, 1.0))

    # Radius 1: the centre and its four face neighbours.
    assert volume[10].sum() == 5


def test_redilation_prefers_edt_over_fwhm():
    G = _single_edge(edt_diameter_um=2.0, fwhm_diameter_um=8.0)

    volume = bench.redilate_skeleton_to_volume(G, (20, 20, 20), (1.0, 1.0, 1.0))

    assert volume[10].sum() == 5


def test_redilation_raises_on_a_bad_preferred_diameter_rather_than_falling_through():
    G = _single_edge(assigned_diameter_um=0.0, edt_diameter_um=4.0)

    with pytest.raises(ValueError, match=r"\(0, 1, 0\).*non-positive assigned_diameter_um"):
        bench.redilate_skeleton_to_volume(G, (20, 20, 20), (1.0, 1.0, 1.0))


def test_redilation_raises_on_an_edge_with_no_points():
    """It skipped such an edge, so its vessel was missing from the reconstruction."""
    G = _single_edge(assigned_diameter_um=4.0)
    G.add_edge(1, 2, key=0, pts=[], voxels=[], assigned_diameter_um=4.0)

    with pytest.raises(ValueError, match=r"\(1, 2, 0\).*no centreline points"):
        bench.redilate_skeleton_to_volume(G, (20, 20, 20), (1.0, 1.0, 1.0))


def test_redilation_raises_on_a_point_outside_the_volume():
    G = _single_edge(assigned_diameter_um=4.0)
    G.add_edge(1, 2, key=0, pts=[(19, 10, 10), (25, 10, 10)], assigned_diameter_um=4.0)

    with pytest.raises(ValueError, match=r"\(1, 2, 0\).*outside the volume"):
        bench.redilate_skeleton_to_volume(G, (20, 20, 20), (1.0, 1.0, 1.0))


# --- Preprocessing suite: no catch, and no perfect score on an empty mask ------------------

EMPTY = np.zeros((10, 10, 10), dtype=bool)


def _cube_mask():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    return mask


def test_preprocessing_suite_raises_a_named_error_on_an_empty_mask():
    """The tuner prunes exactly this error; anything else stops the study."""
    prob = np.full((10, 10, 10), 0.4, dtype=np.float32)

    with pytest.raises(bench.EmptyMaskError, match="empty"):
        bench.run_all_preprocessing_benchmarks(prob, EMPTY, None)

    assert issubclass(bench.EmptyMaskError, ValueError)


def test_preprocessing_suite_lets_a_metric_error_reach_the_caller():
    """It logged each metric's error and carried on, dropping that metric's key."""
    prob = np.full((5, 5, 5), 0.4, dtype=np.float32)  # does not match the mask's shape

    with pytest.raises(IndexError):
        bench.run_all_preprocessing_benchmarks(prob, _cube_mask(), None)


def test_preprocessing_suite_reports_every_key_on_a_good_mask():
    prob = np.where(_cube_mask(), 0.9, 0.1).astype(np.float32)
    entropy = np.full((10, 10, 10), 0.5, dtype=np.float32)

    results = bench.run_all_preprocessing_benchmarks(prob, _cube_mask(), entropy)

    assert set(results) == {
        "confidence", "probability_yield", "mean_uncertainty", "high_uncertainty_fraction",
        "crispness", "fragmentation", "surface_area_ratio", "euler_characteristic",
        "foreground_fraction", "median_radius_um", "p90_radius_um", "p99_radius_um", "max_radius_um",
    }


def test_euler_characteristic_lets_an_error_reach_the_caller(monkeypatch):
    """It returned 0 on any error, a value a real mask can have."""
    import skimage.measure

    def broken(*args, **kwargs):
        raise RuntimeError("euler_number failed")

    monkeypatch.setattr(skimage.measure, "euler_number", broken)

    with pytest.raises(RuntimeError, match="euler_number failed"):
        bench.evaluate_preprocessing_euler_characteristic(_cube_mask())


@pytest.mark.parametrize("metric", [
    lambda: bench.evaluate_preprocessing_uncertainty(np.full((10, 10, 10), 0.5), EMPTY),
    lambda: bench.evaluate_preprocessing_compactness(EMPTY),
], ids=["uncertainty", "compactness"])
def test_a_metric_whose_empty_value_reads_as_perfect_raises(metric):
    """Zero uncertainty and a zero surface-to-volume ratio would both read as the best score."""
    with pytest.raises(ValueError, match="empty"):
        metric()


def test_an_empty_mask_keeps_its_worst_or_true_values():
    """Kept: 0 confidence, yield and crispness are the worst scores; 0 components and an
    Euler characteristic of 0 are the true values for an empty set."""
    prob = np.full((10, 10, 10), 0.4, dtype=np.float32)

    assert bench.evaluate_preprocessing_confidence(prob, EMPTY) == {"confidence": 0.0, "probability_yield": 0.0}
    assert bench.evaluate_preprocessing_crispness(prob, EMPTY) == 0.0
    assert bench.evaluate_preprocessing_fragmentation(EMPTY) == 0
    assert bench.evaluate_preprocessing_euler_characteristic(EMPTY) == 0
