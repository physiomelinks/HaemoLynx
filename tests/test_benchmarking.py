import pytest
import numpy as np
import networkx as nx
import ImageLynx.statistics.benchmarking as bench

def create_straight_vessel_graph(length=10, radius=3.0):
    """Helper fixture to create a perfectly straight 1D graph."""
    G = nx.MultiGraph()
    pts = [(z, 10, 10) for z in range(length)]
    G.add_edge(0, 1, key=0, pts=pts, assigned_diameter_um=radius*2.0, fwhm_diameter_um=radius*2.0)
    return G

def test_volumetric_reconstruction_perfect_match():
    """Phase 2: Core Metric - Volumetric Redilation"""
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
    """Phase 2: Core Metric - Distance Transform Centricity"""
    G = create_straight_vessel_graph(length=20, radius=4.0)
    shape = (20, 20, 20)
    voxel_size = (1.0, 1.0, 1.0)
    
    perfect_mask = bench.redilate_skeleton_to_volume(G, shape, voxel_size)
    results = bench.evaluate_centerline_centricity(G, perfect_mask, voxel_size)
    
    # Because of discrete voxel bounding boxes vs perfect spheres, 
    # EDT might have a tiny sub-voxel error, but it should be extremely close to 0.0
    assert results["mean_centricity_error_um"] < 1.0, f"Centricity error too high: {results['mean_centricity_error_um']}"

def test_topological_preservation_components_and_loops():
    """Phase 2: Core Metric - Euler/Betti Topologies"""
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
    """Phase 2: Core Metric - Orphaned Volume Detection"""
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
    """Phase 3: Graceful Degradation - Missing pts or diameter keys"""
    G = nx.MultiGraph()
    # Missing 'pts', using 'voxels'. Missing 'fwhm_diameter_um', relying on 'assigned_diameter_um'
    G.add_edge(0, 1, key=0, voxels=[(5, 5, 5)], assigned_diameter_um=4.0)
    
    mask = bench.redilate_skeleton_to_volume(G, (10, 10, 10), (1.0, 1.0, 1.0))
    
    res_vol = bench.evaluate_volumetric_reconstruction(G, mask, (1.0, 1.0, 1.0))
    assert res_vol["dice_coefficient"] == 1.0
    
    res_comp = bench.evaluate_completeness_and_overpruning(G, mask, (1.0, 1.0, 1.0))
    assert res_comp["orphaned_volume_fraction"] == 0.0

def test_empty_binary_mask():
    """Phase 3: Graceful Degradation - Total Anoxia / Empty Volumes"""
    G = nx.MultiGraph()
    mask = np.zeros((10, 10, 10), dtype=bool)
    voxel_size = (1.0, 1.0, 1.0)
    
    # Should safely return 0.0 without ZeroDivisionError
    res_vol = bench.evaluate_volumetric_reconstruction(G, mask, voxel_size)
    assert res_vol["dice_coefficient"] == 0.0
    assert res_vol["jaccard_index"] == 0.0
    
    res_cen = bench.evaluate_centerline_centricity(G, mask, voxel_size)
    assert res_cen["mean_centricity_error_um"] == 0.0
    
    res_comp = bench.evaluate_completeness_and_overpruning(G, mask, voxel_size)
    assert res_comp["orphaned_volume_fraction"] == 0.0