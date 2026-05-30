import pytest
import networkx as nx
import numpy as np
from ImageLynx.haemodynamics.automated import measure_edge_diameters_edt_from_binary_mask
from ImageLynx.haemodynamics.poiseuille import PoiseuilleModel

def test_edt_diameter_measurement():
    # 1. Create a mock binary mask: a simple tube of radius 3 centered in a 11x11x11 volume
    mask = np.zeros((11, 11, 11), dtype=bool)
    center = 5
    for z in range(11):
        for y in range(11):
            for x in range(11):
                if (y - center)**2 + (x - center)**2 <= 3**2:
                    mask[z, y, x] = True

    # 2. Create a mock graph with a single edge running through the center
    G = nx.MultiGraph()
    voxel_size_xyz = (1.0, 1.0, 1.0)
    
    # Voxels from z=1 to z=9 at (y,x) = (5,5)
    voxels_phys = [(float(z), float(center), float(center)) for z in range(1, 10)]
    
    G.add_node(0, pos=(1.0, center, center))
    G.add_node(1, pos=(9.0, center, center))
    G.add_edge(0, 1, length=8.0, voxels=voxels_phys)
    
    # 3. Measure EDT diameters
    stats = measure_edge_diameters_edt_from_binary_mask(G, mask, voxel_size_xyz)
    
    assert stats["edges_measured"] == 1
    assert stats["edges_skipped"] == 0
    
    # The radius should be 3, so diameter should be 6
    edge_data = G[0][1][0]
    assert "edt_diameter_um" in edge_data
    # Allow small floating point difference due to discrete grid and EDT interpolation
    assert np.isclose(edge_data["edt_diameter_um"], 6.0, atol=0.5)

def test_poiseuille_edt_mode():
    G = nx.MultiGraph()
    G.add_node(0)
    G.add_node(1)
    
    # Pre-assign EDT diameter
    G.add_edge(0, 1, branch_order="B01", length=100.0, edt_diameter_um=12.0)
    
    diameter_by_branch_order = {"B01": {"d1": 4.0, "d2": 4.0}}
    
    model = PoiseuilleModel(constriction_length=5.0, constriction_spacing=100.0)
    
    # Test flat resistance
    G_res, stats = model.set_poiseuille_resistances(
        G, 
        diameter_by_branch_order,
        radius_assignment_mode="edt_radius"
    )
    
    assert stats["resistances_set"] == 1
    assert G_res[0][1][0]["assigned_diameter_um"] == 12.0
    
    # Test constricted resistance
    G_constrict, stats_constrict = model.set_poiseuille_resistances_with_constrictions(
        G, 
        diameter_by_branch_order,
        radius_assignment_mode="edt_radius"
    )
    
    assert stats_constrict["resistances_set"] == 1
    assert G_constrict[0][1][0]["assigned_diameter_um"] == 12.0
