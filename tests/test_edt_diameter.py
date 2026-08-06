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


def test_edt_radius_raises_when_nothing_was_measured():
    """edt_radius used to pass validation and then silently fabricate every diameter.

    Nothing populated edt_diameter_um anywhere in the codebase, so selecting the mode
    produced the synthetic branch-order law with no error and no warning. Half a calibre
    distribution being fabricated is not something that should happen quietly.
    """
    G = nx.MultiGraph()
    G.add_edge(0, 1, branch_order="B01", length=100.0)  # no edt_diameter_um
    diameter_by_branch_order = {"B01": {"d1": 4.0, "d2": 4.0}}
    model = PoiseuilleModel(constriction_length=5.0, constriction_spacing=100.0)

    with pytest.raises(ValueError, match="edt_diameter_um"):
        model.set_poiseuille_resistances(
            G, diameter_by_branch_order, radius_assignment_mode="edt_radius"
        )

    with pytest.raises(ValueError, match="edt_diameter_um"):
        model.set_poiseuille_resistances_with_constrictions(
            G, diameter_by_branch_order, radius_assignment_mode="edt_radius"
        )


def test_fwhm_radius_does_not_raise_when_nothing_was_measured():
    """fwhm_radius deliberately does not raise, unlike edt_radius.

    FWHM legitimately fails on individual edges - roughly half of them on the Ilastik
    probability field, whose flat in-vessel plateau is the documented failure case for
    Gaussian fitting - so an empty measurement is a real outcome rather than proof that the
    measurement step never ran. The fabricated fraction is surfaced through the provenance
    counts instead of being turned into an error.
    """
    G = nx.MultiGraph()
    G.add_edge(0, 1, branch_order="B01", length=100.0)
    model = PoiseuilleModel(constriction_length=5.0, constriction_spacing=100.0)

    _, stats = model.set_poiseuille_resistances(
        G, {"B01": {"d1": 4.0, "d2": 4.0}}, radius_assignment_mode="fwhm_radius"
    )
    assert stats["diameter_provenance_counts"] == {"synthetic_branch_order": 1}


def test_diameter_provenance_distinguishes_measured_from_synthetic():
    """assigned_diameter_um recorded measured and fabricated diameters identically.

    Section 1.2 is a distributional claim, so a mixed distribution has to be separable.
    """
    G = nx.MultiGraph()
    G.add_edge(0, 1, branch_order="B01", length=100.0, edt_diameter_um=12.0)  # measured
    G.add_edge(1, 2, branch_order="B01", length=100.0)                        # falls back
    model = PoiseuilleModel(constriction_length=5.0, constriction_spacing=100.0)

    _, stats = model.set_poiseuille_resistances(
        G, {"B01": {"d1": 4.0, "d2": 4.0}}, radius_assignment_mode="edt_radius"
    )

    assert stats["diameter_provenance_counts"] == {
        "measured_edt": 1, "synthetic_branch_order": 1,
    }
    assert G[0][1][0]["diameter_provenance"] == "measured_edt"
    assert G[1][2][0]["diameter_provenance"] == "synthetic_branch_order"
    assert G[0][1][0]["assigned_diameter_um"] == 12.0
    assert G[1][2][0]["assigned_diameter_um"] == 4.0


# --- The default radius estimator (#98 Phase 3) -------------------------------------------
#
# Re-measured on the repaired pipeline over the same 1330 edges: EDT covered 100.0% with a
# median diameter of 6.37 um, FWHM covered 76.5% with a median of 8.20 um and a maximum of
# 39.16 um. Pearson r = +0.245, Spearman rho = +0.284, median FWHM/EDT ratio 1.359. The
# assessment's r = 0.079 / rho = 0.141 was measured on a bridge_gaps-inflated mask and is
# superseded; repairing it roughly tripled the correlation, but the two still disagree.

def test_default_radius_assignment_mode_is_edt():
    """H1 section 1.2 specifies EDT, and the default used to contradict that."""
    C = pytest.importorskip("carotid_image_to_model")
    assert C.HaemodynamicsConfig().radius_assignment_mode == "edt_radius"


def test_default_radius_mode_yields_measured_not_synthetic_provenance():
    """The failure 79baf86 closed: a mode that validates and then fabricates diameters.

    With the default mode, an edge carrying an EDT measurement must be recorded as measured,
    never as synthetic_branch_order.
    """
    C = pytest.importorskip("carotid_image_to_model")

    G = nx.MultiGraph()
    # The measured median on the repaired pipeline, so the fixture is a real capillary.
    G.add_edge(0, 1, branch_order="B01", length=100.0, edt_diameter_um=6.37)
    model = PoiseuilleModel(constriction_length=5.0, constriction_spacing=100.0)

    _, stats = model.set_poiseuille_resistances(
        G, {"B01": {"d1": 4.0, "d2": 4.0}},
        radius_assignment_mode=C.HaemodynamicsConfig().radius_assignment_mode,
    )

    assert stats["diameter_provenance_counts"] == {"measured_edt": 1}
    assert all(d["diameter_provenance"] == "measured_edt" for _, _, d in G.edges(data=True))
