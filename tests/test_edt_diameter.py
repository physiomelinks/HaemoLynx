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


# --- Junction-proximity exclusion (#98 Phase A) -------------------------------------------
#
# Within roughly one radius of a bifurcation the EDT reports the junction's inscribed sphere
# rather than the vessel's, so every radius sampled there is biased upward, and resistance
# carries the error as r^-4.
#
# The fixture below isolates the mechanism: one short edge against a deliberately large
# junction sphere, where it is worth 34% on diameter and 3.2x on resistance. On the real
# WKY-A network the effect is about 8% on resistance, because the median segment is only 3.9
# voxels long and 61% of edges are too short to trim at all. The length-dependence these
# tests demonstrate is real but does not dominate there - every segment in that network is
# short relative to the exclusion. See HaemodynamicsConfig for the measured sweep.

def _junction_fixture():
    """A z-tube with a spherical swelling at a degree-3 node, plus a transverse branch.

    Voxel size is 1 um so index and physical coordinates coincide and the arithmetic in the
    assertions stays readable.

        A = (20, 16, 16)  junction, degree 3, sitting at the centre of a radius-5 sphere
        B = (26, 16, 16)  free end,  6 um from A - short enough for the junction to dominate
        C = ( 0, 16, 16)  free end, 20 um from A - long enough for it not to
        D = (20, 31, 16)  free end on the transverse branch
    """
    mask = np.zeros((48, 32, 32), dtype=bool)
    zz, yy, xx = np.ogrid[:48, :32, :32]

    mask |= ((yy - 16) ** 2 + (xx - 16) ** 2) <= 2 ** 2                    # z-tube, radius 2
    mask |= (((zz - 20) ** 2 + (xx - 16) ** 2) <= 2 ** 2) & (yy >= 16)     # y-branch, radius 2
    mask |= ((zz - 20) ** 2 + (yy - 16) ** 2 + (xx - 16) ** 2) <= 5 ** 2   # junction swelling

    G = nx.MultiGraph()
    for name, pos in (("A", (20, 16, 16)), ("B", (26, 16, 16)),
                      ("C", (0, 16, 16)), ("D", (20, 31, 16))):
        G.add_node(name, pos=tuple(float(c) for c in pos))

    G.add_edge("A", "B", voxels=[(float(z), 16.0, 16.0) for z in range(20, 27)])
    G.add_edge("A", "C", voxels=[(float(z), 16.0, 16.0) for z in range(20, -1, -1)])
    G.add_edge("A", "D", voxels=[(20.0, float(y), 16.0) for y in range(16, 32)])
    return mask, G


def test_junction_proximity_exclusion_lowers_the_radius_of_a_short_junction_edge():
    """The bias the exclusion exists to remove, on the segment length where it bites."""
    mask, G = _junction_fixture()
    voxel = (1.0, 1.0, 1.0)

    measure_edge_diameters_edt_from_binary_mask(G, mask, voxel)
    untrimmed = G["A"]["B"][0]["edt_diameter_um"]
    # A-C runs down the same radius-2 tube but is long enough that the junction samples
    # cannot move its median, so it reports the calibre A-B should have reported.
    honest = G["A"]["C"][0]["edt_diameter_um"]

    measure_edge_diameters_edt_from_binary_mask(
        G, mask, voxel, junction_proximity_exclusion_um=5.0
    )
    trimmed = G["A"]["B"][0]["edt_diameter_um"]

    # Two identical tubes, read 34% apart purely because one segment is short. Poiseuille
    # takes that to (6.0 / 4.47)^4 = 3.2x on the segment's resistance.
    assert untrimmed == pytest.approx(6.0)
    assert honest == pytest.approx(np.sqrt(5.0) * 2.0)
    assert trimmed == pytest.approx(honest), "trimming should recover the tube's own calibre"
    assert G["A"]["B"][0]["edt_junction_trim"] == "trimmed"


def test_junction_proximity_exclusion_is_off_by_default():
    """The default has to reproduce the previous behaviour exactly.

    Every measured number in the #98 sweep was taken without it, so a silent default would
    make those figures irreproducible from the code that claims to have produced them.
    """
    mask, G = _junction_fixture()
    measure_edge_diameters_edt_from_binary_mask(G, mask, (1.0, 1.0, 1.0))

    assert G["A"]["B"][0]["edt_junction_trim"] == "not_applied"


def test_a_reversed_voxel_list_still_trims_the_junction_end():
    """Edge voxel lists are conventionally u -> v, and nothing enforces it.

    If the ends are assumed rather than matched, a reversed list trims the free end and
    keeps the junction, which inflates the radius instead of correcting it - the opposite of
    the intended effect, and silent.
    """
    mask, G = _junction_fixture()
    forward = G["A"]["B"][0]["voxels"]

    reversed_G = nx.MultiGraph()
    reversed_G.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        payload = dict(data)
        if {u, v} == {"A", "B"}:
            payload["voxels"] = list(reversed(forward))
        reversed_G.add_edge(u, v, **payload)

    kwargs = dict(junction_proximity_exclusion_um=5.0)
    measure_edge_diameters_edt_from_binary_mask(G, mask, (1.0, 1.0, 1.0), **kwargs)
    measure_edge_diameters_edt_from_binary_mask(reversed_G, mask, (1.0, 1.0, 1.0), **kwargs)

    assert reversed_G["A"]["B"][0]["edt_diameter_um"] == pytest.approx(
        G["A"]["B"][0]["edt_diameter_um"]
    )


def test_a_long_edge_is_barely_moved_by_the_same_exclusion():
    """The bias is length-dependent, which is why it cannot be corrected by a global factor."""
    mask, G = _junction_fixture()

    measure_edge_diameters_edt_from_binary_mask(G, mask, (1.0, 1.0, 1.0))
    long_before, short_before = G["A"]["C"][0]["edt_diameter_um"], G["A"]["B"][0]["edt_diameter_um"]

    measure_edge_diameters_edt_from_binary_mask(
        G, mask, (1.0, 1.0, 1.0), junction_proximity_exclusion_um=5.0
    )
    long_after, short_after = G["A"]["C"][0]["edt_diameter_um"], G["A"]["B"][0]["edt_diameter_um"]

    assert abs(long_before - long_after) < abs(short_before - short_after)


def test_free_ends_are_not_trimmed():
    """Only degree > 2 nodes carry the junction inscribed sphere; tips are ordinary vessel."""
    mask = np.zeros((21, 21, 21), dtype=bool)
    zz, yy, xx = np.ogrid[:21, :21, :21]
    mask |= ((yy - 10) ** 2 + (xx - 10) ** 2) <= 3 ** 2

    G = nx.MultiGraph()
    G.add_node(0, pos=(2.0, 10.0, 10.0))
    G.add_node(1, pos=(18.0, 10.0, 10.0))
    G.add_edge(0, 1, voxels=[(float(z), 10.0, 10.0) for z in range(2, 19)])

    measure_edge_diameters_edt_from_binary_mask(G, mask, (1.0, 1.0, 1.0))
    untrimmed = G[0][1][0]["edt_diameter_um"]
    measure_edge_diameters_edt_from_binary_mask(
        G, mask, (1.0, 1.0, 1.0), junction_proximity_exclusion_um=5.0
    )

    assert G[0][1][0]["edt_diameter_um"] == pytest.approx(untrimmed)
    assert G[0][1][0]["edt_junction_trim"] == "no_junction"


def test_an_edge_shorter_than_the_exclusion_is_tagged_rather_than_discarded():
    """Discarding them would delete the capillary population section 1.2 is about.

    A segment running between two bifurcations can be shorter than twice the exclusion, in
    which case no sample survives. Dropping the edge would bias the reported distribution
    towards long vessels; the untrimmed median is kept and tagged so the inflated fraction
    stays countable instead of disappearing into the measured population.
    """
    mask = np.ones((12, 12, 12), dtype=bool)
    mask[0, :, :] = mask[-1, :, :] = False

    G = nx.MultiGraph()
    for name, pos in (("J1", (4.0, 6.0, 6.0)), ("J2", (7.0, 6.0, 6.0))):
        G.add_node(name, pos=pos)
    for spur in ("s1", "s2", "s3", "s4"):
        G.add_node(spur, pos=(6.0, 6.0, 6.0))
    G.add_edge("J1", "J2", voxels=[(float(z), 6.0, 6.0) for z in range(4, 8)])
    G.add_edge("J1", "s1", voxels=[(4.0, 6.0, 6.0), (4.0, 5.0, 6.0)])
    G.add_edge("J1", "s2", voxels=[(4.0, 6.0, 6.0), (4.0, 7.0, 6.0)])
    G.add_edge("J2", "s3", voxels=[(7.0, 6.0, 6.0), (7.0, 5.0, 6.0)])
    G.add_edge("J2", "s4", voxels=[(7.0, 6.0, 6.0), (7.0, 7.0, 6.0)])

    summary = measure_edge_diameters_edt_from_binary_mask(
        G, mask, (1.0, 1.0, 1.0), junction_proximity_exclusion_um=20.0
    )

    assert G["J1"]["J2"][0]["edt_junction_trim"] == "untrimmed_too_short"
    assert G["J1"]["J2"][0]["edt_diameter_um"] is not None
    assert summary["junction_trim_counts"]["untrimmed_too_short"] >= 1


def test_summary_records_the_exclusion_that_was_actually_applied():
    """A frozen parameter set is only frozen if the artefact says what it was."""
    mask, G = _junction_fixture()
    summary = measure_edge_diameters_edt_from_binary_mask(
        G, mask, (1.0, 1.0, 1.0), junction_proximity_exclusion_um=3.5
    )
    assert summary["junction_proximity_exclusion_um"] == 3.5
    assert sum(summary["junction_trim_counts"].values()) == summary["edges_measured"]
