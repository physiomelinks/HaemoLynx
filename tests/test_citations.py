"""Run citations: which packages/models a run's own settings actually used.

Every entry in PACKAGE_CITATIONS/MODEL_CITATIONS carries its own ``applies``
predicate; these tests pin that predicate against the settings that should
and should not trigger it, rather than trusting the citation text is right
(that is a domain-knowledge concern, not something a test can check) or that
"some citations came back" is enough (a predicate stuck at ``True`` would
still pass a test like that).
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from haemolynx.pipeline import default_schema, resolve_settings
from haemolynx.pipeline.citations import (
    MODEL_CITATIONS,
    PACKAGE_CITATIONS,
    SOFTWARE_CITATION,
    render_citations,
    used_citations,
    write_citations,
)
from haemolynx.pipeline.stages import (
    HaemodynamicModel,
    SkeletonisedVolume,
    Solution,
    VesselNetwork,
    export_results,
)

SCHEMA = default_schema()


def _settings(tmp_path: Path, **overrides) -> dict:
    values = SCHEMA.defaults()
    values.update(
        {
            "input_path": tmp_path / "input.tif",
            "vtk_output_prefix": tmp_path / "out" / "run",
            "plot_dir": tmp_path / "plots",
        }
    )
    values.update(overrides)
    return resolve_settings(values, schema=SCHEMA, config_path=None)


def _names(citations) -> set[str]:
    return {citation.name for citation in citations}


# --- packages that never depend on a setting --------------------------------


def test_numpy_scipy_networkx_are_always_cited(tmp_path):
    on = _settings(tmp_path, run_haemodynamics=False)
    names = _names(used_citations(on, PACKAGE_CITATIONS))
    assert {"NumPy", "SciPy", "NetworkX"} <= names


# --- packages gated on a setting ---------------------------------------------


def test_scikit_image_and_skan_follow_the_stage_toggles(tmp_path):
    on = _settings(tmp_path, do_skeletonize=True, do_graph_building=True)
    off = _settings(tmp_path, do_skeletonize=False, do_graph_building=False)
    assert "scikit-image" in _names(used_citations(on, PACKAGE_CITATIONS))
    assert "skan" in _names(used_citations(on, PACKAGE_CITATIONS))
    assert "scikit-image" not in _names(used_citations(off, PACKAGE_CITATIONS))
    assert "skan" not in _names(used_citations(off, PACKAGE_CITATIONS))


def test_pandas_is_cited_for_napari_and_not_for_the_statistics(tmp_path, monkeypatch):
    """Regression: pandas was cited for "statistics tables and CSV export",
    which the csv module writes; only napari's layer features use it."""
    settings = _settings(tmp_path, statistics=True)
    monkeypatch.delitem(sys.modules, "napari", raising=False)
    assert "pandas" not in _names(used_citations(settings, PACKAGE_CITATIONS))
    monkeypatch.setitem(sys.modules, "napari", object())
    assert "pandas" in _names(used_citations(settings, PACKAGE_CITATIONS))


def test_the_statistics_csv_writer_does_not_use_pandas():
    import inspect

    from haemolynx.statistics import csv_export

    assert "pandas" not in inspect.getsource(csv_export)


SWEEP = {"name": "p", "type": "pressure_sweep", "overrides": {}}
RE_SOLVE = {"name": "a", "type": "arteriole_diameter_change", "overrides": {}}


def _perturbed(tmp_path: Path, entries, **overrides) -> dict:
    values = dict(run_haemodynamics=True, run_perturbations=True, perturbations=entries)
    values.update(overrides)
    return _settings(tmp_path, **values)


def test_matplotlib_follows_visualize_results(tmp_path):
    on = _settings(tmp_path, visualize_results=True)
    off = _settings(tmp_path, visualize_results=False)
    assert "Matplotlib" in _names(used_citations(on, PACKAGE_CITATIONS))
    assert "Matplotlib" not in _names(used_citations(off, PACKAGE_CITATIONS))


@pytest.mark.parametrize("entry", [SWEEP, RE_SOLVE])
def test_matplotlib_is_cited_for_a_perturbations_plots(tmp_path, entry):
    """A sweep draws its curves and a single re-solve its figures whatever
    the baseline's visualize_results says."""
    on = _perturbed(tmp_path, [entry], visualize_results=False)
    assert "Matplotlib" in _names(used_citations(on, PACKAGE_CITATIONS))
    for off in (
        _perturbed(tmp_path, [entry], visualize_results=False, run_perturbations=False),
        _perturbed(tmp_path, [entry], visualize_results=False, run_haemodynamics=False),
        _perturbed(tmp_path, [], visualize_results=False),
    ):
        assert "Matplotlib" not in _names(used_citations(off, PACKAGE_CITATIONS))


def test_plotly_needs_both_visualize_results_and_3d_render_mode(tmp_path):
    both = _settings(tmp_path, visualize_results=True, final_render_mode="3d")
    only_2d = _settings(tmp_path, visualize_results=True, final_render_mode="2d")
    only_off = _settings(tmp_path, visualize_results=False, final_render_mode="3d")
    assert "Plotly" in _names(used_citations(both, PACKAGE_CITATIONS))
    assert "Plotly" not in _names(used_citations(only_2d, PACKAGE_CITATIONS))
    assert "Plotly" not in _names(used_citations(only_off, PACKAGE_CITATIONS))


def test_plotly_is_cited_for_a_single_re_solves_3d_overlay(tmp_path):
    re_solve = _perturbed(tmp_path, [RE_SOLVE], visualize_results=False, final_render_mode="3d")
    in_2d = _perturbed(tmp_path, [RE_SOLVE], visualize_results=False, final_render_mode="2d")
    sweep = _perturbed(tmp_path, [SWEEP], visualize_results=False, final_render_mode="3d")
    assert "Plotly" in _names(used_citations(re_solve, PACKAGE_CITATIONS))
    assert "Plotly" not in _names(used_citations(in_2d, PACKAGE_CITATIONS))
    assert "Plotly" not in _names(used_citations(sweep, PACKAGE_CITATIONS))


def test_plotly_is_cited_for_the_vessel_mask_views_when_their_masks_are_in_use(tmp_path):
    quiet = dict(visualize_results=False, automated_vessel_assignment=True)
    small = _settings(
        tmp_path,
        **quiet,
        use_small_vessel_masks_for_boundary_assignment=True,
        write_small_vessel_boundary_labelling_3d_html=True,
    )
    small_off = _settings(
        tmp_path,
        **quiet,
        use_small_vessel_masks_for_boundary_assignment=True,
        write_small_vessel_boundary_labelling_3d_html=False,
    )
    small_unused = _settings(
        tmp_path,
        visualize_results=False,
        automated_vessel_assignment=False,
        use_small_vessel_masks_for_boundary_assignment=True,
        write_small_vessel_boundary_labelling_3d_html=True,
    )
    large = dict(
        **quiet,
        use_large_vessel_masks=True,
        write_fast_mode_preassignment_large_vessel_debug_3d_html=True,
    )
    assert "Plotly" in _names(used_citations(small, PACKAGE_CITATIONS))
    assert "Plotly" not in _names(used_citations(small_off, PACKAGE_CITATIONS))
    assert "Plotly" not in _names(used_citations(small_unused, PACKAGE_CITATIONS))
    fast = _settings(tmp_path, **large, automated_vessel_assignment_fast_mode=True)
    slow = _settings(tmp_path, **large, automated_vessel_assignment_fast_mode=False)
    assert "Plotly" in _names(used_citations(fast, PACKAGE_CITATIONS))
    assert "Plotly" not in _names(used_citations(slow, PACKAGE_CITATIONS))


def test_pyvista_needs_both_run_haemodynamics_and_vtk_export(tmp_path):
    both = _settings(tmp_path, run_haemodynamics=True, vtk_export=True)
    no_vtk = _settings(tmp_path, run_haemodynamics=True, vtk_export=False)
    no_haemo = _settings(tmp_path, run_haemodynamics=False, vtk_export=True)
    assert "PyVista / VTK" in _names(used_citations(both, PACKAGE_CITATIONS))
    assert "PyVista / VTK" not in _names(used_citations(no_vtk, PACKAGE_CITATIONS))
    assert "PyVista / VTK" not in _names(used_citations(no_haemo, PACKAGE_CITATIONS))


def test_pyvista_is_cited_for_the_tissue_surface(tmp_path):
    """The tissue surface is a .vtp written whatever vtk_export says."""
    tissue = _settings(
        tmp_path, run_haemodynamics=False, vtk_export=False, measure_tissue_volume=True
    )
    assert "PyVista / VTK" in _names(used_citations(tissue, PACKAGE_CITATIONS))


def test_h5py_and_tifffile_follow_the_input_suffix(tmp_path):
    h5 = _settings(tmp_path, input_path=tmp_path / "input.h5")
    tif = _settings(tmp_path, input_path=tmp_path / "input.tif")
    assert "h5py" in _names(used_citations(h5, PACKAGE_CITATIONS))
    assert "tifffile" not in _names(used_citations(h5, PACKAGE_CITATIONS))
    assert "tifffile" in _names(used_citations(tif, PACKAGE_CITATIONS))
    assert "h5py" not in _names(used_citations(tif, PACKAGE_CITATIONS))


def _readers(settings) -> set[str]:
    return _names(used_citations(settings, PACKAGE_CITATIONS)) & {"h5py", "tifffile"}


def test_the_raw_image_is_read_as_a_tiff_beside_an_hdf5_segmentation(tmp_path):
    settings = _settings(
        tmp_path, input_path=tmp_path / "input.h5", raw_image_path=tmp_path / "raw.tif"
    )
    assert _readers(settings) == {"h5py", "tifffile"}


def test_the_vessel_masks_count_only_while_their_role_is_in_use(tmp_path):
    masks = dict(
        input_path=tmp_path / "input.tif",
        large_arteriole_mask_path=tmp_path / "art.h5",
        large_venule_mask_path=tmp_path / "ven.h5",
        use_large_vessel_masks=True,
    )
    in_use = _settings(tmp_path, automated_vessel_assignment=True, **masks)
    unused = _settings(tmp_path, automated_vessel_assignment=False, **masks)
    assert _readers(in_use) == {"h5py", "tifffile"}
    assert _readers(unused) == {"tifffile"}


def test_ilastiks_output_is_read_in_the_format_it_is_written(tmp_path):
    """With ilastik segmenting, the segmented image read is its output (by
    ilastik_output_suffix), not a stale input_path; the raw image is a TIFF."""
    settings = _settings(
        tmp_path,
        input_path=tmp_path / "stale.tif",
        raw_image_path=tmp_path / "raw.tif",
        use_ilastik_segmentation=True,
        ilastik_output_suffix=".h5",
    )
    assert _readers(settings) == {"h5py", "tifffile"}
    no_raw = _settings(
        tmp_path,
        input_path=tmp_path / "input.tif",
        use_ilastik_segmentation=True,
        ilastik_output_suffix=".h5",
    )
    assert _readers(no_raw) == {"tifffile"}


def test_the_cell_and_perturbation_masks_are_counted_when_read(tmp_path):
    base = dict(input_path=tmp_path / "input.tif")
    cells = _settings(
        tmp_path, **base, measurement_3d_to_cell_mask=True, cell_mask_path=tmp_path / "cells.h5"
    )
    cells_off = _settings(
        tmp_path, **base, measurement_3d_to_cell_mask=False, cell_mask_path=tmp_path / "cells.h5"
    )
    assert _readers(cells) == {"h5py", "tifffile"}
    assert _readers(cells_off) == {"tifffile"}
    blocks = _perturbed(
        tmp_path,
        [{"name": "b", "type": "capillary_block_mask", "overrides": {}}],
        **base,
        capillary_block_mask_path=tmp_path / "blocks.h5",
    )
    assert _readers(blocks) == {"h5py", "tifffile"}
    pericyte = {"name": "c", "type": "pericyte_diameter_change"}
    mask_on = _perturbed(
        tmp_path,
        [{**pericyte, "overrides": {"use_pericyte_mask_constriction": True}}],
        **base,
        pericyte_mask_path=tmp_path / "pericytes.h5",
    )
    mask_off = _perturbed(
        tmp_path,
        [{**pericyte, "overrides": {}}],
        **base,
        pericyte_mask_path=tmp_path / "pericytes.h5",
    )
    assert _readers(mask_on) == {"h5py", "tifffile"}
    assert _readers(mask_off) == {"tifffile"}


def test_ilastik_is_cited_for_the_main_image_flag_alone(tmp_path):
    on = _settings(tmp_path, use_ilastik_segmentation=True)
    off = _settings(tmp_path)
    assert "ilastik" in _names(used_citations(on, PACKAGE_CITATIONS))
    assert "ilastik" not in _names(used_citations(off, PACKAGE_CITATIONS))


@pytest.mark.parametrize(
    "flag, prerequisite",
    [
        ("use_ilastik_large_vessel_segmentation", "use_large_vessel_masks"),
        (
            "use_ilastik_small_vessel_segmentation",
            "use_small_vessel_masks_for_boundary_assignment",
        ),
    ],
)
def test_ilastik_vessel_mask_flags_need_their_own_prerequisite_too(
    tmp_path, flag, prerequisite
):
    """A vessel-mask ilastik flag left on with its own prerequisite off is
    exactly the schema's own "set but ineffective" case -- crediting ilastik
    for it here would repeat that mistake in the citation list."""
    inactive = _settings(
        tmp_path, **{flag: True, "automated_vessel_assignment": True}
    )
    active = _settings(
        tmp_path,
        **{flag: True, prerequisite: True, "automated_vessel_assignment": True},
    )
    assert "ilastik" not in _names(used_citations(inactive, PACKAGE_CITATIONS))
    assert "ilastik" in _names(used_citations(active, PACKAGE_CITATIONS))


def test_used_ilastik_asks_the_same_question_as_preflight():
    """Regression: _used_ilastik used to hardcode its own copy of the
    large/small-vessel-mask prerequisite chains -- two sources of truth that
    could silently drift apart. It asks the one function preflight's ilastik
    check asks, so a change to when a mask is segmented reaches both."""
    from haemolynx.pipeline.checks import ilastik_segments_vessel_masks
    from haemolynx.pipeline.citations import ilastik_segments_vessel_masks as via_citations

    assert via_citations is ilastik_segments_vessel_masks


def test_napari_is_cited_only_when_napari_is_already_imported(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.delitem(sys.modules, "napari", raising=False)
    assert "napari" not in _names(used_citations(settings, PACKAGE_CITATIONS))
    monkeypatch.setitem(sys.modules, "napari", object())
    assert "napari" in _names(used_citations(settings, PACKAGE_CITATIONS))


# --- haemodynamics models -----------------------------------------------------


def test_hagen_poiseuille_follows_run_haemodynamics(tmp_path):
    on = _settings(tmp_path, run_haemodynamics=True)
    off = _settings(tmp_path, run_haemodynamics=False)
    assert "Hagen-Poiseuille law" in _names(used_citations(on, MODEL_CITATIONS))
    assert "Hagen-Poiseuille law" not in _names(used_citations(off, MODEL_CITATIONS))


def test_no_model_citations_without_haemodynamics(tmp_path):
    off = _settings(tmp_path, run_haemodynamics=False)
    assert used_citations(off, MODEL_CITATIONS) == []


PRIES_1992 = "Pries in vitro viscosity law (Pries et al., 1992)"
PRIES_1994 = "Pries in vivo viscosity law (Pries et al., 1994)"
PRIES_SECOMB_2005 = "Pries-Secomb in vivo viscosity law (Pries and Secomb, 2005)"
POWER_LAW = "Legacy capillary power-law viscosity (viscosity_law='capillary_power_law')"
VISCOSITY_CITATIONS = {PRIES_1992, PRIES_1994, PRIES_SECOMB_2005, POWER_LAW}


@pytest.mark.parametrize(
    "viscosity_law, form, expected",
    [
        # auto reads each diameter through the form its basis calls for.
        ("pries", "auto", {PRIES_1992, PRIES_1994}),
        ("pries", "in_vitro_1992", {PRIES_1992}),
        ("pries", "in_vivo_1994", {PRIES_1994}),
        # The 2005 model is the in vitro law at the layer's free diameter.
        ("pries", "in_vivo_2005", {PRIES_SECOMB_2005, PRIES_1992}),
        ("capillary_power_law", "in_vivo_2005", {POWER_LAW}),
    ],
)
def test_viscosity_law_citation_matches_the_chosen_law(tmp_path, viscosity_law, form, expected):
    settings = _settings(
        tmp_path, run_haemodynamics=True, viscosity_law=viscosity_law, pries_viscosity_form=form
    )
    names = set(_names(used_citations(settings, MODEL_CITATIONS)))
    assert names & VISCOSITY_CITATIONS == expected


def test_constant_viscosity_law_cites_neither_viscosity_model(tmp_path):
    settings = _settings(tmp_path, run_haemodynamics=True, viscosity_law="constant")
    names = _names(used_citations(settings, MODEL_CITATIONS))
    assert not any("viscosity" in name.lower() for name in names)


def test_haematocrit_distribution_citation_follows_haematocrit_model(tmp_path):
    fixed = _settings(tmp_path, run_haemodynamics=True, haematocrit_model="fixed")
    distributed = _settings(
        tmp_path, run_haemodynamics=True, haematocrit_model="distributed_iterative"
    )
    name = "Pries-Secomb bifurcation haematocrit distribution (haematocrit_model='distributed_iterative')"
    assert name not in _names(used_citations(fixed, MODEL_CITATIONS))
    assert name in _names(used_citations(distributed, MODEL_CITATIONS))


@pytest.mark.parametrize(
    "rule, cited",
    [
        ("no_separation", True),
        ("sequential_bifurcations", True),
        # This pipeline's own split, not Secomb's rule.
        ("split_junctions", False),
    ],
)
def test_secombs_junction_rules_are_cited_when_a_run_uses_one(tmp_path, rule, cited):
    name = "Secomb's network-flow junction rules (haematocrit_junction_rule)"
    distributed = _settings(
        tmp_path,
        run_haemodynamics=True,
        haematocrit_model="distributed_iterative",
        haematocrit_junction_rule=rule,
    )
    fixed = _settings(
        tmp_path, run_haemodynamics=True, haematocrit_model="fixed", haematocrit_junction_rule=rule
    )
    assert (name in _names(used_citations(distributed, MODEL_CITATIONS))) is cited
    assert name not in _names(used_citations(fixed, MODEL_CITATIONS))


@pytest.mark.parametrize("model, cited", [("estimated", True), ("closed", False), ("terminal_resistance", False)])
def test_fry_is_cited_when_the_open_ends_are_estimated(tmp_path, model, cited):
    name = "Estimated boundary conditions at open ends (open_end_model='estimated')"
    settings = _settings(tmp_path, run_haemodynamics=True, open_end_model=model)
    assert (name in _names(used_citations(settings, MODEL_CITATIONS))) is cited
    assert name in _names(used_citations(_settings(tmp_path, run_haemodynamics=True), MODEL_CITATIONS)), (
        "the estimate is the default"
    )


def test_pericyte_constriction_citation_follows_configured_perturbations(tmp_path):
    none_configured = _settings(tmp_path, run_perturbations=True, perturbations=[])
    non_pericyte = _settings(
        tmp_path,
        run_perturbations=True,
        perturbations=[{"name": "a", "type": "arteriole_diameter_change", "overrides": {}}],
    )
    pericyte = _settings(
        tmp_path,
        run_perturbations=True,
        perturbations=[{"name": "a", "type": "pericyte_diameter_change", "overrides": {}}],
    )
    disabled = _settings(
        tmp_path,
        run_perturbations=False,
        perturbations=[{"name": "a", "type": "pericyte_diameter_change", "overrides": {}}],
    )
    name = "Focal (pericyte) constriction model"
    assert name not in _names(used_citations(none_configured, MODEL_CITATIONS))
    assert name not in _names(used_citations(non_pericyte, MODEL_CITATIONS))
    assert name in _names(used_citations(pericyte, MODEL_CITATIONS))
    assert name not in _names(used_citations(disabled, MODEL_CITATIONS))


# --- rendering and writing ----------------------------------------------------


def test_render_citations_has_both_sections_and_the_software_line(tmp_path):
    settings = _settings(tmp_path)
    text = render_citations(settings)
    assert SOFTWARE_CITATION in text
    assert "Software packages" in text
    assert "Haemodynamics models and methods" in text
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_render_citations_says_so_when_haemodynamics_did_not_run(tmp_path):
    settings = _settings(tmp_path, run_haemodynamics=False)
    text = render_citations(settings)
    assert "none detected -- haemodynamics was not run" in text


def test_write_citations_writes_the_rendered_text(tmp_path):
    settings = _settings(tmp_path)
    out_path = tmp_path / "nested" / "run_citations.txt"
    result = write_citations(settings, out_path)
    assert result == out_path
    assert out_path.read_text(encoding="utf-8") == render_citations(settings)


# --- the schema setting itself -------------------------------------------------


def test_export_citations_schema_default_and_section():
    setting = SCHEMA["export_citations"]
    assert setting.kind == "bool"
    assert setting.default is True
    assert setting.section == "Solver and output"


def test_export_citations_lives_on_the_export_tab():
    from haemolynx.gui.tabs import assign_to_stages

    assert assign_to_stages(SCHEMA)["export_citations"] == "10. Export"


# --- wired into export_results -----------------------------------------------


def _minimal_network(tmp_path: Path) -> VesselNetwork:
    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    volume = SkeletonisedVolume(
        image=np.zeros((2, 2, 2), dtype=np.uint8),
        skeleton=np.zeros((2, 2, 2), dtype=bool),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=tmp_path,
    )
    return VesselNetwork(graph=graph, volume=volume)


def _export_results_with(tmp_path: Path, *, export_citations: bool):
    settings = _settings(
        tmp_path,
        run_haemodynamics=False,
        statistics=False,
        measurement_3d_to_cell_mask=False,
        visualize_results=False,
        export_citations=export_citations,
    )
    network = _minimal_network(tmp_path)
    model = HaemodynamicModel(graph=network.graph)
    solution = Solution(graph=network.graph)
    export_results(settings, network, model, solution)
    return tmp_path / "input_citations.txt"


def test_export_results_writes_the_citations_file_when_the_setting_is_on(tmp_path):
    citations_path = _export_results_with(tmp_path, export_citations=True)
    assert citations_path.is_file()
    assert SOFTWARE_CITATION in citations_path.read_text(encoding="utf-8")


def test_export_results_skips_the_citations_file_when_the_setting_is_off(tmp_path):
    citations_path = _export_results_with(tmp_path, export_citations=False)
    assert not citations_path.exists()


def test_a_synthetic_run_cites_the_generator_and_not_the_image_tools(tmp_path):
    from haemolynx.pipeline.citations import NETWORK_GENERATION_CITATIONS, render_citations

    settings = {
        **SCHEMA.defaults(),
        "network_source": "synthetic",
        "synthetic_template_path": tmp_path / "run.haemorun",
        "input_path": tmp_path / "stale.tif",
        "do_skeletonize": True,
        "do_graph_building": True,
    }
    text = render_citations(settings)
    assert "Network generation methods" in text
    assert "Smith, A.F." in text and "fphys.2019.00233" in text
    assert "- skan" not in text and "- tifffile" not in text and "- scikit-image" not in text
    assert all(c.applies(settings) for c in NETWORK_GENERATION_CITATIONS)
    image_run = {**settings, "network_source": "segmented_image"}
    assert "Network generation methods" not in render_citations(image_run)
    assert "- skan" in render_citations(image_run)
    # Tissue measurement still credits scikit-image (it is read even without a skeleton).
    assert "- scikit-image" in render_citations({**settings, "measure_tissue_volume": True})


def test_the_fahraeus_relation_is_cited_when_red_cells_are_predicted(tmp_path):
    """The flux heterogeneity statistic and the capillary-block comparison
    predict each vessel's tube haematocrit, red-cell velocity and flux by
    Pries et al. 1990's relation; nothing else reads it."""
    name = "Fahraeus effect: tube haematocrit from discharge haematocrit (Pries et al., 1990)"
    on = _settings(
        tmp_path,
        run_haemodynamics=True,
        statistics=True,
        statistics_network_analysis=True,
        statistics_flux_heterogeneity=True,
    )
    assert name in _names(used_citations(on, MODEL_CITATIONS))
    for off in (
        "statistics_flux_heterogeneity",
        "statistics_network_analysis",
        "statistics",
        "run_haemodynamics",
    ):
        overrides = dict(
            run_haemodynamics=True,
            statistics=True,
            statistics_network_analysis=True,
            statistics_flux_heterogeneity=True,
        )
        overrides[off] = False
        settings = _settings(tmp_path, **overrides)
        assert name not in _names(used_citations(settings, MODEL_CITATIONS)), off
    block = _settings(
        tmp_path,
        run_haemodynamics=True,
        statistics=False,
        run_perturbations=True,
        perturbations=[{"name": "stall", "type": "capillary_block"}],
    )
    assert name in _names(used_citations(block, MODEL_CITATIONS))
    other = _settings(
        tmp_path,
        run_haemodynamics=True,
        statistics=False,
        run_perturbations=True,
        perturbations=[{"name": "dilate", "type": "pericyte_dilation_sweep"}],
    )
    assert name not in _names(used_citations(other, MODEL_CITATIONS))


# --- centreline and network-statistics methods ---------------------------------


def _method_names(settings) -> set[str]:
    from haemolynx.pipeline.citations import CENTRELINE_CITATIONS, NETWORK_STATISTICS_CITATIONS

    return _names(used_citations(settings, CENTRELINE_CITATIONS + NETWORK_STATISTICS_CITATIONS))


LEE = "Lee's 3D medial-axis thinning (skeletonize method='lee')"
TAUBIN = "Taubin smoothing (centreline_smoothing_method='taubin')"
CHAIKIN = "Chaikin corner cutting (centreline_smoothing_method='chaikin')"
MURRAY = "Murray's law"
STRAHLER = "Horton-Strahler ordering (statistics_strahler)"
MODULARITY = "Greedy modularity communities (Clauset-Newman-Moore)"
KAPLAN_MEIER = "Kaplan-Meier estimator (statistics_capillary_bed_joins)"
FIEDLER = "Algebraic connectivity (statistics_algebraic_connectivity)"
BRANDES = "Brandes' betweenness centrality"


def test_lee_thinning_follows_the_skeletonise_stage(tmp_path):
    assert LEE in _method_names(_settings(tmp_path, do_skeletonize=True))
    assert LEE not in _method_names(_settings(tmp_path, do_skeletonize=False))


@pytest.mark.parametrize("method, cited, other", [("taubin", TAUBIN, CHAIKIN), ("chaikin", CHAIKIN, TAUBIN)])
def test_the_centreline_smoother_that_ran_is_cited(tmp_path, method, cited, other):
    on = _settings(tmp_path, smooth_centrelines=True, centreline_smoothing_method=method)
    assert cited in _method_names(on) and other not in _method_names(on)
    for off in (
        _settings(tmp_path, smooth_centrelines=False, centreline_smoothing_method=method),
        _settings(tmp_path, do_graph_building=False, centreline_smoothing_method=method),
    ):
        assert not _method_names(off) & {TAUBIN, CHAIKIN}


def test_a_synthetic_run_cites_no_centreline_method_but_cites_murray(tmp_path):
    settings = {
        **SCHEMA.defaults(),
        "network_source": "synthetic",
        "synthetic_template_path": tmp_path / "run.haemorun",
        "statistics": False,
    }
    names = _method_names(settings)
    assert not names & {LEE, TAUBIN, CHAIKIN}
    assert MURRAY in names


#: Each statistics citation, the statistics_* toggle that runs it, and
#: whether the report's always-written weighted betweenness/community JSONs
#: also call on it.
STATISTIC_METHODS = [
    (MURRAY, "murray_law", False),
    (STRAHLER, "strahler", False),
    (KAPLAN_MEIER, "capillary_bed_joins", False),
    (FIEDLER, "algebraic_connectivity", False),
    (BRANDES, "betweenness", True),
    (MODULARITY, "community", True),
]


@pytest.mark.parametrize("name, measure, always_with_statistics", STATISTIC_METHODS)
def test_each_statistics_method_follows_its_toggle(tmp_path, name, measure, always_with_statistics):
    on = _settings(tmp_path, statistics=True, statistics_network_analysis=True)
    assert name in _method_names(on)
    toggled_off = _settings(
        tmp_path, statistics=True, statistics_network_analysis=True, **{f"statistics_{measure}": False}
    )
    assert (name in _method_names(toggled_off)) is always_with_statistics
    assert name not in _method_names(_settings(tmp_path, statistics=False))


@pytest.mark.parametrize("name, measure, _always", STATISTIC_METHODS)
def test_the_network_analysis_gate_matches_the_statistics_package(tmp_path, name, measure, _always):
    """A single re-solve computes the run's own selection with statistics
    off, so it shows the toggle alone; a network analysis also needs
    statistics_network_analysis, exactly as statistics_arguments decides."""
    from haemolynx.statistics import NETWORK_ANALYSIS_MEASURES

    gated = _perturbed(
        tmp_path, [RE_SOLVE], statistics=False, statistics_network_analysis=False,
        **{f"statistics_{measure}": True},
    )
    open_gate = _perturbed(
        tmp_path, [RE_SOLVE], statistics=False, statistics_network_analysis=True,
        **{f"statistics_{measure}": True},
    )
    assert name in _method_names(open_gate)
    assert (name in _method_names(gated)) is (measure not in NETWORK_ANALYSIS_MEASURES)


def test_a_sweep_computes_no_statistics(tmp_path):
    sweep = _perturbed(tmp_path, [SWEEP], statistics=False, statistics_network_analysis=True)
    assert not _method_names(sweep) & {name for name, _m, _a in STATISTIC_METHODS}


def test_the_vessels_layer_communities_cite_greedy_modularity(tmp_path):
    settings = _settings(tmp_path, statistics=False, compute_vascular_communities=True)
    assert MODULARITY in _method_names(settings)


def test_the_new_sections_render_only_when_a_method_in_them_ran(tmp_path):
    full = render_citations(
        _settings(tmp_path, statistics=True, do_skeletonize=True, smooth_centrelines=True)
    )
    assert "Centreline methods" in full and "Network statistics methods" in full
    assert "https://doi.org/10.1006/cgip.1994.1042" in full
    bare = render_citations(
        _settings(
            tmp_path,
            statistics=False,
            do_skeletonize=False,
            do_graph_building=False,
            compute_vascular_communities=False,
        )
    )
    assert "Centreline methods" not in bare and "Network statistics methods" not in bare
