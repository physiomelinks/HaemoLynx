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


def test_pandas_is_cited_only_when_statistics_is_on(tmp_path):
    on = _settings(tmp_path, statistics=True)
    off = _settings(tmp_path, statistics=False)
    assert "pandas" in _names(used_citations(on, PACKAGE_CITATIONS))
    assert "pandas" not in _names(used_citations(off, PACKAGE_CITATIONS))


def test_matplotlib_follows_visualize_results(tmp_path):
    on = _settings(tmp_path, visualize_results=True)
    off = _settings(tmp_path, visualize_results=False)
    assert "Matplotlib" in _names(used_citations(on, PACKAGE_CITATIONS))
    assert "Matplotlib" not in _names(used_citations(off, PACKAGE_CITATIONS))


def test_plotly_needs_both_visualize_results_and_3d_render_mode(tmp_path):
    both = _settings(tmp_path, visualize_results=True, final_render_mode="3d")
    only_2d = _settings(tmp_path, visualize_results=True, final_render_mode="2d")
    only_off = _settings(tmp_path, visualize_results=False, final_render_mode="3d")
    assert "Plotly" in _names(used_citations(both, PACKAGE_CITATIONS))
    assert "Plotly" not in _names(used_citations(only_2d, PACKAGE_CITATIONS))
    assert "Plotly" not in _names(used_citations(only_off, PACKAGE_CITATIONS))


def test_pyvista_needs_both_run_haemodynamics_and_vtk_export(tmp_path):
    both = _settings(tmp_path, run_haemodynamics=True, vtk_export=True)
    no_vtk = _settings(tmp_path, run_haemodynamics=True, vtk_export=False)
    no_haemo = _settings(tmp_path, run_haemodynamics=False, vtk_export=True)
    assert "PyVista / VTK" in _names(used_citations(both, PACKAGE_CITATIONS))
    assert "PyVista / VTK" not in _names(used_citations(no_vtk, PACKAGE_CITATIONS))
    assert "PyVista / VTK" not in _names(used_citations(no_haemo, PACKAGE_CITATIONS))


def test_h5py_and_tifffile_follow_the_input_suffix(tmp_path):
    h5 = _settings(tmp_path, input_path=tmp_path / "input.h5")
    tif = _settings(tmp_path, input_path=tmp_path / "input.tif")
    assert "h5py" in _names(used_citations(h5, PACKAGE_CITATIONS))
    assert "tifffile" not in _names(used_citations(h5, PACKAGE_CITATIONS))
    assert "tifffile" in _names(used_citations(tif, PACKAGE_CITATIONS))
    assert "h5py" not in _names(used_citations(tif, PACKAGE_CITATIONS))


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


@pytest.mark.parametrize(
    "viscosity_law, expected_name",
    [
        ("pries", "Pries apparent-viscosity law (viscosity_law='pries')"),
        (
            "capillary_power_law",
            "Legacy capillary power-law viscosity (viscosity_law='capillary_power_law')",
        ),
    ],
)
def test_viscosity_law_citation_matches_the_chosen_law(tmp_path, viscosity_law, expected_name):
    settings = _settings(tmp_path, run_haemodynamics=True, viscosity_law=viscosity_law)
    names = _names(used_citations(settings, MODEL_CITATIONS))
    assert expected_name in names
    other = "pries" if viscosity_law != "pries" else "capillary_power_law"
    other_name = (
        "Pries apparent-viscosity law (viscosity_law='pries')"
        if other == "pries"
        else "Legacy capillary power-law viscosity (viscosity_law='capillary_power_law')"
    )
    assert other_name not in names


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

    assert assign_to_stages(SCHEMA)["export_citations"] == "9. Export"


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
