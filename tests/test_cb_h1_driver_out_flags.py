"""``--out`` lets cb_h1_figures, cb_h1_vtk and cb_h1_renders write elsewhere (C5).

The driver-snapshot tool runs each driver with ``--out`` pointing into a snapshot folder. With
no flag every driver must still write where it always did.
"""
from types import SimpleNamespace

import pytest

pv = pytest.importorskip("pyvista")

import cb_h1_figures  # noqa: E402
import cb_h1_renders  # noqa: E402
import cb_h1_vtk  # noqa: E402

FIGURE_FILES = {"figure1_network_density.png", "figure2_diameter_distribution.png",
                "figure3_threshold_sensitivity.png", "figure7_node_degree.png",
                "figure8_segment_length.png"}


@pytest.fixture
def stub_figures(tmp_path, monkeypatch):
    """Every figure function writes a marker file; the batch folder is a tmp folder."""
    batch = tmp_path / "batch"
    monkeypatch.setattr(cb_h1_figures, "OUTPUT_DIR", batch)
    monkeypatch.setattr(cb_h1_figures, "_load_diameters", lambda: None)
    for name in ("figure_density", "figure_sensitivity", "figure_degree",
                 "figure_segment_length"):
        monkeypatch.setattr(cb_h1_figures, name, lambda path: path.write_bytes(b"x"))
    monkeypatch.setattr(cb_h1_figures, "figure_diameter",
                        lambda path, diameters: path.write_bytes(b"x"))
    return batch


def test_figures_default_to_the_batch_folder(stub_figures):
    cb_h1_figures.main([])
    assert {p.name for p in stub_figures.iterdir()} == FIGURE_FILES


def test_figures_write_into_out_and_leave_the_batch_folder_alone(stub_figures, tmp_path):
    out = tmp_path / "snap" / "cb_h1_figures"
    cb_h1_figures.main(["--out", str(out)])
    assert {p.name for p in out.iterdir()} == FIGURE_FILES
    assert not stub_figures.exists()


@pytest.fixture
def stub_vtk(tmp_path, monkeypatch):
    monkeypatch.setattr(cb_h1_vtk, "OUTPUT", tmp_path / "paraview")
    monkeypatch.setattr(cb_h1_vtk, "SPECIMENS", [SimpleNamespace(specimen_id="WKY-A",
                                                                 group="WKY")])
    run = SimpleNamespace(edge_table=lambda: {})
    monkeypatch.setattr(cb_h1_vtk, "open_batch_run", lambda specimen: run)
    monkeypatch.setattr(cb_h1_vtk, "enrich_vessels", lambda run, edges, report: None)
    monkeypatch.setattr(cb_h1_vtk, "build_nodes", lambda run, edges, report: None)
    monkeypatch.setattr(cb_h1_vtk, "build_skeleton", lambda run, report: None)
    monkeypatch.setattr(cb_h1_vtk, "build_surface", lambda run, report: (None, None))
    return tmp_path / "paraview"


def test_vtk_defaults_to_the_paraview_folder(stub_vtk):
    cb_h1_vtk.main([])
    assert (stub_vtk / "export_summary.json").exists()


def test_vtk_writes_into_out_and_leaves_the_paraview_folder_alone(stub_vtk, tmp_path, capsys):
    out = tmp_path / "snap" / "cb_h1_vtk"
    cb_h1_vtk.main(["--out", str(out)])
    assert (out / "export_summary.json").exists()
    assert not stub_vtk.exists()
    assert f"Wrote {out}" in capsys.readouterr().out


@pytest.fixture
def renders_calls(monkeypatch):
    calls = []
    for name in ("figure_reconstruction", "figure_measured", "figure_skeleton_detail"):
        monkeypatch.setattr(cb_h1_renders, name,
                            lambda path, *, vtk_dir, _name=name: calls.append((_name, path, vtk_dir)))
    return calls


def test_renders_default_to_the_paraview_and_batch_folders(renders_calls, tmp_path, monkeypatch):
    monkeypatch.setattr(cb_h1_renders, "OUT", tmp_path / "batch")
    cb_h1_renders.main([])
    assert [(n, p.parent, v) for n, p, v in renders_calls] == [
        (n, tmp_path / "batch", cb_h1_renders.VTK)
        for n in ("figure_reconstruction", "figure_measured", "figure_skeleton_detail")]


def test_renders_read_from_vtk_dir_and_write_into_out(renders_calls, tmp_path):
    out, vtk = tmp_path / "out", tmp_path / "vtk"
    cb_h1_renders.main(["--out", str(out), "--vtk-dir", str(vtk)])
    assert {p.parent for _, p, _ in renders_calls} == {out}
    assert {v for _, _, v in renders_calls} == {vtk}
