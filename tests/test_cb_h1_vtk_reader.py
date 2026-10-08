"""cb_h1_vtk.py reads through the batch-run reader and joins rows onto cells strictly.

Batch-run reader ticket 04. A ``.vtp`` cell with no edge-table row used to be written as
NaN, 0 or "", so a missing row looked like a measurement. It now raises. The placed ROI is
checked by opening the run, before the edge table is read.
"""
from types import SimpleNamespace

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

import cb_h1_vtk  # noqa: E402

EDGES = [(0, 1, 0), (1, 2, 0), (1, 2, 1)]


def _row(diameter):
    return {
        "length_um": "12.5", "euclidean_um": "10.0", "tortuosity": "1.25", "curvature": "0.1",
        "edt_diameter_um": "6.0", "fwhm_diameter_um": "", "assigned_diameter_um": diameter,
        "n_centreline_points": "7", "reconnected": "True",
        "diameter_provenance": "measured_edt", "edt_junction_trim": "trimmed",
        "centreline_smoothing": "bspline",
    }


def _run(tmp_path, cell_keys):
    """A run folder holding a .vtp whose cells carry ``cell_keys`` as (u, v, key)."""
    points = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    lines = np.hstack([[2, i % 2, i % 2 + 1] for i in range(len(cell_keys))])
    mesh = pv.PolyData(points, lines=lines)
    u, v, k = (np.array(column) for column in zip(*cell_keys))
    mesh.cell_data["edge_u"], mesh.cell_data["edge_v"], mesh.cell_data["edge_key"] = u, v, k
    mesh.save(tmp_path / "resistance_network_vessels.vtp")
    specimen = SimpleNamespace(specimen_id="WKY-A", group="WKY",
                               probabilities_path=tmp_path / "missing.h5")
    return SimpleNamespace(specimen=specimen, run_dir=tmp_path)


def test_every_cell_gets_its_own_row(tmp_path):
    run = _run(tmp_path, EDGES)
    edges = {edge: _row(str(d)) for edge, d in zip(EDGES, (4.0, 6.0, 8.0))}
    report = {}
    mesh = cb_h1_vtk.enrich_vessels(run, edges, report)
    assert report == {"vessels_cells": 3, "vessels_matched": 3}
    assert mesh.cell_data["assigned_diameter_um"].tolist() == [4.0, 6.0, 8.0]
    assert mesh.cell_data["radius_um"].tolist() == [2.0, 3.0, 4.0]
    assert mesh.cell_data["reconnected"].tolist() == [1, 1, 1]
    assert mesh.cell_data["diameter_provenance_code"].tolist() == [0, 0, 0]


def test_a_cell_with_no_row_raises_rather_than_being_filled(tmp_path):
    run = _run(tmp_path, EDGES)
    edges = {edge: _row("5.0") for edge in EDGES[:2]}
    with pytest.raises(ValueError, match=r"\(1, 2, 1\)"):
        cb_h1_vtk.enrich_vessels(run, edges, {})


def _stub_builders(monkeypatch, events):
    monkeypatch.setattr(cb_h1_vtk, "SPECIMENS", [SimpleNamespace(specimen_id="WKY-A",
                                                                 group="WKY")])
    monkeypatch.setattr(cb_h1_vtk, "enrich_vessels",
                        lambda run, edges, report: events.append("vessels"))
    monkeypatch.setattr(cb_h1_vtk, "build_nodes",
                        lambda run, edges, report: events.append("nodes"))
    monkeypatch.setattr(cb_h1_vtk, "build_skeleton", lambda run, report: events.append("skeleton"))
    monkeypatch.setattr(cb_h1_vtk, "build_surface",
                        lambda run, report: events.append("surface") or (None, None))
    monkeypatch.setattr("sys.argv", ["cb_h1_vtk.py", "--verify"])


def test_the_run_is_opened_before_the_edge_table_is_read(monkeypatch):
    events = []

    class _Run:
        def edge_table(self):
            events.append("edge_table")
            return {}

    def _open(specimen):
        events.append("open")
        return _Run()

    _stub_builders(monkeypatch, events)
    monkeypatch.setattr(cb_h1_vtk, "open_batch_run", _open)
    cb_h1_vtk.main()
    assert events[:2] == ["open", "edge_table"]


def test_a_refused_run_raises_before_anything_is_read(monkeypatch):
    events = []

    def _open(specimen):
        raise ValueError("cropped at another box")

    _stub_builders(monkeypatch, events)
    monkeypatch.setattr(cb_h1_vtk, "open_batch_run", _open)
    with pytest.raises(ValueError, match="another box"):
        cb_h1_vtk.main()
    assert events == []
