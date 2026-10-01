"""The render slab pins its surface-extraction algorithm (re-run package R)."""
import warnings

import pyvista as pv

import cb_h1_renders


def _mesh():
    return pv.ImageData(dimensions=(12, 12, 40), spacing=(1.0, 1.0, 1.0)).cast_to_unstructured_grid()


def test_slab_raises_no_pyvista_future_warning():
    mesh = _mesh()
    with warnings.catch_warnings():
        warnings.simplefilter("error", pv.PyVistaFutureWarning)
        surface = cb_h1_renders.slab(mesh, mesh)
    assert isinstance(surface, pv.PolyData)
    assert surface.n_cells > 0


def test_slab_uses_the_dataset_surface_algorithm():
    mesh = _mesh()
    b = mesh.bounds
    centre = (b[4] + b[5]) / 2.0
    half = cb_h1_renders.SLAB_UM / 2
    box = (b[0], b[1], b[2], b[3], centre - half, centre + half)
    expected = mesh.clip_box(box, invert=False).extract_surface(algorithm="dataset_surface")
    surface = cb_h1_renders.slab(mesh, mesh)
    assert surface.n_points == expected.n_points
    assert surface.n_cells == expected.n_cells
