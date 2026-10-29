"""The render slab pins its surface-extraction algorithm."""
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
