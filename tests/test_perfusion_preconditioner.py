"""The linear solve inside the perfusion Picard loop.

Conjugate gradient assumes its preconditioner is symmetric positive definite. An incomplete-LU
factorisation is a general-purpose approximation and carries no such guarantee, and given a
preconditioner that is neither, CG does not converge slowly, it diverges.
"""
import logging

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from ImageLynx.haemodynamics.perfusion import _jacobi_preconditioner


def _diffusion_matrix(n=400, coupling=1.0, extra_diag=1e-6):
    """A 1D Laplacian with a small regulariser: the shape the ADR assembly produces."""
    main = np.full(n, 2.0 * coupling + extra_diag)
    main[0] = main[-1] = coupling + extra_diag
    A = sp.diags([np.full(n - 1, -coupling), main, np.full(n - 1, -coupling)],
                 [-1, 0, 1], format="csr")
    return A


def test_the_preconditioner_is_the_inverse_diagonal():
    A = _diffusion_matrix(50)
    M = _jacobi_preconditioner(A)
    v = np.arange(1.0, 51.0)
    np.testing.assert_allclose(M @ v, v / A.diagonal())


def test_it_is_symmetric_positive_definite_which_is_what_cg_requires():
    A = _diffusion_matrix(60)
    M = _jacobi_preconditioner(A)
    dense = np.column_stack([M @ e for e in np.eye(60)])
    np.testing.assert_allclose(dense, dense.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(dense) > 0)


def test_cg_converges_under_it_on_a_representative_system():
    A = _diffusion_matrix(500)
    rng = np.random.default_rng(0)
    b = rng.normal(size=500)
    x, info = splinalg.cg(A, b, M=_jacobi_preconditioner(A), rtol=1e-8, maxiter=5000)
    assert info == 0
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-7


def test_cg_converges_on_a_production_like_ill_conditioned_system():
    """The condition the fix had to survive.

    The assembled matrix is a Laplacian plus a 1e-6 regulariser against face conductances of
    order 1e4, so the condition number is around 1e10. That is what defeated the previous
    preconditioner: measured on the real 10 um grid for WKY-C, ILU-preconditioned CG reached a
    relative residual of 19 after 1000 iterations, where the initial residual is 1 by
    construction, and the resulting PO2 field was 4e-4 mmHg everywhere against an arterial 100.
    Unpreconditioned and Jacobi-preconditioned CG both reached 8.8e-7 in 0.05 s on the same
    system. The measurement is recorded in S24; what is pinned here is that the preconditioner
    this module returns does not stand in the way of convergence on that shape of problem.
    """
    A = _diffusion_matrix(2000, coupling=1.5e4, extra_diag=1e-6)
    rng = np.random.default_rng(2)
    b = rng.normal(size=2000)
    b -= b.mean()          # orthogonal to the near-null constant vector, as the real RHS is

    x, info = splinalg.cg(A, b, M=_jacobi_preconditioner(A), rtol=1e-6, maxiter=5000)
    assert info == 0, f"CG did not converge, info={info}"
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-5


def test_a_non_positive_diagonal_declines_rather_than_forming_a_bad_preconditioner(caplog):
    A = _diffusion_matrix(20)
    A = A.tolil()
    A[5, 5] = -1.0
    with caplog.at_level(logging.WARNING):
        assert _jacobi_preconditioner(A.tocsr()) is None
    assert "non-positive" in caplog.text


def test_the_perfusion_matrix_diagonal_is_positive_by_construction():
    """Face conductances plus a positive regulariser plus a non-negative washout."""
    import networkx as nx

    from ImageLynx.haemodynamics.perfusion import (
        PerfusionGrid, build_adr_matrix, map_vessels_to_grid)

    class Cfg:
        sigma_diff, M_max, k_reduce, C_arterial = 1.5e-9, 0.05, 0.1, 0.13

    G = nx.MultiGraph()
    G.add_node(1, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([30.0, 30.0, 30.0]))
    G.add_edge(1, 2, key=0, length=20.0, flow_abs=5.0, assigned_diameter_um=8.0,
               voxels=[np.array([t, t, t]) for t in np.linspace(2.0, 28.0, 6)])
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    A, _q, _s = build_adr_matrix(grid, map_vessels_to_grid(G, grid), Cfg())

    assert np.all(A.diagonal() > 0)
    assert _jacobi_preconditioner(A) is not None
