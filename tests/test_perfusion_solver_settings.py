"""Open item 6: every perfusion tier takes its step cap and tolerance from the config.

Tier 1 hard-coded 50 Newton steps and 1e-5, Tier 2 50 passes and 1e-4, while
``PerfusionConfig.picard_tolerance`` said 1e-4 and only Tier 3 read it. Now all three read
``picard_max_iterations`` and ``picard_tolerance`` without a default, and the pipeline config and
``cb_settings.PerfusionSettings`` both hold 1e-5, the value behind the published H2 numbers.
"""
import logging
from dataclasses import dataclass, fields, replace

import numpy as np
import pytest

from ImageLynx import cb_settings
from ImageLynx.haemodynamics.perfusion import solve_perfusion_steady_state
from test_perfusion_config_values import _Config as _BarrierConfig, _coupled
from test_perfusion_tier1_solubility import _NetworkConfig, _capillary_network

_LOGGER = "ImageLynx.haemodynamics.perfusion"


def _tier1(config):
    grid, A, q, s, h = _capillary_network()
    return solve_perfusion_steady_state(grid, A, q, s, config, cell_hematocrit=h,
                                        return_info=True)


def _without_solver_fields(config_cls):
    """The same config class with the two stop settings removed."""
    kept = [(f.name, f.type, f.default) for f in fields(config_cls)
            if f.name not in ("picard_max_iterations", "picard_tolerance")]
    return dataclass(type(f"No{config_cls.__name__}", (), {
        "__annotations__": {n: t for n, t, _ in kept}, **{n: d for n, _, d in kept}}))()


def test_tier1_refuses_a_config_without_the_stop_settings():
    with pytest.raises(AttributeError, match="picard_max_iterations"):
        _tier1(_without_solver_fields(_NetworkConfig))


def test_tier2_refuses_a_config_without_the_stop_settings():
    with pytest.raises(AttributeError, match="picard_max_iterations"):
        _coupled(_without_solver_fields(_BarrierConfig))


def test_tier1_stops_at_the_configured_tolerance():
    loose_po2, loose = _tier1(replace(_NetworkConfig(), picard_tolerance=1e-2))
    tight_po2, tight = _tier1(replace(_NetworkConfig(), picard_tolerance=1e-10))
    assert loose["converged"] and tight["converged"]
    assert loose["residual"] < 1e-2 and tight["residual"] < 1e-10
    assert loose["iterations"] < tight["iterations"]
    # The loose stop is a real stop, not the tight answer reached early.
    assert not np.array_equal(loose_po2, tight_po2)


def test_tier1_stops_at_the_configured_cap_and_says_so(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _, info = _tier1(replace(_NetworkConfig(), picard_max_iterations=2,
                                 picard_tolerance=1e-12))
    assert not info["converged"]
    assert info["iterations"] == 2
    assert "hit max_iter (2)" in caplog.text


def test_tier2_stops_at_the_configured_cap_and_says_so(caplog):
    config = _BarrierConfig(M_max=1e-4)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _coupled(replace(config, picard_max_iterations=1, picard_tolerance=1e-12))
    assert "Coupled 1D-3D Picard iteration hit max_iter (1)" in caplog.text


def test_tier2_converged_run_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _coupled(replace(_BarrierConfig(M_max=1e-4), picard_max_iterations=200,
                         picard_tolerance=1e-4))
    assert "hit max_iter" not in caplog.text


def test_tier2_tolerance_changes_where_it_stops():
    config = replace(_BarrierConfig(M_max=1e-4), picard_max_iterations=200)
    # The first pass's step from the zero start is exactly 1, so above 1 it stops there.
    loose = _coupled(replace(config, picard_tolerance=1.5))
    tight = _coupled(replace(config, picard_tolerance=1e-10))
    assert not np.array_equal(loose, tight)


def test_the_h2_settings_carry_tier1s_old_hard_coded_values():
    """Tier 1 used 50 and 1e-5 for every published H2 number; moving them must not move one."""
    settings = cb_settings.PerfusionSettings()
    assert settings.picard_max_iterations == 50
    assert settings.picard_tolerance == 1e-5
