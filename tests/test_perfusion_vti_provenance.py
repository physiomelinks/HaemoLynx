"""The pipeline's perfusion .vti says which solver and inputs made it (open item T).

The pipeline defaults to Tier 3, but the published H2 hypoxia numbers come from Tier 1 run
directly by the H2 drivers with cb_settings inputs. Without a tag the two fields look alike.
"""
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from carotid_image_to_model import (
    HaemodynamicsConfig,
    PerfusionConfig,
    _perfusion_provenance,
    _tag_perfusion_vti,
)

SOLVERS = {
    1: "solve_perfusion_steady_state",
    2: "solve_coupled_1d3d_perfusion",
    3: "solve_multi_species_perfusion",
}


_INFO = {"converged": True, "iterations": 11, "residual_o2": 2.4e-5, "residual_co2": 7.0e-6}


def _provenance(tier, info=_INFO):
    return _perfusion_provenance(tier, SOLVERS[tier], PerfusionConfig(), HaemodynamicsConfig(),
                                 info if tier == 3 else None)


def test_tier_3_records_its_solver_rate_pressures_and_tolerance():
    perf, hemo = PerfusionConfig(), HaemodynamicsConfig()
    tags = _provenance(3)

    assert tags["perfusion_tier"] == 3
    assert tags["perfusion_solver"] == "solve_multi_species_perfusion"
    assert tags["perfusion_M_max"] == perf.M_max
    assert tags["perfusion_picard_tolerance"] == perf.picard_tolerance
    # Config pressures are in mPa; the tag is in mmHg. 13.332e6 mPa is 100 mmHg, 0.27e6 is ~2.03.
    assert tags["perfusion_inlet_pressure_mmHg"] == pytest.approx(100.0, abs=0.01)
    assert tags["perfusion_outlet_pressure_mmHg"] == pytest.approx(2.025, abs=0.01)
    assert tags["perfusion_inlet_pressure_mmHg"] * 133322.387415 == pytest.approx(hemo.input_p_bc)


@pytest.mark.parametrize("tier", [1, 2])
def test_tiers_that_hard_code_their_tolerance_do_not_claim_the_config_one(tier):
    tags = _provenance(tier)
    assert tags["perfusion_tier"] == tier
    assert tags["perfusion_solver"] == SOLVERS[tier]
    assert "perfusion_picard_tolerance" not in tags


@pytest.mark.parametrize("tier", [1, 2, 3])
def test_every_tag_says_it_is_not_the_h2_field(tier):
    note = _provenance(tier)["perfusion_note"]
    assert "not the H2 field" in note
    assert "Tier 1" in note and "solve_perfusion_steady_state" in note


def test_a_config_missing_M_max_is_refused_not_defaulted():
    class Partial:
        picard_tolerance = 1e-4

    with pytest.raises(AttributeError, match="M_max"):
        _perfusion_provenance(3, SOLVERS[3], Partial(), HaemodynamicsConfig(), _INFO)


def test_tier_3_records_whether_its_loop_converged():
    """Open item 23: an unconverged Tier 3 field must say so in the file."""
    tags = _provenance(3)
    assert tags["perfusion_converged"] == 1
    assert tags["perfusion_picard_iterations"] == 11
    assert tags["perfusion_final_residual"] == 2.4e-5
    unconverged = _provenance(3, dict(_INFO, converged=False, iterations=50))
    assert unconverged["perfusion_converged"] == 0
    assert unconverged["perfusion_picard_iterations"] == 50


def test_tier_3_without_its_convergence_info_is_refused():
    with pytest.raises(ValueError, match="convergence info"):
        _perfusion_provenance(3, SOLVERS[3], PerfusionConfig(), HaemodynamicsConfig())


@pytest.mark.parametrize("tier", [1, 2])
def test_tiers_1_and_2_carry_no_convergence_tags(tier):
    tags = _provenance(tier)
    assert not {"perfusion_converged", "perfusion_picard_iterations",
                "perfusion_final_residual"} & set(tags)


def test_the_tags_survive_a_write_and_read_and_keep_the_cell_arrays(tmp_path):
    path = tmp_path / "specimen_perfusion.vti"
    img = pv.ImageData(dimensions=(3, 3, 3))
    po2 = np.arange(img.n_cells, dtype=float)
    img.cell_data["PO2_mmHg"] = po2
    img.save(path)

    tags = _provenance(3)
    _tag_perfusion_vti(path, tags)
    vol = pv.read(path)

    np.testing.assert_array_equal(vol.cell_data["PO2_mmHg"], po2)
    for key, value in tags.items():
        stored = vol.field_data[key][0]
        if isinstance(value, str):
            assert str(stored) == value
        else:
            assert stored == pytest.approx(value)
