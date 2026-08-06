"""The probabilistic pericyte cohort must be reproducible.

Before the seed existed, every run drew a fresh cohort from an unseeded
generator, so two runs of identical code returned different resistances and a
published number could not be reproduced. These tests pin that down at each
level it has to hold: the draw, the resistance assignment, the run-level config,
and the schema the config file is generated from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT / "examples"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ImageLynx.haemodynamics.apply import (  # noqa: E402
    DEFAULT_PERICYTE_CONSTRICTION_SEED,
    HaemodynamicsApplyConfig,
    apply_poiseuille_haemodynamics,
)
from ImageLynx.haemodynamics.pericyte_comparison import (  # noqa: E402
    compare_baseline_vs_pericyte_constriction,
)
from ImageLynx.haemodynamics.probability import (  # noqa: E402
    resolve_generator,
    select_active_pericyte_indices,
    set_poiseuille_resistances_with_probabilistic_periodic_constrictions,
)

#: Long edges so each carries ~10 candidate pericyte sites at the default
#: 100 um spacing: enough that two seeds disagree on the cohort.
EDGE_LENGTH_UM = 1000.0
DIAMETERS = {"B01": 6.0}
CONSTRICTION_FACTORS = {"B01": 0.6}


def _chain_graph(num_edges: int = 3) -> nx.MultiGraph:
    """A straight capillary chain with physical positions and lengths."""
    graph = nx.MultiGraph()
    for node in range(num_edges + 1):
        graph.add_node(node, pos=np.asarray([0.0, 0.0, node * EDGE_LENGTH_UM]))
    for node in range(num_edges):
        graph.add_edge(
            node,
            node + 1,
            key=0,
            length=EDGE_LENGTH_UM,
            branch_order="B01",
            voxels=[[0.0, 0.0, node * EDGE_LENGTH_UM], [0.0, 0.0, (node + 1) * EDGE_LENGTH_UM]],
        )
    return graph


def _resistances(graph: nx.MultiGraph) -> list[float]:
    return [float(data["resistance"]) for _u, _v, data in graph.edges(data=True)]


def _constrict(**kwargs) -> tuple[list[float], dict]:
    graph, results = set_poiseuille_resistances_with_probabilistic_periodic_constrictions(
        _chain_graph(),
        diameter_by_branch_order=DIAMETERS,
        constriction_factor_by_branch_order=CONSTRICTION_FACTORS,
        constriction_probability=0.5,
        **kwargs,
    )
    return _resistances(graph), results["active_center_indices_by_edge"]


# ----------------------------------------------------------------------------
# The draw itself
# ----------------------------------------------------------------------------


def test_same_seed_selects_the_same_pericytes():
    first = select_active_pericyte_indices(200, 0.5, seed=7)
    second = select_active_pericyte_indices(200, 0.5, seed=7)
    assert first == second
    assert 0 < len(first) < 200  # a real draw, not all-in or all-out


def test_different_seeds_select_different_pericytes():
    assert select_active_pericyte_indices(200, 0.5, seed=7) != (
        select_active_pericyte_indices(200, 0.5, seed=8)
    )


def test_explicit_generator_beats_the_seed():
    from_generator = select_active_pericyte_indices(
        200,
        0.5,
        rng=np.random.default_rng(8),
        seed=7,
    )
    assert from_generator == select_active_pericyte_indices(200, 0.5, seed=8)
    assert from_generator != select_active_pericyte_indices(200, 0.5, seed=7)


def test_resolve_generator_without_seed_is_unseeded():
    """seed=None keeps the old opt-out behaviour: a fresh cohort each call."""
    assert resolve_generator(None, None).random() != resolve_generator(None, None).random()


# ----------------------------------------------------------------------------
# Resistance assignment
# ----------------------------------------------------------------------------


def test_same_seed_gives_identical_resistances():
    first_resistances, first_cohort = _constrict(seed=1234)
    second_resistances, second_cohort = _constrict(seed=1234)
    assert first_cohort == second_cohort
    assert first_resistances == second_resistances  # bit for bit, not approx


def test_different_seed_gives_a_different_cohort_and_resistances():
    first_resistances, first_cohort = _constrict(seed=1234)
    other_resistances, other_cohort = _constrict(seed=4321)
    assert first_cohort != other_cohort
    assert first_resistances != other_resistances


def test_explicit_generator_beats_the_seed_when_assigning_resistances():
    _seeded_resistances, seeded_cohort = _constrict(seed=4321)
    _passed_resistances, passed_cohort = _constrict(
        rng=np.random.default_rng(4321),
        seed=1234,
    )
    assert passed_cohort == seeded_cohort


def test_seeded_comparison_writes_the_same_csv_twice(tmp_path):
    """The published number comes off this CSV, so it is what must repeat."""

    def _run(seed: int, name: str) -> tuple[bytes, float]:
        csv_path = tmp_path / f"{name}.csv"
        summary = compare_baseline_vs_pericyte_constriction(
            _chain_graph(),
            diameter_by_branch_order=DIAMETERS,
            constriction_factor_by_branch_order=CONSTRICTION_FACTORS,
            resistance_node_pair=(0, 3),
            output_csv_path=csv_path,
            use_probabilistic_pericyte_constriction=True,
            pericyte_constriction_probability=0.5,
            seed=seed,
        )
        return csv_path.read_bytes(), float(summary["constricted_resistance"])

    first_csv, first_resistance = _run(99, "first")
    repeat_csv, repeat_resistance = _run(99, "repeat")
    other_csv, other_resistance = _run(100, "other")

    assert first_csv == repeat_csv
    assert first_resistance == repeat_resistance
    assert first_csv != other_csv
    assert first_resistance != other_resistance


# ----------------------------------------------------------------------------
# Run-level config
# ----------------------------------------------------------------------------


def _apply_config(**diameters) -> HaemodynamicsApplyConfig:
    return HaemodynamicsApplyConfig(
        diameters={
            "diameter_by_branch_order": DIAMETERS,
            "constriction_by_branch_order": CONSTRICTION_FACTORS,
            "do_pericyte_construction": True,
            "use_probabilistic_pericyte_constriction": True,
            "pericyte_constriction_probability": 0.5,
            **diameters,
        }
    )


def test_config_seeds_the_run_by_default():
    """Settings that say nothing about the seed still get a repeatable run."""
    assert _apply_config().pericyte_constriction_seed == DEFAULT_PERICYTE_CONSTRICTION_SEED


def test_config_seed_can_be_switched_off():
    assert _apply_config(pericyte_constriction_seed=None).pericyte_constriction_seed is None


def test_config_rng_beats_the_config_seed():
    config = _apply_config(pericyte_constriction_seed=1234)
    config.rng = np.random.default_rng(4321)
    drawn = config.pericyte_rng().random()
    assert drawn == np.random.default_rng(4321).random()


def test_apply_poiseuille_haemodynamics_repeats_under_one_seed():
    first, _ = apply_poiseuille_haemodynamics(
        _chain_graph(),
        config=_apply_config(pericyte_constriction_seed=2024),
    )
    repeat, _ = apply_poiseuille_haemodynamics(
        _chain_graph(),
        config=_apply_config(pericyte_constriction_seed=2024),
    )
    other, _ = apply_poiseuille_haemodynamics(
        _chain_graph(),
        config=_apply_config(pericyte_constriction_seed=2025),
    )

    assert _resistances(first) == _resistances(repeat)
    assert _resistances(first) != _resistances(other)


def test_apply_poiseuille_haemodynamics_unseeded_still_varies():
    """The opt-out has to keep working, or 'fresh cohort' becomes unreachable."""
    runs = [
        _resistances(
            apply_poiseuille_haemodynamics(
                _chain_graph(),
                config=_apply_config(pericyte_constriction_seed=None),
            )[0]
        )
        for _ in range(4)
    ]
    assert any(run != runs[0] for run in runs[1:])


# ----------------------------------------------------------------------------
# Schema and config file
# ----------------------------------------------------------------------------


def test_schema_declares_the_seed_beside_the_other_pericyte_settings():
    schema = __import__("resistance_pipeline_schema").SCHEMA
    setting = schema["pericyte_constriction_seed"]

    assert setting.kind == "int"
    assert setting.default == DEFAULT_PERICYTE_CONSTRICTION_SEED
    assert setting.section == schema["pericyte_constriction_probability"].section
    assert setting.requires == ("use_probabilistic_pericyte_constriction",)


def test_seed_survives_the_config_round_trip_and_reaches_the_run(tmp_path):
    yaml_module = pytest.importorskip("yaml")
    from ImageLynx.parsers import dump_config, load_config

    schema = __import__("resistance_pipeline_schema").SCHEMA
    config_path = tmp_path / "config.yaml"

    dump_config(
        config_path,
        schema,
        values={
            "use_probabilistic_pericyte_constriction": True,
            "pericyte_constriction_seed": 4242,
        },
    )
    raw = yaml_module.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["diameters_and_pericytes"]["pericyte_constriction_seed"] == 4242

    settings = load_config(config_path, schema)
    assert settings["pericyte_constriction_seed"] == 4242

    diameters = schema.section_values(settings, "Diameters and pericytes")
    assert diameters["pericyte_constriction_seed"] == 4242
    # The run builds its diameter table separately; everything else in the
    # section reaches the haemodynamics config exactly as the file wrote it.
    config = HaemodynamicsApplyConfig(
        diameters={**diameters, "diameter_by_branch_order": DIAMETERS}
    )
    assert config.pericyte_constriction_seed == 4242


def test_null_seed_survives_the_config_round_trip(tmp_path):
    from ImageLynx.parsers import dump_config, load_config

    schema = __import__("resistance_pipeline_schema").SCHEMA
    config_path = tmp_path / "config.yaml"

    dump_config(
        config_path,
        schema,
        values={
            "use_probabilistic_pericyte_constriction": True,
            "pericyte_constriction_seed": None,
        },
    )
    settings = load_config(config_path, schema)

    assert settings["pericyte_constriction_seed"] is None
    diameters = schema.section_values(settings, "Diameters and pericytes")
    config = HaemodynamicsApplyConfig(
        diameters={**diameters, "diameter_by_branch_order": DIAMETERS}
    )
    assert config.pericyte_constriction_seed is None
