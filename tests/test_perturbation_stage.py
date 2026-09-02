"""Running the perturbations: one baseline, N independent re-solves.

The stage's whole promise is independence. Each perturbation answers a
question about the finished network -- what if the arterioles dilate, what if
the pericytes tighten -- and the answer is only meaningful if it was asked of
the same network as every other, and if asking it did not edit the network the
run goes on to export. Two perturbations that composed would report the second
one's effect as the pair's, and nobody would see it happen.

So what is pinned here is mostly what does *not* change: the baseline's
resistances after the stage, and one perturbation's result with and without
another beside it.

Independence is cheaper than it looks. `_perturbation_copy` is a *shallow*
graph copy, so a perturbation holds the very same `voxels` lists and `pos`
arrays as the baseline -- only the attribute dicts around them are new. That is
sound exactly as long as nothing on a perturbation path mutates one of those
values in place, so the geometry guards below snapshot them before the stage
and compare element by element after it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haemolynx.haemodynamics import PERTURBATION_TYPES, PoiseuilleModel  # noqa: E402
from haemolynx.pipeline import (  # noqa: E402
    BoundaryNodes,
    HaemodynamicModel,
    PerturbationRun,
    default_schema,
    resolve_settings,
    run_perturbations,
)
from haemolynx.pipeline.progress import STEP, ProgressEvent, RunProgress  # noqa: E402
from haemolynx.pipeline.stages import _perturbation_copy  # noqa: E402

SCHEMA = default_schema()

#: A capillary bed between an arteriole and a venule, long enough that the
#: periodic constriction model has somewhere to put a pericyte.
EDGE_LENGTH_UM = 400.0
DIAMETERS = {"Art1": 20.0, "B01": 6.0, "Ven1": 20.0}
BRANCH_ORDERS = ("Art1", "B01", "B01", "Ven1")


def _network() -> nx.MultiGraph:
    graph = nx.MultiGraph()
    for node in range(len(BRANCH_ORDERS) + 1):
        graph.add_node(node, pos=np.asarray([0.0, 0.0, node * EDGE_LENGTH_UM]))
    for node, branch_order in enumerate(BRANCH_ORDERS):
        graph.add_edge(
            node,
            node + 1,
            key=0,
            length=EDGE_LENGTH_UM,
            branch_order=branch_order,
            voxels=[
                [0.0, 0.0, node * EDGE_LENGTH_UM],
                [0.0, 0.0, (node + 1) * EDGE_LENGTH_UM],
            ],
        )
    return graph


def _model() -> HaemodynamicModel:
    """The baseline: a solved network, as `build_haemodynamic_model` leaves it."""
    graph, _results = PoiseuilleModel(
        constriction_length=40.0,
        constriction_spacing=100.0,
        viscosity_law="constant",
    ).set_poiseuille_resistances(_network(), DIAMETERS)
    return HaemodynamicModel(graph=graph)


def _boundaries() -> BoundaryNodes:
    last = len(BRANCH_ORDERS)
    return BoundaryNodes(
        inlet_nodes=[0],
        outlet_nodes=[last],
        resistance_node_pair=(0, last),
    )


def _settings(tmp_path: Path, perturbations: list[dict], **extra) -> dict:
    values = {setting.name: setting.default for setting in SCHEMA}
    values.update(
        {
            "input_path": tmp_path / "input.tif",
            "vtk_output_prefix": tmp_path / "out" / "run",
            "plot_dir": tmp_path / "plots",
            "run_haemodynamics": True,
            "run_perturbations": True,
            "perturbations": perturbations,
            "diameter_by_branch_order": dict(DIAMETERS),
            "constriction_by_branch_order": {order: 0.6 for order in DIAMETERS},
            "use_fwhm_edge_diameters": False,
            "viscosity_law": "constant",
            # A four-point sweep: enough to draw a curve, quick enough to run
            # in a unit test.
            "pericyte_dilation_min_percent": 1,
            "pericyte_dilation_max_percent": 2,
            "pericyte_dilation_step_percent": 1,
            "arteriole_dilation_min_percent": 0,
            "arteriole_dilation_max_percent": 10,
            "arteriole_dilation_step_percent": 10,
            "capillary_dilation_min_percent": 0,
            "capillary_dilation_max_percent": 10,
            "capillary_dilation_step_percent": 10,
            "constriction_spacing_min_um": 50.0,
            "constriction_spacing_max_um": 100.0,
            "constriction_spacing_step_um": 50.0,
            "constriction_length_min_um": 20.0,
            "constriction_length_max_um": 40.0,
            "constriction_length_step_um": 20.0,
            "pericyte_geometry_dilation_percent": 0,
            "inlet_pressure_min_pa": 4500,
            "inlet_pressure_max_pa": 5000,
            "inlet_pressure_step_pa": 500,
        }
    )
    values.update(extra)
    return resolve_settings(values, schema=SCHEMA, config_path=None)


def _run(tmp_path: Path, perturbations: list[dict], **extra) -> PerturbationRun:
    return run_perturbations(
        _settings(tmp_path, perturbations, **extra),
        _model(),
        _boundaries(),
        SCHEMA,
    )


def _resistances(graph: nx.MultiGraph) -> dict[tuple, float]:
    return {
        (u, v, key): float(data["resistance"])
        for u, v, key, data in graph.edges(keys=True, data=True)
    }


def _geometry(graph: nx.MultiGraph) -> dict[Any, np.ndarray]:
    """Every node's `pos` and every edge's `voxels`, copied out of the graph.

    Copied, because a snapshot has to survive an in-place edit of the thing it
    is a snapshot of -- holding the graph's own objects would compare them with
    themselves and pass whatever happened to them.
    """
    snapshot: dict[Any, np.ndarray] = {
        node: np.asarray(data["pos"], dtype=float).copy()
        for node, data in graph.nodes(data=True)
    }
    snapshot.update(
        {
            (u, v, key): np.asarray(data["voxels"], dtype=float).copy()
            for u, v, key, data in graph.edges(keys=True, data=True)
        }
    )
    return snapshot


def _assert_geometry_unmoved(
    graph: nx.MultiGraph, before: dict[Any, np.ndarray]
) -> None:
    """Element-by-element, not object identity: the sharing is expected."""
    assert _geometry(graph).keys() == before.keys()
    for node, data in graph.nodes(data=True):
        np.testing.assert_array_equal(
            np.asarray(data["pos"], dtype=float), before[node], f"node {node} moved"
        )
    for u, v, key, data in graph.edges(keys=True, data=True):
        np.testing.assert_array_equal(
            np.asarray(data["voxels"], dtype=float),
            before[(u, v, key)],
            f"the centreline of edge ({u}, {v}, {key}) moved",
        )


def _attribute_names(graph: nx.MultiGraph) -> dict[Any, frozenset]:
    """What each node and edge carries, so a leaked write shows up as a new key."""
    names: dict[Any, frozenset] = {
        node: frozenset(data) for node, data in graph.nodes(data=True)
    }
    names.update(
        {
            (u, v, key): frozenset(data)
            for u, v, key, data in graph.edges(keys=True, data=True)
        }
    )
    return names


ARTERIOLE_DILATION = {
    "name": "art_dilate_20",
    "type": "arteriole_diameter_change",
    "overrides": {"arteriole_diameter_change_percent": 20},
}
ARTERIOLE_CONSTRICTION = {
    "name": "art_narrow_20",
    "type": "arteriole_diameter_change",
    "overrides": {"arteriole_diameter_change_percent": -20},
}
PERICYTE_TONE = {
    "name": "pericytes_tighten",
    "type": "pericyte_diameter_change",
    "overrides": {
        "do_pericyte_construction": True,
        "constriction_by_branch_order": {"Art1": 1.0, "B01": 0.5, "Ven1": 1.0},
    },
}
ARTERIOLE_AND_PERICYTE = {
    "name": "art_and_pericytes",
    "type": "arteriole_and_pericyte_diameter_change",
    "overrides": {
        "arteriole_diameter_change_percent": 20,
        "do_pericyte_construction": True,
        "constriction_by_branch_order": {"Art1": 1.0, "B01": 0.5, "Ven1": 1.0},
    },
}
DILATION_SWEEP = {
    "name": "dilation_only",
    "type": "pericyte_dilation_sweep",
    "overrides": {},
}
PRESSURE_SWEEP = {
    "name": "pressure_only",
    "type": "pressure_sweep",
    "overrides": {},
}
PRESSURE_AND_PERICYTE_SWEEP = {
    "name": "dilation_and_pressure",
    "type": "pressure_and_pericyte_sweep",
    "overrides": {},
}
ARTERIOLE_DIAMETER_SWEEP = {
    "name": "arteriole_dilation_only",
    "type": "arteriole_diameter_sweep",
    "overrides": {},
}
PRESSURE_AND_ARTERIOLE_SWEEP = {
    "name": "arteriole_and_pressure",
    "type": "pressure_and_arteriole_sweep",
    "overrides": {},
}
CAPILLARY_DIAMETER_SWEEP = {
    "name": "capillary_dilation_only",
    "type": "capillary_diameter_sweep",
    "overrides": {},
}
PRESSURE_AND_CAPILLARY_SWEEP = {
    "name": "capillary_and_pressure",
    "type": "pressure_and_capillary_sweep",
    "overrides": {},
}
SPACING_SWEEP = {
    "name": "spacing_only",
    "type": "pericyte_spacing_sweep",
    "overrides": {
        "constriction_spacing_min_um": 50.0,
        "constriction_spacing_max_um": 100.0,
        "constriction_spacing_step_um": 50.0,
        "constriction_length_um": 40.0,
        "pericyte_geometry_dilation_percent": 0,
    },
}
LENGTH_SWEEP = {
    "name": "length_only",
    "type": "pericyte_length_sweep",
    "overrides": {
        "constriction_length_min_um": 20.0,
        "constriction_length_max_um": 40.0,
        "constriction_length_step_um": 20.0,
        "constriction_spacing_um": 100.0,
        "pericyte_geometry_dilation_percent": 0,
    },
}
NO_OP = {"name": "placeholder", "type": "none", "overrides": {}}

#: One worked entry per type, keyed by the type it exercises. What the guards
#: below are parametrised over is `PERTURBATION_TYPES` itself, and this is how
#: each of those names becomes something runnable -- so a fifth type added to
#: the module is guarded the day it appears, instead of being quietly exempt
#: because the parametrise list still spells out today's four.
ENTRY_FOR_TYPE: dict[str, dict] = {
    "none": NO_OP,
    "pressure_sweep": PRESSURE_SWEEP,
    "pressure_and_pericyte_sweep": PRESSURE_AND_PERICYTE_SWEEP,
    "pericyte_dilation_sweep": DILATION_SWEEP,
    "arteriole_diameter_change": ARTERIOLE_DILATION,
    "arteriole_diameter_sweep": ARTERIOLE_DIAMETER_SWEEP,
    "pressure_and_arteriole_sweep": PRESSURE_AND_ARTERIOLE_SWEEP,
    "capillary_diameter_sweep": CAPILLARY_DIAMETER_SWEEP,
    "pressure_and_capillary_sweep": PRESSURE_AND_CAPILLARY_SWEEP,
    "pericyte_spacing_sweep": SPACING_SWEEP,
    "pericyte_length_sweep": LENGTH_SWEEP,
    "pericyte_diameter_change": PERICYTE_TONE,
    "arteriole_and_pericyte_diameter_change": ARTERIOLE_AND_PERICYTE,
}

#: Every type but `none`, which by definition re-solves nothing and writes
#: nothing; `test_a_none_perturbation_produces_nothing` is what covers it.
TYPES_THAT_RUN = tuple(name for name in PERTURBATION_TYPES if name != "none")

#: One entry of every type, for the tests that run the whole list at once.
EVERY_TYPE_ONCE = [
    ENTRY_FOR_TYPE[name] for name in PERTURBATION_TYPES if name in ENTRY_FOR_TYPE
]


def _entry_for(perturbation_type: str) -> dict:
    entry = ENTRY_FOR_TYPE.get(perturbation_type)
    if entry is None:
        pytest.fail(
            f"perturbation type {perturbation_type!r} has no entry in "
            "ENTRY_FOR_TYPE, so nothing in this file runs it and the guards "
            "below are not guarding it. Add one."
        )
    return entry


def test_every_type_has_something_that_exercises_it():
    """The guards are only as complete as this table is.

    Parametrising over `PERTURBATION_TYPES` is what makes a new type covered
    automatically, and this is what makes that cover real rather than an empty
    parametrise case.
    """
    assert set(ENTRY_FOR_TYPE) == set(PERTURBATION_TYPES)
    for perturbation_type, entry in ENTRY_FOR_TYPE.items():
        assert entry["type"] == perturbation_type


# --- nothing configured, nothing done ----------------------------------------


def test_the_stage_does_nothing_when_it_is_switched_off(tmp_path):
    run = _run(tmp_path, [ARTERIOLE_DILATION], run_perturbations=False)

    assert run.results == []
    assert run.output_dir is None
    assert not (tmp_path / "out" / "perturbations").exists()


def test_the_stage_does_nothing_without_haemodynamics(tmp_path):
    """There are no resistances to re-solve, so there is nothing to perturb."""
    run = _run(tmp_path, [ARTERIOLE_DILATION], run_haemodynamics=False)

    assert run.results == []
    assert not (tmp_path / "out" / "perturbations").exists()


def test_a_none_perturbation_produces_nothing(tmp_path):
    """It is the type an entry has before a user has chosen one."""
    run = _run(tmp_path, [NO_OP])

    assert [result.name for result in run.results] == ["placeholder"]
    result = run.results[0]
    assert result.ok
    assert result.graph is None
    assert result.output_dir is None
    assert result.outputs == []
    assert not (tmp_path / "out" / "perturbations" / "placeholder").exists()


# --- each type writes its own output -----------------------------------------


@pytest.mark.parametrize("perturbation_type", TYPES_THAT_RUN)
def test_each_type_writes_its_own_directory_and_files(tmp_path, perturbation_type):
    entry = _entry_for(perturbation_type)
    run = _run(tmp_path, [entry])

    result = run.results[0]
    assert result.error is None, result.error
    assert result.output_dir == tmp_path / "out" / "perturbations" / entry["name"]
    assert result.output_dir.is_dir()
    written = sorted(path.name for path in result.output_dir.iterdir())
    assert f"{entry['name']}_summary.csv" in written
    assert f"{entry['name']}_edges.csv" in written
    for path in result.outputs:
        assert path.exists(), f"{path} was reported but not written"
    # A perturbation is a number to compare, not a second published model.
    assert not list(result.output_dir.glob("*.vtp"))


def test_a_pressure_and_pericyte_sweep_writes_its_combined_csv(tmp_path):
    run = _run(tmp_path, [PRESSURE_AND_PERICYTE_SWEEP])

    written = {path.name for path in run.results[0].output_dir.iterdir()}
    assert "pericyte_dilation_pressure_sweep.csv" in written
    assert any(name.endswith(".png") for name in written), "no curves were drawn"


def test_a_pericyte_only_sweep_writes_its_dilation_csv(tmp_path):
    run = _run(tmp_path, [DILATION_SWEEP])

    written = {path.name for path in run.results[0].output_dir.iterdir()}
    assert "pericyte_dilation_sweep.csv" in written
    summary = run.results[0].summary
    # Fixed pressure: one pressure column value across the dilation axis.
    assert summary["sweep_points"] == 2  # min=1, max=2, step=1 from _settings


def test_a_pressure_only_sweep_writes_its_pressure_csv(tmp_path):
    run = _run(tmp_path, [PRESSURE_SWEEP])

    written = {path.name for path in run.results[0].output_dir.iterdir()}
    assert "inlet_pressure_sweep.csv" in written
    assert run.results[0].summary["sweep_points"] == 2  # 4500 and 5000


def test_an_arteriole_only_sweep_writes_its_dilation_csv(tmp_path):
    """Whole-branch arteriole percent sweep at fixed inlet pressure."""
    run = _run(tmp_path, [ARTERIOLE_DIAMETER_SWEEP])

    written = {path.name for path in run.results[0].output_dir.iterdir()}
    assert "arteriole_dilation_sweep.csv" in written
    # min=0, max=10, step=10 from _settings -> 0% and 10%
    assert run.results[0].summary["sweep_points"] == 2
    csv_text = (run.results[0].output_dir / "arteriole_dilation_sweep.csv").read_text(
        encoding="utf-8"
    )
    assert "dilation_percent" in csv_text
    # One fixed pressure column across the dilation axis.
    pressures = {
        line.split(",")[2]
        for line in csv_text.splitlines()[1:]
        if line.strip()
    }
    assert len(pressures) == 1


def test_a_pressure_and_arteriole_sweep_writes_its_combined_csv(tmp_path):
    """Arteriole percent and inlet pressure both vary."""
    run = _run(tmp_path, [PRESSURE_AND_ARTERIOLE_SWEEP])

    written = {path.name for path in run.results[0].output_dir.iterdir()}
    assert "arteriole_dilation_pressure_sweep.csv" in written
    # 2 dilations x 2 pressures
    assert run.results[0].summary["sweep_points"] == 4
    assert any(name.endswith(".png") for name in written), "no curves were drawn"


def test_a_capillary_only_sweep_writes_its_dilation_csv(tmp_path):
    """Passive whole-capillary percent sweep at fixed inlet pressure."""
    run = _run(tmp_path, [CAPILLARY_DIAMETER_SWEEP])

    written = {path.name for path in run.results[0].output_dir.iterdir()}
    assert "capillary_dilation_sweep.csv" in written
    assert run.results[0].summary["sweep_points"] == 2
    csv_text = (run.results[0].output_dir / "capillary_dilation_sweep.csv").read_text(
        encoding="utf-8"
    )
    assert "dilation_percent" in csv_text


def test_a_pressure_and_capillary_sweep_writes_its_combined_csv(tmp_path):
    """Capillary percent and inlet pressure both vary."""
    run = _run(tmp_path, [PRESSURE_AND_CAPILLARY_SWEEP])

    written = {path.name for path in run.results[0].output_dir.iterdir()}
    assert "capillary_dilation_pressure_sweep.csv" in written
    assert run.results[0].summary["sweep_points"] == 4


def test_the_summary_says_what_it_did_and_what_it_did_it_to(tmp_path):
    run = _run(tmp_path, [ARTERIOLE_DILATION])

    summary = (
        run.results[0].output_dir / "art_dilate_20_summary.csv"
    ).read_text(encoding="utf-8")
    header, row = summary.splitlines()[:2]
    assert "baseline_equivalent_resistance" in header
    assert "delta_vs_baseline" in header
    assert "art_dilate_20" in row
    assert "arteriole_diameter_change_percent" in row, "the overrides are not recorded"


def test_wider_arterioles_lower_the_networks_resistance(tmp_path):
    """The stage has to move the number it reports, not just write a file."""
    run = _run(tmp_path, [ARTERIOLE_DILATION, ARTERIOLE_CONSTRICTION])

    dilated, narrowed = run.results
    baseline = run.baseline["equivalent_resistance"]
    assert dilated.summary["equivalent_resistance"] < baseline
    assert narrowed.summary["equivalent_resistance"] > baseline


def test_combined_arteriole_and_pericyte_applies_both_mechanisms(tmp_path):
    """Whole-branch arteriole scale and focal pericyte sites both land.

    Art1 constriction factor is 1.0 so arteriole edges match art-only after
    scaling; B01 is 0.5 so capillaries match pericyte-only. The combined
    graph is therefore the composition of the two mechanisms, not either alone.
    """
    run = _run(tmp_path, [ARTERIOLE_AND_PERICYTE, ARTERIOLE_DILATION, PERICYTE_TONE])
    by_name = {result.name: result for result in run.results}
    combined = by_name[ARTERIOLE_AND_PERICYTE["name"]]
    art_only = by_name[ARTERIOLE_DILATION["name"]]
    peri_only = by_name[PERICYTE_TONE["name"]]

    assert combined.ok and art_only.ok and peri_only.ok
    assert combined.summary.get("strategy")
    assert combined.summary.get("arteriole_diameter_change_percent") == 20.0
    assert _resistances(combined.graph) != _resistances(art_only.graph)
    assert _resistances(combined.graph) != _resistances(peri_only.graph)

    for u, v, key, data in combined.graph.edges(keys=True, data=True):
        order = data["branch_order"]
        edge_key = (u, v, key)
        if order.startswith("Art"):
            assert data["resistance"] == pytest.approx(
                art_only.graph.edges[edge_key]["resistance"]
            )
        elif order.startswith("B"):
            assert data["resistance"] == pytest.approx(
                peri_only.graph.edges[edge_key]["resistance"]
            )


# --- independence ------------------------------------------------------------


def test_the_baseline_graph_is_unchanged_by_the_stage(tmp_path):
    """The guarantee everything else rests on.

    `export_results` runs after this stage and writes out the run's own graph;
    a perturbation that edited it in place would publish a perturbed network
    under the baseline's name.
    """
    model = _model()
    before = _resistances(model.graph)
    edges_before = frozenset(model.graph.edges(keys=True))
    geometry_before = _geometry(model.graph)
    attributes_before = _attribute_names(model.graph)

    run_perturbations(
        _settings(tmp_path, EVERY_TYPE_ONCE), model, _boundaries(), SCHEMA
    )

    assert _resistances(model.graph) == before
    assert frozenset(model.graph.edges(keys=True)) == edges_before
    _assert_geometry_unmoved(model.graph, geometry_before)
    # The re-solve writes pressures and flows onto whatever it solves; none of
    # them may appear here.
    assert _attribute_names(model.graph) == attributes_before


@pytest.mark.parametrize("perturbation_type", PERTURBATION_TYPES)
def test_no_perturbation_type_edits_the_baselines_geometry(
    tmp_path, perturbation_type
):
    """The invariant `_perturbation_copy` rests on, per type.

    A perturbation holds the baseline's own `voxels` lists and `pos` arrays --
    a shallow graph copy shares them deliberately, because duplicating the
    centrelines of a whole-brain network costs minutes and nothing writes to
    them. Anything that started mutating one in place instead of rebinding it
    would silently rewrite the network the run exports, and this is what would
    notice.

    Over every type the module declares, not over the ones that existed when
    this was written: a type added later shares the same lists and arrays, and
    would otherwise be the one type nothing here watches.
    """
    entry = _entry_for(perturbation_type)
    model = _model()
    before = _geometry(model.graph)

    run = run_perturbations(_settings(tmp_path, [entry]), model, _boundaries(), SCHEMA)

    assert run.results[0].error is None, run.results[0].error
    _assert_geometry_unmoved(model.graph, before)


# --- the copy a perturbation is given ----------------------------------------


def test_the_perturbation_copy_keeps_its_rebindings_to_itself():
    """New attribute dicts are what make a shallow copy safe to write on."""
    original = _network()
    original.graph["viscosity_law"] = "constant"
    copy = _perturbation_copy(original)
    node = next(iter(original.nodes))
    edge = next(iter(original.edges(keys=True)))

    copy.nodes[node]["pressure"] = 4500.0
    copy.edges[edge]["resistance"] = 1.0
    copy.edges[edge]["voxels"] = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    copy.graph["viscosity_law"] = "pries"

    assert "pressure" not in original.nodes[node]
    assert "resistance" not in original.edges[edge]
    assert len(original.edges[edge]["voxels"]) == 2
    assert original.edges[edge]["voxels"][1][2] == EDGE_LENGTH_UM
    assert original.graph["viscosity_law"] == "constant"


def test_the_perturbation_copy_shares_the_values_nothing_writes_to():
    """The saving: the centrelines are not duplicated, only pointed at.

    Stated as identity on purpose. If this ever has to become equality, the
    copy got deeper and the run got slower, and someone should have meant it.
    """
    original = _network()
    copy = _perturbation_copy(original)
    node = next(iter(original.nodes))
    edge = next(iter(original.edges(keys=True)))

    assert copy.nodes[node]["pos"] is original.nodes[node]["pos"]
    assert copy.edges[edge]["voxels"] is original.edges[edge]["voxels"]


def test_the_perturbation_copy_has_a_structure_of_its_own():
    """Shallow is about the attribute values, not about the nodes and edges."""
    original = _network()
    copy = _perturbation_copy(original)

    copy.add_edge("extra", "another", key=0)
    copy.remove_node(next(iter(original.nodes)))

    assert "extra" not in original
    assert original.number_of_edges() == len(BRANCH_ORDERS)
    assert original.number_of_nodes() == len(BRANCH_ORDERS) + 1


def test_a_perturbation_gets_its_own_graph_not_the_baselines(tmp_path):
    model = _model()
    run = run_perturbations(
        _settings(tmp_path, [ARTERIOLE_DILATION]), model, _boundaries(), SCHEMA
    )

    perturbed = run.results[0].graph
    assert perturbed is not model.graph
    changed = {
        edge
        for edge, resistance in _resistances(perturbed).items()
        if resistance != _resistances(model.graph)[edge]
    }
    assert changed, "the perturbation changed nothing"
    assert all(
        perturbed.edges[edge]["branch_order"].startswith("Art") for edge in changed
    ), "an arteriole dilation moved a capillary"


def test_two_perturbations_do_not_compose(tmp_path):
    """Each runs from the baseline, so the list order cannot change an answer."""
    alone = _run(tmp_path / "alone", [PERICYTE_TONE])
    together = _run(tmp_path / "together", [ARTERIOLE_DILATION, PERICYTE_TONE])

    solo = alone.results[0]
    beside_another = next(
        result for result in together.results if result.name == PERICYTE_TONE["name"]
    )
    assert _resistances(beside_another.graph) == _resistances(solo.graph)
    assert beside_another.summary["equivalent_resistance"] == pytest.approx(
        solo.summary["equivalent_resistance"]
    )


def test_two_pericyte_entries_keep_independent_constriction_knobs(tmp_path):
    """Length, spacing and probability are per entry, not shared run settings."""
    factors = {"Art1": 1.0, "B01": 0.5, "Ven1": 1.0}
    short_sparse = {
        "name": "short_sparse",
        "type": "pericyte_diameter_change",
        "overrides": {
            "do_pericyte_construction": True,
            "constriction_by_branch_order": factors,
            "constriction_length_um": 20.0,
            "constriction_spacing_um": 200.0,
            "use_probabilistic_pericyte_constriction": True,
            "pericyte_constriction_probability": 0.25,
            "pericyte_constriction_seed": 11,
        },
    }
    long_dense = {
        "name": "long_dense",
        "type": "pericyte_diameter_change",
        "overrides": {
            "do_pericyte_construction": True,
            "constriction_by_branch_order": factors,
            "constriction_length_um": 80.0,
            "constriction_spacing_um": 50.0,
            "use_probabilistic_pericyte_constriction": True,
            "pericyte_constriction_probability": 1.0,
            "pericyte_constriction_seed": 22,
        },
    }
    run = _run(tmp_path, [short_sparse, long_dense])

    by_name = {result.name: result for result in run.results}
    assert by_name["short_sparse"].ok and by_name["long_dense"].ok
    for name, entry in (
        ("short_sparse", short_sparse),
        ("long_dense", long_dense),
    ):
        overrides = by_name[name].summary["overrides"]
        for key in (
            "constriction_length_um",
            "constriction_spacing_um",
            "pericyte_constriction_probability",
        ):
            assert overrides[key] == entry["overrides"][key]
    assert _resistances(by_name["short_sparse"].graph) != _resistances(
        by_name["long_dense"].graph
    )


def test_two_pericyte_entries_with_different_branch_order_factors_differ(tmp_path):
    """constriction_by_branch_order is per entry: different tables, different R."""
    shared = {
        "do_pericyte_construction": True,
        "constriction_length_um": 40.0,
        "constriction_spacing_um": 100.0,
        "use_probabilistic_pericyte_constriction": False,
        "pericyte_constriction_probability": 1.0,
    }
    mild = {
        "name": "mild_tone",
        "type": "pericyte_diameter_change",
        "overrides": {
            **shared,
            "constriction_by_branch_order": {
                "Art1": 1.0,
                "B01": 0.8,
                "Ven1": 1.0,
            },
        },
    }
    tight = {
        "name": "tight_tone",
        "type": "pericyte_diameter_change",
        "overrides": {
            **shared,
            "constriction_by_branch_order": {
                "Art1": 1.0,
                "B01": 0.3,
                "Ven1": 1.0,
            },
        },
    }
    run = _run(tmp_path, [mild, tight])

    by_name = {result.name: result for result in run.results}
    assert by_name["mild_tone"].ok and by_name["tight_tone"].ok
    assert by_name["mild_tone"].summary["overrides"]["constriction_by_branch_order"] == (
        mild["overrides"]["constriction_by_branch_order"]
    )
    assert by_name["tight_tone"].summary["overrides"]["constriction_by_branch_order"] == (
        tight["overrides"]["constriction_by_branch_order"]
    )
    mild_r = _resistances(by_name["mild_tone"].graph)
    tight_r = _resistances(by_name["tight_tone"].graph)
    assert mild_r != tight_r
    # Capillary edges (B01) must be the ones that moved with the factor table.
    capillary = [
        edge
        for edge in mild_r
        if by_name["mild_tone"].graph.edges[edge]["branch_order"] == "B01"
    ]
    assert capillary
    assert any(mild_r[edge] != tight_r[edge] for edge in capillary)


def test_base_constriction_factor_with_branch_order_override(tmp_path):
    """Global 0.8 + {B01: 1.0} → only B01 differs from empty-map global."""
    shared = {
        "do_pericyte_construction": True,
        "pericyte_constriction_factor": 0.8,
        "constriction_length_um": 40.0,
        "constriction_spacing_um": 100.0,
        "use_probabilistic_pericyte_constriction": False,
        "pericyte_constriction_probability": 1.0,
    }
    with_override = {
        "name": "base_with_b01_open",
        "type": "pericyte_diameter_change",
        "overrides": {**shared, "constriction_by_branch_order": {"B01": 1.0}},
    }
    empty_map = {
        "name": "global_only",
        "type": "pericyte_diameter_change",
        "overrides": {**shared, "constriction_by_branch_order": {}},
    }
    run = _run(tmp_path, [with_override, empty_map])
    by_name = {result.name: result for result in run.results}
    assert by_name["base_with_b01_open"].ok and by_name["global_only"].ok

    open_g = by_name["base_with_b01_open"].graph
    open_r = _resistances(open_g)
    global_r = _resistances(by_name["global_only"].graph)

    for u, v, k, data in open_g.edges(keys=True, data=True):
        edge = (u, v, k)
        if data["branch_order"] == "B01":
            assert open_r[edge] < global_r[edge]
        else:
            assert open_r[edge] == pytest.approx(global_r[edge])


def test_empty_constriction_map_matches_omitted_map(tmp_path):
    """Empty override map is pure global behaviour — same as leaving it unset.

    The run's baseline map must itself be empty; otherwise omitting the key
    inherits a non-empty baseline table, which is a different question.
    """
    shared = {
        "do_pericyte_construction": True,
        "pericyte_constriction_factor": 0.8,
        "constriction_length_um": 40.0,
        "constriction_spacing_um": 100.0,
        "use_probabilistic_pericyte_constriction": False,
        "pericyte_constriction_probability": 1.0,
    }
    empty = {
        "name": "empty_map",
        "type": "pericyte_diameter_change",
        "overrides": {**shared, "constriction_by_branch_order": {}},
    }
    omitted = {
        "name": "omitted_map",
        "type": "pericyte_diameter_change",
        "overrides": dict(shared),
    }
    run = _run(
        tmp_path,
        [empty, omitted],
        constriction_by_branch_order={},
        pericyte_constriction_factor=0.8,
    )
    by_name = {result.name: result for result in run.results}
    assert by_name["empty_map"].ok and by_name["omitted_map"].ok
    assert _resistances(by_name["empty_map"].graph) == _resistances(
        by_name["omitted_map"].graph
    )


def test_a_pericyte_sweep_honours_entry_length_and_spacing(tmp_path):
    """Sweep geometry knobs merge into the settings the existing helper reads."""
    short = {
        "name": "short_sites",
        "type": "pericyte_dilation_sweep",
        "overrides": {
            "constriction_length_um": 10.0,
            "constriction_spacing_um": 200.0,
            "use_probabilistic_pericyte_constriction": True,
            "pericyte_constriction_probability": 0.3,
        },
    }
    long = {
        "name": "long_sites",
        "type": "pericyte_dilation_sweep",
        "overrides": {
            "constriction_length_um": 80.0,
            "constriction_spacing_um": 50.0,
            "use_probabilistic_pericyte_constriction": True,
            "pericyte_constriction_probability": 0.9,
        },
    }
    run = _run(tmp_path, [short, long])

    by_name = {result.name: result for result in run.results}
    assert by_name["short_sites"].ok and by_name["long_sites"].ok
    for name, entry in (("short_sites", short), ("long_sites", long)):
        overrides = by_name[name].summary["overrides"]
        for key, value in entry["overrides"].items():
            assert overrides[key] == value
    # Entries stay independent: one entry's knobs do not leak into the other.
    assert (
        by_name["short_sites"].summary["overrides"]["constriction_length_um"]
        != by_name["long_sites"].summary["overrides"]["constriction_length_um"]
    )


def test_the_order_they_are_listed_in_does_not_matter(tmp_path):
    forwards = _run(tmp_path / "forwards", [ARTERIOLE_DILATION, PERICYTE_TONE])
    backwards = _run(tmp_path / "backwards", [PERICYTE_TONE, ARTERIOLE_DILATION])

    by_name = {result.name: result for result in backwards.results}
    for result in forwards.results:
        assert _resistances(result.graph) == _resistances(by_name[result.name].graph)


# --- one bad entry does not lose the run -------------------------------------


def test_a_failing_perturbation_is_reported_and_the_others_still_run(tmp_path):
    """An hour of graph building must not be lost to one bad entry."""
    broken = {
        "name": "broken",
        "type": "pericyte_diameter_change",
        # Asking for the mask strategy without a mask: it raises when it runs.
        "overrides": {
            "do_pericyte_construction": True,
            "use_pericyte_mask_constriction": True,
        },
    }
    run = _run(tmp_path, [broken, ARTERIOLE_DILATION])

    failed, succeeded = run.results
    assert failed.name == "broken"
    assert not failed.ok
    assert "pericyte_mask_path" in failed.error
    assert failed.graph is None
    assert succeeded.ok and succeeded.graph is not None
    assert [result.name for result in run.failures] == ["broken"]
    assert [result.name for result in run.solved] == ["art_dilate_20"]


# --- the blood model a perturbation may not change ---------------------------


@pytest.mark.parametrize(
    "name,value",
    (
        ("viscosity_law", "pries"),
        ("diameter_basis", "anatomical"),
        ("haematocrit", 0.3),
    ),
)
def test_a_perturbation_may_not_change_the_blood_model(tmp_path, name, value):
    """Its resistances would not be comparable with the baseline's.

    Switching the viscosity law roughly doubles a capillary's resistance, so a
    perturbation that changed one would report that as its own effect. Refused
    rather than warned about: the CSV outlives the log.
    """
    run = _run(
        tmp_path,
        [
            {
                "name": "sneaky",
                "type": "arteriole_diameter_change",
                "overrides": {"arteriole_diameter_change_percent": 20, name: value},
            },
            ARTERIOLE_DILATION,
        ],
    )

    refused, allowed = run.results
    assert not refused.ok
    assert name in refused.error
    assert refused.output_dir is None or not list(refused.output_dir.iterdir())
    assert allowed.ok, "the refusal stopped the entries that were fine"


def test_an_override_a_type_does_not_read_is_not_applied(tmp_path):
    """`unused_overrides` calls it unread; the run has to agree.

    Every perturbation re-solves, so an `inlet_p_bc` riding along in the
    settings would change the answer while the checks reported it as having no
    effect.
    """
    plain = _run(tmp_path / "plain", [ARTERIOLE_DILATION])
    with_extra = _run(
        tmp_path / "extra",
        [
            {
                **ARTERIOLE_DILATION,
                "overrides": {**ARTERIOLE_DILATION["overrides"], "inlet_p_bc": 9000.0},
            }
        ],
    )

    assert with_extra.results[0].summary["equivalent_resistance"] == pytest.approx(
        plain.results[0].summary["equivalent_resistance"]
    )


# --- progress ----------------------------------------------------------------


def test_one_step_is_reported_per_entry(tmp_path):
    events: list[ProgressEvent] = []
    run = RunProgress(events.append)

    with run.stage("run_perturbations") as reporter:
        run_perturbations(
            _settings(tmp_path, EVERY_TYPE_ONCE),
            _model(),
            _boundaries(),
            SCHEMA,
            progress=reporter,
        )

    steps = [event for event in events if event.kind == STEP]
    # One per entry, whatever its type: a `none` entry is still a line the
    # panel counts through.
    assert [event.step for event in steps] == [
        entry["name"] for entry in EVERY_TYPE_ONCE
    ]
    assert [event.step_index for event in steps] == list(range(len(EVERY_TYPE_ONCE)))
    assert {event.step_total for event in steps} == {len(EVERY_TYPE_ONCE)}
    assert {event.stage for event in steps} == {"run_perturbations"}
