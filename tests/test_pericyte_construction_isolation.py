"""``do_pericyte_construction`` is inert outside typed pericyte paths.

Baseline Haemodynamics and non-pericyte perturbations must stay on uniform
Poiseuille even when the global flag is True. Only pericyte-typed entries
place focal constrictions, via ``set_resistances_for_constriction_strategy``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haemolynx.haemodynamics import PoiseuilleModel  # noqa: E402
from haemolynx.haemodynamics.apply import (  # noqa: E402
    HaemodynamicsApplyConfig,
    apply_poiseuille_haemodynamics,
)
from haemolynx.pipeline import (  # noqa: E402
    BoundaryNodes,
    HaemodynamicModel,
    default_schema,
    resolve_settings,
    run_perturbations,
)
from haemolynx.pipeline.stages import (  # noqa: E402
    SkeletonisedVolume,
    VesselNetwork,
    assign_diameters,
)

SCHEMA = default_schema()

EDGE_LENGTH_UM = 400.0
DIAMETERS = {"Art1": 20.0, "B01": 6.0, "Ven1": 20.0}
BRANCH_ORDERS = ("Art1", "B01", "B01", "Ven1")
CONSTRICTION = {"Art1": 1.0, "B01": 0.5, "Ven1": 1.0}


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


def _vessel_network(tmp_path: Path, graph: nx.MultiGraph | None = None) -> VesselNetwork:
    image = np.zeros((8, 8, 8), dtype=np.uint8)
    volume = SkeletonisedVolume(
        image=image,
        skeleton=image.astype(bool),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=tmp_path / "out",
    )
    return VesselNetwork(graph=graph if graph is not None else _network(), volume=volume)


def _boundaries() -> BoundaryNodes:
    last = len(BRANCH_ORDERS)
    return BoundaryNodes(
        inlet_nodes=[0],
        outlet_nodes=[last],
        arteriole_boundary_nodes=[0],
        venule_boundary_nodes=[last],
        resistance_node_pair=(0, last),
    )


def _settings(tmp_path: Path, **extra) -> dict:
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    values = {setting.name: setting.default for setting in SCHEMA}
    values.update(
        {
            "input_path": tmp_path / "input.tif",
            "vtk_output_prefix": tmp_path / "out" / "run",
            "plot_dir": plot_dir,
            "run_haemodynamics": True,
            "inlet_nodes": [0],
            "outlet_nodes": [len(BRANCH_ORDERS)],
            "arteriole_boundary_nodes": [0],
            "venule_boundary_nodes": [len(BRANCH_ORDERS)],
            "use_fwhm_edge_diameters": False,
            "viscosity_law": "constant",
            "do_pericyte_construction": True,
            "run_pericyte_resistance_comparison": True,
            "strict_branch_order_assignment": False,
            "automated_vessel_assignment": False,
            "use_small_vessel_masks_for_boundary_assignment": False,
        }
    )
    values.update(extra)
    return resolve_settings(values, schema=SCHEMA, config_path=None)


def _resistances(graph: nx.MultiGraph) -> dict[tuple, float]:
    return {
        (u, v, key): float(data["resistance"])
        for u, v, key, data in graph.edges(keys=True, data=True)
    }


def _has_focal_site_attrs(graph: nx.MultiGraph) -> bool:
    """Mask/probabilistic paths write these; the periodic integral path may not."""
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        if "pericyte_count_assigned" in data or "constriction_sites" in data:
            return True
        if "pericyte_centers_um" in data:
            return True
    return False


def _uniform_baseline_model() -> HaemodynamicModel:
    graph, _results = PoiseuilleModel(
        constriction_length=40.0,
        constriction_spacing=100.0,
        viscosity_law="constant",
    ).set_poiseuille_resistances(_network(), dict(DIAMETERS))
    return HaemodynamicModel(graph=graph, results={})


def test_assign_diameters_ignores_do_pericyte_construction(tmp_path):
    """Baseline with the flag True matches uniform Poiseuille (flag False)."""
    with_flag = assign_diameters(
        _settings(tmp_path / "on", do_pericyte_construction=True),
        _vessel_network(tmp_path / "on"),
        _boundaries(),
        SCHEMA,
    )
    without_flag = assign_diameters(
        _settings(tmp_path / "off", do_pericyte_construction=False),
        _vessel_network(tmp_path / "off"),
        _boundaries(),
        SCHEMA,
    )

    assert _resistances(with_flag.graph) == pytest.approx(
        _resistances(without_flag.graph)
    )
    assert not _has_focal_site_attrs(with_flag.graph)
    assert "pericyte_comparison" not in with_flag.results
    assert "poiseuille" in with_flag.results.get("resistances", {})


def test_library_apply_still_honours_the_flag_when_asked():
    """``apply_poiseuille_haemodynamics`` keeps the library opt-in path."""
    on, summary_on = apply_poiseuille_haemodynamics(
        _network(),
        config=HaemodynamicsApplyConfig(
            diameters={
                "diameter_by_branch_order": dict(DIAMETERS),
                "constriction_by_branch_order": dict(CONSTRICTION),
                "do_pericyte_construction": True,
                "use_pericyte_mask_constriction": False,
                "use_probabilistic_pericyte_constriction": False,
                "viscosity_law": "constant",
            }
        ),
    )
    off, summary_off = apply_poiseuille_haemodynamics(
        _network(),
        config=HaemodynamicsApplyConfig(
            diameters={
                "diameter_by_branch_order": dict(DIAMETERS),
                "constriction_by_branch_order": dict(CONSTRICTION),
                "do_pericyte_construction": False,
                "viscosity_law": "constant",
            }
        ),
    )
    assert _resistances(on) != _resistances(off)
    assert "constrictions" in summary_on.get("resistances", {})
    assert "poiseuille" in summary_off.get("resistances", {})


def test_arteriole_perturbation_ignores_global_pericyte_flag(tmp_path):
    """A non-pericyte perturbation must not inherit top-level pericyte tone."""
    with_flag = run_perturbations(
        _settings(
            tmp_path / "on",
            do_pericyte_construction=True,
            run_perturbations=True,
            perturbations=[
                {
                    "name": "art_dilate",
                    "type": "arteriole_diameter_change",
                    "overrides": {"arteriole_diameter_change_percent": 20},
                }
            ],
        ),
        _uniform_baseline_model(),
        _boundaries(),
        SCHEMA,
    )
    without_flag = run_perturbations(
        _settings(
            tmp_path / "off",
            do_pericyte_construction=False,
            run_perturbations=True,
            perturbations=[
                {
                    "name": "art_dilate",
                    "type": "arteriole_diameter_change",
                    "overrides": {"arteriole_diameter_change_percent": 20},
                }
            ],
        ),
        _uniform_baseline_model(),
        _boundaries(),
        SCHEMA,
    )

    assert with_flag.results[0].ok and without_flag.results[0].ok
    assert _resistances(with_flag.results[0].graph) == _resistances(
        without_flag.results[0].graph
    )
    assert not _has_focal_site_attrs(with_flag.results[0].graph)


def test_pericyte_perturbation_still_applies_constrictions(tmp_path):
    """Typed pericyte entries place constrictions via their strategy path."""
    baseline = _uniform_baseline_model()
    run = run_perturbations(
        _settings(
            tmp_path,
            do_pericyte_construction=False,
            run_perturbations=True,
            perturbations=[
                {
                    "name": "tone",
                    "type": "pericyte_diameter_change",
                    "overrides": {
                        "constriction_by_branch_order": dict(CONSTRICTION),
                    },
                }
            ],
        ),
        baseline,
        _boundaries(),
        SCHEMA,
    )

    assert run.results[0].ok
    assert run.results[0].summary.get("strategy")
    assert _resistances(run.results[0].graph) != _resistances(baseline.graph)
