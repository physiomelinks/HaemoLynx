"""One declared name for the constriction geometry, read by run and sweep alike.

`constriction_length_um` and `constriction_spacing_um` are declared in the
`Diameters and pericytes` section (their panel rows live on Perturbations).
`pericyte_sweep.py` read them under those names; `apply.py` -- which is what
the main run goes through -- read `constriction_length` and
`constriction_spacing`, names no schema has, so they always fell through to
`DIAMETER_DEFAULTS`. A config setting the declared name therefore moved the
dilation sweep's constrictions and silently not the run's, and the two halves
of one run disagreed about how long a constriction is.

These tests pin the property that was broken: the declared setting has to reach
the resistances the main run produces.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src",):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from haemolynx.haemodynamics.apply import (  # noqa: E402
    DIAMETER_DEFAULTS,
    HaemodynamicsApplyConfig,
    apply_poiseuille_haemodynamics,
)
from haemolynx.pipeline import default_schema  # noqa: E402

APPLY_SOURCE = REPO_ROOT / "src" / "haemolynx" / "haemodynamics" / "apply.py"
SWEEP_SOURCE = REPO_ROOT / "src" / "haemolynx" / "haemodynamics" / "pericyte_sweep.py"

#: Long enough to hold several constrictions at the default 100 um spacing, so
#: the geometry settings have somewhere to show up.
EDGE_LENGTH_UM = 600.0

GEOMETRY_SETTINGS = ("constriction_length_um", "constriction_spacing_um")


def _chain_graph(num_edges: int = 3) -> nx.MultiGraph:
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
            voxels=[
                [0.0, 0.0, node * EDGE_LENGTH_UM],
                [0.0, 0.0, (node + 1) * EDGE_LENGTH_UM],
            ],
        )
    return graph


def _run(**geometry: float) -> list[float]:
    """Resistances the main run produces for one constriction geometry."""
    config = HaemodynamicsApplyConfig(
        diameters={
            "diameter_by_branch_order": {"B01": 6.0},
            "constriction_by_branch_order": {"B01": 0.6},
            "do_pericyte_construction": True,
            "use_pericyte_mask_constriction": False,
            "use_probabilistic_pericyte_constriction": False,
            "viscosity_law": "constant",
            **geometry,
        },
        fwhm={},
    )
    graph, _summary = apply_poiseuille_haemodynamics(_chain_graph(), config=config)
    return [float(data["resistance"]) for _u, _v, data in graph.edges(data=True)]


# --- the property that was broken -------------------------------------------


def test_the_declared_constriction_length_changes_the_runs_resistances():
    """The main run, not the sweep: the setting has to move these numbers."""
    default_length = float(DIAMETER_DEFAULTS["constriction_length_um"])
    shorter = _run(constriction_length_um=default_length / 5.0)
    longer = _run(constriction_length_um=default_length)

    assert len(shorter) == len(longer) == 3
    # A constriction held over a longer stretch of vessel resists more.
    for narrow, wide in zip(shorter, longer):
        assert wide > narrow


def test_the_declared_constriction_spacing_changes_the_runs_resistances():
    default_spacing = float(DIAMETER_DEFAULTS["constriction_spacing_um"])
    sparse = _run(constriction_spacing_um=default_spacing)
    dense = _run(constriction_spacing_um=default_spacing / 4.0)

    # Four times as many pericytes on the same vessel resist more.
    for few, many in zip(sparse, dense):
        assert many > few


def test_omitting_them_is_the_documented_default():
    """A caller passing a bare dict still gets the numbers it used to."""
    omitted = _run()
    spelled_out = _run(
        constriction_length_um=float(DIAMETER_DEFAULTS["constriction_length_um"]),
        constriction_spacing_um=float(DIAMETER_DEFAULTS["constriction_spacing_um"]),
    )
    assert omitted == pytest.approx(spelled_out)


# --- one name, not two -------------------------------------------------------


@pytest.mark.parametrize("name", GEOMETRY_SETTINGS)
def test_the_geometry_is_declared_in_the_schema(name: str):
    schema = default_schema()
    assert name in schema.names
    assert name in schema.section_names("Diameters and pericytes")


@pytest.mark.parametrize("name", GEOMETRY_SETTINGS)
def test_the_unsuffixed_name_is_gone_from_the_defaults(name: str):
    """Both spellings present is the bug: one of them would go unread."""
    assert name in DIAMETER_DEFAULTS
    assert name[: -len("_um")] not in DIAMETER_DEFAULTS


@pytest.mark.parametrize("source", (APPLY_SOURCE, SWEEP_SOURCE))
def test_no_settings_lookup_uses_the_unsuffixed_name(source: Path):
    """A settings key is a string literal here; the model's argument is not.

    `PoiseuilleModel(constriction_length=...)` keeps its own parameter name --
    it takes a number, not a settings group -- so only the quoted spellings are
    lookups, and only those have to agree with the schema.
    """
    text = source.read_text(encoding="utf-8")
    for name in GEOMETRY_SETTINGS:
        unsuffixed = name[: -len("_um")]
        assert f'"{unsuffixed}"' not in text, (
            f"{source.name} looks up '{unsuffixed}', which no schema declares; "
            f"it would always read the default instead of the user's {name}."
        )
        assert f'"{name}"' in text, f"{source.name} no longer reads {name}"
