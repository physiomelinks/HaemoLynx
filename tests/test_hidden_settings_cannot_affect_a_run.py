"""A hidden/greyed GUI setting must never change what a real run produces.

The GUI hides or greys a setting's row exactly when its schema ``requires``
prerequisite is unmet (see ``gui/form.py``'s ``HIDE_WHEN_UNMET_SECTIONS`` /
``is_visible``, and ``gui/boundary_picking.py``'s ``role_manual_controls_enabled``
for the boundary-role tabs). ``tests/test_pipeline_schema_api.py`` already
proves, for 100+ such settings, that ``Schema.ineffective_settings`` flags a
non-default value while its prerequisite is off -- but that is a static,
declarative check on the schema's own bookkeeping. It says the schema *thinks*
the value is unused; it does not run a single line of pipeline code, so a stage
function that reads a gated setting without independently re-checking its own
prerequisite -- a typo'd condition, a value read above the guard that is meant
to protect it, a setting whose ``requires`` no longer matches what the code
actually checks -- would sail through that check unnoticed and only show up as
a silently wrong run.

This file closes that gap behaviourally: for every gated setting, it finds
which pipeline stage owns it (``pipeline.progress.STAGES``, the same table the
GUI's tabs are built from) and actually runs that one stage twice -- once with
the setting at its schema default, once at a different ("probe") value --
both times with the setting's own first prerequisite forced unmet, on a tiny
synthetic input built fresh for each call. If the stage's real output differs
between the two runs, the setting reached the run despite being hidden, and
the test fails naming exactly which setting and which stage.

Covers ``segment``, ``skeletonise``, ``build_network``, ``assign_boundaries``,
``assign_diameters`` and ``solve`` by comparing their returned dataclass
structurally (graphs by topology + every node/edge attribute, arrays by
value); ``export_results`` separately, by comparing the files it writes,
since its return value is always the same ``Solution`` object handed in,
unchanged, so comparing it proves nothing. ``build_haemodynamic_model`` owns
no gated setting of its own and is only used here to build inputs for
``solve``. A minority of settings are skipped with a documented reason: no
generic probe value can be built for a mapping/list-kind setting; a
prerequisite that can only be unmet by requiring real ilastik infrastructure
or a real, quality-checked automated-vessel-assignment mask setup this
harness does not build; or ownership by ``run_perturbations`` (the pericyte
constriction/comparison family), which needs a fuller perturbation-run setup
not built here either. Two floor assertions guard against either skip list
silently growing.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
import pytest
import tifffile

from haemolynx.pipeline import (
    BoundaryNodes,
    SkeletonisedVolume,
    VesselNetwork,
    default_schema,
    resolve_settings,
)
from haemolynx.pipeline.progress import STAGES
from haemolynx.pipeline.stages import (
    SegmentedInputs,
    assign_boundaries,
    assign_diameters,
    build_haemodynamic_model,
    build_network,
    export_results,
    segment,
    skeletonise,
    solve,
)

from test_diameter_assignment import _network as _diameter_line_graph  # noqa: E402
from test_pipeline_schema_api import (  # noqa: E402
    _first_unmet_prerequisite,
    _non_default_probe_value,
)

SCHEMA = default_schema()


# --- a generic, structural "did the output actually change" comparator -----


def _stage_outputs_equal(a: Any, b: Any) -> bool:
    """Whether *a* and *b* -- two stage outputs, or parts of them -- match.

    Plain ``==`` breaks on a dataclass or dict holding a numpy array (the
    array's own ``==`` returns an array, not a bool) and treats two
    separately-built ``nx.Graph`` objects as unequal by identity regardless
    of content, so both are handled explicitly; everything else falls
    through to plain equality. Assumes the two runs are deterministic given
    identical settings, matching the byte-identical assertions already used
    for this codebase's other pipeline invariants -- no numeric tolerance.
    """
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a_arr, b_arr = np.asarray(a), np.asarray(b)
        return a_arr.shape == b_arr.shape and bool(
            np.array_equal(a_arr, b_arr, equal_nan=True)
        )
    if isinstance(a, nx.Graph) or isinstance(b, nx.Graph):
        if not (isinstance(a, nx.Graph) and isinstance(b, nx.Graph)):
            return False
        return _graphs_equal(a, b)
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
        if type(a) is not type(b):
            return False
        return all(
            _stage_outputs_equal(getattr(a, f.name), getattr(b, f.name))
            for f in dataclasses.fields(a)
        )
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_stage_outputs_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            _stage_outputs_equal(x, y) for x, y in zip(a, b)
        )
    return a == b


def _graphs_equal(g1: nx.Graph, g2: nx.Graph) -> bool:
    if type(g1) is not type(g2):
        return False
    if frozenset(g1.nodes) != frozenset(g2.nodes):
        return False
    if not _stage_outputs_equal(dict(g1.graph), dict(g2.graph)):
        return False
    for node in g1.nodes:
        if not _stage_outputs_equal(dict(g1.nodes[node]), dict(g2.nodes[node])):
            return False

    def _edge_items(g: nx.Graph) -> dict[Any, dict]:
        if g.is_multigraph():
            return {
                (frozenset((u, v)), key): data
                for u, v, key, data in g.edges(keys=True, data=True)
            }
        return {frozenset((u, v)): data for u, v, data in g.edges(data=True)}

    e1, e2 = _edge_items(g1), _edge_items(g2)
    return set(e1) == set(e2) and all(_stage_outputs_equal(e1[k], e2[k]) for k in e1)


# --- which stage owns a given gated setting ---------------------------------


def _setting_owner_stage(setting) -> str | None:
    """The ``Stage.call`` whose tab this setting appears on.

    Mirrors ``Stage``'s own claim order (its docstring): every stage's
    ``settings`` names are checked first, then every stage's whole-``section``
    claims -- the same table ``gui/tabs.py`` uses to place a setting on a tab.
    """
    for stage in STAGES:
        if setting.name in stage.settings:
            return stage.call
    for stage in STAGES:
        if setting.section in stage.sections:
            return stage.call
    return None


# --- minimal, fast, synthetic inputs for each stage -------------------------


def _base_values(tmp_dir: Path, **extra) -> dict:
    (tmp_dir / "out").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "plots").mkdir(parents=True, exist_ok=True)
    values = SCHEMA.defaults()
    values.update(
        {
            "input_path": tmp_dir / "input.tif",
            "vtk_output_prefix": tmp_dir / "out" / "run",
            "plot_dir": tmp_dir / "plots",
        }
    )
    values.update(extra)
    return values


def _settings_for(tmp_dir: Path, overrides: Mapping[str, Any], **extra) -> dict:
    values = _base_values(tmp_dir, **extra)
    values.update(overrides)
    return resolve_settings(values, schema=SCHEMA, config_path=None)


def _write_mask(path: Path, shape=(6, 6, 6)) -> Path:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[2:4, 2:4, :] = 255
    tifffile.imwrite(path, mask)
    return path


def _run_segment(overrides: Mapping[str, Any], tmp_dir: Path):
    mask_path = _write_mask(tmp_dir / "input.tif")
    settings = _settings_for(tmp_dir, overrides, input_path=mask_path)
    return segment(settings)


def _run_skeletonise(overrides: Mapping[str, Any], tmp_dir: Path):
    # `do_skeletonize=False` is a *resume* flag -- it loads the skeleton a
    # previous, real run already saved, rather than "skip this feature". A
    # setting gated on it needs that file to already exist, so it is
    # produced once, on demand, before the real call.
    if overrides.get("do_skeletonize") is False:
        skeleton_path = tmp_dir / "out" / "input_skeleton.npy"
        if not skeleton_path.exists():
            _run_skeletonise({"do_skeletonize": True}, tmp_dir)
    mask_path = _write_mask(tmp_dir / "input.tif")
    settings = _settings_for(tmp_dir, overrides, input_path=mask_path)
    inputs = SegmentedInputs(
        image_path=mask_path, output_dir=tmp_dir / "out", input_format="tif"
    )
    return skeletonise(settings, inputs)


def _tiny_skeleton() -> np.ndarray:
    """A short trunk with one side branch -- more than one edge to build."""
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[2:8, 5, 5] = True
    skeleton[5, 5, 2:8] = True
    return skeleton


def _run_build_network(overrides: Mapping[str, Any], tmp_dir: Path):
    # Same resume-flag shape as do_skeletonize above: do_graph_building=False
    # loads a previously-pickled graph instead of building one.
    if overrides.get("do_graph_building") is False:
        graph_path = tmp_dir / "out" / "input_graph.pkl"
        if not graph_path.exists():
            _run_build_network({"do_graph_building": True}, tmp_dir)
    skeleton = _tiny_skeleton()
    volume = SkeletonisedVolume(
        image=np.zeros(skeleton.shape, dtype=np.uint8),
        skeleton=skeleton,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=tmp_dir / "out",
    )
    settings = _settings_for(tmp_dir, overrides)
    return build_network(settings, volume, SCHEMA)


def _synthetic_network(tmp_dir: Path, graph: nx.MultiGraph | None = None) -> VesselNetwork:
    """A small, fast, real ``VesselNetwork`` every stage below shares.

    Defaults to the same pinned 4-edge line ``test_diameter_assignment.py``
    already uses for stage-level unit tests; the volume's image/skeleton
    content is never read by any of these stages beyond ``.shape``.
    """
    volume = SkeletonisedVolume(
        image=np.zeros((2, 2, 2), dtype=np.uint8),
        skeleton=np.zeros((2, 2, 2), dtype=bool),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=tmp_dir / "out",
    )
    return VesselNetwork(graph=graph if graph is not None else _diameter_line_graph(), volume=volume)


#: assign_boundaries' default inlet/outlet method (edge_percent) needs a
#: graph with an unambiguous single terminal at each end -- the diameter
#: fixture's straight line (pos varying only in z) puts both terminals at the
#: same percentile and edge_percent picks them both as "inlet", leaving no
#: outlet. This Y-shape (test_boundary_node_defaults.py's own fixture for
#: exactly this stage) resolves cleanly to one inlet, two outlets.
_BOUNDARY_IMAGE_SHAPE = (48, 48, 48)


def _boundary_test_network() -> nx.MultiGraph:
    positions = {
        0: (24.0, 8.0, 24.0),
        1: (24.0, 20.0, 24.0),
        2: (24.0, 32.0, 12.0),
        3: (24.0, 32.0, 36.0),
        4: (24.0, 41.0, 8.0),
        5: (24.0, 41.0, 40.0),
    }
    G = nx.MultiGraph()
    for node_id, position in positions.items():
        G.add_node(node_id, pos=np.asarray(position, dtype=float))
    for u, v in ((0, 1), (1, 2), (1, 3), (2, 4), (3, 5)):
        G.add_edge(u, v, length=1.0)
    return G


def _run_assign_boundaries(overrides: Mapping[str, Any], tmp_dir: Path):
    volume = SkeletonisedVolume(
        image=np.zeros(_BOUNDARY_IMAGE_SHAPE, dtype=np.uint8),
        skeleton=np.zeros(_BOUNDARY_IMAGE_SHAPE, dtype=bool),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=tmp_dir / "out",
    )
    network = VesselNetwork(graph=_boundary_test_network(), volume=volume)
    settings = _settings_for(tmp_dir, overrides)
    return assign_boundaries(settings, network)


#: The line graph's own node ids -- see test_diameter_assignment.BRANCH_ORDERS.
_LAST_NODE = 4


def _diameter_boundaries() -> BoundaryNodes:
    return BoundaryNodes(
        inlet_nodes=[0],
        outlet_nodes=[_LAST_NODE],
        arteriole_boundary_nodes=[0],
        venule_boundary_nodes=[_LAST_NODE],
        resistance_node_pair=(0, _LAST_NODE),
    )


_DIAMETER_BASE_OVERRIDES = {
    "run_haemodynamics": True,
    "inlet_nodes": [0],
    "outlet_nodes": [_LAST_NODE],
    "arteriole_boundary_nodes": [0],
    "venule_boundary_nodes": [_LAST_NODE],
    "use_fwhm_edge_diameters": False,
    "automated_vessel_assignment": False,
    "use_small_vessel_masks_for_boundary_assignment": False,
}


def _run_assign_diameters(overrides: Mapping[str, Any], tmp_dir: Path):
    network = _synthetic_network(tmp_dir)
    boundaries = _diameter_boundaries()
    settings = _settings_for(tmp_dir, overrides, **_DIAMETER_BASE_OVERRIDES)
    return assign_diameters(settings, network, boundaries, SCHEMA)


def _solve_inputs(tmp_dir: Path):
    network = _synthetic_network(tmp_dir)
    boundaries = _diameter_boundaries()
    settings = _settings_for(tmp_dir, {}, **_DIAMETER_BASE_OVERRIDES)
    diameters = assign_diameters(settings, network, boundaries, SCHEMA)
    model = build_haemodynamic_model(settings, diameters, SCHEMA)
    return model, boundaries


def _run_solve(overrides: Mapping[str, Any], tmp_dir: Path):
    model, boundaries = _solve_inputs(tmp_dir)
    settings = _settings_for(tmp_dir, overrides, **_DIAMETER_BASE_OVERRIDES)
    return solve(settings, model, boundaries)


#: One runner per stage this test can exercise behaviourally. A gated setting
#: owned by a stage not listed here -- export_results (handled separately
#: below via file-diffing) or run_perturbations (the pericyte/comparison
#: family; needs a fuller perturbation-run setup this harness does not build)
#: -- is skipped with a documented reason; the floor assertion below catches
#: that list growing silently.
STAGE_RUNNERS: dict[str, Callable[[Mapping[str, Any], Path], Any]] = {
    "segment": _run_segment,
    "skeletonise": _run_skeletonise,
    "build_network": _run_build_network,
    "assign_boundaries": _run_assign_boundaries,
    "assign_diameters": _run_assign_diameters,
    "solve": _run_solve,
}

#: Settings whose "prerequisite unmet" state requires real infrastructure or
#: a real, meaningful mask/classifier setup this generic synthetic-fixture
#: harness does not build -- constructing one just to satisfy these would be
#: disproportionate (a real automated-vessel-assignment run needs masks that
#: actually overlap the graph's terminals with plausible quality metrics, or
#: a real ilastik executable and classifier). The schema-level check in
#: test_pipeline_schema_api.py still covers every one of these; only the
#: behavioural, real-stage-execution check here cannot.
NEEDS_EXTERNAL_INFRASTRUCTURE = frozenset(
    {
        # requires=("!use_ilastik_segmentation",): unmet means running WITH
        # ilastik on, which needs a real executable + trained classifier.
        "input_path",
        # requires include "automated_vessel_assignment": unmet (for the two
        # "!automated_vessel_assignment" ones) or met (for the other) means
        # running WITH automated_vessel_assignment=True, which raises unless
        # given real, non-None large arteriole/venule masks with plausible
        # terminal-node overlap -- see assign_boundaries's own ValueError.
        "inlet_node_selection_method",
        "outlet_node_selection_method",
        "use_small_vessel_masks_for_boundary_assignment",
    }
)


def _stage_level_cases():
    cases: list[tuple[str, str, Any]] = []
    skipped: list[tuple[str, str]] = []
    for setting in SCHEMA:
        if not setting.requires:
            continue
        owner = _setting_owner_stage(setting)
        if owner == "export_results":
            continue  # handled by the separate file-diffing test below
        if owner not in STAGE_RUNNERS:
            skipped.append((setting.name, f"owner stage {owner!r} has no runner"))
            continue
        if setting.name in NEEDS_EXTERNAL_INFRASTRUCTURE:
            skipped.append((setting.name, "needs real infrastructure this harness lacks"))
            continue
        probe = _non_default_probe_value(setting)
        if probe is None:
            skipped.append((setting.name, "no generic probe value for its kind"))
            continue
        cases.append((setting.name, owner, probe))
    return cases, skipped


_STAGE_CASES, _STAGE_SKIPPED = _stage_level_cases()


@pytest.mark.slow
@pytest.mark.parametrize(
    "name, owner, probe", _STAGE_CASES, ids=[case[0] for case in _STAGE_CASES]
)
def test_a_hidden_setting_cannot_change_a_real_stage_run(name, owner, probe, tmp_path):
    setting = SCHEMA[name]
    unmet_name, unmet_value = _first_unmet_prerequisite(setting)
    runner = STAGE_RUNNERS[owner]

    default_output = runner(
        {name: setting.coerce(setting.default), unmet_name: unmet_value}, tmp_path
    )
    probe_output = runner(
        {name: setting.coerce(probe), unmet_name: unmet_value}, tmp_path
    )

    assert _stage_outputs_equal(default_output, probe_output), (
        f"{name!r} (owned by the {owner!r} stage) changed that stage's real "
        f"output even though its prerequisite {unmet_name!r}={unmet_value!r} "
        "is unmet -- a hidden/greyed GUI setting must never reach a real "
        "pipeline run."
    )


def test_the_stage_level_check_covers_most_gated_settings():
    """A floor, not the exact count -- see test_every_gated_setting_is_
    ineffective_when_its_prerequisite_is_off in test_pipeline_schema_api.py
    for the equivalent schema-level floor and why one is used instead of an
    exact count."""
    total = len(_STAGE_CASES) + len(_STAGE_SKIPPED)
    assert len(_STAGE_CASES) > 100, (
        f"only checked {len(_STAGE_CASES)} of {total} non-export-owned gated "
        f"settings behaviourally (skipped: {_STAGE_SKIPPED})"
    )


# --- export_results: compare written files, not its inert return value -----

#: export_results' own plotting/VTK paths are expensive and, for the 3D
#: Plotly HTML in particular, not byte-reproducible across two separate
#: output directories (see the docstring on _run_export_results) -- neither
#: is under test here, so both are switched off regardless of which setting
#: is being probed, isolating the file diff to whatever that setting can
#: actually reach.
_EXPORT_BASE_OVERRIDES = {
    **_DIAMETER_BASE_OVERRIDES,
    "visualize_results": False,
    "vtk_export": False,
}


def _export_results_inputs(tmp_dir: Path):
    network = _synthetic_network(tmp_dir)
    boundaries = _diameter_boundaries()
    settings = _settings_for(tmp_dir, {}, **_DIAMETER_BASE_OVERRIDES)
    diameters = assign_diameters(settings, network, boundaries, SCHEMA)
    model = build_haemodynamic_model(settings, diameters, SCHEMA)
    solution = solve(settings, model, boundaries)
    return network, model, solution


def _directory_signature(root: Path) -> dict[str, bytes]:
    """Every file under *root*, keyed by its path relative to *root*."""
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run_export_results(overrides: Mapping[str, Any], tmp_dir: Path) -> dict[str, bytes]:
    network, model, solution = _export_results_inputs(tmp_dir)
    out_dir = tmp_dir / "export"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    settings = _settings_for(
        tmp_dir,
        overrides,
        vtk_output_prefix=out_dir / "run",
        plot_dir=out_dir / "plots",
        **_EXPORT_BASE_OVERRIDES,
    )
    export_results(settings, network, model, solution)
    return _directory_signature(out_dir)


def _export_results_cases():
    cases: list[tuple[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    for setting in SCHEMA:
        if not setting.requires or _setting_owner_stage(setting) != "export_results":
            continue
        probe = _non_default_probe_value(setting)
        if probe is None:
            skipped.append((setting.name, "no generic probe value for its kind"))
            continue
        cases.append((setting.name, probe))
    return cases, skipped


_EXPORT_CASES, _EXPORT_SKIPPED = _export_results_cases()


@pytest.mark.slow
@pytest.mark.parametrize(
    "name, probe", _EXPORT_CASES, ids=[case[0] for case in _EXPORT_CASES]
)
def test_a_hidden_export_setting_cannot_change_written_output(name, probe, tmp_path):
    """``export_results`` always returns the same ``Solution`` object it was
    handed, unchanged -- see this module's docstring -- so the only place a
    setting's effect (or lack of one) is observable is the files it writes."""
    setting = SCHEMA[name]
    unmet_name, unmet_value = _first_unmet_prerequisite(setting)

    default_files = _run_export_results(
        {name: setting.coerce(setting.default), unmet_name: unmet_value}, tmp_path / "default"
    )
    probe_files = _run_export_results(
        {name: setting.coerce(probe), unmet_name: unmet_value}, tmp_path / "probe"
    )

    assert default_files == probe_files, (
        f"{name!r} changed export_results' written output even though its "
        f"prerequisite {unmet_name!r}={unmet_value!r} is unmet."
    )


def test_the_export_results_check_covers_every_settable_export_setting():
    total = len(_EXPORT_CASES) + len(_EXPORT_SKIPPED)
    assert len(_EXPORT_CASES) > 10, (
        f"only checked {len(_EXPORT_CASES)} of {total} export_results-owned "
        f"gated settings (skipped: {_EXPORT_SKIPPED})"
    )
