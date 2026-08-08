"""Which settings belong to which pipeline stage.

The panel shows one tab per stage, in the order
``examples/resistance_network_pipeline.py`` runs them, so a user configures a
run the way the pipeline executes it rather than the way the config file is
laid out.

The schema groups settings by *section*, which is close but not the same: a
section can span stages (`Pipeline stages` holds the skeleton thresholds, the
graph thresholds and the output prefix), and a stage can span sections
(`assign_boundaries` reads the boundary section plus toggles from elsewhere).
So the split is declared here rather than derived -- but it was *built* from
which stage function actually reads which setting, and
:mod:`tests.test_gui_tabs` fails if a setting reaches no tab or more than one.

Claims are resolved in two passes: named settings first, in stage order, then
whole sections. A stage can therefore take one setting out of a section another
stage owns, which is what `base_plot_dir` (declared under boundary assignment,
used for output) needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from haemolynx.gui.form import Field, fields_for
from haemolynx.parsers.schema import Schema


@dataclass(frozen=True)
class Stage:
    """One tab: a pipeline stage, and the settings that steer it."""

    #: The stage function in `haemolynx.pipeline`, or None for a tab that is
    #: not a stage.
    call: str | None
    title: str
    summary: str
    #: Settings claimed by name, before any section is claimed.
    settings: tuple[str, ...] = ()
    #: Whole schema sections claimed after every named claim is settled.
    sections: tuple[str, ...] = ()


#: One tab per stage, in the order the pipeline runs them.
STAGES: tuple[Stage, ...] = (
    Stage(
        call="segment",
        title="1. Input",
        summary="Which image to analyse, and whether ilastik segments it first.",
        sections=("Input and segmentation",),
    ),
    Stage(
        call="skeletonise",
        title="2. Skeletonise",
        summary="Load the volume, resolve its voxel size, reduce vessels to a skeleton.",
        settings=(
            "do_skeletonize",
            "skeleton_closing_radius",
            "skeleton_bridge_gap_size",
            "skeleton_min_branch_length",
            "skeleton_max_bridge_distance",
            "skeleton_component_connectivity",
            "skeleton_min_component_percent",
        ),
    ),
    Stage(
        call="build_network",
        title="3. Graph",
        summary="Turn the skeleton into a graph and repair its topology.",
        settings=(
            "do_graph_building",
            "graph_reconnect_threshold",
            "final_orphan_reconnect_threshold",
            "cluster_collapse_distance",
            "min_stub_length",
        ),
        sections=("Vessel masks",),
    ),
    Stage(
        call="assign_boundaries",
        title="4. Boundaries",
        summary="Where flow enters and leaves, and where vessel types change.",
        sections=("Boundary assignment",),
    ),
    Stage(
        call="assign_diameters",
        title="5. Diameters",
        summary="Branch orders, then the diameter each vessel is modelled with.",
        settings=(
            "use_fwhm_edge_diameters",
            "strict_branch_order_assignment",
            "max_branch_order",
            "all_diams_const",
            "default_diameter",
            "manual_capillary_diameter_by_branch_order",
            "manual_arteriole_diameter_by_branch_order",
            "manual_venule_diameter_by_branch_order",
            "diameter_by_branch_order",
            "custom_edges",
        ),
        sections=("FWHM diameter measurement",),
    ),
    Stage(
        call="build_haemodynamic_model",
        title="6. Resistances",
        summary="Poiseuille resistance per vessel, and any pericyte constriction.",
        settings=("run_haemodynamics", "constriction_by_branch_order"),
        sections=("Diameters and pericytes",),
    ),
    Stage(
        call="solve",
        title="7. Solve",
        summary="Pressures and flows, from the boundary pressures.",
        settings=("input_p_bc", "output_p_bc", "do_equiv_resistance_calculation"),
    ),
    Stage(
        call="export_results",
        title="8. Export",
        summary="VTK, statistics and plots.",
        settings=("vtk_output_prefix", "base_plot_dir", "verbose_logging"),
        sections=("Solver and output", "Statistics and measurements"),
    ),
)


def assign_to_stages(schema: Schema) -> dict[str, str]:
    """Setting name -> the title of the tab it belongs on.

    Named claims win over section claims, so a stage can take one setting out
    of a section another stage owns.
    """
    owner: dict[str, str] = {}
    for stage in STAGES:
        for name in stage.settings:
            if name in schema:
                owner.setdefault(name, stage.title)
    for stage in STAGES:
        for section in stage.sections:
            for setting in schema:
                if setting.section == section:
                    owner.setdefault(setting.name, stage.title)
    return owner


def unassigned(schema: Schema) -> list[str]:
    """Settings no tab claims. Must be empty: a new setting needs a home."""
    owner = assign_to_stages(schema)
    return sorted(name for name in schema.names if name not in owner)


@dataclass
class Tab:
    """A stage and the form rows shown on its tab."""

    stage: Stage
    fields: list[Field] = field(default_factory=list)


def tabs_for(schema: Schema, values=None) -> list[Tab]:
    """The panel's tabs, each carrying its own rows, in pipeline order."""
    owner = assign_to_stages(schema)
    by_title = {stage.title: Tab(stage=stage) for stage in STAGES}
    for row in fields_for(schema, values):
        title = owner.get(row.name)
        if title is not None:
            by_title[title].fields.append(row)
    return list(by_title.values())
