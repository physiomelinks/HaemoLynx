"""What a run is made of, and how it reports its way through it.

A run is a fixed sequence of stages -- :data:`STAGES` -- and everything that
watches one wants the same few facts: which stage started, how many there are,
and when each finished. So a caller hands :func:`run_pipeline_stages` a
``progress`` callback and gets one :class:`ProgressEvent` per boundary::

    run_pipeline_stages(settings, schema, progress=log_progress)   # console
    run_pipeline_stages(settings, schema, progress=events.append)  # a test
    run_pipeline_stages(settings, schema, progress=bar.update)     # a panel

The callback is a plain function taking one event, so what "showing progress"
means is the consumer's decision: nothing here imports napari, Qt or tqdm, and
nothing here writes to the console unless asked to (:func:`log_progress`).

Graph building reports a second, finer level -- the eleven topology steps of
:func:`haemolynx.graph.build_graph_from_skeleton` -- through the same callback,
so a consumer that only cares about stages can ignore ``kind == "step"``.

:data:`STAGES` lives here rather than in the panel because it is the pipeline's
own running order: the napari panel draws one tab per entry, and a progress bar
counts them. Keeping one list means a stage cannot be added to the run and
missed by everything that reports it.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Sequence

logger = logging.getLogger(__name__)

#: Pericyte / constriction knobs declared under Diameters and pericytes in the
#: schema (so apply.py's section_values still finds them) but claimed here so
#: their panel rows sit on Perturbations. Keep in sync with
#: ``PERICYTE_CONSTRICTION_SETTINGS`` in haemodynamics.perturbations -- the
#: drift test in test_gui_tabs pins that.
_PERICYTE_SETTINGS_ON_PERTURBATIONS_TAB: tuple[str, ...] = (
    "pericyte_constriction_factor",
    "constriction_by_branch_order",
    "constriction_length_um",
    "constriction_spacing_um",
    "use_pericyte_mask_constriction",
    "pericyte_mask_path",
    "pericyte_mask_h5_dataset_name",
    "pericyte_max_assignment_distance_um",
    "pericyte_min_diameter_um",
    "pericyte_max_diameter_um",
    "use_probabilistic_pericyte_constriction",
    "pericyte_constriction_probability",
    "pericyte_constriction_seed",
)

#: Legacy baseline flags and comparison-CSV knobs. Still in the Diameters
#: schema section for apply.py / CLI, but claimed here so they leave the
#: Diameters tab without becoming always-on or typed-entry rows. The pipeline
#: forces ``do_pericyte_construction`` and ``run_pericyte_resistance_comparison``
#: False on baseline and every perturbation merge.
_LEGACY_SETTINGS_HIDDEN_FROM_DIAMETERS: tuple[str, ...] = (
    "do_pericyte_construction",
    "run_pericyte_resistance_comparison",
    "pericyte_comparison_baseline_value",
    "pericyte_comparison_constricted_value",
    "reuse_comparison_pericyte_cohort_for_main_run",
)

#: Alias kept for older test imports; prefer
#: :data:`_LEGACY_SETTINGS_HIDDEN_FROM_DIAMETERS`.
_COMPARISON_SETTINGS_HIDDEN_FROM_DIAMETERS = _LEGACY_SETTINGS_HIDDEN_FROM_DIAMETERS


@dataclass(frozen=True)
class Stage:
    """One stage of a run: what runs it, and how to describe it to a user."""

    #: The stage function in :mod:`haemolynx.pipeline`, or None for something
    #: the panel shows that the pipeline does not run.
    call: Optional[str]
    title: str
    summary: str
    #: Settings this stage steers, claimed by name before any section claim.
    settings: tuple[str, ...] = ()
    #: Whole schema sections it steers, claimed after every named claim.
    sections: tuple[str, ...] = ()
    #: The tab this stage's settings appear on. None means a tab of its own,
    #: named `title`. A stage naming another stage's tab contributes its rows
    #: to that tab and opens no tab of its own.
    tab: Optional[str] = None


#: Every stage of a run, in the order :func:`run_pipeline_stages` runs them.
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
            "use_thick_vessel_skeletonisation",
            "skeleton_thick_vessel_min_radius_um",
            "skeleton_fill_mask_holes_before_thickness",
            "skeleton_thick_vessel_wall_absorption_um",
            "skeleton_thick_vessel_flake_filter_um",
            "skeleton_thick_vessel_max_bridge_radius_multiple",
            "skeleton_thick_vessel_bridge_radius_smoothing_um",
            # A read-only check on the tree this stage just built -- see
            # preprocessing.thick_vessel_braid_guard.
            "detect_thick_vessel_braiding",
            "thick_vessel_braid_factor_limit",
            "thick_vessel_braid_min_occupied_slices",
            "skeleton_closing_radius",
            "skeleton_bridge_gap_size",
            "skeleton_min_branch_length",
            "skeleton_max_bridge_distance",
            "skeleton_component_connectivity",
            "skeleton_min_component_percent",
            # A read-only check on the skeleton this stage just built -- see
            # preprocessing.skeleton_consistency.
            "skeleton_mask_consistency_warn_below",
            "missing_vessel_min_voxels",
            "skeleton_missing_vessel_warn_below",
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
            "cluster_collapse_method",
            "cluster_collapse_max_radial_dispersion",
            "cluster_collapse_persistence_search_multiple",
            "min_stub_length",
            "save_step_artifacts",
            # Centreline smoothing is the last thing graph building does.
            "smooth_centrelines",
            "centreline_smoothing_method",
            "centreline_smoothing_iterations",
            "centreline_max_deviation",
            # A read-only check on the graph this stage just built -- see
            # graph.cartwheel_guard.
            "detect_cartwheel_hub_artifacts",
            "cartwheel_hub_min_degree",
            "cartwheel_hub_max_radial_dispersion",
            "cartwheel_hub_tangent_length_um",
            # Read-only checks on the graph this stage just built -- see
            # graph.diagnostics.diagnose_skeleton_graph_consistency and
            # diagnose_graph_mask_consistency.
            "skeleton_graph_consistency_warn_below",
            "graph_mask_consistency_warn_below",
            "graph_missing_vessel_warn_below",
        ),
    ),
    Stage(
        call="assign_boundaries",
        title="4. Boundaries",
        summary="Where flow enters and leaves, and where vessel types change.",
        # Vessel masks stay in their own schema section so build_network's
        # section_values still finds them; the panel shows them here under
        # automated assignment, above the manual methods.
        sections=("Vessel masks", "Boundary assignment"),
    ),
    Stage(
        call="assign_diameters",
        title="5. Diameters",
        summary=(
            "Branch orders, the diameter each vessel is modelled with, and "
            "the blood viscosity law."
        ),
        # Declared under boundary assignment, but it is branch-order
        # assignment that reads it, which happens here.
        settings=("strict_branch_order_assignment",),
        # This is the stage that reads them: it hands the whole
        # `Diameters and pericytes` section to the haemodynamics as one group
        # (see `pipeline/stages.py`), so the settings and the tab agree.
        # Pericyte / constriction knobs live under Perturbation runs: they
        # configure a typed perturbation, not the baseline diameter model.
        sections=("Diameters and pericytes", "FWHM diameter measurement"),
    ),
    Stage(
        call="build_haemodynamic_model",
        title="6. Haemodynamics",
        summary="Whether to solve the flow, and the pressures to solve it at.",
        settings=("run_haemodynamics",),
    ),
    Stage(
        # Its rows belong beside the haemodynamics they configure, so this
        # stage opens no tab of its own; it is still a stage a run reports.
        call="solve",
        title="Solve",
        summary="Pressures and flows, from the boundary pressures.",
        settings=("inlet_p_bc", "outlet_p_bc", "do_equiv_resistance_calculation"),
        tab="6. Haemodynamics",
    ),
    Stage(
        call="run_perturbations",
        title="7. Perturbations",
        summary="What to re-solve the finished network for.",
        # Pericyte / constriction knobs stay in the Diameters schema section
        # (apply.py reads that group by name) but their *rows* belong here:
        # they are options of a typed perturbation, revealed only when one is
        # chosen. Legacy baseline / comparison flags are claimed here too so
        # they leave the Diameters tab without becoming always-on or typed
        # rows (the pipeline forces those flags False on baseline and merges).
        # Named claims beat the Diameters section claim below.
        settings=(
            *_PERICYTE_SETTINGS_ON_PERTURBATIONS_TAB,
            *_LEGACY_SETTINGS_HIDDEN_FROM_DIAMETERS,
        ),
        sections=("Perturbation runs",),
    ),
    Stage(
        call="export_results",
        title="8. Export",
        summary="VTK, statistics and plots.",
        settings=("vtk_output_prefix", "base_plot_dir", "verbose_logging"),
        sections=("Solver and output", "Statistics and measurements"),
    ),
)

#: A stage is about to run.
STAGE_STARTED = "stage_started"
#: A stage returned.
STAGE_FINISHED = "stage_finished"
#: A stage raised. The exception is re-raised, so a run reports this at most once.
STAGE_FAILED = "stage_failed"
#: A step *inside* a stage finished -- graph building's topology passes.
STEP = "step"

#: Every kind an event can be, for a consumer that wants to check.
KINDS: tuple[str, ...] = (STAGE_STARTED, STAGE_FINISHED, STAGE_FAILED, STEP)


@dataclass(frozen=True)
class ProgressEvent:
    """One point in a run at which there is something new to show.

    ``index`` and ``total`` describe the stage in every kind of event, so a bar
    across the whole run needs nothing else. ``step_index`` and ``step_total``
    are filled in only for ``kind == STEP``, and describe the step just done.
    """

    kind: str
    #: The stage function's name, e.g. ``"build_network"``.
    stage: str
    #: How to name that stage to a user, e.g. ``"3. Graph"``.
    title: str
    #: 0-based position of the stage in the run.
    index: int
    #: How many stages the run has.
    total: int
    #: For ``STEP``: the step's label, as `build_graph_from_skeleton` reports it.
    step: Optional[str] = None
    #: For ``STEP``: 0-based position of the step just finished.
    step_index: Optional[int] = None
    #: For ``STEP``: how many steps the stage has, when it knows.
    step_total: Optional[int] = None
    #: For ``STAGE_FAILED``: what was raised.
    error: Optional[BaseException] = None

    @property
    def completed(self) -> int:
        """How many stages are done: what a bar across the run should read."""
        return self.index + 1 if self.kind == STAGE_FINISHED else self.index


#: What a consumer supplies. One event in, nothing out.
ProgressCallback = Callable[[ProgressEvent], None]


class StageProgress:
    """The stage that is running, for whatever reports steps inside it.

    A stage that has no inner steps never touches this; graph building calls
    :meth:`step` once per topology pass, which is the only finer level of
    progress the pipeline has today.
    """

    def __init__(self, run: "RunProgress", stage: Stage, index: int) -> None:
        self._run = run
        self._stage = stage
        self._index = index
        self._done = 0
        self._total: Optional[int] = None

    @property
    def steps_done(self) -> int:
        """How many steps this stage has reported so far."""
        return self._done

    def step(self, label: str, total: Optional[int] = None) -> None:
        """Report that the step named *label* has just finished.

        *total* is how many steps the stage will report in all; it is passed on
        every call because the caller usually has the list to hand, and the
        most recent value is the one reported. Without it a consumer gets a
        step count but no end point, and should show an indeterminate bar.
        """
        if total is not None:
            self._total = total
        self._run.emit(
            ProgressEvent(
                kind=STEP,
                stage=self._stage.call or "",
                title=self._stage.title,
                index=self._index,
                total=self._run.total,
                step=label,
                step_index=self._done,
                step_total=self._total,
            )
        )
        self._done += 1


class RunProgress:
    """Turns a progress callback into the events a run reports.

    The stage list lives here rather than in each stage function, which is what
    makes ``index`` and ``total`` right without every stage having to count.
    With no callback every method is a no-op, so a run that nobody is watching
    does exactly what it did before.
    """

    def __init__(
        self,
        callback: Optional[ProgressCallback] = None,
        stages: Sequence[Stage] = STAGES,
    ) -> None:
        self._callback = callback
        #: Only stages the pipeline actually runs; a panel-only tab is not one.
        self._stages = tuple(stage for stage in stages if stage.call)

    @property
    def total(self) -> int:
        """How many stages this run has."""
        return len(self._stages)

    @property
    def stages(self) -> tuple[Stage, ...]:
        return self._stages

    def emit(self, event: ProgressEvent) -> None:
        """Hand *event* to the callback, if there is one."""
        if self._callback is not None:
            self._callback(event)

    def _locate(self, name: str) -> tuple[int, Stage]:
        for index, stage in enumerate(self._stages):
            if stage.call == name:
                return index, stage
        known = ", ".join(str(stage.call) for stage in self._stages)
        raise KeyError(f"{name!r} is not a pipeline stage. Known stages: {known}.")

    @contextmanager
    def stage(self, name: str) -> Iterator[StageProgress]:
        """Report *name* starting, then finishing -- or failing, and re-raise.

        The failure event is what lets a progress bar stop at the stage that
        broke rather than sit at the last one that worked.
        """
        index, stage = self._locate(name)
        reporter = StageProgress(self, stage, index)

        def event(kind: str, **extra) -> ProgressEvent:
            return ProgressEvent(
                kind=kind,
                stage=name,
                title=stage.title,
                index=index,
                total=self.total,
                **extra,
            )

        self.emit(event(STAGE_STARTED))
        try:
            yield reporter
        except BaseException as error:
            self.emit(event(STAGE_FAILED, error=error))
            raise
        self.emit(event(STAGE_FINISHED))


def log_progress(event: ProgressEvent) -> None:
    """A ready-made consumer: the run's progress, through `logging`.

    For a command line or a notebook, where a bar would need a dependency this
    library does not have::

        run_pipeline_stages(settings, schema, progress=log_progress)
    """
    position = f"{event.index + 1}/{event.total}"
    if event.kind == STAGE_STARTED:
        logger.info(f"Stage {position}: {event.title}")
    elif event.kind == STAGE_FINISHED:
        logger.info(f"Stage {position} done: {event.title}")
    elif event.kind == STAGE_FAILED:
        logger.error(f"Stage {position} failed: {event.title}: {event.error}")
    elif event.kind == STEP:
        of_total = f"/{event.step_total}" if event.step_total else ""
        logger.debug(
            f"Stage {position} step {(event.step_index or 0) + 1}{of_total}: {event.step}"
        )
