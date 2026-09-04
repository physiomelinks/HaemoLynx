"""The image-to-model pipeline: its settings, and the stages of a run.

Every setting is declared once, in :mod:`haemolynx.pipeline.schema`. Start from
that declaration to write yourself a config file, then run the whole thing::

    write_default_config("my_config.yaml")      # commented, every setting
    schema = default_schema()
    settings = resolve_settings(schema=schema, config_path="my_config.yaml")
    graph = run_pipeline_stages(settings, schema)

or call the stages yourself, which is what the examples do so that each step of
a run is visible and can be intervened in::

    inputs   = segment(settings)
    volume   = skeletonise(settings, inputs)
    network  = build_network(settings, volume, SCHEMA)
    ...

Either way a run says where it has got to, if anything is listening::

    run_pipeline_stages(settings, schema, progress=log_progress)
"""
from .checks import preflight
from .progress import (
    KINDS,
    STAGE_FAILED,
    STAGE_FINISHED,
    STAGE_STARTED,
    STAGES,
    STEP,
    ProgressCallback,
    ProgressEvent,
    RunProgress,
    Stage,
    StageProgress,
    log_progress,
)
from .schema import SCHEMA, default_schema, write_default_config
from .settings import fill_derived_settings, resolve_settings
from .stages import (
    BoundaryNodes,
    HaemodynamicModel,
    PerturbationResult,
    PerturbationRun,
    PipelineResume,
    SegmentedInputs,
    SkeletonisedVolume,
    Solution,
    VesselNetwork,
    assign_boundaries,
    assign_diameters,
    build_haemodynamic_model,
    build_network,
    export_results,
    run_perturbations,
    run_pipeline_stages,
    segment,
    skeletonise,
    solve,
)

__all__ = [
    "KINDS",
    "SCHEMA",
    "STAGES",
    "STAGE_FAILED",
    "STAGE_FINISHED",
    "STAGE_STARTED",
    "STEP",
    "BoundaryNodes",
    "HaemodynamicModel",
    "PerturbationResult",
    "PerturbationRun",
    "PipelineResume",
    "ProgressCallback",
    "ProgressEvent",
    "RunProgress",
    "SegmentedInputs",
    "SkeletonisedVolume",
    "Solution",
    "Stage",
    "StageProgress",
    "VesselNetwork",
    "assign_boundaries",
    "assign_diameters",
    "build_haemodynamic_model",
    "build_network",
    "default_schema",
    "export_results",
    "log_progress",
    "preflight",
    "fill_derived_settings",
    "resolve_settings",
    "run_perturbations",
    "run_pipeline_stages",
    "segment",
    "skeletonise",
    "solve",
    "write_default_config",
]
