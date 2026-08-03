"""The image-to-model pipeline: settings resolution and the run stages.

Run the whole thing::

    settings = resolve_settings(schema=SCHEMA, config_path="my_config.yaml")
    graph = run_pipeline_stages(settings, SCHEMA)

or call the stages yourself, which is what the examples do so that each step of
a run is visible and can be intervened in::

    inputs   = segment(settings)
    volume   = skeletonise(settings, inputs)
    network  = build_network(settings, volume, SCHEMA)
    ...
"""
from .settings import fill_derived_settings, resolve_settings
from .stages import (
    BoundaryNodes,
    HaemodynamicModel,
    SegmentedInputs,
    SkeletonisedVolume,
    Solution,
    VesselNetwork,
    assign_boundaries,
    assign_diameters,
    build_haemodynamic_model,
    build_network,
    export_results,
    run_pipeline_stages,
    segment,
    skeletonise,
    solve,
)

__all__ = [
    "BoundaryNodes",
    "HaemodynamicModel",
    "SegmentedInputs",
    "SkeletonisedVolume",
    "Solution",
    "VesselNetwork",
    "assign_boundaries",
    "assign_diameters",
    "build_haemodynamic_model",
    "build_network",
    "export_results",
    "fill_derived_settings",
    "resolve_settings",
    "run_pipeline_stages",
    "segment",
    "skeletonise",
    "solve",
]
