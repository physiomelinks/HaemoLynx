"""The image-to-model pipeline: settings resolution and the run stages.

    from ImageLynx.pipeline import resolve_settings, run_pipeline_stages

    settings = resolve_settings(schema=SCHEMA, config_path="my_config.yaml")
    graph = run_pipeline_stages(settings, SCHEMA)
"""
from .settings import fill_derived_settings, resolve_settings
from .stages import run_pipeline_stages

__all__ = [
    "fill_derived_settings",
    "resolve_settings",
    "run_pipeline_stages",
]
