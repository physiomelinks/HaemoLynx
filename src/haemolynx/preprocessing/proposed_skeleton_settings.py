"""Proposed bundle-into-paths GUI settings — declared here, not wired.

``preprocess_skeleton_for_graph`` already accepts ``bundle_scan_size``,
``bundle_density_fraction``, ``bundle_max_connections_per_hub`` and
``bundle_hub_min_spacing``. The skeletonise stage already forwards every
``skeleton_*`` setting whose name matches those parameters
(``prefixed_arguments(..., "skeleton_", ...)``). They never appear in the
napari form because they are absent from :func:`haemolynx.pipeline.default_schema`.
Adding :data:`BUNDLE_INTO_PATHS_SETTINGS` to the schema, and the names to the
``skeletonise`` entry of :data:`haemolynx.pipeline.progress.STAGES`, is the
whole GUI wiring.

Thickness-gated skeletonisation is live: see
``use_thick_vessel_skeletonisation`` on the Skeletonise tab.
"""
from __future__ import annotations

from haemolynx.parsers import Setting

_PIPELINE_STAGES = "Pipeline stages"
_REQUIRES_SKELETONIZE = ("do_skeletonize",)

#: ``skeleton_<param>`` names that ``prefixed_arguments`` already maps onto
#: :func:`haemolynx.preprocessing.skeletonize_voxel_bundles_into_paths` via
#: :func:`haemolynx.preprocessing.preprocess_skeleton_for_graph`.
BUNDLE_INTO_PATHS_SETTINGS: tuple[Setting, ...] = (
    Setting(
        name="skeleton_bundle_scan_size",
        kind="int",
        default=9,
        help=(
            "Sliding-window size (voxels) used to detect dense skeleton "
            "bundles and collapse each into a hub with clean in/out paths"
        ),
        section=_PIPELINE_STAGES,
        unit="voxels",
        minimum=3,
        requires=_REQUIRES_SKELETONIZE,
    ),
    Setting(
        name="skeleton_bundle_density_fraction",
        kind="float",
        default=0.35,
        help=(
            "Mark a window as a dense bundle when this fraction of its voxels "
            "are skeleton foreground"
        ),
        section=_PIPELINE_STAGES,
        minimum=0.0,
        maximum=1.0,
        requires=_REQUIRES_SKELETONIZE,
    ),
    Setting(
        name="skeleton_bundle_max_connections_per_hub",
        kind="int",
        default=8,
        help="Keep at most this many directional links when reconnecting paths to a bundle hub",
        section=_PIPELINE_STAGES,
        minimum=1,
        requires=_REQUIRES_SKELETONIZE,
    ),
    Setting(
        name="skeleton_bundle_hub_min_spacing",
        kind="int",
        default=4,
        help=(
            "Minimum spacing between bundle hub centres; 4 matches the "
            "automatic half-window spacing used when scan size is 9"
        ),
        section=_PIPELINE_STAGES,
        unit="voxels",
        minimum=1,
        requires=_REQUIRES_SKELETONIZE,
    ),
)

PROPOSED_SKELETONISE_SETTING_NAMES: tuple[str, ...] = tuple(
    setting.name for setting in BUNDLE_INTO_PATHS_SETTINGS
)

#: Names to append to the ``skeletonise`` stage's ``settings`` tuple in
#: ``pipeline/progress.py`` when this is wired. Bundle rows belong on the
#: Skeletonise tab next to the existing closing / bridging knobs.
BUNDLE_INTO_PATHS_STAGE_SETTING_NAMES: tuple[str, ...] = tuple(
    setting.name for setting in BUNDLE_INTO_PATHS_SETTINGS
)


def proposed_skeleton_schema_extension() -> tuple[Setting, ...]:
    """Settings that would be appended to the pipeline schema on wiring."""
    return BUNDLE_INTO_PATHS_SETTINGS
