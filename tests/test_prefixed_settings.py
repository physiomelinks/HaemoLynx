"""Every setting handed over by its prefix reaches something.

``parsers.prefixed_arguments`` passes a group of settings named
``<prefix><parameter>`` to a function, keeping only the names the function
accepts. A setting whose parameter is renamed or removed is dropped without a
word: it stays in the schema, the config file and the panel, and changes
nothing. These tests hold each group the pipeline hands over that way to its
function: every setting with the prefix is one of its parameters, or is read
by its own name somewhere else in the package.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from haemolynx import preprocessing
from haemolynx.haemodynamics import automated
from haemolynx.parsers import parameters_of
from haemolynx.pipeline import default_schema

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "haemolynx"

#: Each prefix and the function it is handed to, as the pipeline calls them:
#: ``haemodynamics/apply.py`` (FWHM), ``pipeline/stages.py``'s skeletonise
#: (cleanup and skeleton).
PREFIXED_GROUPS = {
    "fwhm_": automated.measure_edge_diameters_fwhm_from_raw_tiff,
    "skeleton_": preprocessing.preprocess_skeleton_for_graph,
    "segmentation_cleanup_": preprocessing.clean_segmented_mask_for_skeletonisation,
}


@pytest.fixture(scope="module")
def source() -> str:
    """The package's source, less the schema, which declares every setting,
    and the panel, whose reading a setting for its own previews puts nothing
    into a run."""
    schema = PACKAGE / "pipeline" / "schema.py"
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(PACKAGE.rglob("*.py"))
        if path != schema and "gui" not in path.relative_to(PACKAGE).parts
    )


def _reaching_nothing(prefix: str, names, source: str) -> list[str]:
    """The *names* that are no parameter of *prefix*'s function and that the
    *source* never reads by name."""
    parameters = set(parameters_of(PREFIXED_GROUPS[prefix]))
    return [
        name
        for name in names
        if name[len(prefix):] not in parameters
        and not re.search(r"[\"']%s[\"']" % re.escape(name), source)
    ]


@pytest.mark.parametrize("prefix", sorted(PREFIXED_GROUPS))
def test_every_prefixed_setting_reaches_its_function_or_is_read_by_name(prefix, source):
    names = [s.name for s in default_schema().settings if s.name.startswith(prefix)]
    parameters = set(parameters_of(PREFIXED_GROUPS[prefix]))

    assert any(name[len(prefix):] in parameters for name in names), (
        f"no {prefix}* setting is a parameter of {PREFIXED_GROUPS[prefix].__name__}: "
        "the group is no longer handed over this way, so drop it from this test"
    )
    assert _reaching_nothing(prefix, names, source) == []


@pytest.mark.parametrize("prefix", sorted(PREFIXED_GROUPS))
def test_a_setting_whose_parameter_is_gone_is_caught(prefix, source):
    """The check itself: a prefixed name the function does not take and the
    package never reads is reported, while one of the function's own
    parameters is not."""
    orphan = f"{prefix}a_parameter_that_was_renamed"
    taken = prefix + parameters_of(PREFIXED_GROUPS[prefix])[-1]

    assert _reaching_nothing(prefix, [orphan, taken], source) == [orphan]
