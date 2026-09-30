"""One "Diameter measurement" choice on the Diameters tab, over two settings.

FWHM on the plasma label (``use_fwhm_edge_diameters``) and the internal
diameter inside an endothelial stain's wall (``use_endothelial_diameters``) are
alternatives: each is the first source of its own diameter chain, and
preflight refuses a run with both on
(:func:`haemolynx.pipeline.checks.check_one_primary_diameter_measurement`).
Two checkboxes let a user tick both; one drop-down cannot.

The settings themselves stay two bools, so every config file, preset and
command line that names them keeps working. The panel shows this choice in
their place and writes them; this module is what it reads and writes, with no
Qt, so it is tested on its own.
"""
from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "BOTH_ON_NOTE",
    "DIAMETER_SOURCES",
    "DIAMETER_SOURCE_LABEL",
    "DIAMETER_SOURCE_SETTINGS",
    "ENDOTHELIAL",
    "FWHM",
    "NO_MEASUREMENT",
    "both_on",
    "settings_for",
    "source_from",
]

NO_MEASUREMENT = "none"
FWHM = "fwhm"
ENDOTHELIAL = "endothelial"

#: The two settings the choice stands for, in the order the panel shows them.
DIAMETER_SOURCE_SETTINGS: tuple[str, str] = (
    "use_fwhm_edge_diameters",
    "use_endothelial_diameters",
)

#: The row's label, and its ``(display text, value)`` choices: short enough
#: to read whole in a docked panel; the row's tooltip says the rest.
DIAMETER_SOURCE_LABEL = "Diameter measurement"
DIAMETER_SOURCES: tuple[tuple[str, str], ...] = (
    ("None (table or mask estimate)", NO_MEASUREMENT),
    ("FWHM of the raw image", FWHM),
    ("Endothelial wall", ENDOTHELIAL),
)

#: What the panel says when a loaded config turns both on.
BOTH_ON_NOTE = (
    "This config turns on both FWHM and endothelial diameters, which cannot run "
    "together (preflight refuses it), so the panel kept FWHM and turned the "
    "endothelial measurement off. Choose Diameter measurement on the Diameters tab "
    "to change it."
)


def source_from(values: Mapping[str, Any]) -> str:
    """The choice two settings amount to. FWHM wins if both are on."""
    if bool(values.get("use_fwhm_edge_diameters")):
        return FWHM
    if bool(values.get("use_endothelial_diameters")):
        return ENDOTHELIAL
    return NO_MEASUREMENT


def settings_for(source: str) -> dict[str, bool]:
    """The two settings one choice stands for."""
    if source not in {value for _label, value in DIAMETER_SOURCES}:
        raise ValueError(
            f"Unknown diameter source {source!r}. "
            f"Sources are: {[value for _label, value in DIAMETER_SOURCES]}."
        )
    return {
        "use_fwhm_edge_diameters": source == FWHM,
        "use_endothelial_diameters": source == ENDOTHELIAL,
    }


def both_on(values: Mapping[str, Any]) -> bool:
    """True for the one combination the choice cannot show, and a run refuses."""
    return all(bool(values.get(name)) for name in DIAMETER_SOURCE_SETTINGS)
