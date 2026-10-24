"""Segmentation source follows a synthetic template's diameter basis.

When the user picks a template run (``synthetic_template_path``), or switches
the Input to a synthetic network with one already picked, the panel sets the
Segmentation source row to the basis the template's diameters were measured on
(``synthetic.diameter_basis_of_run``, what the template records as
``NetworkTemplate.diameter_basis``). The drawn vessels carry that basis
whatever the row says; with the row matching, a width given by hand in Post
processing is read on the same basis, and preflight has no mismatch to warn
about (Harvey, 2026-10-24).

Loading a config or a saved run never changes the row: the panel's restoring
guard skips it, as it skips the vessel-mask search, since a loaded file says
what the user asked for. A template that cannot be read changes nothing;
preflight says why.

Nothing here imports Qt: settings in, the value to set and the line to show out.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from haemolynx.graph.boundaries import network_is_synthetic

#: The settings whose change can set the row.
TRIGGER_SETTINGS = ("network_source", "synthetic_template_path")
#: The row set.
TARGET_SETTING = "segmentation_source"


@dataclass(frozen=True)
class BasisUpdate:
    """``value``: what to set ``segmentation_source`` to, None to leave it;
    ``line``: what to tell the user."""

    value: Optional[str] = None
    line: Optional[str] = None


def template_basis_update(before: Mapping[str, Any], values: Mapping[str, Any]) -> BasisUpdate:
    """What to set ``segmentation_source`` to now *values* replace *before*:
    the template's basis when a trigger changed, the network is synthetic, the
    template reads, and its basis is not the row's already."""
    if all(before.get(name) == values.get(name) for name in TRIGGER_SETTINGS):
        return BasisUpdate()
    if not network_is_synthetic(values):
        return BasisUpdate()
    path = values.get("synthetic_template_path")
    if path in (None, ""):
        return BasisUpdate()
    file = Path(path)
    if file.suffix.lower() != ".haemorun" or not file.is_file():
        return BasisUpdate()
    from haemolynx.gui.run_snapshot import RunSnapshotError
    from haemolynx.synthetic import diameter_basis_of_run

    try:
        basis = diameter_basis_of_run(file)
    except (RunSnapshotError, OSError, ValueError):
        return BasisUpdate()
    if basis is None or basis == values.get(TARGET_SETTING):
        return BasisUpdate()
    return BasisUpdate(
        value=basis,
        line=f"Segmentation source set to {basis}: the template run's diameters were measured on it.",
    )


__all__ = ["BasisUpdate", "TARGET_SETTING", "TRIGGER_SETTINGS", "template_basis_update"]
