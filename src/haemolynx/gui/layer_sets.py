"""Which network the viewer shows: the baseline, or one perturbation.

A perturbation is the baseline's geometry with its own flows on it, so its
layers lie exactly on top of the baseline's and whichever is drawn last wins.
Comparing them one layer tick at a time means finding the right three of a
dozen layers each time. The view panel's "Showing" menu swaps the whole set
instead: the chosen network's vessels, nodes and flow-direction arrows, every
other network's hidden.

What carries across a swap is *which kinds* of layer were on -- nodes on in
the baseline stay on in the perturbation -- so flipping back and forth
compares like with like. The vessels are always on for the network shown,
exactly as the baseline's are today (the tubes/lines drawing keeps them so).

Pure: layer names and visibilities in, visibilities out. A set is named by
its perturbation, and :data:`BASELINE` (``None``) is the baseline, so a
perturbation called "Baseline" is still its own entry.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from haemolynx.gui.results import (
    FLOW_DIRECTION,
    NODES,
    VESSELS,
    perturbation_flow_direction_layer_name,
    perturbation_layer_names,
)
from haemolynx.gui.vessel_tubes import vessel_tubes_layer_name

__all__ = [
    "BASELINE",
    "BASELINE_LABEL",
    "ROLES",
    "layer_set",
    "layer_set_choices",
    "role_visibility",
    "set_layer_names",
    "visibility_for",
]

#: The baseline network's key: ``None``, so no perturbation name can collide.
BASELINE = None
BASELINE_LABEL = "Baseline"

#: The kinds of layer one network has, in the order they are listed.
ROLES = ("vessels", "nodes", "flow direction")


def layer_set(key: str | None) -> dict[str, str]:
    """``{role: layer name}`` for the baseline (``None``) or one perturbation."""
    if key is BASELINE:
        return {"vessels": VESSELS, "nodes": NODES, "flow direction": FLOW_DIRECTION}
    vessels, nodes = perturbation_layer_names(key)
    return {
        "vessels": vessels,
        "nodes": nodes,
        "flow direction": perturbation_flow_direction_layer_name(key),
    }


def set_layer_names(key: str | None) -> tuple[str, ...]:
    """Every layer that belongs to one network, its vessel tubes included."""
    roles = layer_set(key)
    return (*roles.values(), vessel_tubes_layer_name(roles["vessels"]))


def layer_set_choices(perturbations: Iterable[str]) -> list[tuple[str, str | None]]:
    """``(label, key)`` menu entries: the baseline, then each perturbation once."""
    return [(BASELINE_LABEL, BASELINE)] + [
        (str(name), str(name)) for name in dict.fromkeys(perturbations)
    ]


def role_visibility(key: str | None, visible: Mapping[str, bool]) -> dict[str, bool]:
    """Which kinds of layer are on for one network, among those it has.

    *visible* is every layer's visibility by name. The vessels count as on
    when either their lines or their tubes are drawn.
    """
    roles = layer_set(key)
    shown: dict[str, bool] = {}
    for role, name in roles.items():
        if role == "vessels":
            tubes = vessel_tubes_layer_name(name)
            if name in visible or tubes in visible:
                shown[role] = bool(visible.get(name)) or bool(visible.get(tubes))
        elif name in visible:
            shown[role] = bool(visible[name])
    return shown


def visibility_for(
    key: str | None,
    keys: Sequence[str | None],
    roles: Mapping[str, bool],
    present: Iterable[str],
) -> dict[str, bool]:
    """The visibility to give each *present* layer to show network *key*.

    Every other network in *keys* (the baseline always counts) is hidden,
    tubes and all. The shown network's vessels go on, and each of its other
    layers takes *roles*' answer for its kind, keeping its own visibility
    when *roles* has none. Its tubes are left to the tubes/lines drawing,
    which knows whether the vessels are drawn as tubes or as lines.
    """
    present = set(present)
    result: dict[str, bool] = {}
    for other in dict.fromkeys([BASELINE, *keys]):
        if other == key:
            continue
        for name in set_layer_names(other):
            if name in present:
                result[name] = False
    for role, name in layer_set(key).items():
        if name not in present:
            continue
        if role == "vessels":
            result[name] = True
        elif role in roles:
            result[name] = bool(roles[role])
    return result
