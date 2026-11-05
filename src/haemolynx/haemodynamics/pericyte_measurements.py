"""Measurements around each pericyte of a pericyte mask (``measure_pericytes``).

Each pericyte is placed as the pericyte perturbations place it
(:func:`~haemolynx.haemodynamics.pericyte_mask.place_pericytes`): its centroid
projected onto the nearest capillary's centreline, kept when it lies within the
assignment distance of it and is of a pericyte's size. The point it projects to
is its midline on the vessel, and from there:

* **a diameter profile** -- the vessel's width at the midline and every *step*
  um each way along it, out to *reach* um or to the vessel's end, whichever
  comes first. A vessel ends at its end nodes: a junction or a dead end. The
  width is the segmentation's (the mask-diameter reader,
  :func:`~haemolynx.haemodynamics.edt_diameter.mask_diameters_at_points`:
  :func:`mask_reader`) or the raw image's FWHM across the vessel there, fitted
  as the Diameters stage fits its samples (:func:`fwhm_reader`);
* **the distance to the vessel's nearest junction** -- the nearer of its two
  ends that is one, along the vessel and in a straight line. A junction is a
  node three or more vessels meet at, or where the vessel opens into a thick
  vessel (a zero-resistance bridge's end);
* **each branch's resistance** -- as the uniform tube the network is solved
  with, and with the profiles measured along it. The measured profile scales
  the branch's own diameter: at each point, the branch's diameter times the
  reading there over the branch's typical width in the segmentation (the
  median of its widths along its whole length,
  :func:`branch_reference_diameters`), so a branch whose readings are all
  alike keeps its resistance, and the narrowing is put on the run's own
  diameters (FWHM, endothelial or mask) rather than mixing the mask's widths
  with them. Where several pericytes' profiles cover a point, the nearest
  pericyte's is read; beyond every profile the branch keeps its diameter.
  Readings under
  :data:`~haemolynx.haemodynamics.edt_diameter.MIN_RESOLVABLE_DIAMETER_UM`, and
  planes the vessel's section does not close in (a junction's, a dead end's),
  are not widths and are bridged by their neighbours.

The ``pericyte_measured_constriction`` perturbation re-solves the network with
each branch's resistance scaled by its measured-constriction ratio
(:func:`apply_measured_constriction`).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import networkx as nx
import numpy as np

from .constriction import profile_resistance
from .edt_diameter import MIN_RESOLVABLE_DIAMETER_UM, _centreline_tangents, mask_diameters_at_points
from .fwhm_diameter import _arc_length_parameterize, _interpolate_centerline
from .pericyte_mask import PericytePlacement, _project_point_to_polyline
from .poiseuille import (
    PoiseuilleModel,
    baseline_edge_diameter,
    edge_diameter_basis,
    set_edge_resistance,
)

__all__ = [
    "BRANCH_COLUMNS",
    "DEFAULT_PROFILE_REACH_UM",
    "DEFAULT_PROFILE_STEP_UM",
    "PERICYTE_COLUMNS",
    "PROFILE_COLUMNS",
    "PROFILE_FOLDER_SUFFIX",
    "PROFILE_READINGS",
    "USABLE_METHODS",
    "BranchResistance",
    "DiameterProfile",
    "DiameterReader",
    "JunctionDistance",
    "PericyteMeasurements",
    "PlacedPericyte",
    "ReadingRequest",
    "apply_measured_constriction",
    "branch_reference_diameters",
    "branch_resistances",
    "diameter_profiles",
    "fwhm_reader",
    "is_junction",
    "junction_distances",
    "mask_reader",
    "measure_pericytes",
    "placed_pericytes",
    "profile_file_name",
    "record_pericyte_sites",
    "write_branch_csv",
    "write_pericyte_measurements",
]

#: The profile's step and reach when the settings give none (um).
DEFAULT_PROFILE_STEP_UM = 2.0
DEFAULT_PROFILE_REACH_UM = 50.0

#: Points the resistance integral takes per profile step along a branch: the
#: trapezoid over a 3 -> 6 um ramp read every 2 um is then within 0.02% of
#: its closed form (8 points: 1.7%).
INTEGRATION_POINTS_PER_STEP = 32

#: Where a profile's diameters are read: the segmentation's cross-section, or
#: a FWHM across the vessel in the raw image (``pericyte_profile_reading``).
PROFILE_READINGS = ("segmentation", "raw_image_fwhm")

#: The readings a branch's resistance reads: a closed cross-section of the
#: segmentation, or a FWHM fit that passed its checks.
USABLE_METHODS = ("cross_section", "fwhm")

#: The folder each pericyte's profile CSV goes in: ``{stem}`` + this.
PROFILE_FOLDER_SUFFIX = "_pericyte_profiles"

#: Columns of one pericyte's profile CSV, ``pericyte_<NNNN>.csv``.
PROFILE_COLUMNS = (
    "offset_um",
    "toward_node",
    "arc_position_um",
    "z_um",
    "y_um",
    "x_um",
    "diameter_um",
    "diameter_method",
    "resolved",
    "diameter_change_um",
    "diameter_ratio",
    "distance_to_vessel_end_um",
)

#: Columns of ``{stem}_pericytes.csv``: one row per pericyte of the mask.
PERICYTE_COLUMNS = (
    "pericyte",
    "centroid_z_um",
    "centroid_y_um",
    "centroid_x_um",
    "equivalent_diameter_um",
    "placed",
    "reason",
    "branch_id",
    "u",
    "v",
    "key",
    "branch_order",
    "distance_to_vessel_um",
    "arc_position_um",
    "vessel_length_um",
    "midline_z_um",
    "midline_y_um",
    "midline_x_um",
    "profile_file",
    "profile_readings",
    "diameter_at_pericyte_um",
    "narrowest_diameter_um",
    "narrowest_offset_um",
    "nearest_junction_node",
    "junction_end",
    "junction_vessels",
    "junction_distance_along_vessel_um",
    "junction_straight_line_distance_um",
    "branch_length_um",
    "branch_diameter_um",
    "branch_haematocrit",
    "branch_pericytes",
    "branch_uniform_resistance_pa_s_per_m3",
    "branch_constricted_resistance_pa_s_per_m3",
    "branch_resistance_ratio",
)

#: Columns of ``{stem}_pericyte_branches.csv``: one row per branch holding a
#: pericyte whose profile was measured.
BRANCH_COLUMNS = (
    "branch_id",
    "u",
    "v",
    "key",
    "branch_order",
    "length_um",
    "diameter_um",
    "haematocrit",
    "pericytes",
    "pericyte_numbers",
    "resolved_readings",
    "reference_diameter_um",
    "narrowest_diameter_um",
    "uniform_resistance_pa_s_per_m3",
    "constricted_resistance_pa_s_per_m3",
    "resistance_ratio",
)


@dataclass(frozen=True)
class PlacedPericyte:
    """One pericyte of the mask, and where it sits.

    ``number`` counts the mask's components from 1, as
    ``{name}_mask_objects.csv`` numbers them. A placed pericyte has its
    vessel ``edge`` ``(u, v, key)``, its ``arc_position_um`` from ``u`` along
    the vessel's centreline (``vessel_length_um`` long, run ``u`` to ``v``)
    and its ``midline_um`` there; one not placed says why in ``reason``.
    """

    number: int
    centroid_um: tuple[float, float, float]
    equivalent_diameter_um: float
    reason: str = ""
    edge: tuple[Any, Any, Any] | None = None
    branch_id: int | None = None
    branch_order: str | None = None
    distance_to_vessel_um: float | None = None
    arc_position_um: float | None = None
    vessel_length_um: float | None = None
    midline_um: tuple[float, float, float] | None = None

    @property
    def placed(self) -> bool:
        return self.edge is not None


@dataclass(frozen=True)
class DiameterProfile:
    """The widths read along one pericyte's vessel: ``offsets_um`` from its
    midline (negative toward ``u``, positive toward ``v``), each reading's
    arc position from ``u``, point, diameter (NaN where none) and method."""

    pericyte: PlacedPericyte
    offsets_um: np.ndarray
    arc_positions_um: np.ndarray
    points_um: np.ndarray
    diameters_um: np.ndarray
    methods: tuple[str, ...]

    @property
    def resolved(self) -> np.ndarray:
        """Which readings are widths: at least the narrowest resolvable."""
        return np.isfinite(self.diameters_um) & (self.diameters_um >= MIN_RESOLVABLE_DIAMETER_UM)

    @property
    def usable(self) -> np.ndarray:
        """The resolved readings a branch's resistance reads
        (:data:`USABLE_METHODS`): a closed cross-section of the vessel's own,
        or a FWHM fit that passed its checks. Where the segmentation's plane
        is not closed -- a junction's, cutting the vessel it joins, or a dead
        end's -- the reading is the inscribed radius's, a junction's sphere or
        a tapering tip, not the vessel's width."""
        return self.resolved & np.asarray([method in USABLE_METHODS for method in self.methods], dtype=bool)

    @property
    def diameter_at_pericyte_um(self) -> float:
        at = np.flatnonzero(np.isclose(self.offsets_um, 0.0))
        return float(self.diameters_um[at[0]]) if len(at) else float("nan")

    def narrowest(self) -> tuple[float, float]:
        """``(diameter, offset)`` of the narrowest resolved reading, or NaNs."""
        rows = np.flatnonzero(self.resolved)
        if not len(rows):
            return float("nan"), float("nan")
        row = rows[int(np.argmin(self.diameters_um[rows]))]
        return float(self.diameters_um[row]), float(self.offsets_um[row])


@dataclass(frozen=True)
class JunctionDistance:
    """How far a pericyte's midline is from its vessel's nearest junction:
    ``node``, which end of the vessel it is (``"u"`` or ``"v"``), how many
    vessels meet there, and the distance along the vessel and in a straight
    line. All None when neither end of the vessel is a junction."""

    pericyte: PlacedPericyte
    node: Any = None
    end: str | None = None
    vessels: int | None = None
    along_vessel_um: float | None = None
    straight_line_um: float | None = None


@dataclass(frozen=True)
class BranchResistance:
    """One branch holding measured pericytes: its uniform and measured-
    constriction resistances (Pa.s/m^3), and what they were computed from."""

    branch_id: int
    edge: tuple[Any, Any, Any]
    branch_order: str | None
    length_um: float
    diameter_um: float | None
    haematocrit: float
    pericytes: tuple[int, ...]
    resolved_readings: int
    reference_diameter_um: float | None
    narrowest_diameter_um: float | None
    uniform_resistance: float | None
    constricted_resistance: float | None

    @property
    def ratio(self) -> float | None:
        if not self.uniform_resistance or self.constricted_resistance is None:
            return None
        return float(self.constricted_resistance) / float(self.uniform_resistance)


@dataclass
class PericyteMeasurements:
    """Everything :func:`measure_pericytes` measured. ``profiles`` and
    ``junctions`` are keyed by pericyte number; ``branches`` by edge."""

    pericytes: list[PlacedPericyte]
    profiles: dict[int, DiameterProfile] = field(default_factory=dict)
    junctions: dict[int, JunctionDistance] = field(default_factory=dict)
    branches: dict[tuple[Any, Any, Any], BranchResistance] = field(default_factory=dict)
    #: Each measured branch's typical width (:func:`branch_reference_diameters`).
    reference_diameters: dict[tuple[Any, Any, Any], float | None] = field(default_factory=dict)
    step_um: float = DEFAULT_PROFILE_STEP_UM
    reach_um: float = DEFAULT_PROFILE_REACH_UM

    def summary(self) -> dict[str, Any]:
        placed = [p for p in self.pericytes if p.placed]
        ratios = [b.ratio for b in self.branches.values() if b.ratio is not None]
        along = [j.along_vessel_um for j in self.junctions.values() if j.along_vessel_um is not None]
        return {
            "pericytes": len(self.pericytes),
            "placed": len(placed),
            "profiles": len(self.profiles),
            "junction_distances": len(along),
            "junction_distance_median_um": float(np.median(along)) if along else None,
            "branches": len(self.branches),
            "resistance_ratio_median": float(np.median(ratios)) if ratios else None,
        }


def _branch_ids(G: nx.MultiGraph) -> dict[tuple[Any, Any, Any], int]:
    """Each edge's branch ID: its place in the graph's edge order."""
    return {(u, v, key): index for index, (u, v, key) in enumerate(G.edges(keys=True))}


def _oriented_centreline(G: nx.MultiGraph, edge: tuple[Any, Any, Any]) -> np.ndarray:
    from haemolynx.visualization.geometry import edge_polyline

    u, v, key = edge
    return edge_polyline(G, u, v, G.edges[u, v, key])


def placed_pericytes(G: nx.MultiGraph, placement: PericytePlacement, *,
                     max_assignment_distance_um: float | None,
                     min_pericyte_diameter_um: float | None,
                     max_pericyte_diameter_um: float | None) -> list[PlacedPericyte]:
    """Every pericyte of *placement* (``place_pericytes`` with the same
    limits), each placed pericyte at its midline on its capillary's
    centreline run ``u`` to ``v``, and every other with why it is not."""
    from haemolynx.geometry import cumulative_lengths

    ids = _branch_ids(G)
    eligible = set(placement.eligible)
    pericytes: list[PlacedPericyte] = []
    centreline_cache: dict[tuple[Any, Any, Any], tuple[np.ndarray, np.ndarray]] = {}
    for index in range(int(placement.centroids_um.shape[0])):
        centroid = tuple(float(c) for c in placement.centroids_um[index])
        diameter = float(placement.equivalent_diameters_um[index])
        common = {"number": index + 1, "centroid_um": centroid, "equivalent_diameter_um": diameter}
        projection = placement.projections.get(index)
        if projection is None:
            pericytes.append(PlacedPericyte(**common, reason="no capillary in the network"))
            continue
        edge, _s, distance = projection
        if index not in eligible:
            if max_assignment_distance_um is not None and distance > float(max_assignment_distance_um):
                reason = "too far from a capillary"
            elif min_pericyte_diameter_um is not None and diameter < float(min_pericyte_diameter_um):
                reason = "smaller than a pericyte"
            else:
                reason = "larger than a pericyte"
            pericytes.append(PlacedPericyte(**common, reason=reason, distance_to_vessel_um=float(distance)))
            continue
        edge = tuple(edge)
        if edge not in centreline_cache:
            points = _oriented_centreline(G, edge)
            centreline_cache[edge] = (points, cumulative_lengths(points))
        points, lengths = centreline_cache[edge]
        s, distance = _project_point_to_polyline(np.asarray(centroid), points, lengths)
        midline = _interpolate_centerline(points, lengths, np.array([s]))[0]
        pericytes.append(
            PlacedPericyte(
                **common,
                edge=edge,
                branch_id=ids.get(edge),
                branch_order=G.edges[edge].get("branch_order"),
                distance_to_vessel_um=float(distance),
                arc_position_um=float(s),
                vessel_length_um=float(lengths[-1]),
                midline_um=tuple(float(c) for c in midline),
            )
        )
    return pericytes


def _reading_offsets(position_um: float, length_um: float, step_um: float, reach_um: float) -> np.ndarray:
    """The midline and every *step_um* each way, out to *reach_um* or the
    vessel's ends (0 and *length_um*), whichever comes first."""
    count = int(np.floor(float(reach_um) / float(step_um) + 1e-9))
    offsets = np.arange(-count, count + 1, dtype=float) * float(step_um)
    at = position_um + offsets
    keep = (at >= -1e-9) & (at <= length_um + 1e-9)
    return offsets[keep]


@dataclass(frozen=True)
class ReadingRequest:
    """Points on one edge's centreline to read the vessel's width at: their
    positions (physical ``(z, y, x)`` microns) and the centreline's direction
    there."""

    edge: tuple[Any, Any, Any]
    points_um: np.ndarray
    tangents: np.ndarray


#: Reads each request's widths: per request, the diameters (NaN where none)
#: and each one's method (``""`` where none).
DiameterReader = Callable[[Sequence[ReadingRequest]], list[tuple[np.ndarray, list[str]]]]


def mask_reader(
    vessel_mask: np.ndarray, voxel_size_zyx: Sequence[float], *, workers: int = 1, memmap_directory=None
) -> DiameterReader:
    """The segmentation's width at each point (*vessel_mask*, binary, on the
    graph's grid): its cross-section's area-equivalent diameter, else twice
    its inscribed radius (:func:`~haemolynx.haemodynamics.edt_diameter.mask_diameters_at_points`)."""

    def read(requests: Sequence[ReadingRequest]) -> list[tuple[np.ndarray, list[str]]]:
        return mask_diameters_at_points(
            vessel_mask, tuple(float(v) for v in voxel_size_zyx),
            [(request.points_um, request.tangents) for request in requests],
            workers=workers, memmap_directory=memmap_directory,
        )

    return read


def fwhm_reader(
    G: nx.MultiGraph, measure: Callable[[dict[tuple[Any, Any, Any], np.ndarray]], Mapping[Any, Sequence[float]]]
) -> DiameterReader:
    """The raw image's FWHM width at each point, method ``"fwhm"``.
    *measure* takes ``{edge: arc lengths}`` -- microns along each edge's
    stored ``voxels``, where the FWHM measurement samples -- and returns each
    edge's width at each (NaN where no fit passed), as
    :func:`~haemolynx.haemodynamics.apply.measure_fwhm_diameters` does with
    ``at_positions``. Every request is read in one call."""
    from haemolynx.geometry import cumulative_lengths

    from .pericyte_mask import _edge_centerline_points

    def read(requests: Sequence[ReadingRequest]) -> list[tuple[np.ndarray, list[str]]]:
        asked: dict[tuple[Any, Any, Any], list[float]] = {}
        spans: list[tuple[tuple[Any, Any, Any], int, int]] = []
        for request in requests:
            edge = tuple(request.edge)
            stored = _edge_centerline_points(G, edge[0], edge[1], G.edges[edge])
            lengths = cumulative_lengths(stored)
            at = [
                _project_point_to_polyline(np.asarray(point, dtype=float), stored, lengths)[0]
                for point in np.asarray(request.points_um, dtype=float).reshape(-1, 3)
            ]
            positions = asked.setdefault(edge, [])
            spans.append((edge, len(positions), len(at)))
            positions.extend(at)
        widths = measure({edge: np.asarray(at, dtype=float) for edge, at in asked.items()}) if asked else {}
        results = []
        for edge, start, count in spans:
            got = np.full(count, np.nan)
            read_here = np.asarray(widths.get(edge, ()), dtype=float)[start:start + count]
            got[: len(read_here)] = read_here
            results.append((got, ["fwhm" if np.isfinite(d) else "" for d in got]))
        return results

    return read


def diameter_profiles(
    G: nx.MultiGraph,
    pericytes: Sequence[PlacedPericyte],
    reader: DiameterReader,
    *,
    step_um: float = DEFAULT_PROFILE_STEP_UM,
    reach_um: float = DEFAULT_PROFILE_REACH_UM,
) -> dict[int, DiameterProfile]:
    """Each placed pericyte's diameter profile, read by *reader*
    (:func:`mask_reader`, :func:`fwhm_reader`), keyed by pericyte number."""
    if step_um <= 0:
        raise ValueError(f"The profile step must be positive, got {step_um}.")
    if reach_um < 0:
        raise ValueError(f"The profile reach must not be negative, got {reach_um}.")
    placed = [p for p in pericytes if p.placed]
    requests: list[ReadingRequest] = []
    layouts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for pericyte in placed:
        poly = _oriented_centreline(G, pericyte.edge)
        s, total = _arc_length_parameterize(poly)
        offsets = _reading_offsets(pericyte.arc_position_um, total, step_um, reach_um)
        at = np.clip(pericyte.arc_position_um + offsets, 0.0, total)
        points = np.asarray(_interpolate_centerline(poly, s, at), dtype=float).reshape(-1, 3)
        requests.append(ReadingRequest(pericyte.edge, points, _centreline_tangents(poly, s, total, at)))
        layouts.append((offsets, at, points))
    readings = reader(requests) if requests else []
    return {
        pericyte.number: DiameterProfile(
            pericyte=pericyte,
            offsets_um=offsets,
            arc_positions_um=at,
            points_um=points,
            diameters_um=diameters,
            methods=tuple(methods),
        )
        for pericyte, (offsets, at, points), (diameters, methods) in zip(placed, layouts, readings)
    }


def is_junction(G: nx.MultiGraph, node: Any) -> bool:
    """Whether *node* is a junction: three or more vessels meet there, or a
    vessel opens into a thick vessel there (a zero-resistance bridge's end)."""
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    if int(G.degree(node)) >= 3:
        return True
    return any(data.get(IS_ZERO_RESISTANCE) for _a, _b, data in G.edges(node, data=True))


def junction_distances(G: nx.MultiGraph, pericytes: Sequence[PlacedPericyte]) -> dict[int, JunctionDistance]:
    """Each placed pericyte's distance to its vessel's nearest junction,
    keyed by pericyte number."""
    distances: dict[int, JunctionDistance] = {}
    for pericyte in pericytes:
        if not pericyte.placed:
            continue
        u, v, _key = pericyte.edge
        ends = []
        if is_junction(G, u):
            ends.append((pericyte.arc_position_um, u, "u"))
        if is_junction(G, v):
            ends.append((pericyte.vessel_length_um - pericyte.arc_position_um, v, "v"))
        if not ends:
            distances[pericyte.number] = JunctionDistance(pericyte)
            continue
        along, node, end = min(ends, key=lambda item: item[0])
        position = np.asarray(G.nodes[node]["pos"], dtype=float)
        distances[pericyte.number] = JunctionDistance(
            pericyte,
            node=node,
            end=end,
            vessels=int(G.degree(node)),
            along_vessel_um=float(max(along, 0.0)),
            straight_line_um=float(np.linalg.norm(position - np.asarray(pericyte.midline_um))),
        )
    return distances


def _relative_profile(
    grid_um: np.ndarray, profiles: Sequence[DiameterProfile], reference_um: float
) -> np.ndarray:
    """The branch's diameter over its own, at each of *grid_um* (arc lengths
    along its centreline from ``u``): the nearest pericyte's measured profile
    over *reference_um* where one covers the point, 1 elsewhere."""
    ratio = np.ones(len(grid_um))
    best = np.full(len(grid_um), np.inf)
    for profile in profiles:
        rows = np.flatnonzero(profile.usable)
        if not len(rows):
            continue
        at = profile.arc_positions_um[rows]
        widths = profile.diameters_um[rows] / float(reference_um)
        covered = (grid_um >= at.min() - 1e-9) & (grid_um <= at.max() + 1e-9)
        distance = np.abs(grid_um - float(profile.pericyte.arc_position_um))
        nearer = covered & (distance < best)
        if np.any(nearer):
            ratio[nearer] = np.interp(grid_um[nearer], at, widths)
            best[nearer] = distance[nearer]
    return ratio


def branch_reference_diameters(
    G: nx.MultiGraph,
    edges: Sequence[tuple[Any, Any, Any]],
    reader: DiameterReader,
    *,
    step_um: float = DEFAULT_PROFILE_STEP_UM,
) -> dict[tuple[Any, Any, Any], float | None]:
    """Each of *edges*' typical width as *reader* reads it: the median of its
    usable readings (:data:`USABLE_METHODS`, at least the narrowest
    resolvable), taken every *step_um* along its whole centreline. What a
    profile's readings are scaled against: the branch's own diameter is
    likewise a median over its whole length, so a branch read only where its
    pericyte narrows it does not take the narrowing for its width. None
    where nothing reads."""
    requests: list[ReadingRequest] = []
    for edge in edges:
        poly = _oriented_centreline(G, edge)
        s, total = _arc_length_parameterize(poly)
        at = np.linspace(0.0, total, max(2, int(np.floor(total / float(step_um))) + 1))
        requests.append(ReadingRequest(
            tuple(edge), np.asarray(_interpolate_centerline(poly, s, at)).reshape(-1, 3),
            _centreline_tangents(poly, s, total, at),
        ))
    readings = reader(requests) if requests else []
    references: dict[tuple[Any, Any, Any], float | None] = {}
    for edge, (diameters, methods) in zip(edges, readings):
        kept = [
            float(d) for d, method in zip(diameters, methods)
            if method in USABLE_METHODS and d >= MIN_RESOLVABLE_DIAMETER_UM
        ]
        references[tuple(edge)] = float(np.median(kept)) if kept else None
    return references


def branch_resistances(
    G: nx.MultiGraph,
    profiles: Mapping[int, DiameterProfile],
    *,
    model: PoiseuilleModel,
    diameter_by_branch_order: Mapping[str, Any] | None = None,
    prefer_edge_fwhm_diameter: bool = False,
    step_um: float = DEFAULT_PROFILE_STEP_UM,
    reference_diameters: Mapping[tuple[Any, Any, Any], float | None] | None = None,
) -> dict[tuple[Any, Any, Any], BranchResistance]:
    """Each branch holding a measured pericyte, as a uniform tube and with its
    measured profiles (read every *step_um*), by *model*'s viscosity law and
    the branch's own haematocrit (its ``discharge_haematocrit``, else the
    model's). The branch's diameter is the one its resistance was solved
    with (``baseline_edge_diameter``); one with none gets no resistances.
    The readings are scaled against each branch's *reference_diameters*
    (:func:`branch_reference_diameters`), or, for a branch with none, the
    median of its profiles' own readings."""
    ids = _branch_ids(G)
    by_edge: dict[tuple[Any, Any, Any], list[DiameterProfile]] = {}
    for profile in profiles.values():
        by_edge.setdefault(tuple(profile.pericyte.edge), []).append(profile)

    branches: dict[tuple[Any, Any, Any], BranchResistance] = {}
    for edge, on_branch in by_edge.items():
        data = G.edges[edge]
        order = data.get("branch_order")
        length = float(data.get("length") or on_branch[0].pericyte.vessel_length_um)
        centreline = float(on_branch[0].pericyte.vessel_length_um)
        haematocrit = float(data.get("discharge_haematocrit", model.haematocrit))
        basis = edge_diameter_basis(data, model.diameter_basis)
        readings = np.concatenate([p.diameters_um[p.usable] for p in on_branch])
        reference = (reference_diameters or {}).get(edge)
        if reference is None and len(readings):
            reference = float(np.median(readings))
        try:
            diameter, _from_fwhm = baseline_edge_diameter(
                data, str(order), dict(diameter_by_branch_order or {}),
                fwhm_fallback=prefer_edge_fwhm_diameter,
            )
        except ValueError:
            diameter = None
        uniform = constricted = narrowest = None
        if diameter is not None and length > 0:
            uniform = model.resistance_of_uniform_segment(
                length, diameter, haematocrit=haematocrit, diameter_basis=basis
            )
            constricted, narrowest = uniform, diameter
            if reference and centreline > 0:
                count = max(3, int(np.ceil(centreline * INTEGRATION_POINTS_PER_STEP / float(step_um))) + 1)
                grid = np.linspace(0.0, centreline, count)
                widths = diameter * _relative_profile(grid, on_branch, reference)
                # Along the centreline drawn u to v, stretched to the length
                # the branch's resistance is solved over (the two differ by
                # where the ends were snapped to the nodes).
                constricted = profile_resistance(
                    grid * (length / centreline),
                    widths,
                    viscosity_law=model.viscosity_law,
                    haematocrit=haematocrit,
                    diameter_basis=basis,
                )
                narrowest = float(widths.min())
        branches[edge] = BranchResistance(
            branch_id=ids[edge],
            edge=edge,
            branch_order=order,
            length_um=length,
            diameter_um=diameter,
            haematocrit=haematocrit,
            pericytes=tuple(sorted(p.pericyte.number for p in on_branch)),
            resolved_readings=int(len(readings)),
            reference_diameter_um=reference,
            narrowest_diameter_um=narrowest,
            uniform_resistance=uniform,
            constricted_resistance=constricted,
        )
    return branches


def measure_pericytes(
    G: nx.MultiGraph,
    pericyte_mask: np.ndarray,
    voxel_size_zyx: Sequence[float],
    *,
    vessel_mask: np.ndarray | None = None,
    reader: DiameterReader | None = None,
    profiles: bool = True,
    junctions: bool = True,
    step_um: float = DEFAULT_PROFILE_STEP_UM,
    reach_um: float = DEFAULT_PROFILE_REACH_UM,
    max_assignment_distance_um: float | None = 3.0,
    min_pericyte_diameter_um: float | None = 5.0,
    max_pericyte_diameter_um: float | None = 12.0,
    model: PoiseuilleModel | None = None,
    diameter_by_branch_order: Mapping[str, Any] | None = None,
    prefer_edge_fwhm_diameter: bool = False,
    workers: int = 1,
    memmap_directory=None,
) -> PericyteMeasurements:
    """The pericytes of *pericyte_mask* (a volume on *G*'s grid) placed on
    *G*, with their diameter profiles and their branches' resistances
    (*profiles*; the resistances need *model*) and their distances to their
    vessels' nearest junctions (*junctions*). The profiles and the branches'
    reference widths are read by *reader* (:func:`fwhm_reader` for the raw
    image), by default off *vessel_mask*, the segmentation
    (:func:`mask_reader`)."""
    from .pericyte_mask import place_pericytes

    placement = place_pericytes(
        G,
        pericyte_mask,
        voxel_size_zyx,
        max_assignment_distance_um=max_assignment_distance_um,
        min_pericyte_diameter_um=min_pericyte_diameter_um,
        max_pericyte_diameter_um=max_pericyte_diameter_um,
    )
    pericytes = placed_pericytes(
        G, placement,
        max_assignment_distance_um=max_assignment_distance_um,
        min_pericyte_diameter_um=min_pericyte_diameter_um,
        max_pericyte_diameter_um=max_pericyte_diameter_um,
    )
    result = PericyteMeasurements(pericytes=pericytes, step_um=float(step_um), reach_um=float(reach_um))
    if profiles:
        if reader is None:
            if vessel_mask is None:
                raise ValueError(
                    "The pericytes' diameter profiles are read off the run's segmented image, "
                    "and there is none."
                )
            reader = mask_reader(vessel_mask, voxel_size_zyx, workers=workers, memmap_directory=memmap_directory)
        result.profiles = diameter_profiles(G, pericytes, reader, step_um=step_um, reach_um=reach_um)
        if model is not None:
            edges = sorted({tuple(p.pericyte.edge) for p in result.profiles.values()}, key=str)
            result.reference_diameters = branch_reference_diameters(G, edges, reader, step_um=step_um)
            result.branches = branch_resistances(
                G, result.profiles, model=model,
                diameter_by_branch_order=diameter_by_branch_order,
                prefer_edge_fwhm_diameter=prefer_edge_fwhm_diameter,
                step_um=step_um,
                reference_diameters=result.reference_diameters,
            )
    if junctions:
        result.junctions = junction_distances(G, pericytes)
    return result


def apply_measured_constriction(
    G: nx.MultiGraph, branches: Mapping[tuple[Any, Any, Any], BranchResistance]
) -> int:
    """Scale each branch of *branches*' resistance on *G* by its measured-
    constriction ratio (constricted over uniform), and record its pericytes
    where the pericyte layer reads them (``pericyte_count_assigned``,
    ``pericyte_centers_um``). Returns how many branches changed."""
    changed = 0
    for edge, branch in branches.items():
        if branch.ratio is None or not G.has_edge(*edge):
            continue
        data = G.edges[edge]
        resistance = data.get("resistance")
        if resistance is None:
            continue
        set_edge_resistance(data, float(resistance) * float(branch.ratio))
        changed += 1
    return changed


def record_pericyte_sites(G: nx.MultiGraph, pericytes: Sequence[PlacedPericyte]) -> None:
    """Put each placed pericyte where the viewer's pericyte layer reads
    pericytes: ``pericyte_count_assigned`` and ``pericyte_centers_um``, its
    midline's arc length along the edge's centreline drawn ``u`` to ``v``
    (``gui.results.pericyte_points``). Rebinds, never edits in place: a
    perturbation's graph shares its values with the baseline's."""
    sites: dict[tuple[Any, Any, Any], list[float]] = {}
    for pericyte in pericytes:
        if pericyte.placed:
            sites.setdefault(pericyte.edge, []).append(float(pericyte.arc_position_um))
    for edge, centres in sites.items():
        G.edges[edge]["pericyte_count_assigned"] = len(centres)
        G.edges[edge]["pericyte_centers_um"] = sorted(centres)


# -- the CSVs --------------------------------------------------------------------


def _number(value: Any) -> Any:
    """A float for a CSV: blank for None or NaN."""
    if value is None:
        return None
    value = float(value)
    return None if not np.isfinite(value) else value


def _profile_rows(profile: DiameterProfile) -> list[dict[str, Any]]:
    pericyte = profile.pericyte
    u, v, _key = pericyte.edge
    at_pericyte = profile.diameter_at_pericyte_um
    resolved = profile.resolved
    rows = []
    for index, offset in enumerate(profile.offsets_um):
        diameter = float(profile.diameters_um[index])
        known = np.isfinite(diameter)
        compared = known and np.isfinite(at_pericyte) and at_pericyte > 0
        rows.append(
            {
                "offset_um": float(offset),
                "toward_node": "" if offset == 0 else (u if offset < 0 else v),
                "arc_position_um": float(profile.arc_positions_um[index]),
                "z_um": float(profile.points_um[index][0]),
                "y_um": float(profile.points_um[index][1]),
                "x_um": float(profile.points_um[index][2]),
                "diameter_um": diameter if known else None,
                "diameter_method": profile.methods[index],
                "resolved": bool(resolved[index]),
                "diameter_change_um": diameter - at_pericyte if compared else None,
                "diameter_ratio": diameter / at_pericyte if compared else None,
                "distance_to_vessel_end_um": float(
                    min(profile.arc_positions_um[index], pericyte.vessel_length_um - profile.arc_positions_um[index])
                ),
            }
        )
    return rows


def profile_file_name(number: int) -> str:
    """One pericyte's profile CSV, in its run's profile folder."""
    return f"pericyte_{int(number):04d}.csv"


def _branch_row(branch: BranchResistance) -> dict[str, Any]:
    u, v, key = branch.edge
    return {
        "branch_id": branch.branch_id,
        "u": u,
        "v": v,
        "key": key,
        "branch_order": branch.branch_order,
        "length_um": branch.length_um,
        "diameter_um": _number(branch.diameter_um),
        "haematocrit": branch.haematocrit,
        "pericytes": len(branch.pericytes),
        "pericyte_numbers": json.dumps(list(branch.pericytes)),
        "resolved_readings": branch.resolved_readings,
        "reference_diameter_um": _number(branch.reference_diameter_um),
        "narrowest_diameter_um": _number(branch.narrowest_diameter_um),
        "uniform_resistance_pa_s_per_m3": _number(branch.uniform_resistance),
        "constricted_resistance_pa_s_per_m3": _number(branch.constricted_resistance),
        "resistance_ratio": _number(branch.ratio),
    }


def write_branch_csv(branches: Mapping[Any, BranchResistance], path: Path | str) -> Path:
    """*branches*, one row each in branch-ID order, as :data:`BRANCH_COLUMNS`."""
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BRANCH_COLUMNS))
        writer.writeheader()
        for branch in sorted(branches.values(), key=lambda item: item.branch_id):
            writer.writerow(_branch_row(branch))
    return path


def _pericyte_row(measurements: PericyteMeasurements, pericyte: PlacedPericyte) -> dict[str, Any]:
    row: dict[str, Any] = {name: None for name in PERICYTE_COLUMNS}
    row.update(
        {
            "pericyte": pericyte.number,
            "centroid_z_um": pericyte.centroid_um[0],
            "centroid_y_um": pericyte.centroid_um[1],
            "centroid_x_um": pericyte.centroid_um[2],
            "equivalent_diameter_um": pericyte.equivalent_diameter_um,
            "placed": pericyte.placed,
            "reason": pericyte.reason,
            "distance_to_vessel_um": _number(pericyte.distance_to_vessel_um),
        }
    )
    if not pericyte.placed:
        return row
    u, v, key = pericyte.edge
    row.update(
        {
            "branch_id": pericyte.branch_id,
            "u": u,
            "v": v,
            "key": key,
            "branch_order": pericyte.branch_order,
            "arc_position_um": pericyte.arc_position_um,
            "vessel_length_um": pericyte.vessel_length_um,
            "midline_z_um": pericyte.midline_um[0],
            "midline_y_um": pericyte.midline_um[1],
            "midline_x_um": pericyte.midline_um[2],
        }
    )
    profile = measurements.profiles.get(pericyte.number)
    if profile is not None:
        narrowest, offset = profile.narrowest()
        row.update(
            {
                "profile_file": profile_file_name(pericyte.number),
                "profile_readings": len(profile.offsets_um),
                "diameter_at_pericyte_um": _number(profile.diameter_at_pericyte_um),
                "narrowest_diameter_um": _number(narrowest),
                "narrowest_offset_um": _number(offset),
            }
        )
    junction = measurements.junctions.get(pericyte.number)
    if junction is not None:
        row.update(
            {
                "nearest_junction_node": junction.node,
                "junction_end": junction.end,
                "junction_vessels": junction.vessels,
                "junction_distance_along_vessel_um": _number(junction.along_vessel_um),
                "junction_straight_line_distance_um": _number(junction.straight_line_um),
            }
        )
    branch = measurements.branches.get(pericyte.edge)
    if branch is not None:
        row.update(
            {
                "branch_length_um": branch.length_um,
                "branch_diameter_um": _number(branch.diameter_um),
                "branch_haematocrit": branch.haematocrit,
                "branch_pericytes": len(branch.pericytes),
                "branch_uniform_resistance_pa_s_per_m3": _number(branch.uniform_resistance),
                "branch_constricted_resistance_pa_s_per_m3": _number(branch.constricted_resistance),
                "branch_resistance_ratio": _number(branch.ratio),
            }
        )
    return row


def write_pericyte_measurements(
    measurements: PericyteMeasurements, output_dir: Path | str, stem: str
) -> list[Path]:
    """``{stem}_pericytes.csv`` (one row per pericyte of the mask), with
    profiles ``{stem}_pericyte_branches.csv`` and one ``pericyte_<NNNN>.csv``
    per measured pericyte in ``{stem}_pericyte_profiles/``. Returns the files
    written."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    summary_path = output_dir / f"{stem}_pericytes.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PERICYTE_COLUMNS))
        writer.writeheader()
        for pericyte in measurements.pericytes:
            writer.writerow(_pericyte_row(measurements, pericyte))
    written.append(summary_path)
    if measurements.profiles:
        folder = output_dir / f"{stem}{PROFILE_FOLDER_SUFFIX}"
        folder.mkdir(parents=True, exist_ok=True)
        for number, profile in sorted(measurements.profiles.items()):
            path = folder / profile_file_name(number)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(PROFILE_COLUMNS))
                writer.writeheader()
                writer.writerows(_profile_rows(profile))
            written.append(path)
    if measurements.branches:
        written.append(write_branch_csv(measurements.branches, output_dir / f"{stem}_pericyte_branches.csv"))
    return written
