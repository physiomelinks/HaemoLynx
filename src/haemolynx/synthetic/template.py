"""What a synthetic network is matched to: a template read from a finished run.

A :class:`NetworkTemplate` is the general characteristics of one network -- its
box, where its inlets and outlets sit, how many vessels of each class and
branch order it has and how long and wide they are, its angles, Murray
exponent, intercapillary distance, loops and dead ends -- as plain numbers a
generator can draw a new network to and a comparison can score it against.

It is read from a ``.haemorun`` (``gui.run_snapshot``) of an image run in one
of two ways, chosen by the ``synthetic_template_source`` setting:

* ``statistics`` (:func:`template_from_statistics`): the run's saved
  statistics report, the sections ``export_results`` leaves on the graph
  (``pipeline.stages.STATISTICS_REPORT``). A section the run did not compute
  is a :class:`TemplateError` naming the ``statistics_*`` setting to switch on
  before running it again (:data:`SECTION_TOGGLES`).
* ``graph`` (:func:`template_from_graph`): the same sections measured again
  from the graph in the file, plus every vessel's own length, diameter and
  tortuosity (:class:`Samples`), so the generator draws from the run's
  distributions rather than lognormal fits to its means.

Both go through the one reader, :func:`template_from_report`, so the two
sources cannot disagree on what a row means; the comparison on the synthetic
run (``synthetic.compare``) reads the synthetic run's own report with it too.
"""
from __future__ import annotations

import logging
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import networkx as nx
import numpy as np

from haemolynx.graph.boundaries import NETWORK_SOURCE_SETTING, NETWORK_SOURCE_SYNTHETIC
from haemolynx.graph.branch_order import (
    BRANCH_ORDER_CATEGORY_SEQUENCE,
    UNKNOWN_VESSEL_CLASSES,
    flow_path_key,
)
from haemolynx.graph.dead_ends import (
    DEAD_END_KIND,
    LEAVES_IMAGE,
    chain_length_um,
    chain_to_junction,
    graph_image_extent_um,
    node_leaves_image,
)
from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.graph.vessel_size_classes import size_convention_of
from haemolynx.haemodynamics.poiseuille import edge_diameter_basis
from haemolynx.haemodynamics.viscosity import (
    DIAMETER_BASES,
    SEGMENTATION_SOURCE_SETTING,
    normalise_diameter_basis,
)
from haemolynx.statistics.bifurcation import _normalize_branch_order_tag, branch_order_category
from haemolynx.statistics.dead_ends import DENSITY_KINDS, _KIND_ROWS
from haemolynx.statistics.vessel_classes import (
    VESSEL_CLASS_TITLES,
    VESSEL_CLASSES,
    vessel_class_scheme,
)

from .organisation import Organisation, measure_organisation

logger = logging.getLogger(__name__)

#: The two ``synthetic_template_source`` choices.
SOURCE_STATISTICS = "statistics"
SOURCE_GRAPH = "graph"
TEMPLATE_SOURCES: tuple[str, ...] = (SOURCE_STATISTICS, SOURCE_GRAPH)

#: Where ``export_results`` leaves the report on the graph, and the report's
#: branch-order section: ``pipeline.stages.STATISTICS_REPORT`` and
#: ``BRANCH_ORDER_SECTION``, repeated here so this module does not import the
#: stages (which import it); ``tests/test_synthetic_template.py`` pins them.
STATISTICS_REPORT_KEY = "statistics_report"
BRANCH_ORDERS_SECTION = "branch_orders"

#: What the generator writes on its graph: the ``network_source`` the graph
#: came from, and the template it was matched to (``to_dict`` plus the seed
#: and options). A graph carrying the first is refused as a template.
GRAPH_NETWORK_SOURCE = "network_source"
GRAPH_TEMPLATE_RECORD = "synthetic_template"

#: The six node roles a template counts and the generator marks.
ROLES: tuple[str, ...] = (
    "inlet",
    "outlet",
    "arteriole_boundary",
    "venule_boundary",
    "large_arteriole_boundary",
    "large_venule_boundary",
)

#: The report sections the ``statistics`` source needs, and the settings that
#: switch each on (the network-analysis ones sit under
#: ``statistics_network_analysis`` as well as their own toggle).
REQUIRED_SECTIONS: tuple[str, ...] = (
    "basic",
    "vessel_density",
    "vessel_classes",
    BRANCH_ORDERS_SECTION,
    "tortuosity",
    "murray_law",
    "daughter_angles",
    "intercapillary_distance",
    "dead_ends",
    "loop_hierarchy",
)
#: The layout sections (``statistics.layout``): read from the report when it
#: has them, else measured off the run's graph (a report saved before
#: 2026-10-25 has none), so never required.
LAYOUT_SECTIONS: tuple[str, ...] = ("junction_shape", "tissue_coverage", "vessel_columns")
#: Sections read when present, never required.
OPTIONAL_SECTIONS: tuple[str, ...] = ("branching", "cyclomatic_number") + LAYOUT_SECTIONS
SECTION_TOGGLES: dict[str, tuple[str, ...]] = {
    "basic": ("statistics",),
    BRANCH_ORDERS_SECTION: ("statistics",),
    "tortuosity": ("statistics_tortuosity",),
    "branching": ("statistics_branching",),
    "vessel_density": ("statistics_vessel_density",),
    "vessel_classes": ("statistics_vessel_classes",),
    "murray_law": ("statistics_murray_law",),
    "daughter_angles": ("statistics_daughter_angles",),
    "intercapillary_distance": ("statistics_intercapillary_distance",),
    "dead_ends": ("statistics_dead_ends",),
    "junction_shape": ("statistics_junction_shape",),
    "tissue_coverage": ("statistics_tissue_coverage",),
    "vessel_columns": ("statistics_vessel_columns",),
    "loop_hierarchy": ("statistics_network_analysis", "statistics_loop_hierarchy"),
    "cyclomatic_number": ("statistics_network_analysis", "statistics_cyclomatic_number"),
}

#: What the generator falls back on when the template cannot say.
DEFAULT_MURRAY_EXPONENT = 3.0
#: The daughter-daughter angle of a symmetric bifurcation under Murray's law.
DEFAULT_DAUGHTER_ANGLE_DEG = 75.0
DEFAULT_DAUGHTER_ANGLE_SD_DEG = 15.0
#: Coefficient of variation taken for a diameter the template gives no spread for.
DEFAULT_DIAMETER_CV = 0.3

#: The template's faces: axis letter and which end, ``"z0"`` ... ``"x1"``.
FACES: tuple[str, ...] = ("z0", "y0", "x0", "z1", "y1", "x1")
INTERIOR = "interior"

#: The ``synthetic_scale_axes`` choices: the axes ``synthetic_scale`` grows
#: the box along. ``xy`` keeps the template's depth (a wider field of the same
#: tissue), ``xyz`` grows all three.
SCALE_AXES_XYZ = "xyz"
SCALE_AXES_XY = "xy"
SCALE_AXES: tuple[str, ...] = (SCALE_AXES_XYZ, SCALE_AXES_XY)

_SIDES = {"arterial": ("large_arteriole", "arteriole"), "venous": ("large_venule", "venule")}


class TemplateError(ValueError):
    """The ``.haemorun`` cannot be a template, and why.

    ``reason`` is one of ``not_a_run``, ``synthetic_run``, ``no_branch_orders``,
    ``global_classes``, ``no_boundaries``, ``no_diameters`` or
    ``no_statistics``; for the last, ``missing_statistics`` names the
    ``statistics_*`` settings to switch on.
    """

    def __init__(self, message: str, *, reason: str, missing_statistics: Sequence[str] = ()):
        super().__init__(message)
        self.reason = reason
        self.missing_statistics = tuple(missing_statistics)


# ---------------------------------------------------------------------------
# The data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Endpoint:
    """One inlet or outlet: the face it sits on (:data:`FACES`, or
    :data:`INTERIOR`) and its position as a fraction of the extent, (z, y, x)."""

    face: str
    position_rel: tuple[float, float, float]


@dataclass(frozen=True)
class Samples:
    """Per-vessel values read off a template graph (the ``graph`` source)."""

    lengths_um: tuple[float, ...] = ()
    diameters_um: tuple[float, ...] = ()
    tortuosities: tuple[float, ...] = ()


@dataclass(frozen=True)
class ClassTarget:
    """One vessel class (a category of :data:`BRANCH_ORDER_CATEGORY_SEQUENCE`)."""

    category: str
    count: int
    total_length_um: float
    mean_length_um: Optional[float]
    median_length_um: Optional[float]
    mean_diameter_um: Optional[float]
    median_diameter_um: Optional[float]
    mean_tortuosity: Optional[float]
    median_tortuosity: Optional[float]
    samples: Optional[Samples] = None
    #: The class's 90th percentile tortuosity: how far its most winding
    #: tenth wander (the report's row, else read off the graph).
    p90_tortuosity: Optional[float] = None


@dataclass(frozen=True)
class OrderTarget:
    """One branch-order label (``Large_Art1``, ``Art3``, ``BO2``, ``Ven1`` ...)."""

    label: str
    category: str
    index: int
    edge_count: int
    mean_length_um: Optional[float]
    mean_tortuosity: Optional[float]
    mean_emergence_angle_deg: Optional[float]
    mean_diameter_um: Optional[float]
    diameter_cv: Optional[float]
    samples: Optional[Samples] = None


@dataclass(frozen=True)
class LoopTarget:
    independent_loops: Optional[int]
    independent_loops_per_mm3: Optional[float]
    #: ``{"P10", "P25", "Median", "P75", "P90"}`` -> um, those the report had.
    length_percentiles_um: dict = field(default_factory=dict)
    vessels_on_no_loop_fraction: Optional[float] = None
    loop_length_samples_um: tuple[float, ...] = ()


@dataclass(frozen=True)
class DeadEndTarget:
    total: int
    #: Dead ends at the image faces (inlets and outlets on a face included).
    leaves_image: int
    #: Ends inside the tissue that could be the tissue's own: the kinds a
    #: dead-end density counts (``statistics.dead_ends.DENSITY_KINDS``).
    interior: int
    artefacts: int
    island_ends: int
    interior_per_mm: Optional[float]
    chain_length_median_um: Optional[float]
    chain_length_p10_um: Optional[float]
    chain_length_p90_um: Optional[float]
    chain_length_samples_um: tuple[float, ...] = ()


@dataclass(frozen=True)
class LayoutTarget:
    """How the template's vessels lie (``statistics.layout``; Harvey,
    2026-10-25): its junctions' shape, how far its tissue is from a vessel,
    and its columns along z. Each the report's row of that name."""

    junction_smallest_angle_median_deg: Optional[float] = None
    junction_smallest_angle_p25_deg: Optional[float] = None
    junction_largest_angle_median_deg: Optional[float] = None
    acute_junction_share: Optional[float] = None
    junction_angle_sum_median_deg: Optional[float] = None
    box_within_20_share: Optional[float] = None
    box_beyond_50_share: Optional[float] = None
    distance_median_um: Optional[float] = None
    distance_p90_um: Optional[float] = None
    column_length_share: Optional[float] = None
    columns_per_mm2: Optional[float] = None
    column_spacing_median_um: Optional[float] = None
    column_clark_evans: Optional[float] = None
    column_tortuosity_median: Optional[float] = None


#: Each :class:`LayoutTarget` field's section and row of the report.
LAYOUT_ROWS: dict[str, tuple[str, str]] = {
    "junction_smallest_angle_median_deg": ("junction_shape", "Junction Smallest Angle Median (degrees)"),
    "junction_smallest_angle_p25_deg": ("junction_shape", "Junction Smallest Angle 25th Percentile (degrees)"),
    "junction_largest_angle_median_deg": ("junction_shape", "Junction Largest Angle Median (degrees)"),
    "acute_junction_share": ("junction_shape", "Acute Junction Share (fraction)"),
    "junction_angle_sum_median_deg": ("junction_shape", "Junction Angle Sum Median (degrees)"),
    "box_within_20_share": ("tissue_coverage", "Box Within 20 um of a Vessel (fraction)"),
    "box_beyond_50_share": ("tissue_coverage", "Box Beyond 50 um of a Vessel (fraction)"),
    "distance_median_um": ("tissue_coverage", "Median Distance to Nearest Vessel (microns)"),
    "distance_p90_um": ("tissue_coverage", "90th Percentile Distance to Nearest Vessel (microns)"),
    "column_length_share": ("vessel_columns", "Column Share of Vessel Length (fraction)"),
    "columns_per_mm2": ("vessel_columns", "Columns per mm² of Image Plane"),
    "column_spacing_median_um": ("vessel_columns", "Column Spacing Median (microns)"),
    "column_clark_evans": ("vessel_columns", "Column Spacing Clark-Evans Ratio"),
    "column_tortuosity_median": ("vessel_columns", "Median Column Tortuosity Index"),
}


@dataclass(frozen=True)
class Provenance:
    template_path: Optional[str]
    source: str
    haemolynx_version: Optional[str]
    statistics_mode: Optional[str]
    run_version: Optional[str]
    run_commit: Optional[str]
    #: Every fallback taken, in words.
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class NetworkTemplate:
    """The characteristics a synthetic network is drawn to. See the module."""

    extent_um: tuple[float, float, float]
    voxel_size_zyx: tuple[float, float, float]
    image_shape: tuple[int, int, int]
    inlets: tuple
    outlets: tuple
    arteriole_boundary_count: int
    venule_boundary_count: int
    large_arteriole_boundary_count: int
    large_venule_boundary_count: int
    #: ``statistics.vessel_classes.vessel_class_scheme``: hierarchical or capillary.
    scheme: str
    classes: dict
    orders: tuple
    total_length_um: float
    node_count: Optional[int]
    edge_count: Optional[int]
    average_degree: Optional[float]
    branching_points: Optional[int]
    mean_tortuosity: Optional[float]
    daughter_angle_deg: Optional[float]
    daughter_angle_sd_deg: Optional[float]
    murray_exponent: Optional[float]
    murray_iqr: Optional[tuple]
    intercapillary_distance_um: Optional[float]
    intercapillary_distance_median_um: Optional[float]
    loops: LoopTarget
    dead_ends: DeadEndTarget
    tissue_volume_um3: Optional[float]
    provenance: Provenance
    #: What the template's diameters measure (``viscosity.DIAMETER_BASES``):
    #: the basis most of its vessels' viscosity was read on
    #: (:func:`template_diameter_basis`). Every drawn vessel carries it, so its
    #: viscosity takes the template's form of the law; None when the template
    #: run records none (a template saved before this was read).
    diameter_basis: Optional[str] = None
    #: Where the vessels lie and which way the capillaries run
    #: (``synthetic.organisation``), read off the graph; None for a template
    #: written before 2026-10-25 or one with no capillaries.
    organisation: Optional[Organisation] = None
    #: Its junctions' shape, coverage and columns (:class:`LayoutTarget`);
    #: None for a template written before 2026-10-25.
    layout: Optional[LayoutTarget] = None

    # -- reading -----------------------------------------------------------

    @property
    def image_volume_um3(self) -> float:
        """``prod(shape * voxel)``, the report's "Total Image Volume"."""
        return float(np.prod(np.asarray(self.image_shape, float) * np.asarray(self.voxel_size_zyx, float)))

    @property
    def vessel_count(self) -> int:
        return int(sum(target.count for target in self.classes.values()))

    @property
    def inlet_count(self) -> int:
        return len(self.inlets)

    @property
    def outlet_count(self) -> int:
        return len(self.outlets)

    def class_target(self, category: str) -> Optional[ClassTarget]:
        return self.classes.get(category)

    def orders_on(self, side: str) -> tuple[tuple, tuple]:
        """``(large tier, small tier)`` of the ``"arterial"`` or ``"venous"``
        side, each in generation order from the root (index ascending: Ven
        orders are numbered from the outlet, so Ven1 is the root's)."""
        large, small = _SIDES[side]
        by_index = sorted(self.orders, key=lambda o: o.index)
        return (
            tuple(o for o in by_index if o.category == large),
            tuple(o for o in by_index if o.category == small),
        )

    def summary(self) -> dict:
        """A few numbers for a log line or a manifest."""
        return {
            "template_path": self.provenance.template_path,
            "source": self.provenance.source,
            "extent_um": list(self.extent_um),
            "voxel_size_zyx": list(self.voxel_size_zyx),
            "image_shape": list(self.image_shape),
            "scheme": self.scheme,
            "diameter_basis": self.diameter_basis,
            "vessels": self.vessel_count,
            "total_length_um": self.total_length_um,
            "inlets": self.inlet_count,
            "outlets": self.outlet_count,
            "orders": [o.label for o in self.orders],
            "notes": list(self.provenance.notes),
        }

    # -- scaling -----------------------------------------------------------

    def scaled(self, factor: float, axes: str = SCALE_AXES_XYZ) -> "NetworkTemplate":
        """The same network in a box *factor* times as wide along *axes*
        (:data:`SCALE_AXES`): counts and total lengths go as the volume
        (``factor**3``, or ``factor**2`` along x and y), inlets and outlets as
        the area of the face each is on (each face keeping at least one it
        had), dead ends leaving the image as the box's surface, every
        per-vessel figure and density stays."""
        factors = scale_factors_zyx(factor, axes)
        if factors == (1.0, 1.0, 1.0):
            return self
        volume = float(np.prod(factors))
        surface = _surface_factor(self.extent_um, factors)
        extent = tuple(float(e) * f for e, f in zip(self.extent_um, factors))
        shape = tuple(int(round((s - 1) * f)) + 1 for s, f in zip(self.image_shape, factors))
        along = "" if axes == SCALE_AXES_XYZ else " along x and y, its depth kept"
        faces = _face_factors(factors, surface)
        return replace(
            self._with_volume(volume, surface, f"scaled by {float(factor):g} from the template's box{along}"),
            extent_um=extent,
            image_shape=shape,
            inlets=_scaled_endpoints(self.inlets, faces),
            outlets=_scaled_endpoints(self.outlets, faces),
        )

    def around_hole(self, share: float) -> "NetworkTemplate":
        """The same network with *share* of its box a hole it does not fill
        (``synthetic.hole``: the image's own box, in a surround): counts and
        total lengths go as the volume left, its inlets, outlets and the dead
        ends at its faces stay, every per-vessel figure and density stays."""
        share = float(share)
        if not (math.isfinite(share) and 0.0 <= share < 1.0):
            raise ValueError(f"a hole's share of the box must be from 0 to under 1, got {share!r}")
        if share == 0.0:
            return self
        return self._with_volume(1.0 - share, 1.0, f"{share:.0%} of the box left to a hole")

    def _with_volume(self, volume: float, surface: float, note: str) -> "NetworkTemplate":
        """Counts and total lengths times *volume*, the dead ends leaving the
        image times *surface*, and *note* added to the provenance."""
        classes = {
            key: replace(
                target,
                count=_scaled_count(target.count, volume),
                total_length_um=target.total_length_um * volume,
            )
            for key, target in self.classes.items()
        }
        orders = tuple(replace(o, edge_count=_scaled_count(o.edge_count, volume)) for o in self.orders)
        loops = replace(
            self.loops,
            independent_loops=(
                None if self.loops.independent_loops is None else _scaled_count(self.loops.independent_loops, volume)
            ),
        )
        parts = {
            "leaves_image": _scaled_count(self.dead_ends.leaves_image, surface),
            "interior": _scaled_count(self.dead_ends.interior, volume),
            "artefacts": _scaled_count(self.dead_ends.artefacts, volume),
            "island_ends": _scaled_count(self.dead_ends.island_ends, volume),
        }
        dead_ends = replace(self.dead_ends, total=sum(parts.values()), **parts)
        return replace(
            self,
            arteriole_boundary_count=_scaled_count(self.arteriole_boundary_count, volume),
            venule_boundary_count=_scaled_count(self.venule_boundary_count, volume),
            large_arteriole_boundary_count=_scaled_count(self.large_arteriole_boundary_count, volume),
            large_venule_boundary_count=_scaled_count(self.large_venule_boundary_count, volume),
            classes=classes,
            orders=orders,
            total_length_um=self.total_length_um * volume,
            node_count=None if self.node_count is None else _scaled_count(self.node_count, volume),
            edge_count=None if self.edge_count is None else _scaled_count(self.edge_count, volume),
            branching_points=(
                None if self.branching_points is None else _scaled_count(self.branching_points, volume)
            ),
            loops=loops,
            dead_ends=dead_ends,
            tissue_volume_um3=None if self.tissue_volume_um3 is None else self.tissue_volume_um3 * volume,
            provenance=replace(self.provenance, notes=self.provenance.notes + (note,)),
        )

    # -- JSON --------------------------------------------------------------

    def to_dict(self) -> dict:
        """Plain JSON types only (tuples become lists; NaN becomes null)."""
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NetworkTemplate":
        return _template_from_dict(data)

    def to_json(self, path: Path | str) -> Path:
        import json

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return dest

    @classmethod
    def from_json(cls, path: Path | str) -> "NetworkTemplate":
        import json

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _scaled_count(count: int, factor: float) -> int:
    return int(round(int(count) * factor)) if count else 0


def scale_factors_zyx(factor: float, axes: str = SCALE_AXES_XYZ) -> tuple[float, float, float]:
    """The box's growth along z, y and x: *factor* along the axes *axes*
    names (:data:`SCALE_AXES`), 1 along the others."""
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"scale factor must be positive, got {factor!r}")
    if axes not in SCALE_AXES:
        raise ValueError(f"scale axes must be one of {', '.join(SCALE_AXES)}, not {axes!r}")
    return tuple(factor if letter in axes else 1.0 for letter in "zyx")


def _surface_factor(extent_um: Sequence[float], factors: Sequence[float]) -> float:
    """How much the box's surface grows: each pair of faces' area weighted
    by its own growth (a box with no surface grows as a cube's face would)."""
    fz, fy, fx = (float(f) for f in factors)
    if fz == fy == fx:
        return fz**2
    ez, ey, ex = (float(e) for e in extent_um)
    areas = np.array([ey * ex, ez * ex, ez * ey])
    if areas.sum() <= 0:
        return (fz * fy * fx) ** (2.0 / 3.0)
    return float((areas * np.array([fy * fx, fz * fx, fz * fy])).sum() / areas.sum())


def _face_factors(factors: Sequence[float], surface: float) -> dict[str, float]:
    """Each face's growth (:data:`FACES`: the product of the other two axes'
    factors), and an endpoint inside the box's, the surface's."""
    fz, fy, fx = (float(f) for f in factors)
    growth = {"z": fy * fx, "y": fz * fx, "x": fz * fy}
    return {**{face: growth[face[0]] for face in FACES}, INTERIOR: surface}


def _scaled_endpoints(endpoints: Sequence[Endpoint], factors: Mapping[str, float]) -> tuple:
    """Each face's endpoints times its growth (*factors*, by face), never
    below one on a face the template had one on, cycling through the ones
    there are."""
    if not endpoints:
        return ()
    wanted = {
        face: max(1, int(round(count * factors.get(face, factors[INTERIOR]))))
        for face, count in Counter(end.face for end in endpoints).items()
    }
    total = sum(wanted.values())
    out: list[Endpoint] = []
    i = 0
    while len(out) < total:
        end = endpoints[i % len(endpoints)]
        if wanted[end.face]:
            out.append(end)
            wanted[end.face] -= 1
        i += 1
    return tuple(out)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _samples_from_dict(data: Any) -> Optional[Samples]:
    if data is None:
        return None
    return Samples(
        lengths_um=tuple(float(v) for v in data.get("lengths_um", ())),
        diameters_um=tuple(float(v) for v in data.get("diameters_um", ())),
        tortuosities=tuple(float(v) for v in data.get("tortuosities", ())),
    )


def _opt_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _template_from_dict(data: Mapping[str, Any]) -> NetworkTemplate:
    classes = {
        key: ClassTarget(
            category=item["category"],
            count=int(item["count"]),
            total_length_um=float(item["total_length_um"]),
            mean_length_um=_opt_float(item.get("mean_length_um")),
            median_length_um=_opt_float(item.get("median_length_um")),
            mean_diameter_um=_opt_float(item.get("mean_diameter_um")),
            median_diameter_um=_opt_float(item.get("median_diameter_um")),
            mean_tortuosity=_opt_float(item.get("mean_tortuosity")),
            median_tortuosity=_opt_float(item.get("median_tortuosity")),
            samples=_samples_from_dict(item.get("samples")),
            p90_tortuosity=_opt_float(item.get("p90_tortuosity")),
        )
        for key, item in data["classes"].items()
    }
    orders = tuple(
        OrderTarget(
            label=item["label"],
            category=item["category"],
            index=int(item["index"]),
            edge_count=int(item["edge_count"]),
            mean_length_um=_opt_float(item.get("mean_length_um")),
            mean_tortuosity=_opt_float(item.get("mean_tortuosity")),
            mean_emergence_angle_deg=_opt_float(item.get("mean_emergence_angle_deg")),
            mean_diameter_um=_opt_float(item.get("mean_diameter_um")),
            diameter_cv=_opt_float(item.get("diameter_cv")),
            samples=_samples_from_dict(item.get("samples")),
        )
        for item in data["orders"]
    )
    loops_data = data["loops"]
    loops = LoopTarget(
        independent_loops=None if loops_data.get("independent_loops") is None else int(loops_data["independent_loops"]),
        independent_loops_per_mm3=_opt_float(loops_data.get("independent_loops_per_mm3")),
        length_percentiles_um={k: float(v) for k, v in (loops_data.get("length_percentiles_um") or {}).items()},
        vessels_on_no_loop_fraction=_opt_float(loops_data.get("vessels_on_no_loop_fraction")),
        loop_length_samples_um=tuple(float(v) for v in loops_data.get("loop_length_samples_um", ())),
    )
    ends = data["dead_ends"]
    dead_ends = DeadEndTarget(
        total=int(ends["total"]),
        leaves_image=int(ends["leaves_image"]),
        interior=int(ends["interior"]),
        artefacts=int(ends.get("artefacts", 0)),
        island_ends=int(ends.get("island_ends", 0)),
        interior_per_mm=_opt_float(ends.get("interior_per_mm")),
        chain_length_median_um=_opt_float(ends.get("chain_length_median_um")),
        chain_length_p10_um=_opt_float(ends.get("chain_length_p10_um")),
        chain_length_p90_um=_opt_float(ends.get("chain_length_p90_um")),
        chain_length_samples_um=tuple(float(v) for v in ends.get("chain_length_samples_um", ())),
    )
    prov = data["provenance"]
    provenance = Provenance(
        template_path=prov.get("template_path"),
        source=prov["source"],
        haemolynx_version=prov.get("haemolynx_version"),
        statistics_mode=prov.get("statistics_mode"),
        run_version=prov.get("run_version"),
        run_commit=prov.get("run_commit"),
        notes=tuple(prov.get("notes", ())),
    )
    # A template written before 2026-10-24 carries a "vessel_box_um" (the
    # box its nodes spanned), dropped since: the organisation's per-cell
    # fields replaced it. It is read past, like any other key.
    basis = normalise_diameter_basis(data.get("diameter_basis"))
    organisation = data.get("organisation")
    layout = data.get("layout")
    return NetworkTemplate(
        extent_um=tuple(float(v) for v in data["extent_um"]),
        voxel_size_zyx=tuple(float(v) for v in data["voxel_size_zyx"]),
        image_shape=tuple(int(v) for v in data["image_shape"]),
        inlets=tuple(Endpoint(e["face"], tuple(float(v) for v in e["position_rel"])) for e in data["inlets"]),
        outlets=tuple(Endpoint(e["face"], tuple(float(v) for v in e["position_rel"])) for e in data["outlets"]),
        arteriole_boundary_count=int(data["arteriole_boundary_count"]),
        venule_boundary_count=int(data["venule_boundary_count"]),
        large_arteriole_boundary_count=int(data["large_arteriole_boundary_count"]),
        large_venule_boundary_count=int(data["large_venule_boundary_count"]),
        scheme=data["scheme"],
        classes=classes,
        orders=orders,
        total_length_um=float(data["total_length_um"]),
        node_count=None if data.get("node_count") is None else int(data["node_count"]),
        edge_count=None if data.get("edge_count") is None else int(data["edge_count"]),
        average_degree=_opt_float(data.get("average_degree")),
        branching_points=None if data.get("branching_points") is None else int(data["branching_points"]),
        mean_tortuosity=_opt_float(data.get("mean_tortuosity")),
        daughter_angle_deg=_opt_float(data.get("daughter_angle_deg")),
        daughter_angle_sd_deg=_opt_float(data.get("daughter_angle_sd_deg")),
        murray_exponent=_opt_float(data.get("murray_exponent")),
        murray_iqr=None if data.get("murray_iqr") is None else tuple(float(v) for v in data["murray_iqr"]),
        intercapillary_distance_um=_opt_float(data.get("intercapillary_distance_um")),
        intercapillary_distance_median_um=_opt_float(data.get("intercapillary_distance_median_um")),
        loops=loops,
        dead_ends=dead_ends,
        tissue_volume_um3=_opt_float(data.get("tissue_volume_um3")),
        provenance=provenance,
        diameter_basis=basis if basis in DIAMETER_BASES else None,
        organisation=None if organisation is None else Organisation.from_dict(organisation),
        layout=None if layout is None else LayoutTarget(**{k: _opt_float(layout.get(k)) for k in LAYOUT_ROWS}),
    )


# ---------------------------------------------------------------------------
# Distribution fits from summary numbers
# ---------------------------------------------------------------------------


def lognormal_from_mean_median(mean: float, median: float) -> tuple[float, float]:
    """``(mu, sigma)`` of the lognormal with this *mean* and *median*:
    ``mu = ln(median)``, ``sigma^2 = 2 ln(mean / median)``. A median at or
    above the mean has no such fit; the spread is then a nominal 0.1."""
    mean, median = float(mean), float(median)
    if not (mean > 0 and median > 0):
        raise ValueError(f"mean and median must be positive, got {mean!r}, {median!r}")
    if median >= mean:
        return math.log(median), 0.1
    return math.log(median), math.sqrt(2.0 * math.log(mean / median))


def lognormal_from_mean_cv(mean: float, cv: float) -> tuple[float, float]:
    """``(mu, sigma)`` of the lognormal with this *mean* and coefficient of
    variation: ``sigma^2 = ln(1 + cv^2)``, ``mu = ln(mean) - sigma^2 / 2``."""
    mean, cv = float(mean), float(cv)
    if mean <= 0:
        raise ValueError(f"mean must be positive, got {mean!r}")
    sigma2 = math.log1p(max(cv, 0.0) ** 2)
    return math.log(mean) - sigma2 / 2.0, math.sqrt(sigma2)


def lognormal_from_median_p10_p90(median: float, p10: float, p90: float) -> tuple[float, float]:
    """``(mu, sigma)`` of the lognormal with this *median* and 10th/90th
    percentiles: ``sigma = (ln p90 - ln p10) / (2 * 1.28155)``."""
    median, p10, p90 = float(median), float(p10), float(p90)
    if not (median > 0 and p10 > 0 and p90 > 0):
        raise ValueError("median and percentiles must be positive")
    if p90 <= p10:
        return math.log(median), 0.1
    return math.log(median), (math.log(p90) - math.log(p10)) / (2.0 * 1.2815515655446004)


# ---------------------------------------------------------------------------
# Reading the snapshot
# ---------------------------------------------------------------------------


def _number(value: Any) -> Optional[float]:
    """A finite number, or None for an ``"N/A (...)"`` string or nothing."""
    if value is None or isinstance(value, (bool, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _has_branch_orders(G: Any) -> bool:
    return isinstance(G, nx.Graph) and any(
        d.get("branch_order") is not None for *_e, d in G.edges(data=True) if not d.get(IS_ZERO_RESISTANCE)
    )


def template_graph(snapshot: Any) -> tuple[Optional[nx.MultiGraph], Optional[Any]]:
    """The graph a template is read from, and the checkpoint its boundary
    lists come from: the ``export_results`` checkpoint's graph (it carries the
    report, the diameters, the orders and the dead-end kinds), else the last
    checkpoint with branch orders, else the results' graph."""
    checkpoints = list(getattr(snapshot, "checkpoints", ()) or ())
    for checkpoint in checkpoints:
        if getattr(checkpoint, "stage", None) == "export_results" and isinstance(checkpoint.graph, nx.Graph):
            return checkpoint.graph, checkpoint
    for checkpoint in reversed(checkpoints):
        if _has_branch_orders(getattr(checkpoint, "graph", None)):
            return checkpoint.graph, checkpoint
    state = getattr(snapshot, "results_state", None) or {}
    graph = state.get("graph") if isinstance(state, Mapping) else None
    if isinstance(graph, nx.Graph):
        return graph, (checkpoints[-1] if checkpoints else None)
    return None, None


def _run_settings(snapshot: Any) -> Mapping[str, Any]:
    state = getattr(snapshot, "results_state", None) or {}
    settings = state.get("settings") if isinstance(state, Mapping) else None
    if isinstance(settings, Mapping) and settings:
        return settings
    return getattr(snapshot, "settings", None) or {}


def boundary_nodes_of(snapshot: Any, checkpoint: Any) -> dict[str, tuple]:
    """The six role lists, from the checkpoint (carried forward from the
    Boundaries stage) else the run's settings (filled while it ran)."""
    settings = _run_settings(snapshot)
    out: dict[str, tuple] = {}
    for role in ROLES:
        name = f"{role}_nodes"
        nodes = getattr(checkpoint, name, None) if checkpoint is not None else None
        if not nodes:
            nodes = settings.get(name) or ()
        out[role] = tuple(nodes)
    return out


def run_diameter_basis_of(G: nx.Graph, settings: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """The template run's own diameter basis, the one its unmarked vessels
    are on: what its haemodynamics recorded on the graph
    (``G.graph["diameter_basis"]``), else what its settings say
    (``segmentation_source``, or an older run's ``diameter_basis``); None
    when neither says."""
    settings = settings or {}
    for value in (G.graph.get("diameter_basis"), *(settings.get(k) for k in (SEGMENTATION_SOURCE_SETTING, "diameter_basis"))):
        basis = normalise_diameter_basis(value)
        if basis in DIAMETER_BASES:
            return basis
    return None


def template_diameter_basis(G: nx.Graph, run_basis: Optional[str], notes: list[str]) -> Optional[str]:
    """The basis the template's diameters are on: each vessel's own, as its
    viscosity read it (``poiseuille.edge_diameter_basis``: its mark, else
    *run_basis*), and of those the one most vessels are on (*run_basis* on a
    tie). A template on two bases says so in *notes*; one whose vessels and
    run name none is *run_basis* (None) with a note."""
    counts: Counter = Counter()
    for *_e, d in G.edges(data=True):
        if d.get(IS_ZERO_RESISTANCE) or not _number(d.get("diameter_um")):
            continue
        basis = edge_diameter_basis(d, run_basis)
        if basis in DIAMETER_BASES:
            counts[basis] += 1
    if not counts:
        notes.append(
            "the template run records no diameter basis; the drawn vessels are on this run's segmentation_source"
        )
        return run_basis
    basis = min(counts, key=lambda b: (-counts[b], b != run_basis, b))
    if len(counts) > 1:
        notes.append(
            "the template's diameters are on two bases ("
            + ", ".join(f"{counts[b]} {b}" for b in sorted(counts))
            + f"); every drawn vessel is on {basis}, the one most are on"
        )
    return basis


def check_template_run(snapshot: Any, G: Optional[nx.Graph], boundaries: Mapping[str, Sequence]) -> None:
    """Raise :class:`TemplateError` unless the run can be a template."""
    if G is None or G.number_of_edges() == 0:
        raise TemplateError(
            "The file holds no network with branch orders: run the template "
            "through its Boundaries stage before using it as a template.",
            reason="not_a_run",
        )
    settings = _run_settings(snapshot)
    if (
        G.graph.get(GRAPH_NETWORK_SOURCE) == NETWORK_SOURCE_SYNTHETIC
        or settings.get(NETWORK_SOURCE_SETTING) == NETWORK_SOURCE_SYNTHETIC
    ):
        raise TemplateError(
            "The template run is itself a synthetic network: choose the "
            "microscopy-image run it was generated from.",
            reason="synthetic_run",
        )
    labels = [d.get("branch_order") for *_e, d in G.edges(data=True) if not d.get(IS_ZERO_RESISTANCE)]
    scheme = vessel_class_scheme(labels)
    if scheme == "global" or any(str(label) in UNKNOWN_VESSEL_CLASSES for label in labels if label is not None):
        how = (
            "by a size convention (classify_vessels_by_size)"
            if size_convention_of(G) is not None
            else "by size alone (global vessel masks: Unknown_Large/Small/Cap)"
        )
        raise TemplateError(
            f"The template run classed its vessels {how}; the generator needs arteriole "
            "and venule branch orders. Re-run it with arteriole and venule masks or "
            "manual inlets, outlets and boundary nodes.",
            reason="global_classes",
        )
    if scheme == "none":
        raise TemplateError(
            "The template run assigned no branch orders: run it through its "
            "Boundaries stage first.",
            reason="no_branch_orders",
        )
    if not boundaries.get("inlet") or not boundaries.get("outlet"):
        raise TemplateError(
            "The template run chose no inlets or outlets, so the synthetic "
            "network would have none: assign them in the template run first.",
            reason="no_boundaries",
        )
    if not any(_number(d.get("diameter_um")) for *_e, d in G.edges(data=True) if not d.get(IS_ZERO_RESISTANCE)):
        raise TemplateError(
            "No vessel of the template run has a diameter: run its Diameters stage first.",
            reason="no_diameters",
        )


def _face_of(position: np.ndarray, extent: np.ndarray, tolerance: np.ndarray) -> str:
    both = np.concatenate([position, extent - position])
    nearest = int(np.argmin(both))
    return FACES[nearest] if both[nearest] <= tolerance[nearest % 3] else INTERIOR


def endpoints_from_nodes(
    G: nx.Graph,
    nodes: Iterable[Any],
    extent_um: Sequence[float],
    voxel_size_zyx: Sequence[float],
) -> tuple:
    """Each node of *nodes* as an :class:`Endpoint`: on a face when within
    two of the finest voxels, or one percent of that axis, of it."""
    extent = np.asarray(extent_um, dtype=float)
    voxel = np.asarray(voxel_size_zyx, dtype=float)
    tolerance = np.maximum(2.0 * float(voxel.min()), 0.01 * extent)
    out = []
    for node in nodes:
        if node not in G or G.nodes[node].get("pos") is None:
            continue
        position = np.asarray(G.nodes[node]["pos"], dtype=float)[:3]
        rel = np.where(extent > 0, position / np.where(extent > 0, extent, 1.0), 0.0)
        out.append(Endpoint(_face_of(position, extent, tolerance), tuple(float(np.clip(v, 0.0, 1.0)) for v in rel)))
    return tuple(out)


def _graph_box(G: nx.Graph, checkpoint: Any, notes: list[str]) -> tuple[np.ndarray, np.ndarray, tuple, Optional[tuple]]:
    """``(extent, voxel, shape, node box)`` of the template graph."""
    positions = np.asarray(
        [G.nodes[n]["pos"] for n in G.nodes if G.nodes[n].get("pos") is not None], dtype=float
    ).reshape(-1, 3)
    box = None
    if len(positions):
        box = (tuple(float(v) for v in positions.min(axis=0)), tuple(float(v) for v in positions.max(axis=0)))
    voxel = G.graph.get("image_voxel_size_zyx") or G.graph.get("voxel_size")
    if voxel is None and checkpoint is not None:
        voxel = getattr(checkpoint, "voxel_size_zyx", None)
    if voxel is None:
        voxel = (1.0, 1.0, 1.0)
        notes.append("the template records no voxel size; 1 um voxels assumed")
    voxel = np.asarray(voxel, dtype=float)[:3]
    extent = graph_image_extent_um(G)
    if extent is None:
        if box is None:
            raise TemplateError("The template graph has no node positions.", reason="not_a_run")
        extent = np.asarray(box[1], dtype=float)
        notes.append("the template records no image extent; the box its nodes span is used")
    shape = tuple(int(round(float(e) / float(v))) + 1 for e, v in zip(extent, voxel))
    return extent, voxel, shape, box


def _read_classes(section: Optional[Mapping[str, Any]], notes: list[str]) -> dict:
    rows = (section or {}).get(VESSEL_CLASSES) if section else None
    out: dict[str, ClassTarget] = {}
    if not isinstance(rows, Mapping):
        return out
    for category in BRANCH_ORDER_CATEGORY_SEQUENCE:
        item = rows.get(VESSEL_CLASS_TITLES[category])
        if not isinstance(item, Mapping):
            continue
        count = int(_number(item.get("Vessel Count")) or 0)
        out[category] = ClassTarget(
            category=category,
            count=count,
            total_length_um=float(_number(item.get("Total Length (microns)")) or 0.0),
            mean_length_um=_number(item.get("Mean Length (microns)")),
            median_length_um=_number(item.get("Median Length (microns)")),
            mean_diameter_um=_number(item.get("Mean Diameter (microns)")),
            median_diameter_um=_number(item.get("Median Diameter (microns)")),
            mean_tortuosity=_number(item.get("Mean Tortuosity Index")),
            median_tortuosity=_number(item.get("Median Tortuosity Index")),
            p90_tortuosity=_number(item.get("90th Percentile Tortuosity Index")),
        )
        if count and out[category].mean_diameter_um is None:
            notes.append(f"{VESSEL_CLASS_TITLES[category]}: no diameter in the template")
    return out


def read_layout(sections: Mapping[str, Any], G: Optional[nx.Graph] = None) -> Optional[LayoutTarget]:
    """The :class:`LayoutTarget` of report *sections*; a layout section the
    report lacks (one saved before 2026-10-25) measured off *G* by the same
    statistics. None when neither has any of it."""
    have = {name: sections.get(name) for name in LAYOUT_SECTIONS if isinstance(sections.get(name), Mapping)}
    missing = [name for name in LAYOUT_SECTIONS if name not in have]
    if missing and G is not None and G.number_of_edges():
        from haemolynx.statistics import compute_vessel_statistics_by_measure

        computed = compute_vessel_statistics_by_measure(G, enabled_measures=frozenset(missing))
        have.update({name: computed.get(name) or {} for name in missing})
    values = {field: _number((have.get(section) or {}).get(name)) for field, (section, name) in LAYOUT_ROWS.items()}
    if all(value is None for value in values.values()):
        return None
    return LayoutTarget(**values)


def _class_p90_from_graph(classes: dict, G: nx.Graph) -> dict:
    """*classes* with each 90th percentile tortuosity a report saved before
    2026-10-25 lacks measured off *G* by the same statistics."""
    if all(t.p90_tortuosity is not None or not t.count for t in classes.values()):
        return classes
    from haemolynx.statistics import compute_vessel_statistics_by_measure

    positions = {n: np.asarray(G.nodes[n]["pos"], float) for n in G.nodes if G.nodes[n].get("pos") is not None}
    section = compute_vessel_statistics_by_measure(
        G, node_positions=positions, enabled_measures=frozenset({"vessel_classes"})
    ).get("vessel_classes")
    measured = _read_classes(section, [])
    return {
        key: target
        if target.p90_tortuosity is not None or measured.get(key) is None
        else replace(target, p90_tortuosity=measured[key].p90_tortuosity)
        for key, target in classes.items()
    }


def _order_index(label: str) -> int:
    digits = "".join(ch for ch in label if ch.isdigit())
    return int(digits) if digits else 0


def _read_orders(section: Optional[Mapping[str, Any]]) -> tuple:
    out: list[OrderTarget] = []
    for raw_label, item in (section or {}).items():
        if not isinstance(item, Mapping):
            continue
        label = _normalize_branch_order_tag(raw_label)
        if label is None:
            continue
        category = branch_order_category(label)
        if category not in BRANCH_ORDER_CATEGORY_SEQUENCE:
            continue
        out.append(
            OrderTarget(
                label=label,
                category=category,
                index=_order_index(label),
                edge_count=int(_number(item.get("Edge Count")) or 0),
                mean_length_um=_number(item.get("Mean Length (microns)")),
                mean_tortuosity=_number(item.get("Mean Tortuosity Index")),
                mean_emergence_angle_deg=_number(item.get("Mean Emergence Angle (degrees)")),
                mean_diameter_um=_number(item.get("Mean Diameter (microns)")),
                diameter_cv=_number(item.get("Diameter Coefficient of Variation")),
            )
        )
    return tuple(sorted(out, key=lambda o: flow_path_key(o.label)))


def _read_dead_ends(section: Optional[Mapping[str, Any]]) -> DeadEndTarget:
    rows = (section or {}).get("Dead Ends") if section else None
    rows = rows if isinstance(rows, Mapping) else {}

    def count(kind: str) -> int:
        return int(_number(rows.get(_KIND_ROWS[kind])) or 0)

    return DeadEndTarget(
        total=int(_number(rows.get("Dead Ends")) or 0),
        leaves_image=count("leaves_image"),
        interior=sum(count(kind) for kind in DENSITY_KINDS),
        artefacts=count("artefact"),
        island_ends=count("island_end"),
        interior_per_mm=_number(rows.get("Interior Dead Ends per mm of Vessel (1/mm)")),
        chain_length_median_um=_number(rows.get("Interior Dead-End Chain Length, median (microns)")),
        chain_length_p10_um=_number(rows.get("Interior Dead-End Chain Length, 10th percentile (microns)")),
        chain_length_p90_um=_number(rows.get("Interior Dead-End Chain Length, 90th percentile (microns)")),
    )


def _read_loops(
    section: Optional[Mapping[str, Any]],
    cyclomatic: Optional[Mapping[str, Any]],
    image_volume_mm3: float,
    notes: list[str],
) -> LoopTarget:
    rows = section or {}
    percentiles = {}
    for name in ("P10", "P25", "Median", "P75", "P90"):
        value = _number(rows.get(f"Loop Length {name} (um)"))
        if value is not None:
            percentiles[name] = value
    per_mm3 = _number(rows.get("Independent Loops Per mm^3"))
    independent = None
    if cyclomatic is not None and _number(cyclomatic.get("Cyclomatic Number")) is not None:
        independent = int(round(_number(cyclomatic.get("Cyclomatic Number"))))
    elif per_mm3 is not None:
        independent = int(round(per_mm3 * image_volume_mm3))
    elif section is not None:
        notes.append("the template's loop count is unknown; a bed of mean degree 3 is assumed")
    return LoopTarget(
        independent_loops=independent,
        independent_loops_per_mm3=per_mm3,
        length_percentiles_um=percentiles,
        vessels_on_no_loop_fraction=_number(rows.get("Vessels On No Loop Fraction")),
    )


def missing_sections(sections: Mapping[str, Any], require: Sequence[str] = REQUIRED_SECTIONS) -> tuple[str, ...]:
    return tuple(name for name in require if not isinstance(sections.get(name), Mapping))


def required_statistics_for(sections: Iterable[str]) -> tuple[str, ...]:
    """The ``statistics_*`` settings that switch *sections* on, each once,
    in the order the schema lists them."""
    out: list[str] = []
    for name in sections:
        for toggle in SECTION_TOGGLES.get(name, ("statistics",)):
            if toggle not in out:
                out.append(toggle)
    return tuple(out)


def _run_manifest(path: Optional[Path], settings: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """``(haemolynx version, git commit)`` from the run manifest beside the
    template run's outputs, when it is there."""
    prefix = settings.get("vtk_output_prefix")
    image = settings.get("input_path")
    if not prefix or not image:
        return None, None
    manifest = Path(prefix).parent / f"{Path(image).stem}_run_manifest.json"
    if path is not None and not manifest.exists():
        manifest = Path(path).parent / manifest.name
    if not manifest.exists():
        return None, None
    try:
        import json

        data = json.loads(manifest.read_text(encoding="utf-8"))
        git = data.get("git") or {}
        return data.get("haemolynx_version"), git.get("commit") if isinstance(git, Mapping) else None
    except Exception:  # noqa: BLE001 -- provenance only; never fails a template
        return None, None


def template_from_report(
    sections: Mapping[str, Any],
    G: nx.Graph,
    *,
    boundaries: Mapping[str, Sequence],
    checkpoint: Any = None,
    require: Sequence[str] = REQUIRED_SECTIONS,
    source: str = SOURCE_STATISTICS,
    path: Optional[Path | str] = None,
    statistics_mode: Optional[str] = None,
    run_version: Optional[str] = None,
    run_commit: Optional[str] = None,
    run_basis: Optional[str] = None,
    notes: Sequence[str] = (),
) -> NetworkTemplate:
    """A template from report *sections* (``{section: {row: value}}``) and
    the graph they describe. Raises :class:`TemplateError` for a required
    section that is missing, naming the settings that switch it on.
    *run_basis* is the template run's own diameter basis
    (:func:`run_diameter_basis_of`; read off *G* when None)."""
    notes_out = list(notes)
    missing = missing_sections(sections, require)
    if missing:
        toggles = required_statistics_for(missing)
        raise TemplateError(
            "The template run's statistics lack "
            + ", ".join(f"'{name}'" for name in missing)
            + ": switch on "
            + ", ".join(toggles)
            + " on the template run's Additional measurements tab and run its Export stage again.",
            reason="no_statistics",
            missing_statistics=toggles,
        )
    extent, voxel, shape, _box = _graph_box(G, checkpoint, notes_out)
    image_volume_um3 = float(np.prod(np.asarray(shape, float) * voxel))

    basic = sections.get("basic") or {}
    density = sections.get("vessel_density") or {}
    reported_volume = _number(density.get("Total Image Volume (micron³)"))
    if reported_volume is not None and image_volume_um3 > 0:
        if abs(reported_volume - image_volume_um3) > 0.02 * image_volume_um3:
            notes_out.append(
                f"the report's image volume ({reported_volume:.4g} um^3) differs from the "
                f"graph's extent ({image_volume_um3:.4g} um^3); the extent is used"
            )

    labels = [d.get("branch_order") for *_e, d in G.edges(data=True) if not d.get(IS_ZERO_RESISTANCE)]
    scheme = vessel_class_scheme(labels)
    classes = _class_p90_from_graph(_read_classes(sections.get("vessel_classes"), notes_out), G)
    orders = _read_orders(sections.get(BRANCH_ORDERS_SECTION))

    murray = sections.get("murray_law") or {}
    exponent = _number(murray.get("Median Fitted Murray Exponent"))
    if exponent is None:
        notes_out.append(f"no fitted Murray exponent in the template; {DEFAULT_MURRAY_EXPONENT:g} is used")
    low, high = (
        _number(murray.get("Fitted Murray Exponent IQR Low")),
        _number(murray.get("Fitted Murray Exponent IQR High")),
    )
    angles = sections.get("daughter_angles") or {}
    daughter = _number(angles.get("Mean Daughter-Daughter Angle (degrees)"))
    if daughter is None:
        notes_out.append(f"no daughter angle in the template; {DEFAULT_DAUGHTER_ANGLE_DEG:g} degrees is used")
    icd = sections.get("intercapillary_distance") or {}
    tortuosity = _number((sections.get("tortuosity") or {}).get("Average Tortuosity Index"))
    branching = _number((sections.get("branching") or {}).get("Number of Branching Points"))

    total_length = _number(density.get("Total Vessel Length (microns)"))
    if total_length is None:
        total_length = _number(basic.get("Total Edge Length (microns)"))
    if total_length is None:
        total_length = float(sum(float(d.get("length") or 0.0) for *_e, d in G.edges(data=True)))

    loops = _read_loops(
        sections.get("loop_hierarchy"), sections.get("cyclomatic_number"), image_volume_um3 / 1e9, notes_out
    )
    basis = template_diameter_basis(G, run_basis or run_diameter_basis_of(G), notes_out)
    provenance = Provenance(
        template_path=None if path is None else str(path),
        source=source,
        haemolynx_version=_package_version(),
        statistics_mode=statistics_mode,
        run_version=run_version,
        run_commit=run_commit,
        notes=tuple(notes_out),
    )
    inlets = endpoints_from_nodes(G, boundaries.get("inlet", ()), extent, voxel)
    outlets = endpoints_from_nodes(G, boundaries.get("outlet", ()), extent, voxel)
    return NetworkTemplate(
        extent_um=tuple(float(v) for v in extent),
        voxel_size_zyx=tuple(float(v) for v in voxel),
        image_shape=shape,
        inlets=inlets,
        outlets=outlets,
        arteriole_boundary_count=len(boundaries.get("arteriole_boundary", ())),
        venule_boundary_count=len(boundaries.get("venule_boundary", ())),
        large_arteriole_boundary_count=len(boundaries.get("large_arteriole_boundary", ())),
        large_venule_boundary_count=len(boundaries.get("large_venule_boundary", ())),
        scheme=scheme,
        classes=classes,
        orders=orders,
        total_length_um=float(total_length),
        node_count=None if _number(basic.get("Total Nodes")) is None else int(_number(basic.get("Total Nodes"))),
        edge_count=None if _number(basic.get("Total Edges")) is None else int(_number(basic.get("Total Edges"))),
        average_degree=_number(basic.get("Average Degree")),
        branching_points=None if branching is None else int(branching),
        mean_tortuosity=tortuosity,
        daughter_angle_deg=daughter,
        daughter_angle_sd_deg=None,
        murray_exponent=exponent,
        murray_iqr=None if low is None or high is None else (low, high),
        intercapillary_distance_um=_number(icd.get("Mean Intercapillary Distance (microns)")),
        intercapillary_distance_median_um=_number(icd.get("Median Intercapillary Distance (microns)")),
        loops=loops,
        dead_ends=_read_dead_ends(sections.get("dead_ends")),
        tissue_volume_um3=_number(density.get("Tissue Volume (micron³)")),
        provenance=provenance,
        diameter_basis=basis,
        organisation=measure_organisation(
            G, extent, roots=list(boundaries.get("inlet", ())) + list(boundaries.get("outlet", ()))
        ),
        layout=read_layout(sections, G),
    )


def _package_version() -> Optional[str]:
    try:
        import haemolynx

        return getattr(haemolynx, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def template_from_statistics(
    snapshot: Any,
    *,
    path: Optional[Path | str] = None,
    require: Sequence[str] = REQUIRED_SECTIONS,
) -> NetworkTemplate:
    """The template the run's saved statistics report describes."""
    G, checkpoint = template_graph(snapshot)
    boundaries = boundary_nodes_of(snapshot, checkpoint)
    check_template_run(snapshot, G, boundaries)
    report = G.graph.get(STATISTICS_REPORT_KEY)
    sections = report.get("sections") if isinstance(report, Mapping) else None
    if not isinstance(sections, Mapping) or not sections:
        raise TemplateError(
            "The template run was saved with Statistics off, so it has no "
            "statistics report: switch on statistics on its Additional "
            "measurements tab (with "
            + ", ".join(t for t in required_statistics_for(require) if t != "statistics")
            + ") and run its Export stage again.",
            reason="no_statistics",
            missing_statistics=required_statistics_for(require),
        )
    version, commit = _run_manifest(None if path is None else Path(path), _run_settings(snapshot))
    return template_from_report(
        sections,
        G,
        boundaries=boundaries,
        checkpoint=checkpoint,
        require=require,
        source=SOURCE_STATISTICS,
        path=path,
        statistics_mode=report.get("mode") if isinstance(report, Mapping) else None,
        run_version=version,
        run_commit=commit,
        run_basis=run_diameter_basis_of(G, _run_settings(snapshot)),
    )


def _vessel_tortuosity(G: nx.Graph, u: Any, v: Any, length: float) -> Optional[float]:
    pu, pv = G.nodes[u].get("pos"), G.nodes[v].get("pos")
    if pu is None or pv is None or u == v:
        return None
    chord = float(np.linalg.norm(np.asarray(pu, float) - np.asarray(pv, float)))
    return length / chord if chord > 0 and length > 0 else None


def _collect_samples(G: nx.Graph) -> tuple[dict, dict]:
    """``(by order label, by class category)`` -> :class:`Samples`."""
    by_order: dict[str, dict[str, list]] = {}
    by_class: dict[str, dict[str, list]] = {}
    for u, v, d in G.edges(data=True):
        if d.get(IS_ZERO_RESISTANCE):
            continue
        length = _number(d.get("length"))
        if length is None or length <= 0:
            continue
        label = _normalize_branch_order_tag(d.get("branch_order"))
        if label is None:
            continue
        category = branch_order_category(label)
        diameter = _number(d.get("diameter_um"))
        tortuosity = _vessel_tortuosity(G, u, v, length)
        for store, key in ((by_order, label), (by_class, category)):
            bucket = store.setdefault(key, {"lengths_um": [], "diameters_um": [], "tortuosities": []})
            bucket["lengths_um"].append(length)
            if diameter is not None and diameter > 0:
                bucket["diameters_um"].append(diameter)
            if tortuosity is not None:
                bucket["tortuosities"].append(tortuosity)
    as_samples = lambda bucket: Samples(  # noqa: E731
        lengths_um=tuple(bucket["lengths_um"]),
        diameters_um=tuple(bucket["diameters_um"]),
        tortuosities=tuple(bucket["tortuosities"]),
    )
    return {k: as_samples(b) for k, b in by_order.items()}, {k: as_samples(b) for k, b in by_class.items()}


def _dead_end_chain_samples(G: nx.MultiGraph, extent_um: Sequence[float]) -> tuple[float, ...]:
    """The chain length of each interior dead end, as the Dead Ends section
    measures it (its kind from the node, else by the face rule)."""
    multi = G if G.is_multigraph() else nx.MultiGraph(G)
    out: list[float] = []
    for node in multi:
        if multi.degree(node) != 1:
            continue
        kind = multi.nodes[node].get(DEAD_END_KIND)
        if kind is None:
            kind = LEAVES_IMAGE if node_leaves_image(multi, node, extent_um) else "unknown"
        if kind not in DENSITY_KINDS:
            continue
        nodes, edges = chain_to_junction(multi, node)
        if multi.degree(nodes[-1]) > 1:
            out.append(float(chain_length_um(multi, edges)))
    return tuple(out)


def measure_template(
    G: nx.Graph,
    *,
    inlet_nodes: Sequence[Any],
    outlet_nodes: Sequence[Any],
    arteriole_boundary_nodes: Sequence[Any] = (),
    venule_boundary_nodes: Sequence[Any] = (),
    large_arteriole_boundary_nodes: Sequence[Any] = (),
    large_venule_boundary_nodes: Sequence[Any] = (),
    image_shape: Optional[Sequence[int]] = None,
    voxel_size_zyx: Optional[Sequence[float]] = None,
    samples: bool = True,
    statistics_mode: str = "fast",
    path: Optional[Path | str] = None,
    source: str = SOURCE_GRAPH,
    checkpoint: Any = None,
    run_basis: Optional[str] = None,
    notes: Sequence[str] = (),
) -> NetworkTemplate:
    """A template measured from *G* itself: the required report sections
    computed again (the statistics modules, so bridges are contracted and
    every row means what the saved report's does), then, with *samples*,
    every vessel's own length, diameter and tortuosity per order and class,
    the loop lengths and the interior dead-end chains."""
    from haemolynx.statistics import (
        compute_branch_order_statistics,
        compute_vessel_statistics_by_measure,
    )

    notes_out = list(notes)
    extent, voxel, shape, _box = _graph_box(G, checkpoint, notes_out)
    if voxel_size_zyx is not None:
        voxel = np.asarray(voxel_size_zyx, dtype=float)[:3]
    if image_shape is not None:
        shape = tuple(int(v) for v in image_shape)
    positions = {n: np.asarray(G.nodes[n]["pos"], float) for n in G.nodes if G.nodes[n].get("pos") is not None}
    enabled = frozenset(
        name for name in REQUIRED_SECTIONS + OPTIONAL_SECTIONS if name != BRANCH_ORDERS_SECTION
    )
    sections = dict(
        compute_vessel_statistics_by_measure(
            G,
            node_positions=positions,
            voxel_size=tuple(float(v) for v in voxel),
            image_dimensions=shape,
            statistics_mode=statistics_mode,
            enabled_measures=enabled,
            inlet_nodes=list(inlet_nodes),
            outlet_nodes=list(outlet_nodes),
            arteriole_boundary_nodes=list(arteriole_boundary_nodes),
            venule_boundary_nodes=list(venule_boundary_nodes),
        )
    )
    sections[BRANCH_ORDERS_SECTION] = compute_branch_order_statistics(G, node_positions=positions)
    boundaries = {
        "inlet": tuple(inlet_nodes),
        "outlet": tuple(outlet_nodes),
        "arteriole_boundary": tuple(arteriole_boundary_nodes),
        "venule_boundary": tuple(venule_boundary_nodes),
        "large_arteriole_boundary": tuple(large_arteriole_boundary_nodes),
        "large_venule_boundary": tuple(large_venule_boundary_nodes),
    }
    template = template_from_report(
        sections,
        G,
        boundaries=boundaries,
        checkpoint=checkpoint,
        source=source,
        path=path,
        statistics_mode=statistics_mode,
        run_basis=run_basis,
        notes=notes_out,
    )
    if not samples:
        return template
    by_order, by_class = _collect_samples(G)
    classes = {
        key: replace(target, samples=by_class.get(key, Samples())) for key, target in template.classes.items()
    }
    orders = tuple(replace(o, samples=by_order.get(o.label, Samples())) for o in template.orders)
    loop_lengths = tuple(
        float(v)
        for *_e, d in G.edges(data=True)
        if not d.get(IS_ZERO_RESISTANCE)
        for v in [_number(d.get("loop_length_um"))]
        if v is not None
    )
    loops = replace(template.loops, loop_length_samples_um=loop_lengths)
    dead_ends = replace(
        template.dead_ends, chain_length_samples_um=_dead_end_chain_samples(G, template.extent_um)
    )
    return replace(template, classes=classes, orders=orders, loops=loops, dead_ends=dead_ends)


def template_from_graph(
    snapshot: Any,
    *,
    path: Optional[Path | str] = None,
    statistics_mode: str = "fast",
) -> NetworkTemplate:
    """The template measured from the graph in the run, with per-vessel samples."""
    G, checkpoint = template_graph(snapshot)
    boundaries = boundary_nodes_of(snapshot, checkpoint)
    check_template_run(snapshot, G, boundaries)
    version, commit = _run_manifest(None if path is None else Path(path), _run_settings(snapshot))
    template = measure_template(
        G,
        inlet_nodes=boundaries["inlet"],
        outlet_nodes=boundaries["outlet"],
        arteriole_boundary_nodes=boundaries["arteriole_boundary"],
        venule_boundary_nodes=boundaries["venule_boundary"],
        large_arteriole_boundary_nodes=boundaries["large_arteriole_boundary"],
        large_venule_boundary_nodes=boundaries["large_venule_boundary"],
        statistics_mode=statistics_mode,
        path=path,
        checkpoint=checkpoint,
        run_basis=run_diameter_basis_of(G, _run_settings(snapshot)),
    )
    # The tissue volume is measured from the raw image, not the graph: the
    # run's saved report has it when the run measured it.
    report = G.graph.get(STATISTICS_REPORT_KEY)
    sections = report.get("sections") if isinstance(report, Mapping) else None
    density = sections.get("vessel_density") if isinstance(sections, Mapping) else None
    tissue = _number(density.get("Tissue Volume (micron³)")) if isinstance(density, Mapping) else None
    return replace(
        template,
        tissue_volume_um3=template.tissue_volume_um3 if tissue is None else tissue,
        provenance=replace(template.provenance, run_version=version, run_commit=commit),
    )


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------

_CACHE: dict[tuple, NetworkTemplate] = {}
_BASIS_CACHE: dict[tuple, Optional[str]] = {}


def clear_template_cache() -> None:
    _CACHE.clear()
    _BASIS_CACHE.clear()


def diameter_basis_of_run(path: Path | str) -> Optional[str]:
    """The diameter basis a template read from the ``.haemorun`` at *path*
    would record (:attr:`NetworkTemplate.diameter_basis`), without reading
    the rest of the template: what the panel sets a synthetic run's
    ``segmentation_source`` to when the template is picked
    (``gui.template_basis``). Remembered per file identity, as
    :func:`load_template` is. Raises ``gui.run_snapshot.RunSnapshotError``
    for a file that is not a run snapshot; None for one with no network or
    no basis to read."""
    file = Path(path)
    stat = os.stat(file)
    key = (str(file.resolve()), stat.st_size, stat.st_mtime_ns)
    if key in _BASIS_CACHE:
        return _BASIS_CACHE[key]
    from .batch_template import batch_member_templates, is_batch_template

    if is_batch_template(file):
        # The basis most of the batch's runs were read on.
        bases = [t.diameter_basis for t in batch_member_templates(file, SOURCE_STATISTICS) if t.diameter_basis]
        basis = max(set(bases), key=bases.count) if bases else None
        _BASIS_CACHE[key] = basis
        return basis
    from haemolynx.gui.run_snapshot import read_run_snapshot

    snapshot = read_run_snapshot(file)
    G, _checkpoint = template_graph(snapshot)
    basis = None
    if G is not None:
        basis = template_diameter_basis(G, run_diameter_basis_of(G, _run_settings(snapshot)), [])
    _BASIS_CACHE[key] = basis
    return basis


def load_template(path: Path | str, source: str = SOURCE_STATISTICS) -> NetworkTemplate:
    """The template of the ``.haemorun`` at *path*, by *source*
    (:data:`TEMPLATE_SOURCES`), or of a batch condition's ``.haemobatch``
    (its runs' templates averaged, ``synthetic.batch_template``). Remembered
    per (file identity, source), so preflight and the Input stage read the
    file once.

    Raises ``gui.run_snapshot.RunSnapshotError`` for a file that is not a run
    snapshot and :class:`TemplateError` for a run that cannot be a template.
    """
    if source not in TEMPLATE_SOURCES:
        raise ValueError(f"source must be one of {TEMPLATE_SOURCES}, got {source!r}")
    file = Path(path)
    stat = os.stat(file)
    key = (str(file.resolve()), stat.st_size, stat.st_mtime_ns, source)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    from .batch_template import is_batch_template, load_batch_template

    if is_batch_template(file):
        # A batch condition's runs, averaged (synthetic.batch_template).
        template = load_batch_template(file, source)
        _CACHE[key] = template
        return template
    # gui.run_snapshot imports the stages, which import this package lazily.
    from haemolynx.gui.run_snapshot import read_run_snapshot

    snapshot = read_run_snapshot(file)
    if source == SOURCE_STATISTICS:
        template = template_from_statistics(snapshot, path=file)
    else:
        template = template_from_graph(snapshot, path=file)
    _CACHE[key] = template
    logger.info(
        "Synthetic template %s (%s): %d vessels, %d inlets, %d outlets, box %s um",
        file,
        source,
        template.vessel_count,
        template.inlet_count,
        template.outlet_count,
        tuple(round(v, 1) for v in template.extent_um),
    )
    return template


__all__ = [
    "BRANCH_ORDERS_SECTION",
    "ClassTarget",
    "DEFAULT_DAUGHTER_ANGLE_DEG",
    "DEFAULT_DAUGHTER_ANGLE_SD_DEG",
    "DEFAULT_DIAMETER_CV",
    "DEFAULT_MURRAY_EXPONENT",
    "DeadEndTarget",
    "Endpoint",
    "FACES",
    "GRAPH_NETWORK_SOURCE",
    "GRAPH_TEMPLATE_RECORD",
    "INTERIOR",
    "LAYOUT_ROWS",
    "LAYOUT_SECTIONS",
    "LayoutTarget",
    "LoopTarget",
    "NetworkTemplate",
    "OPTIONAL_SECTIONS",
    "OrderTarget",
    "Provenance",
    "REQUIRED_SECTIONS",
    "ROLES",
    "SCALE_AXES",
    "SCALE_AXES_XY",
    "SCALE_AXES_XYZ",
    "SECTION_TOGGLES",
    "SOURCE_GRAPH",
    "SOURCE_STATISTICS",
    "STATISTICS_REPORT_KEY",
    "Samples",
    "TEMPLATE_SOURCES",
    "TemplateError",
    "boundary_nodes_of",
    "check_template_run",
    "clear_template_cache",
    "diameter_basis_of_run",
    "endpoints_from_nodes",
    "load_template",
    "lognormal_from_mean_cv",
    "lognormal_from_mean_median",
    "lognormal_from_median_p10_p90",
    "measure_template",
    "missing_sections",
    "read_layout",
    "required_statistics_for",
    "run_diameter_basis_of",
    "scale_factors_zyx",
    "template_diameter_basis",
    "template_from_graph",
    "template_from_report",
    "template_from_statistics",
    "template_graph",
]
