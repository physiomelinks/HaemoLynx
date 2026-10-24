"""What every statistic the CSVs report is, and its unit.

The statistics are dicts of ``"Name (unit)": value``, and only some names
carry a unit; the CSV exporter used to read the unit from a name's trailing
parentheses, so "Fractal Dimension (Node Positions)" had the unit "Node
Positions", most rows had none, and one unit was spelled microns, um and
micron. This is the one place a statistic is defined:

* :data:`GLOSSARY` -- every row's definition and unit, matched on the row's
  name as the statistics dict spells it (a regular expression where the name
  is built: per side, per group, per vessel type);
* :data:`UNITS` -- the units, each spelled one way (ASCII, so a CSV opened in
  any program reads the same), and :data:`UNIT_SPELLINGS`, every spelling a
  name or a unit row uses mapped onto one of them;
* :func:`unit_for` -- a row's unit. A few depend on the run: a route
  analysis measures in Pa.s/m^3, um or vessels as its route weighting says
  (:class:`ByWeighting`, read from the row naming the weighting), and two
  analyses write their unit as a row of its own (:class:`FromRow`).

``statistics.csv_export`` writes each row's unit and definition from here,
``batch_comparison`` reads old and new CSVs through :func:`csv_metric_and_unit`
so they line up, and ``docs/statistics_glossary.md`` is
:func:`glossary_markdown` (``python -m haemolynx.statistics.glossary``
writes it again).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Union

logger = logging.getLogger(__name__)

#: Every unit a statistic is reported in, spelled as the CSVs spell it, and
#: what it means.
UNITS: dict[str, str] = {
    "um": "micrometres (microns)",
    "um^2": "square micrometres",
    "um^3": "cubic micrometres",
    "um/um^3": "micrometres of vessel per cubic micrometre (length density)",
    "um^2/um^3": "square micrometres of vessel wall per cubic micrometre",
    "1/um": "per micrometre",
    "1/mm": "per millimetre of vessel",
    "1/mm^3": "per cubic millimetre",
    "mm/s": "millimetres per second",
    "nL/min": "nanolitres per minute",
    "m^3/s": "cubic metres per second",
    "m^3/(Pa.s)": "hydraulic conductance, cubic metres per pascal second",
    "Pa": "pascals",
    "Pa.s/m^3": "hydraulic resistance, pascal seconds per cubic metre",
    "s": "seconds",
    "fL": "femtolitres",
    "cells/s": "red cells per second",
    "cells/mm": "red cells per millimetre of vessel",
    "degrees": "degrees of angle",
    "vessels": "a number of vessels (graph edges); as a route length, the vessels along it",
    "count": "a number of things other than vessels (nodes, junctions, beds, loops, samples)",
    "fraction": "a share between 0 and 1",
    "dimensionless": "a ratio or index with no unit",
    "intensity": "the raw image's own grey values",
}

#: Every spelling of a unit met in a statistic's name, a unit row or an
#: older CSV, and the one it is now.
UNIT_SPELLINGS: dict[str, str] = {
    **{unit: unit for unit in UNITS},
    "micron": "um",
    "microns": "um",
    "µm": "um",
    "micron²": "um^2",
    "micron³": "um^3",
    "microns/micron³": "um/um^3",
    "micron²/micron³": "um^2/um^3",
    "micron³/micron³": "fraction",
    "m³/s": "m^3/s",
    "conductance (m^3/(Pa.s))": "m^3/(Pa.s)",
    "um (resistance proportional to length)": "um",
    "vessels (unit resistance per vessel)": "vessels",
}


def canonical_unit(text: Any) -> Optional[str]:
    """*text* as a unit of :data:`UNITS`, or None when it is not a unit."""
    if not isinstance(text, str):
        return None
    return UNIT_SPELLINGS.get(text.strip())


#: A term's unit is the one its name spells in its trailing parentheses (the
#: loop and Strahler lengths, in um or vessels by the weighting they ran on).
UNIT_IN_NAME = "unit in the name"


@dataclass(frozen=True)
class ByWeighting:
    """A unit set by the route weighting the row named *row* reports
    (``resistance``, ``length`` or ``topology``, perhaps with a fallback note
    after it)."""

    row: str
    resistance: str
    length: str
    topology: str

    def resolve(self, rows: Mapping[str, Any]) -> str:
        label = rows.get(self.row)
        if not isinstance(label, str) or not label.strip():
            return ""
        return getattr(self, label.split()[0], "")


@dataclass(frozen=True)
class FromRow:
    """A unit the analysis writes as a row of its own, named *row*."""

    row: str

    def resolve(self, rows: Mapping[str, Any]) -> str:
        return canonical_unit(rows.get(self.row)) or ""


Unit = Union[str, ByWeighting, FromRow]


@dataclass(frozen=True)
class Term:
    """One statistic: the names it is reported under (a regular expression,
    matched whole), its unit, what it is, and the topic it is listed under."""

    pattern: str
    unit: Unit
    definition: str
    topic: str
    #: The name as the glossary page shows it.
    shown: str

    @property
    def regex(self) -> re.Pattern:
        return re.compile(self.pattern)


#: A row whose value is text (a method, a source, a list): no unit.
TEXT = ""

#: The resistance, route length and conductance units of the shortest-route
#: analyses, by the weighting row each reads.
def _route_length(row: str) -> ByWeighting:
    return ByWeighting(row, resistance="Pa.s/m^3", length="um", topology="vessels")


_SHUNT_ROUTE = _route_length("Shunt Route Weighting")
_CURRENT_FLOW_RESISTANCE = FromRow("Equivalent Inlet-Outlet Resistance Unit")
_HYPOPERFUSED_LENGTH = FromRow("Occlusion Hypoperfused Length Unit")
_LOOP_LENGTH = ByWeighting("Loop Length Weighting", resistance="vessels", length="um", topology="vessels")
_LAMBDA_2 = ByWeighting(
    "Algebraic Connectivity Route Weighting",
    resistance="m^3/(Pa.s)",
    length="1/um",
    topology="dimensionless",
)

_TERMS: list[Term] = []


def _add(topic: str, name: str, unit: Unit, definition: str, *, pattern: Optional[str] = None) -> None:
    """One term: *name* as shown, matched literally unless *pattern* is given."""
    _TERMS.append(Term(pattern or re.escape(name), unit, definition, topic, name))  # noqa: F821 (module list, deleted once GLOSSARY is built)


# -- Network size -------------------------------------------------------------
T = "Network size"
_add(T, "Total Nodes", "count", "Nodes of the graph the statistics read: junctions, dead ends and the nodes left along vessels, with each zero-resistance bridge contracted to one node.")
_add(T, "Total Edges", "vessels", "Vessels (graph edges) in the network; zero-resistance bridges are not vessels and are not counted.")
_add(T, "Total Edge Length (microns)", "um", "The sum of every vessel's centreline length (the smoothed centreline, as the haemodynamics use it).")
_add(T, "Average Edge Length (microns)", "um", "Total Edge Length divided by Total Edges: the mean vessel length.")
_add(T, "Average Degree", "dimensionless", "The mean number of vessel ends meeting at a node (twice the vessels over the nodes).")
_add(T, "Number of Branching Points", "count", "Junctions: nodes where three or more vessel ends meet.")
_add(T, "Zero-resistance Bridges (not counted as vessels)", "count", "Bridges joining a vessel to a thick vessel's centreline through its lumen, modelled with negligible resistance. Contracted before every other statistic.")
_add(T, "Zero-resistance Bridge Length (microns)", "um", "The summed length of those bridges.")
_add(T, "Statistics Mode", TEXT, "fast: sampled or summarised where a measure is expensive; full: exact, with per-node rows.")

# -- Shape --------------------------------------------------------------------
T = "Shape"
_add(T, "Average Tortuosity Index", "dimensionless", "The mean over vessels of centreline length divided by the straight distance between its two end nodes; 1 is straight.")
_add(T, "Average Curvature", "dimensionless", "The mean over vessels of (centreline length - straight distance) / centreline length: 0 for a straight vessel, nearer 1 the more it bends. Not a curvature in 1/um.")
_add(T, "Tree Asymmetry Index", "dimensionless", "On the network's shortest-length spanning tree, rooted at its highest-degree node: the sum over nodes of the difference between their largest and smallest child subtree (in nodes), over the tree's node count. 0 for a perfectly balanced tree.")
_add(T, "Fractal Dimension (Node Positions)", "dimensionless", "Box-counting dimension of the nodes' positions alone: the slope of log(boxes occupied) against log(1 / box size). Blind to the vessels' paths between nodes.")
_add(T, "Fractal Dimension (Centreline)", "dimensionless", "Box-counting dimension of points every 1 um along every vessel's centreline, the usual vascular measure.")
_add(T, "Fractal Dimension (Node Positions | Centreline) R^2", "dimensionless", "R^2 of the log-log fit the dimension is the slope of; low means the network does not scale over the box sizes fitted.", pattern=r"Fractal Dimension \((Node Positions|Centreline)\) R\^2")
_add(T, "Fractal Dimension (Node Positions | Centreline) Box Sizes (microns)", "um", "The smallest and largest box edge the dimension was fitted over (from twice the sampling step to half the network's extent, while at least 10 boxes are occupied).", pattern=r"Fractal Dimension \((Node Positions|Centreline)\) Box Sizes \(microns\)")
_add(T, "Mean Daughter-Daughter Angle (degrees)", "degrees", "The mean angle between each pair of daughter vessels at a junction with a unique parent, along their tangents over the first 10 um.")
_add(T, "Daughter-Daughter Angle Sample Count", "count", "Daughter pairs measured.")

# -- Density ------------------------------------------------------------------
T = "Density and volume"
_add(T, "Total Vessel Length (microns)", "um", "The sum of every vessel's centreline length (the same as Total Edge Length).")
_add(T, "Vessel Density in Tissue (microns/micron³)", "um/um^3", "Total vessel length over the tissue volume: the tissue measured from the raw image when it was, else the box the network's nodes span (Tissue Volume Source says which).")
_add(T, "Vessel-Occupied Volume (micron³)", "um^3", "The volume of the axis-aligned box the network's nodes span.")
_add(T, "Tissue Volume (micron³)", "um^3", "The tissue's volume, inside the closed surface measured from the raw image (Additional measurements: tissue volume).")
_add(T, "Tissue Volume Source", TEXT, "What the in-tissue densities divide by: the measured tissue, or the bounding box of the network's nodes.")
_add(T, "Vessel Volume in Tissue (micron³)", "um^3", "Segmented vessel voxels inside the measured tissue.")
_add(T, "Vascular Volume Fraction in Tissue", "fraction", "Vessel Volume in Tissue over Tissue Volume.")
_add(T, "Vessel Density in Whole Image (microns/micron³)", "um/um^3", "Total vessel length over the whole image's volume, empty space included.")
_add(T, "Total Image Volume (micron³)", "um^3", "The image's volume: its shape times the voxel size.")
_add(T, "Independent Loops Per mm^3", "1/mm^3", "Independent loops (vessels - nodes + connected pieces) per cubic millimetre of the image.")

# -- Tissue volume (its own CSV) ----------------------------------------------
T = "Tissue volume (the tissue_volume CSV)"
_add(T, "Tissue Volume, Voxel Count (micron³)", "um^3", "The tissue's volume counted in working-grid voxels, a cross-check on the surface's volume.")
_add(T, "Tissue Fraction of Image", "fraction", "Tissue Volume over the image's volume.")
_add(T, "Tissue Surface Area (micron²)", "um^2", "The area of the tissue's own surface, leaving out where the image faces cut it.")
_add(T, "Tissue Surface Cut by Image Faces (micron²)", "um^2", "The area where the image faces cut the tissue.")
_add(T, "Tissue Pieces", "count", "Separate pieces of tissue kept.")
_add(T, "Tissue Threshold Method", TEXT, "How the tissue was split from empty space: Otsu, Li, triangle or a manual intensity.")
_add(T, "Tissue Threshold Split", "intensity", "The intensity the threshold method split the two classes at.")
_add(T, "Tissue Surface Intensity", "intensity", "The intensity the surface passes through: halfway between the two classes' median levels.")
_add(T, "Empty-Space Intensity", "intensity", "Median smoothed intensity of the empty-space class.")
_add(T, "Tissue Intensity", "intensity", "Median smoothed intensity of the tissue class.")
_add(T, "Tissue/Empty-Space Separation", "dimensionless", "Ashman's D between the two classes; low values mean the split is unreliable.")
_add(T, "Tissue Working Voxel Size (z, y, x microns)", "um", "The voxel size of the grid the tissue was measured on, z x y x x.")

# -- Vessel classes -----------------------------------------------------------
T = "Vessel classes (one section per class)"
_add(T, "Vessel Classes From", TEXT, "How the vessels were classed: by branch-order labels (large and small arterioles, capillaries, small and large venules) or by global masks (large and small vessels).")
_add(T, "Vessel Class Density Volume (micron³)", "um^3", "What every class's densities divide by: the volume Vessel Density in Tissue divides by.")
_add(T, "Vessel Class Density Volume Source", TEXT, "Where that volume came from.")
_add(T, "Vessel Class Haemodynamics", TEXT, "Why the flow rows are missing, when they are.")
_add(T, "Branch Orders", TEXT, "The branch-order labels of the class's vessels.")
_add(T, "Vessel Count", "vessels", "Vessels in the class (or group).")
_add(T, "Total Length (microns)", "um", "The class's summed centreline length.")
_add(T, "Share of Network Vessel Length (fraction)", "fraction", "The class's length over the whole network's.")
_add(T, "Mean Length (microns)", "um", "Mean vessel length in the class.")
_add(T, "Median Length (microns)", "um", "Median vessel length in the class.")
_add(T, "Length Density in Tissue (microns/micron³)", "um/um^3", "The class's length over Vessel Class Density Volume; the classes add up to the network's density.")
_add(T, "Length Density in Whole Image (microns/micron³)", "um/um^3", "The class's length over the whole image's volume.")
_add(T, "Mean Diameter (microns)", "um", "Mean of the class's vessel diameters (diameter_um, as the haemodynamics use them).")
_add(T, "Median Diameter (microns)", "um", "Median of the class's vessel diameters.")
_add(T, "Vessel Volume (micron³)", "um^3", "The class's volume, a cylinder per vessel: pi (d/2)^2 L summed.")
_add(T, "Volume Density in Tissue (micron³/micron³)", "fraction", "Vessel Volume over Vessel Class Density Volume.")
_add(T, "Vessel Surface Area (micron²)", "um^2", "The class's wall area, a cylinder per vessel: pi d L summed.")
_add(T, "Surface Density in Tissue (micron²/micron³)", "um^2/um^3", "Vessel Surface Area over Vessel Class Density Volume.")
_add(T, "Mean Tortuosity Index", "dimensionless", "Mean centreline length over straight end-to-end distance of the class's vessels.")
_add(T, "Median Tortuosity Index", "dimensionless", "Median centreline length over straight end-to-end distance of the class's vessels.")
_add(T, "Vessels Carrying Flow", "vessels", "Vessels with a solved flow (in the flux section, above the carrying floor too). Vessels behind dead ends carry exactly none and are left out of every flow row of the class.")
_add(T, "Mean Flow (nL/min)", "nL/min", "Mean |flow| of the class's vessels carrying flow.")
_add(T, "Median Flow (nL/min)", "nL/min", "Median |flow| of the class's vessels carrying flow.")
_add(T, "Mean Velocity (mm/s)", "mm/s", "Mean of each flowing vessel's flow over its lumen's cross-section (plasma mean velocity, not red-cell speed).")
_add(T, "Median Velocity (mm/s)", "mm/s", "Median of each flowing vessel's mean velocity.")
_add(T, "Mean Wall Shear Stress (Pa)", "Pa", "Mean of pressure drop x diameter / (4 x length), a uniform tube's wall shear stress, over the flowing vessels.")
_add(T, "Median Wall Shear Stress (Pa)", "Pa", "Median of the same wall shear stress.")
_add(T, "Mean Pressure (Pa)", "Pa", "Mean over the flowing vessels of each vessel's mean of its two end pressures.")

# -- Bifurcations -------------------------------------------------------------
T = "Bifurcations and Murray's law"
_add(T, "Murray Law Exponent", "dimensionless", "The exponent n the Murray ratios are taken at (3, Murray's law).")
_add(T, "Mean Murray Ratio", "dimensionless", "The mean over junctions of sum(daughter diameter^n) / parent diameter^n; 1 is Murray's law. Only junctions with a measured diameter on every branch count.")
_add(T, "Median Murray Ratio", "dimensionless", "The median of the same ratio.")
_add(T, "Murray Ratio Sample Count", "count", "Junctions the ratio was taken at.")
_add(T, "Murray Junctions Without Measured Diameters", "count", "Junctions left out because a branch had no measured diameter.")
_add(T, "Median Fitted Murray Exponent", "dimensionless", "The median over junctions of the exponent n with parent^n = sum(daughter^n), at junctions of two or more measured daughters all narrower than the parent.")
_add(T, "Fitted Murray Exponent IQR Low", "dimensionless", "The 25th percentile of the fitted exponents.")
_add(T, "Fitted Murray Exponent IQR High", "dimensionless", "The 75th percentile of the fitted exponents.")
_add(T, "Fitted Murray Exponent Sample Count", "count", "Junctions an exponent was fitted at.")

# -- Intercapillary distance --------------------------------------------------
T = "Intercapillary distance"
_add(T, "Mean Intercapillary Distance (microns)", "um", "Each vessel is sampled every 2 um; each sample's distance to the nearest vessel sharing no node with it; the vessel's value is the median along it. The mean over vessels.")
_add(T, "Median Intercapillary Distance (microns)", "um", "The median over vessels of the same spacing.")
_add(T, "Intercapillary Distance Sample Count", "vessels", "Vessels with a spacing.")
_add(T, "Mean Intercapillary Distance Between Flowing Vessels (microns)", "um", "The same spacing over the vessels carrying flow alone, each to its nearest unrelated flowing vessel; the mean.")
_add(T, "Median Intercapillary Distance Between Flowing Vessels (microns)", "um", "The median of that spacing.")
_add(T, "Intercapillary Distance Between Flowing Vessels Sample Count", "vessels", "Flowing vessels with a spacing.")
_add(T, "Flowing Over All Median Intercapillary Distance", "dimensionless", "The flowing vessels' median spacing over every vessel's: above 1 when the vessels not carrying flow sat between them.")

# -- Dead ends ----------------------------------------------------------------
T = "Dead ends"
_add(T, "Dead Ends", "count", "Dead ends: nodes only one vessel reaches.")
_add(T, "Dead Ends Leaving the Image", "count", "Dead ends whose tip touches an image face, or whose segmented vessel runs on to one: open in the tissue, never in a dead-end density.")
_add(T, "Dead Ends the Graph Stopped Short of (mask runs on)", "count", "Dead ends where the segmented vessel runs on past the tip.")
_add(T, "Dead Ends of Vessels Broken by the Segmentation", "count", "Dead ends with another vessel straight ahead across one stretch of background.")
_add(T, "Dead Ends That Are Artefacts", "count", "Dead ends off the mask, inside a neighbouring lumen or along a vessel's wall.")
_add(T, "Blind Ends (marked by hand)", "count", "Dead ends a person marked as real blind ends.")
_add(T, "Interior Dead Ends, Blind or Broken (unknown)", "count", "Interior dead ends no rule could class.")
_add(T, "Ends of Pieces Without a Junction", "count", "Dead ends of pieces of network with no junction.")
_add(T, "Interior Dead Ends per mm of Vessel (1/mm)", "1/mm", "Interior dead ends that could be the tissue's own (blind, unknown, broken, stopped short) per mm of vessel.")
_add(T, "Interior Dead Ends per mm^3 of Tissue (1/mm^3)", "1/mm^3", "The same dead ends per cubic millimetre of measured tissue.")
_add(T, "Interior Dead-End Chain Length, median (microns)", "um", "The median length of those dead ends' chains, tip to junction.", pattern=r"Interior Dead-End Chain Length, median \(microns\)")
_add(T, "Interior Dead-End Chain Length, 10th / 90th percentile (microns)", "um", "The 10th and 90th percentiles of those chain lengths.", pattern=r"Interior Dead-End Chain Length, (10|90)th percentile \(microns\)")
_add(T, "Vessel Length on Chains Cut by the Image Faces (microns)", "um", "Length of the chains ending at a dead end that leaves the image: lengths the image censored.")
_add(T, "Share of Vessel Length Cut by the Image Faces (fraction)", "fraction", "That length over the network's.")
_add(T, "Pieces Without a Junction", "count", "Pieces of network with no junction (lone vessels and chains).")
_add(T, "Length of Pieces Without a Junction (microns)", "um", "Their summed length.")
_add(T, "Vessel Length Carrying Flow (microns)", "um", "Length the solve gave a flow.")
_add(T, "Vessel Length Behind Interior Dead Ends, No Flow (microns)", "um", "Length that carries no flow because it leads only to dead ends inside the tissue.")
_add(T, "Vessel Length Behind Closed Image-Face Ends, No Flow (microns)", "um", "Length that carries no flow because it leads only to image-face ends with no boundary condition.")
_add(T, "Vessel Length Without Inlet and Outlet, No Flow (microns)", "um", "Length in pieces with no inlet and outlet, where no flow is defined.")

# -- Robustness ---------------------------------------------------------------
T = "Robustness"
_add(T, "Bridge Edge Count", "vessels", "Vessels whose occlusion alone splits the network into more pieces (graph bridges).")
_add(T, "Bridge Edge Fraction", "fraction", "Bridge vessels over all vessels.")
_add(T, "Articulation Point Count", "count", "Nodes whose removal splits the network into more pieces.")
_add(T, "Articulation Point Fraction", "fraction", "Articulation points over all nodes.")
_add(T, "Critical Bridge Edge Count", "vessels", "Bridges whose occlusion separates some inlet from some outlet.")
_add(T, "Critical Bridge Edge Fraction", "fraction", "Critical bridges over all bridges.")
_add(T, "Critical Articulation Point Count", "count", "Articulation points whose removal separates some inlet from some outlet.")
_add(T, "Critical Articulation Point Fraction", "fraction", "Critical articulation points over all articulation points.")

# -- Routes: bottlenecks and shunts -------------------------------------------
T = "Inlet-outlet routes"
_add(T, "Bottleneck Min-Cut Edge Count", "vessels", "Vessels in the minimum cut separating every inlet from every outlet.")
_add(T, "Bottleneck Min-Cut Capacity", FromRow("Bottleneck Min-Cut Capacity Unit"), "The capacity of that cut: its vessels under topology or length weighting, their summed conductance under resistance weighting.")
_add(T, "Bottleneck Min-Cut Capacity Unit", TEXT, "The unit of the capacity.")
_add(T, "Bottleneck Route Share Max", "fraction", "The largest share of shortest inlet-outlet routes (one per connected pair) passing through one vessel; 1 on a trunk every route uses.")
_add(T, "Bottleneck Route Share Mean", "fraction", "The mean of that share over vessels.")
_add(T, "Bottleneck Top Edges", TEXT, "The vessels with the largest route share: [node, node, key, share].")
_add(T, "Bottleneck Route Share Method", TEXT, "exact, or the inlets sampled as route sources.")
_add(T, "Shunt Route Fraction Threshold", "fraction", "A vessel is a shunt when the shortest inlet-outlet route through it is at most this share of the median one.")
_add(T, "Shunt Edge Count", "vessels", "Vessels on a shunt route.")
_add(T, "Shunt Edge Fraction", "fraction", "Shunt vessels over all vessels.")
_add(T, "Shunt Pathway Count", "count", "Connected groups of shunt vessels joining an inlet to an outlet.")
_add(T, "Shortest Inlet-Outlet Route", _SHUNT_ROUTE, "The shortest inlet-outlet route through any vessel, measured by the route weighting.")
_add(T, "Median Inlet-Outlet Route", _SHUNT_ROUTE, "The median over vessels of the shortest inlet-outlet route through each: the reference a shunt is judged against.")
_add(T, "Shunt Outflow Fraction", "fraction", "The share of the solved flow at the outlets that leaves through shunt vessels.")

# -- Current flow -------------------------------------------------------------
T = "Current flow"
_add(T, "Equivalent Inlet-Outlet Resistance", _CURRENT_FLOW_RESISTANCE, "The resistance between all inlets and all outlets with the route weighting's conductances (1 / flow at a unit pressure difference).")
_add(T, "Equivalent Inlet-Outlet Resistance Unit", TEXT, "The unit of that resistance.")
_add(T, "Current Flow Share Max", "fraction", "The largest share of the total inlet-outlet flow passing through one vessel when flow splits over every route by its conductance (Kirchhoff).")
_add(T, "Current Flow Share Mean", "fraction", "The mean of that share over vessels.")
_add(T, "Current Flow Top Vessels", TEXT, "The vessels with the largest share: [node, node, key, share].")
_add(T, "Kirchhoff Index", _CURRENT_FLOW_RESISTANCE, "The sum of resistance distances over every node pair of the largest connected piece (full mode, up to 3,000 nodes).")

# -- Occlusion ----------------------------------------------------------------
T = "Occlusion"
_add(T, "Occlusion Impact Method", TEXT, "exact, or how many vessels were evaluated.")
_add(T, "Occlusion Impact Assumptions", TEXT, "What the occlusion model leaves out.")
_add(T, "Occlusion Curves Assumptions", TEXT, "What the occlusion model leaves out.")
_add(T, "Occlusion Hypoperfusion Threshold (fraction)", "fraction", "A vessel counts as hypoperfused when its flow falls by at least this share.")
_add(T, "Occlusion Hypoperfused Length Unit", TEXT, "The unit of the hypoperfused lengths: um, or vessels when a vessel has no length.")
_add(T, "Occlusion Impact (Network) / By Vessel Type / By Branch Order", TEXT, "Sections: the single-vessel occlusion summary for the whole network, per vessel type and per branch order.", pattern=r"Occlusion Impact (\(Network\)|By Vessel Type|By Branch Order)")
_add(T, "Evaluated Count", "vessels", "Vessels whose single occlusion was evaluated.")
_add(T, "Mean Flow Loss", "fraction", "Blocking one vessel alone: the share of total inlet-outlet flow lost (with the route weighting's fixed conductances). The mean over vessels.")
_add(T, "Median Flow Loss", "fraction", "The median of that flow loss.")
_add(T, "P90 Flow Loss", "fraction", "The 90th percentile of that flow loss.")
_add(T, "Max Flow Loss", "fraction", "The largest flow loss from blocking one vessel.")
_add(T, "Mean Hypoperfused Length", _HYPOPERFUSED_LENGTH, "Blocking one vessel alone: the length of other vessels whose flow falls by at least the hypoperfusion threshold. The mean over vessels.")
_add(T, "Max Hypoperfused Length", _HYPOPERFUSED_LENGTH, "The largest such length from blocking one vessel.")
_add(T, "Most Critical Vessel", TEXT, "The vessel whose occlusion loses the most flow: [node, node, key, flow loss].")
_add(T, "Top Vessels", TEXT, "The vessels whose occlusion loses the most flow: [node, node, key, flow loss].")
_add(T, "Vessels Losing Over N% Of Flow", "vessels", "Vessels whose occlusion alone loses more than that share of total inlet-outlet flow.", pattern=r"Vessels Losing Over \d+% Of Flow")
_add(T, "Occlusion Curve (random | highest_current_flow_share | narrowest_diameter)", TEXT, "Sections: vessels blocked cumulatively, at random (seeded, averaged), busiest first, or narrowest first; or why a curve is missing.", pattern=r"Occlusion Curve \([a-z_]+\)")
_add(T, "Robustness Index", "fraction", "The mean remaining flow, as a share of the intact network's, across the removal range; 1 is unaffected.")
_add(T, "Removal Fraction At 50% Flow", "fraction", "The share of vessels removed at which flow halves.")
_add(T, "Curve [fraction removed, flow fraction, outlets reachable]", TEXT, "The curve itself: share of vessels removed, remaining flow share, share of outlets still connected to an inlet.")

# -- Territories --------------------------------------------------------------
T = "Perfusion territories"
_add(T, "Arteriolar / Venular Territory Count", "count", "Inlets (arteriolar) or outlets (venular) that reach at least one vessel first: territories.", pattern=r"(Arteriolar|Venular) Territory Count")
_add(T, "Arteriolar / Venular Watershed Vessel Count", "vessels", "Vessels on a border between two territories: their two ends are reached first from different inlets (or outlets).", pattern=r"(Arteriolar|Venular) Watershed Vessel Count")
_add(T, "Arteriolar / Venular Watershed Length Fraction", "fraction", "Watershed vessels' length over the network's.", pattern=r"(Arteriolar|Venular) Watershed Length Fraction")
_add(T, "Arteriolar / Venular Unreached Vessel Count", "vessels", "Vessels no inlet (or outlet) reaches.", pattern=r"(Arteriolar|Venular) Unreached Vessel Count")
_add(T, "Arteriolar / Venular Territory Size Min / Median / Max (vessels)", "vessels", "The smallest, median and largest territory, in vessels.", pattern=r"(Arteriolar|Venular) Territory Size (Min|Median|Max) \(vessels\)")
_add(T, "Arteriolar / Venular Territory Length Median", "um", "The median territory's summed vessel length.", pattern=r"(Arteriolar|Venular) Territory Length Median")

# -- Capillary bed joins ------------------------------------------------------
T = "Capillary bed joins"
_add(T, "Capillary Beds", "count", "Capillary beds, one per feed (the nodes capillary branch orders are counted from).")
_add(T, "Capillary Beds Joined", "count", "Beds that meet another bed on a watershed vessel.")
_add(T, "Capillary Beds Joined (fraction)", "fraction", "Beds joined over all beds.")
_add(T, "Capillary Bed Join Order, Kaplan-Meier Median", "dimensionless", "A bed's join order is the lowest capillary branch order on a watershed vessel with another bed; the Kaplan-Meier median over beds, with beds that never join censored.")
_add(T, "Capillary Bed Join Distance, Kaplan-Meier Median (microns)", "um", "The same Kaplan-Meier median of the route length from the feed to where its bed joins another.")
_add(T, "Capillary Beds Censored, Leaving the Image / Draining to a Venule / Ending Blind", "count", "Beds that never join another, by why: they leave the image, drain into a venule, or end blind.", pattern=r"Capillary Beds Censored, (Leaving the Image|Draining to a Venule|Ending Blind)")
_add(T, "Capillary Beds Joining Within Their Arteriolar Tree", "count", "Joins between beds fed from the same inlet's tree.")
_add(T, "Capillary Beds Joining Another Arteriolar Tree", "count", "Joins between beds fed from different inlets' trees.")
_add(T, "Capillaries No Feed Reaches", "vessels", "Capillaries no feed reaches.")
_add(T, "Graph Built With Short Loops Removed Under (microns)", "um", "remove_short_loops_under_um when the graph was built: loops shorter than this were broken, so the earliest joins may be missing.")

# -- Transit time -------------------------------------------------------------
T = "Transit time"
_add(T, "Transit Time Assumptions", TEXT, "What the transit-time model assumes.")
_add(T, "Mean Transit Time (s)", "s", "The mean time blood takes from the inlets to the outlets, each vessel's time its volume over its flow, mixed by flow at every junction. Open ends feeding or draining the network count as inlets and outlets; vessels behind dead ends carry no flow and take no part.")
_add(T, "Transit Time SD (s)", "s", "The standard deviation of that inlet-to-outlet time.")
_add(T, "Transit Time CV", "dimensionless", "Transit Time SD over Mean Transit Time.")
_add(T, "Capillary Mean Transit Time (s)", "s", "The mean of the same time spent in capillaries alone.")
_add(T, "Capillary Transit Time Heterogeneity CTH (s)", "s", "The standard deviation of the capillary transit time (CTH).")
_add(T, "Capillary Transit Time CV", "dimensionless", "CTH over the capillary mean transit time.")
_add(T, "Vessel Transit Time By Vessel Type", TEXT, "Section: each vessel type's single-vessel transit times.")
_add(T, "Flow-Weighted Mean (s)", "s", "The flow-weighted mean of the type's single-vessel transit times (volume over flow).")
_add(T, "Median (s)", "s", "The median of the type's single-vessel transit times.")

# -- Flux heterogeneity -------------------------------------------------------
T = "Flux heterogeneity and red cells"
_add(T, "Flux Heterogeneity Assumptions", TEXT, "What the rows assume.")
_add(T, "Carrying Flow Floor (m³/s)", "m^3/s", "A vessel carries flow when the solve reached it and its |flow| is above this: 1e-9 of the network's largest. Vessels behind dead ends carry exactly none, so no row of this section counts them.")
_add(T, "Red Cell Volume (fL)", "fL", "The mean red-cell volume the red-cell rows count cells by (red_cell_volume_fl).")
_add(T, "Haematocrit Source", TEXT, "Which discharge haematocrit the red-cell rows used: the vessels' own, or the run's one.")
_add(T, "Vessels", "vessels", "Vessels in the group.")
_add(T, "Vessels / Capillaries Carrying Flow", "vessels", "Vessels (or capillaries) carrying flow.", pattern=r"(Vessels|Capillaries) Carrying Flow")
_add(T, "Vessels / Capillaries Carrying Flow Fraction", "fraction", "Those over all the group's vessels.", pattern=r"(Vessels|Capillaries) Carrying Flow Fraction")
_add(T, "Vessels / Capillaries Carrying Flow Without Diameter", "vessels", "Carrying vessels with no diameter, left out of the velocity rows.", pattern=r"(Vessels|Capillaries) Carrying Flow Without Diameter")
_add(T, "Vessels / Capillaries Carrying Flow Without Haematocrit", "vessels", "Carrying vessels with no haematocrit, left out of the red-cell rows.", pattern=r"(Vessels|Capillaries) Carrying Flow Without Haematocrit")
_add(T, "Flowing Vessel / Capillary Length (microns)", "um", "The carrying vessels' summed length.", pattern=r"Flowing (Vessel|Capillary) Length \(microns\)")
_add(T, "Flowing Vessel / Capillary Length Fraction", "fraction", "That length over the group's.", pattern=r"Flowing (Vessel|Capillary) Length Fraction")
_add(T, "Flowing Vessel / Capillary Volume (micron³)", "um^3", "The carrying vessels' volume, a cylinder per vessel.", pattern=r"Flowing (Vessel|Capillary) Volume \(micron³\)")
_add(T, "Flowing Vessel / Capillary Volume Fraction", "fraction", "That volume over the group's.", pattern=r"Flowing (Vessel|Capillary) Volume Fraction")
_add(T, "Vessel / Capillary Flow Mean / SD (nL/min)", "nL/min", "Mean and sample standard deviation of |flow| over the carrying vessels.", pattern=r"(Vessel|Capillary) Flow (Mean|SD) \(nL/min\)")
_add(T, "Vessel / Capillary Velocity Mean / SD (mm/s)", "mm/s", "Mean and sample standard deviation of mean velocity (flow over the lumen's cross-section) over the carrying vessels.", pattern=r"(Vessel|Capillary) Velocity (Mean|SD) \(mm/s\)")
_add(T, "Vessel / Capillary Red Cell Flux Mean / SD (cells/s)", "cells/s", "Mean and sample standard deviation of the predicted red-cell flux, flow x discharge haematocrit / red-cell volume. A prediction, not a count.", pattern=r"(Vessel|Capillary) Red Cell Flux (Mean|SD) \(cells/s\)")
_add(T, "Vessel / Capillary Red Cell Velocity Mean / SD (mm/s)", "mm/s", "Mean and sample standard deviation of the predicted red-cell velocity, mean velocity x discharge / tube haematocrit (Fahraeus, Pries et al. 1990).", pattern=r"(Vessel|Capillary) Red Cell Velocity (Mean|SD) \(mm/s\)")
_add(T, "Vessel / Capillary Tube Haematocrit Mean", "fraction", "The mean predicted tube haematocrit: the haematocrit inside the vessel, below the discharge haematocrit (Fahraeus, Pries et al. 1990).", pattern=r"(Vessel|Capillary) Tube Haematocrit Mean")
_add(T, "Vessel / Capillary Red Cell Linear Density Mean (cells/mm)", "cells/mm", "The mean predicted spacing of red cells: tube haematocrit x cross-section / red-cell volume, cells per mm of vessel.", pattern=r"(Vessel|Capillary) Red Cell Linear Density Mean \(cells/mm\)")
_add(T, "Vessel / Capillary Flow CV", "dimensionless", "The coefficient of variation (sample SD over mean) of |flow| over the carrying vessels.", pattern=r"((Vessel|Capillary) )?Flow CV")
_add(T, "Vessel / Capillary Velocity CV", "dimensionless", "The coefficient of variation of mean velocity, flow over the lumen's cross-section (plasma, not red-cell speed), over the carrying vessels.", pattern=r"((Vessel|Capillary) )?Velocity CV")
_add(T, "Vessel / Capillary Red Cell Flux CV", "dimensionless", "The coefficient of variation of the predicted red-cell flux over the carrying vessels; with one haematocrit everywhere it is the flow's CV.", pattern=r"((Vessel|Capillary) )?Red Cell Flux CV")
_add(T, "Vessel / Capillary Red Cell Velocity CV", "dimensionless", "The coefficient of variation of the predicted red-cell velocity (Fahraeus, Pries et al. 1990) over the carrying vessels.", pattern=r"((Vessel|Capillary) )?Red Cell Velocity CV")
_add(T, "Capillary Flux Heterogeneity Status", TEXT, "Why the capillary rows are missing.")
_add(T, "Flux Heterogeneity By Vessel Type", TEXT, "Section: the same rows per vessel type.")

# -- Loops --------------------------------------------------------------------
T = "Loops"
_add(T, "Cyclomatic Number", "count", "Independent loops: vessels - nodes + connected pieces.")
_add(T, "Cyclomatic Number Per Node", "dimensionless", "The cyclomatic number over the node count.")
_add(T, "Loop Length Weighting", TEXT, "What loop lengths are measured in: length (um), or vessels where a vessel has no length.")
_add(T, "Loop Method", TEXT, "exact, or how many junction pairs were sampled.")
_add(T, "Vessels On No Loop", "vessels", "Vessels no loop runs through (bridges).")
_add(T, "Vessels On No Loop Fraction", "fraction", "Those over the vessels evaluated.")
_add(T, "Loop Length P10 / P25 / Median / P75 / P90 (um or vessels)", UNIT_IN_NAME, "Percentiles over vessels of the shortest loop through each: the vessel plus the shortest detour between its ends.", pattern=r"Loop Length (P10|P25|Median|P75|P90) \((um|vessels)\)")
_add(T, "Median Loop Length By Vessel Type", TEXT, "Section: the median shortest loop through each vessel type's vessels.")

# -- Strahler -----------------------------------------------------------------
T = "Strahler orders"
_add(T, "Strahler Method", TEXT, "The arterial and venous trees, or the whole network's tree from the inlets (approximate) without an arteriole/venule classification.")
_add(T, "Strahler Status", TEXT, "Why no orders were found.")
_add(T, "Strahler Arterial / Venous / Network", TEXT, "Sections: one tree's Strahler orders, or why it has none.", pattern=r"Strahler [A-Z][a-z]+")
_add(T, "Max Order", "count", "The tree's highest Strahler order.")
_add(T, "Orders", TEXT, "Section: one row per Strahler order.")
_add(T, "Element Count", "count", "Elements of that order: runs of consecutive vessels of one order.")
_add(T, "Mean Element Length (um or vessels)", UNIT_IN_NAME, "The mean length of the order's elements.", pattern=r"Mean Element Length \((um|vessels)\)")
_add(T, "Mean Element Diameter (um)", "um", "The mean length-weighted diameter of the order's elements.")
_add(T, "Bifurcation Ratio R_B", "dimensionless", "Horton's ratio of element counts between successive orders, from a log-linear fit.")
_add(T, "Length Ratio R_L", "dimensionless", "Horton's ratio of mean element lengths between successive orders.")
_add(T, "Diameter Ratio R_D", "dimensionless", "Horton's ratio of mean element diameters between successive orders.")
_add(T, "Unordered Vessels (anastomoses or unreached)", "vessels", "The tree's vessels with no order: closing a loop within it, or not reached from its roots.")

# -- Spectral -----------------------------------------------------------------
T = "Algebraic connectivity"
_add(T, "Algebraic Connectivity lambda_2", _LAMBDA_2, "The second-smallest eigenvalue of the conductance-weighted Laplacian of the largest connected piece: 0 when it falls apart, larger the harder it is to cut in two.")
_add(T, "Normalised Algebraic Connectivity", "dimensionless", "lambda_2 over the mean weighted degree, comparable across networks.")
_add(T, "Algebraic Connectivity Component Nodes", "count", "Nodes in the piece lambda_2 was found on.")
_add(T, "Fiedler Split Vessel Counts [A, B]", "vessels", "Vessels on each side of the Fiedler vector's split, larger side first.")
_add(T, "Fiedler Cut Vessel Count", "vessels", "Vessels the Fiedler split cuts.")

# -- Other topology -----------------------------------------------------------
T = "Topology"
_add(T, "Degree Assortativity", "dimensionless", "The correlation between the degrees of the two nodes of each vessel, -1 to 1; vascular trees are usually below 0.")
_add(T, "Rich Club Coefficient", "dimensionless", "The density of vessels among nodes of degree above the threshold (unnormalised), at the highest threshold any node passes.")
_add(T, "Rich Club Coefficient Degree Threshold", "count", "That degree threshold.")
_add(T, "Max Core Number", "count", "The highest k for which a k-core (every node with at least k neighbours inside it) exists.")
_add(T, "Mean Core Number", "count", "The mean over nodes of each node's core number.")
_add(T, "Innermost Core Size", "count", "Nodes in the innermost core.")
_add(T, "Flow Hierarchy", "fraction", "The share of vessels, oriented by their solved flow, on no directed cycle; 1 means the flow never recirculates.")
_add(T, "Flow Hierarchy Edges Skipped", "vessels", "Vessels with no determinate flow direction, left out: those the solve's flow state says carry no flow (behind dead ends, in pieces with no inlet and outlet) and those with zero or no flow.")
_add(T, "Flow Hierarchy Vessels Behind Dead Ends (excluded)", "vessels", "Of those, the vessels behind dead ends (inside the tissue, or at an image face with no boundary condition): their flow is exactly 0, so they have no direction.")
_add(T, "Path Efficiency", "1/um", "1 over the mean shortest-path length between node pairs.")
_add(T, "Average Shortest Path Length (microns)", "um", "The mean shortest path, by centreline length, between node pairs.")
_add(T, "Path Efficiency Pair Sample Size", "count", "Node pairs measured.")
_add(T, "Path Efficiency Pair Coverage", "fraction", "Node pairs measured over all node pairs.")
_add(T, "Community Count", "count", "Communities found by greedy modularity on the network's topology.")
_add(T, "Largest Community Size", "count", "Nodes in the largest community.")
_add(T, "Mean Community Size", "count", "The mean nodes per community.")
_add(T, "Community Method", TEXT, "How the communities were found.")
_add(T, "communities", TEXT, "Full mode: every community's nodes.")
_add(T, "Betweenness Mean", "fraction", "The mean over nodes of their betweenness centrality: the share of shortest paths (counted in vessels) between other node pairs passing through the node.")
_add(T, "Betweenness Max", "fraction", "The largest node betweenness centrality.")
_add(T, "Betweenness Top Nodes", TEXT, "The nodes of highest betweenness: [node, value].")
_add(T, "Betweenness Method", TEXT, "exact, or the number of sampled sources.")
_add(T, "A node ID", "fraction", "Full mode: that node's betweenness centrality.", pattern=r"-?\d+")

# -- Haemodynamics ------------------------------------------------------------
T = "Haemodynamics"
_add(T, "Two-Point Equivalent Resistance (Pa.s/m^3)", "Pa.s/m^3", "The resistance between the first inlet and the first outlet: the pressure difference driving a unit flow between them alone.")
_add(T, "Two-Point Resistance Nodes", TEXT, "The two nodes it was found between.")
_add(T, "Network Resistance, Pressure Drop / Total Inflow (Pa.s/m^3)", "Pa.s/m^3", "Inlet pressure minus outlet pressure over the total flow in through every designated inlet; the flow through open ends is reported beside it, not in it.")
_add(T, "Open-End Model", TEXT, "The boundary condition on vessels leaving the image (open_end_model): estimated, terminal_resistance or outlet. Written only when open ends were given one.")
_add(T, "Open Ends Given a Condition", "count", "Dead ends leaving the image, in a piece with a designated inlet or outlet, that the solve gave the boundary condition.")
_add(T, "Inflow Through Open Ends (m^3/s)", "m^3/s", "The flow entering the network through those open ends.")
_add(T, "Outflow Through Open Ends (m^3/s)", "m^3/s", "The flow leaving the network through them.")

# -- The per-bed table (the capillary_bed_joins CSV's columns) ---------------
T = "Capillary bed table (the capillary_bed_joins CSV's columns)"
_add(T, "bed", TEXT, "The bed's label, its feed's (F01, F02, ...).")
_add(T, "feed_node", TEXT, "The node the bed is fed from.")
_add(T, "arteriolar_tree", TEXT, "The arteriolar tree (inlet) the feed belongs to.")
_add(T, "vessel_count", "vessels", "Capillaries in the bed.")
_add(T, "bed_length_um", "um", "The bed's summed capillary length.")
_add(T, "join_order", "dimensionless", "The lowest capillary branch order on a watershed vessel with another bed; empty for a censored bed.")
_add(T, "join_distance_um", "um", "The route length from the feed to where the bed first joins another; empty for a censored bed.")
_add(T, "censored", TEXT, "True when the bed never joins another.")
_add(T, "censor_reason", TEXT, "Why: it leaves the image, drains into a venule, or ends blind.")
_add(T, "joins_bed", TEXT, "The bed it first joins.")
_add(T, "join_class", TEXT, "within_tree when the two beds are fed from one inlet's tree, between_trees otherwise.")

# -- Shared rows --------------------------------------------------------------
T = "Rows every analysis may write"
_add(T, "(analysis) Route Weighting", TEXT, "The conductance model a route analysis used -- topology (one per vessel), length or resistance -- and any fallback; it sets the unit of the analysis's lengths and resistances.", pattern=r".+ Route Weighting")
_add(T, "(analysis) Status", TEXT, "Why an analysis or section reported nothing (N/A and its reason).", pattern=r".+ Status")
_add(T, "(analysis) Nodes Both Inlet And Outlet (excluded)", "count", "Nodes given as both inlet and outlet, left out of the analysis.", pattern=r".+ Nodes Both Inlet And Outlet \(excluded\)")
_add(T, "(analysis) Open Ends Taken As Inlets", "count", "Open ends leaving the image, given a boundary condition by the solve (open_end_model), through which blood enters the network: the analysis took them as inlets beside the designated ones.", pattern=r".+ Open Ends Taken As Inlets")
_add(T, "(analysis) Open Ends Taken As Outlets", "count", "Open ends through which blood leaves the network, taken as outlets beside the designated ones.", pattern=r".+ Open Ends Taken As Outlets")
_add(T, "arteriole, capillary, venule ... (a vessel type)", _LOOP_LENGTH, "Under Median Loop Length By Vessel Type: the median shortest loop through that type's vessels.", pattern=r"[a-z_]+")

# -- Branch order statistics (the branch_statistics CSV's columns) -----------
T = "Branch order statistics (the branch_statistics CSV's columns)"
_add(T, "Branch Order", TEXT, "The branch-order label (Large_Art, Art, BO for capillaries, Ven, Large_Ven, and a number).")
_add(T, "Edge Count", "vessels", "Vessels of that branch order.")
_add(T, "Tortuosity Sample Count", "vessels", "Vessels whose tortuosity was measured.")
_add(T, "Mean Emergence Angle (degrees)", "degrees", "The mean angle between the parent's and each daughter's tangent where a vessel of that order leaves its parent.")
_add(T, "Emergence Angle Sample Count", "count", "Emergence angles measured.")
_add(T, "Mean Pressure Drop (Pa)", "Pa", "The mean pressure drop along the order's vessels carrying flow; vessels behind dead ends and in pieces with no inlet and outlet are left out.")
_add(T, "Total Pressure Drop (Pa)", "Pa", "The summed pressure drop along the same vessels.")
_add(T, "Pressure Drop Sample Count", "vessels", "Vessels with a pressure drop, those behind dead ends left out.")
_add(T, "Diameter Coefficient of Variation", "dimensionless", "The standard deviation over the mean of the order's diameters.")
_add(T, "Diameter Sample Count", "vessels", "Vessels with a diameter.")

#: Every statistic, matched in this order (the first whose pattern matches a
#: row's name is its term: the shared rows' ``.+`` patterns come after every
#: literal name).
GLOSSARY: tuple[Term, ...] = tuple(_TERMS)
del T, _TERMS


@lru_cache(maxsize=None)
def term_for(name: str) -> Optional[Term]:
    """The glossary term of a statistics row named *name* (the dict key, its
    unit in parentheses where it has one), or None. A unit spelled another
    way ("(um)" for "(microns)"), or left off, finds the same term."""
    shown, unit = split_name(name)
    candidates = [name]
    if unit is not None:
        candidates += [f"{shown} ({spelling})" for spelling, same in UNIT_SPELLINGS.items() if same == unit]
        candidates.append(shown)
    for candidate in candidates:
        for term in GLOSSARY:
            if term.regex.fullmatch(candidate):
                return term
    return None


_TRAILING_PARENTHESES = re.compile(r"^(.*?)\s*\(([^()]+)\)\s*$")


def split_name(name: str) -> tuple[str, Optional[str]]:
    """*name* without its trailing unit, and that unit (one of :data:`UNITS`);
    parentheses that are not a unit ("(Node Positions)") stay in the name."""
    match = _TRAILING_PARENTHESES.match(name)
    if match:
        unit = canonical_unit(match.group(2))
        if unit is not None:
            return match.group(1), unit
    return name, None


def unit_for(name: str, rows: Mapping[str, Any] = {}) -> str:
    """The unit of the row named *name*: its term's, read from *rows* (every
    row of the same report by name) where the run decides it; the unit its
    name spells when it has no term; "" for a text row or an unknown one."""
    term = term_for(name)
    in_name = split_name(name)[1]
    if term is None or term.unit == UNIT_IN_NAME:
        return in_name or ""
    if isinstance(term.unit, (ByWeighting, FromRow)):
        return term.unit.resolve(rows)
    return term.unit


def definition_for(name: str) -> str:
    """What the row named *name* is, or "" when the glossary has no term."""
    term = term_for(name)
    return term.definition if term is not None else ""


def csv_metric_and_unit(
    metric: str, unit: str, rows: Mapping[str, Any] = {}
) -> tuple[str, str, str]:
    """A statistics CSV row's (Metric, Unit) read back, from this version's
    CSVs or older ones, as ``(name, metric, unit)``: the row's name as the
    statistics dict spelled it, the Metric column this version writes, and
    its unit.

    Older CSVs took whatever trailing parentheses a name had for its unit
    ("Fractal Dimension" with the unit "Node Positions") and spelled units
    microns or micron³; read through here, a row from either gives the same
    metric and unit, so a batch mixing runs from before and after lines up.
    *rows* is every row of the file by name, for the units the run decides.
    """
    metric, unit = metric.strip(), (unit or "").strip()
    if unit and (canonical_unit(unit) is None or term_for(f"{metric} ({unit})") is not None):
        # Parentheses an older CSV took for a unit, or a unit the name spells.
        name = f"{metric} ({unit})"
    else:
        name = metric
    shown, _in_name = split_name(name)
    return name, shown, unit_for(name, rows) or canonical_unit(unit) or ""


def _unit_text(unit: Unit) -> str:
    if unit == UNIT_IN_NAME:
        return "as named (um or vessels)"
    if isinstance(unit, ByWeighting):
        return (
            f"by the route weighting ({unit.row}): resistance {unit.resistance}, "
            f"length {unit.length}, topology {unit.topology}"
        )
    if isinstance(unit, FromRow):
        return f"given by the row {unit.row}"
    return unit or "(text)"


def glossary_markdown() -> str:
    """The glossary page: the units, then every term by topic."""
    lines = [
        "# Statistics glossary",
        "",
        "<!-- Written by `python -m haemolynx.statistics.glossary` from",
        "     src/haemolynx/statistics/glossary.py; edit that, not this. -->",
        "",
        "Every row of `{stem}_statistics.csv`, `{stem}_statistics_connected.csv` (and a",
        "perturbation's statistics CSV) and `{stem}_tissue_volume.csv`, and every column of",
        "`{stem}_branch_statistics.csv` and `{stem}_capillary_bed_joins.csv`: what it is and",
        "its unit. Each statistics CSV row carries its unit in the Unit column and this",
        "definition in the Notes column. A name with `/` or `N` stands for each of the",
        "names it lists.",
        "",
        "## Units",
        "",
        "| Unit | Meaning |",
        "|---|---|",
    ]
    lines += [f"| `{unit}` | {meaning} |" for unit, meaning in UNITS.items()]
    topic = None
    for term in GLOSSARY:
        if term.topic != topic:
            topic = term.topic
            lines += ["", f"## {topic}", "", "| Statistic | Unit | Definition |", "|---|---|---|"]
        lines.append(f"| {term.shown} | {_unit_text(term.unit)} | {term.definition} |")
    return "\n".join(lines) + "\n"


#: Where the glossary page is kept, from the repository root.
GLOSSARY_PAGE = Path("docs") / "statistics_glossary.md"


def write_glossary_markdown(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.write_text(glossary_markdown(), encoding="utf-8", newline="\n")
    return path


if __name__ == "__main__":
    from haemolynx.parsers import configure_console_logging

    configure_console_logging()
    logger.info("Wrote %s", write_glossary_markdown(Path(__file__).resolve().parents[3] / GLOSSARY_PAGE))
