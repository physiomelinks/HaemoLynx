import numpy as np
import networkx as nx
from numba import jit
import logging
from typing import Optional, Dict, List, Sequence, Tuple, Any

logger = logging.getLogger(__name__)

# Flow leaves the resistance solve in mmHg um^3 / cP, not um^3/s. See the constant's own
# definition for the derivation and for what coupling the two unconverted did.
from .resistance import POISEUILLE_FLOW_TO_UM3_PER_S  # noqa: E402

def calculate_blood_oxygen_content(po2_mmHg: float, hematocrit: float, pco2_mmHg: float = 40.0, ph: float = 7.4) -> float:
    """
    Calculates total oxygen content in blood (mmol/L) using the Hill Equation.
    Incorporates the Bohr Effect: P50 shifts based on PCO2 and pH.
    """
    if po2_mmHg <= 0.0:
        return 0.0
        
    alpha_o2 = 1.34e-3  # Solubility of O2 in plasma (mmol/L per mmHg)
    hill_n = 2.7
    c_hb_max = 0.446 * 20.4 / 0.45 # Scale to pure RBC
    
    # Bohr Effect: Shift P50 based on pH and PCO2
    # Baseline P50 = 26.0 mmHg at pH=7.4, PCO2=40
    # Kelman (1966) / Severinghaus (1979) empirical Bohr shift
    log_p50 = np.log10(26.0) - 0.4 * (ph - 7.4) + 0.06 * np.log10(max(pco2_mmHg, 1.0) / 40.0)
    p50 = 10 ** log_p50
    
    dissolved = alpha_o2 * po2_mmHg
    saturation = (po2_mmHg ** hill_n) / ((po2_mmHg ** hill_n) + (p50 ** hill_n))
    bound = hematocrit * c_hb_max * saturation
    
    return float(dissolved + bound)

def calculate_blood_co2_content(pco2_mmHg: float, hematocrit: float, po2_mmHg: float = 100.0) -> float:
    """
    Calculates total carbon dioxide content in whole blood (mmol/L).

    McHardy's whole-blood CO2 dissociation curve (McHardy 1967, as quoted by Mallat & Vallet
    2021), in vol% and converted to mmol/L. It is total content, so dissolved CO2 is already in
    it. Its Hb term gives the haematocrit dependence; its saturation term is the Haldane effect
    (deoxygenated blood carries more CO2).

    Saturation uses a fixed P50 of 26 mmHg, not the Bohr-shifted P50 of the oxygen curve
    (reference §11 row 19). At H near 0 the Hb term makes content peak near PCO2 135 mmHg, far
    above any tissue PCO2; below that the curve rises with PCO2 for every H.

    The saturation term is scaled by Hb / 15 g/dL. McHardy fitted blood at normal Hb, where the
    factor is 1, so the curve is his at H 0.45. Unscaled, the term gave plasma (H near 0) a full
    Haldane effect with no haemoglobin to cause it: a plasma-skimmed vessel losing O2 gained CO2
    capacity, its PCO2 fell, and tissue CO2 came out below arterial.
    """
    if pco2_mmHg <= 0.0:
        return 0.0

    sat_o2 = (po2_mmHg ** 2.7) / ((po2_mmHg ** 2.7) + (26.0 ** 2.7))

    # g/dL. The same Hb that c_hb_max in calculate_blood_oxygen_content assumes:
    # 20.4 vol% O2 capacity = 1.36 mL/g x 15 g/dL at H = 0.45.
    hb_g_dL = 15.0 * hematocrit / 0.45

    c_co2_vol_pct = (11.02 * pco2_mmHg ** 0.396
                     - (15.0 - hb_g_dL) * 0.015 * pco2_mmHg
                     + (95.0 - 100.0 * sat_o2) * 0.064 * (hb_g_dL / 15.0))

    # One mmol of CO2 is 22.26 mL STPD, so vol% (mL per 100 mL) / 2.226 is mmol/L.
    return float(c_co2_vol_pct / 2.226)

def calculate_ph_from_pco2(pco2_mmHg: float | np.ndarray, hco3_mmol_L: float = 24.0) -> float | np.ndarray:
    """
    Calculates tissue/blood pH using the Henderson-Hasselbalch equation.
    Supports both scalar and numpy array inputs.
    """
    pKa = 6.1
    alpha_co2 = 0.03
    
    # Handle scalar or array
    pco2_safe = np.maximum(pco2_mmHg, 1e-12) if isinstance(pco2_mmHg, np.ndarray) else max(pco2_mmHg, 1e-12)
    ph = pKa + np.log10(hco3_mmol_L / (alpha_co2 * pco2_safe))
    
    if isinstance(ph, np.ndarray):
        return ph
    return float(ph)


class PerfusionGrid:
    """
    A 3D structured grid for tissue diffusion modeling.
    Coordinates are natively handled in [z, y, x] to perfectly align with ImageLynx graph conventions
    and VTK exports without flipping.

    **Extent.** By default the grid spans the graph's node bounding box, so it stops where the
    vasculature stops. Where a segmented tissue volume reaches past that, the tissue in the gap
    is not represented at all: ``mask_fraction_per_cell`` drops it, and the solve describes less
    tissue than was handed in. Two of the six carotid body specimens lose 4.35% and 7.54% of
    their glomus volume that way (S28).

    ``bounds_zyx`` extends the grid to cover a requested region, given as ``(min_zyx, max_zyx)``
    in micrometres. It is a **union** with the node bounding box and never a replacement: a
    requested bound tighter than the vasculature would leave graph nodes outside the grid, where
    ``get_cell_index`` returns -1 and their flow silently stops being a source.

    Padding is a modelling choice rather than a correction. The added cells contain tissue but
    no vessels, because the vessels supplying them were cut off by the region crop rather than
    absent from the organ, so they consume without a local source.

    That was expected to drive the padded faces to artefactual anoxia, and **measured, it does
    not**: on SHR-A and SHR-C, padding moves mean PO2 within TH by -0.77 and -0.66 mmHg, and the
    hypoxic fraction below 10 mmHg stays at exactly zero. The reason is the same property that
    makes H2 section 2.3 inert. The oxygen diffusion length is 20 to 45 um against an
    unvascularised rim of 12 to 13 um, so the added cells are supplied by diffusion from their
    neighbours. Tissue this densely vascularised does not go hypoxic for want of a local vessel.

    Read a padded solve as tissue-complete and supply-incomplete at its edges; the default is
    the reverse. On this cohort the difference is about 2% in PO2 and nothing in hypoxic
    fraction, but that is a fact about carotid body vascular density and should be re-measured
    rather than assumed on a sparser bed.
    """
    def __init__(self, G: nx.MultiGraph, grid_resolution_xyz: Tuple[float, float, float],
                 bounds_zyx: Tuple[Sequence[float], Sequence[float]] | None = None):
        # 1. Get physical bounds from graph nodes
        pos = nx.get_node_attributes(G, "pos")
        if not pos:
            raise ValueError("Graph G must have 'pos' attributes (z, y, x).")
            
        # ImageLynx convention: pos is [z, y, x] in physical units (micrometers)
        nodes_zyx = np.array(list(pos.values()))
        
        # We assume resolution is passed as (x,y,z), so we flip it to (z,y,x) to match
        self.res = np.array([grid_resolution_xyz[2], grid_resolution_xyz[1], grid_resolution_xyz[0]], dtype=float)
        
        # Pad by half resolution to ensure all nodes are inside
        lo = np.min(nodes_zyx, axis=0) - self.res * 0.5             # actually min_zyx here
        hi = np.max(nodes_zyx, axis=0) + self.res * 0.5             # actually max_zyx here

        if bounds_zyx is not None:
            want_lo = np.asarray(bounds_zyx[0], dtype=float)
            want_hi = np.asarray(bounds_zyx[1], dtype=float)
            if want_lo.shape != (3,) or want_hi.shape != (3,):
                raise ValueError(
                    f"bounds_zyx must be two (z, y, x) triples, got shapes "
                    f"{want_lo.shape} and {want_hi.shape}")
            if np.any(want_hi <= want_lo):
                raise ValueError(
                    f"bounds_zyx must have max strictly above min, got {want_lo} to {want_hi}")
            # Union, never replacement. See the class docstring.
            lo = np.minimum(lo, want_lo)
            hi = np.maximum(hi, want_hi)

        self.min_xyz = lo
        self.max_xyz = hi
        
        self.dims = np.ceil((self.max_xyz - self.min_xyz) / self.res).astype(int)
        self.n_cells = int(np.prod(self.dims))
        
        # Calculate volumes for the CellML blueprint
        self.cell_volume = float(np.prod(self.res))
        
        logger.info(f"Generated 3D Perfusion Grid: {self.dims[0]}x{self.dims[1]}x{self.dims[2]} (ZYX) "
                    f"({self.n_cells} cells) at resolution {self.res}µm")

    def get_cell_index(self, xyz: np.ndarray) -> int:
        """Map a physical point (z,y,x) to a linear grid index."""
        return _numba_get_linear_index(xyz, self.min_xyz, self.res, self.dims)

    def get_xyz_from_index(self, index: int) -> np.ndarray:
        """Map a linear index back to physical center-of-cell (z,y,x) coordinates."""
        # index = z + y*nz + x*nz*ny
        nz, ny = self.dims[0], self.dims[1]
        ix = index // (nz * ny)
        iy = (index % (nz * ny)) // nz
        iz = index % nz
        
        indices = np.array([iz, iy, ix], dtype=float)
        return self.min_xyz + (indices + 0.5) * self.res

@jit(nopython=True, cache=True)
def _numba_get_linear_index(pos_xyz, min_xyz, res, dims):
    # pos_xyz and min_xyz are actually (z, y, x)
    rel = pos_xyz - min_xyz
    idx_z = int(rel[0] / res[0])
    idx_y = int(rel[1] / res[1])
    idx_x = int(rel[2] / res[2])
    
    if idx_z < 0 or idx_z >= dims[0] or \
       idx_y < 0 or idx_y >= dims[1] or \
       idx_x < 0 or idx_x >= dims[2]:
        return -1
        
    # Linear index (z fastest)
    return idx_z + idx_y * dims[0] + idx_x * dims[0] * dims[1]

def _edge_diameter_um(data, default_diameter_um):
    """The edge's measured diameter, or the caller's deliberate stand-in.

    This used to substitute 5.0 um silently whenever the attribute was absent or
    non-positive. The value feeds vessel surface area, which drives transvascular flux and so
    every tissue PO2 downstream of it: a fabricated lumen produces a surface area, a flux and
    a PO2 that are each arithmetically fine and none of which mean anything.
    """
    diameter = data.get("assigned_diameter_um", data.get("fwhm_diameter_um"))
    if diameter is not None and float(diameter) > 0:
        return float(diameter)
    return None if default_diameter_um is None else float(default_diameter_um)


def _raise_on_non_finite_flow(G) -> None:
    """
    Raise if any edge's ``flow_abs`` is NaN or infinite.

    A NaN flow is not a missing one to these solvers: it passes the ``is None`` checks, and
    ``np.isclose`` then reports it as a unit mismatch, because NaN never equals NaN. That is
    how the pipeline's all-NaN flows (the flow export read a resistance array that held NaN)
    reached Tier 3 disguised as a conversion-factor error.
    """
    bad = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
           if d.get("flow_abs") is not None and not np.isfinite(float(d["flow_abs"]))]
    if bad:
        shown = ", ".join(str(e) for e in bad[:3])
        raise ValueError(
            f"{len(bad)} of {G.number_of_edges()} edges have a NaN or infinite 'flow_abs', for "
            f"example {shown}. The flow solve gave them no flow: either their resistance was "
            f"missing, or they lie in a component with no pressure boundary, whose pressure "
            f"solve_flow_from_conductance_matrix leaves NaN on purpose.")


#: Tier 3 treats an edge whose |flow| is at most this fraction of the network's largest as
#: stagnant. The flow solve conserves mass at every node to ~3e-14 of the largest flow (double
#: precision, measured on WKY-A); flows below 1e-12 of it are rounding, not blood.
STAGNANT_FLOW_FRACTION = 1e-12


def _raise_on_direction_size_mismatch(G) -> None:
    """
    Raise if an edge's ``flow_signed`` and ``flow_abs`` disagree about whether it carries flow.

    The Tier 2 and Tier 3 marches take flow direction from ``flow_signed`` and size from
    ``flow_abs``. If the two come from different solves, a near-stagnant edge can flow by one
    and not at all by the other, and a node can then send blood with none arriving. The
    pipeline did this: it copied only ``flow_abs`` back from its final flow export and left
    ``flow_signed`` from the rheology loop's last iteration.
    """
    bad = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
           if d.get("flow_abs") is not None
           and (float(d["flow_abs"]) != 0.0) != (float(d.get("flow_signed", 0.0)) != 0.0)]
    if bad:
        shown = ", ".join(str(e) for e in bad[:3])
        raise ValueError(
            f"{len(bad)} of {G.number_of_edges()} edges carry flow by one of 'flow_signed' and "
            f"'flow_abs' but not by the other, for example {shown}. Set both from the same flow "
            f"solve.")


def _edge_flows_um3_per_s(DAG: nx.MultiDiGraph, cell_to_vessels: Dict,
                          flow_to_um3_per_s: float) -> Dict[Tuple[Any, Any, Any], float]:
    """
    Edge flow in um^3/s for the Tier 2 and Tier 3 blood-content march, keyed by edge id.

    The march divides a transmural flux in mmol/L um^3/s by this flow, so it has to be in
    um^3/s. It used to read edge ``flow_abs`` raw, in the flow solve's mmHg um^3 / cP, which
    made q 1.33e5 times too small: blood gave up its gas in the first cell of each edge, and
    the answer did not depend on flow at all.

    Raises if an edge that carries flow has no ``flow_abs``; it was read with a 0.0 default,
    which drops the edge's blood without a word. Also raises if the per-cell flows in
    ``cell_to_vessels`` disagree with the converted edge flow, which means
    ``map_vessels_to_grid`` and the solver were given different conversion factors.

    Keys are the DAG's ``(u, v, key)`` and the reversed ``(v, u, key)``, so both orientations
    of an edge id find it.
    """
    missing = [(u, v, k) for u, v, k, d in DAG.edges(keys=True, data=True)
               if d.get("flow_abs") is None]
    if missing:
        shown = ", ".join(str(e) for e in missing[:3])
        raise ValueError(
            f"{len(missing)} of {DAG.number_of_edges()} flowing edges have no 'flow_abs', for "
            f"example {shown}. Edge flow sets how fast blood gives up its gas along the edge, "
            f"so it is not substituted silently. Run the flow solve first, or set it on every "
            f"edge that has a non-zero 'flow_signed'.")
    _raise_on_non_finite_flow(DAG)

    q_by_edge = {}
    for u, v, k, d in DAG.edges(keys=True, data=True):
        q = abs(float(d["flow_abs"])) * flow_to_um3_per_s
        q_by_edge[(u, v, k)] = q
        q_by_edge[(v, u, k)] = q

    for vessels in cell_to_vessels.values():
        for item in vessels:
            q = q_by_edge.get(item['edge'])
            if q is not None and not np.isclose(item['flow'], q, rtol=1e-9, atol=0.0):
                raise ValueError(
                    f"Edge {item['edge']} has flow {item['flow']:.6g} in cell_to_vessels but "
                    f"{q:.6g} um^3/s from its flow_abs at flow_to_um3_per_s="
                    f"{flow_to_um3_per_s:.6g}. map_vessels_to_grid and the solver were given "
                    f"different conversion factors; pass the same one to both.")
    return q_by_edge


def _implicit_cell_outlet_pressure(content_of_p, c_in: float, g: float, p_tissue: float,
                                   species: str, p_cap: float = 1.0e4) -> float:
    """
    Outlet partial pressure of blood leaving one grid cell, from an implicit exchange step.

    Solves ``content_of_p(P) + g * (P - p_tissue) = c_in`` for P, where ``g`` is
    ``P_perm * A * alpha / q``. The wall flux is taken at the outlet pressure, so blood can at
    most come to equilibrium with the tissue, and ``q * (c_in - c_out)`` equals the flux.

    The step used to take the flux at the inlet pressure (open item 21). Once
    ``P_perm * A * alpha`` exceeded ``q * dC/dP``, one step went past equilibrium; at the
    measured wall permeability that happens at capillary flow, and the Picard loop stalled.

    The left side rises with P, so there is one root, and it lies between ``p_tissue`` and the
    pressure at which ``content_of_p`` equals ``c_in``. The bracket is taken on that side of
    ``p_tissue``. Raises if no bracket is found below ``p_cap`` rather than keeping an old value.
    """
    from scipy.optimize import brentq

    def residual(p):
        return content_of_p(p) + g * (p - p_tissue) - c_in

    f_tissue = residual(p_tissue)
    if f_tissue == 0.0:
        return float(p_tissue)
    if f_tissue > 0.0:
        # Blood holds less than tissue-equilibrium content: it gains gas, P_out <= p_tissue.
        lo, hi = 0.0, p_tissue
        if residual(lo) > 0.0:
            raise ValueError(
                f"Implicit {species} step has no root in [0, {p_tissue:.6g}] mmHg "
                f"(c_in={c_in:.6g}, g={g:.6g}).")
    else:
        # Blood gives up gas, P_out >= p_tissue.
        lo, hi = p_tissue, max(2.0 * p_tissue, 150.0)
        while residual(hi) < 0.0:
            if hi >= p_cap:
                raise ValueError(
                    f"Implicit {species} step found no root below {p_cap:.6g} mmHg "
                    f"(c_in={c_in:.6g}, p_tissue={p_tissue:.6g}, g={g:.6g}).")
            hi = min(2.0 * hi, p_cap)
    return float(brentq(residual, lo, hi, xtol=1e-12))


def _increasing_inverse(content_of_p, c: float, species: str, p_cap: float = 1.0e4) -> float:
    """
    Partial pressure at which ``content_of_p`` equals ``c``, from zero upwards.

    Both content curves are 0 at P <= 0 and rise from there; the CO2 curve peaks (near 135 mmHg
    at H near 0, higher as H rises) and then falls. The bracket grows from 150 mmHg until the
    content reaches ``c``, so it holds exactly one upward crossing. Raises if the content never
    reaches ``c`` below ``p_cap``: that blood cannot hold that much gas at any pressure.
    """
    from scipy.optimize import brentq

    if c <= 0.0:
        return 0.0
    hi = 150.0
    while content_of_p(hi) < c:
        if hi >= p_cap:
            raise ValueError(
                f"No {species} partial pressure below {p_cap:.6g} mmHg gives content {c:.6g} "
                f"mmol/L.")
        hi = min(2.0 * hi, p_cap)
    return float(brentq(lambda p: content_of_p(p) - c, 0.0, hi, xtol=1e-12))


def _mixed_blood_state(inflows: List[Dict[str, Any]], max_iter: int = 100,
                       xtol: float = 1e-10) -> Dict[str, float]:
    """
    Blood state where two or more streams join: the pressures of the flow-weighted mixture.

    Content, flow and red cell flux are summed, giving the mixture's content and haematocrit.
    Its pH is the flow-weighted pH the streams' O2 content was last evaluated at. PO2 and PCO2
    are then solved together (the O2 curve depends on PCO2 through Bohr, the CO2 curve on PO2
    through Haldane), alternating one-dimensional inversions from the flow-weighted pressures.

    Both curves are affine in H at fixed pressures, so the mixture content lies between the
    streams' own curves evaluated at the mixture H, and a root exists between their pressures.
    Raises if the inversion fails or the alternation does not settle.
    """
    q = sum(f["q"] for f in inflows)
    h = sum(f["q"] * f["h"] for f in inflows) / q
    c_o2 = sum(f["q"] * f["c_o2"] for f in inflows) / q
    c_co2 = sum(f["q"] * f["c_co2"] for f in inflows) / q
    ph = sum(f["q"] * f["state"]["o2_ph"] for f in inflows) / q
    po2 = sum(f["q"] * f["state"]["po2"] for f in inflows) / q
    pco2 = sum(f["q"] * f["state"]["pco2"] for f in inflows) / q

    for _ in range(max_iter):
        pco2_new = _increasing_inverse(
            lambda p: calculate_blood_co2_content(p, h, po2), c_co2, "CO2")
        po2_new = _increasing_inverse(
            lambda p: calculate_blood_oxygen_content(p, h, pco2_new, ph), c_o2, "O2")
        settled = abs(po2_new - po2) < xtol and abs(pco2_new - pco2) < xtol
        po2, pco2 = po2_new, pco2_new
        if settled:
            return {"po2": po2, "pco2": pco2, "o2_pco2": pco2, "o2_ph": ph}
    raise ValueError(
        f"Mixed blood PO2/PCO2 did not settle in {max_iter} alternations "
        f"(c_o2={c_o2:.6g}, c_co2={c_co2:.6g}, H={h:.6g}).")


def map_vessels_to_grid(
    G: nx.MultiGraph,
    grid: PerfusionGrid,
    default_diameter_um: float | None = None,
    flow_to_um3_per_s: float = POISEUILLE_FLOW_TO_UM3_PER_S,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Step 2: Map 1D vessel segments (edges) to the 3D tissue grid cells.

    This is where the 1D flow solve meets the 3D tissue, and therefore where the two unit
    systems have to be made to agree. Edge ``flow_abs`` is in the flow solve's own units,
    mmHg um^3 / cP, because ``R = 128 mu L / (pi d^4)`` is evaluated with pressure in mmHg,
    viscosity in cP and lengths in um. The metabolic sink downstream is in mmol/L/s times
    um^3. Coupling them unconverted asked the tissue to consume 2.2e4 times the oxygen the
    blood delivered, and the steady-state PO2 was correctly zero everywhere.

    ``flow_to_um3_per_s=1.0`` leaves flow in solver units, for a caller comparing against
    output produced before the conversion existed.

    Raises if any edge lacks a usable diameter, unless ``default_diameter_um`` is given.
    Passing it is a deliberate choice to model unmeasured vessels at a stated calibre; the
    previous behaviour made that choice silently, at 5.0 um, on the caller's behalf.

    Also raises if any edge lacks a ``hematocrit``. It used to be read with a 0.45 default,
    so a graph that had never been through the rheology solve was given systemic haematocrit
    everywhere and delivered oxygen as if it had.

    Returns:
        Mapping of linear_cell_index -> list of segments passing through that cell.
        Each segment info includes the edge ID, flow in um^3/s, and length in that cell.
    """
    missing = [
        (u, v, key) for u, v, key, data in G.edges(keys=True, data=True)
        if _edge_diameter_um(data, default_diameter_um) is None
    ]
    if missing:
        shown = ", ".join(str(e) for e in missing[:3])
        raise ValueError(
            f"{len(missing)} of {G.number_of_edges()} edges have no usable diameter "
            f"(absent or non-positive), for example {shown}. Diameter feeds vessel surface "
            f"area and therefore every transvascular flux and tissue PO2 computed from it, so "
            f"it is not substituted silently. Assign diameters first, or pass "
            f"default_diameter_um to model the unmeasured edges at a stated calibre."
        )
    no_hct = [
        (u, v, key) for u, v, key, data in G.edges(keys=True, data=True)
        if data.get("hematocrit") is None
    ]
    if no_hct:
        shown = ", ".join(str(e) for e in no_hct[:3])
        raise ValueError(
            f"{len(no_hct)} of {G.number_of_edges()} edges have no 'hematocrit', for example "
            f"{shown}. Haematocrit sets the oxygen content each edge delivers, so it is not "
            f"substituted silently. Run the rheology solve "
            f"(solve_coupled_flow_and_hematocrit) first, or set it on every edge."
        )

    _raise_on_non_finite_flow(G)

    cell_to_vessels = {}
    
    for u, v, key, data in G.edges(keys=True, data=True):
        voxels = data.get("voxels")
        flow = float(data.get("flow_abs", 0.0)) * flow_to_um3_per_s
        edge_len = data.get("length", 0.0)
        
        diameter = _edge_diameter_um(data, default_diameter_um)
        radius = diameter / 2.0
        
        if voxels is None or len(voxels) < 2:
            continue
            
        # ImageLynx edges store 'voxels' natively in physical ZYX space from build.py!
        # No spacing multiplication needed here.
        vox_phys_zyx = np.array(voxels, dtype=float)
        
        # Incremental length per voxel segment
        # In a real model, we'd use line-plane intersection, but for high-res microscopy,
        # point-sampling the voxels is a robust and fast approximation.
        len_per_vox = edge_len / (len(voxels) - 1) if len(voxels) > 1 else 0.0

        for i in range(len(vox_phys_zyx)):
            zyx = vox_phys_zyx[i]
            idx = grid.get_cell_index(zyx)
            
            if idx != -1:
                if idx not in cell_to_vessels:
                    cell_to_vessels[idx] = []
                
                # Check if this edge is already registered in this specific cell
                found = False
                for item in cell_to_vessels[idx]:
                    if item['edge'] == (u, v, key):
                        item['length'] += len_per_vox
                        item['surface_area'] += 2.0 * np.pi * radius * len_per_vox
                        found = True
                        break
                
                if not found:
                    cell_to_vessels[idx].append({
                        'edge': (u, v, key),
                        'flow': flow,
                        'hematocrit': float(data["hematocrit"]),
                        'length': len_per_vox,
                        'surface_area': 2.0 * np.pi * radius * len_per_vox
                    })
                    
    # Share each edge among the cells it crosses, by length.
    #
    # Without this an edge's whole flow is recorded against every cell it passes through, so
    # an edge crossing five cells injects five times its own oxygen. Refining the grid makes
    # edges cross more cells, so the total source grew with resolution: on WKY-C the summed
    # s_incoming went 8.87e6, 1.20e7, 1.54e7, 1.80e7 at 10, 6, 4 and 3 um, in exact proportion
    # to the mean cells crossed per edge. Section 2.3's PO2 rose with refinement for that
    # reason and not a physical one.
    #
    # Normalised against the accumulated length rather than the edge's own `length` attribute,
    # so the shares sum to exactly one even though point sampling counts the endpoints of each
    # sub-segment.
    length_by_edge: Dict[Any, float] = {}
    for vessels in cell_to_vessels.values():
        for item in vessels:
            length_by_edge[item['edge']] = length_by_edge.get(item['edge'], 0.0) + item['length']
    for vessels in cell_to_vessels.values():
        for item in vessels:
            total = length_by_edge.get(item['edge'], 0.0)
            item['length_fraction'] = (item['length'] / total) if total > 0 else 0.0

    logger.info(f"Vessel-to-Grid mapping complete. {len(cell_to_vessels)} tissue cells are perfused by vessels.")
    return cell_to_vessels


def _jacobi_preconditioner(A):
    """Diagonal (Jacobi) preconditioner, which conjugate gradient requires to be SPD.

    This replaces an incomplete-LU preconditioner. ``spilu`` is a general-purpose
    factorisation and carries no guarantee of symmetry or positive definiteness; CG assumes
    both of its preconditioner, and given one that is neither it does not merely converge
    slowly, it diverges. Measured on the production 10 µm grid for WKY-C, CG under the ILU
    preconditioner reached a relative residual of **19** after 1000 iterations, where the
    initial residual is 1 by construction. The resulting PO2 field was 4e-4 mmHg everywhere
    against an arterial 100, so every cell read as hypoxic.

    The diagonal here is a sum of face conductances plus a positive regulariser plus a
    non-negative washout term, so it is strictly positive and its inverse is SPD by
    construction. Measured on the same system: relative residual 8.8e-7 in 0.05 s, against
    5.8 s for the ILU version that failed.
    """
    import scipy.sparse.linalg as splinalg

    diagonal = A.diagonal()
    if np.any(diagonal <= 0):
        # Nothing here should produce a non-positive diagonal; if it does, an SPD
        # preconditioner cannot be formed and unpreconditioned CG is the honest fallback.
        logger.warning(
            "Diffusion matrix has %d non-positive diagonal entries; skipping the Jacobi "
            "preconditioner rather than forming a non-SPD one.",
            int((diagonal <= 0).sum()),
        )
        return None
    inverse = 1.0 / diagonal
    return splinalg.LinearOperator(A.shape, lambda v: inverse * v)


def build_adr_matrix(grid: PerfusionGrid, cell_to_vessels: Dict[int, List[Dict[str, Any]]], perf_config) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Step 4: Build the pure Diffusion sparse matrix and Advection vectors.
    Returns:
        A: scipy.sparse.csr_matrix (Constant LHS matrix for Diffusion ONLY)
        q_total: np.ndarray (Total bulk flow through each voxel)
        s_incoming: np.ndarray (Fixed arterial oxygen content entering each voxel)
    """
    import scipy.sparse as sp
    
    N = grid.n_cells
    # grid.dims is (nz, ny, nx) and grid.res is (rz, ry, rx), as PerfusionGrid's own
    # constructor log records. Both were previously unpacked reversed and the conductances
    # named for the wrong axes; the reshape below reversed them a second time and the two
    # errors cancelled. The arithmetic was right and unguessable from the names.
    nz, ny, nx = grid.dims
    res = grid.res
    
    # Convert diffusion coefficient from m^2/s to µm^2/s
    sigma_diff_um2_s = perf_config.sigma_diff * 1e12
    
    # Diffusive conductance between cells (µm^3/s)
    # Diffusive conductance across a cell face: sigma * (face area) / (spacing normal to
    # it). res is (rz, ry, rx), so the z conductance uses the y-x face.
    D_z = sigma_diff_um2_s * (res[1] * res[2]) / res[0]
    D_y = sigma_diff_um2_s * (res[0] * res[2]) / res[1]
    D_x = sigma_diff_um2_s * (res[0] * res[1]) / res[2]
    
    rows, cols, data = [], [], []
    diag_A = np.zeros(N, dtype=np.float64)
    q_total = np.zeros(N, dtype=np.float64)
    s_incoming = np.zeros(N, dtype=np.float64)
    
    po2_arterial = perf_config.po2_arterial_mmHg  # mmHg
    
    # Advection arrays (Vessel coupling)
    for idx, vessels in cell_to_vessels.items():
        # Each edge contributes the share of its flow that lies in this cell, so summing over
        # cells recovers the edge's flow once rather than once per cell crossed.
        q_total[idx] = sum(v['flow'] * v.get('length_fraction', 1.0) for v in vessels)
        
        # Calculate exactly how much oxygen is delivered to this cell based on the Hill Equation
        # S_incoming = Sum( Q_share * C_blood_arterial )
        total_o2_flux = 0.0
        for v in vessels:
            h = v['hematocrit']
            c_art = calculate_blood_oxygen_content(po2_arterial, h)
            total_o2_flux += v['flow'] * v.get('length_fraction', 1.0) * c_art
            
        s_incoming[idx] = total_o2_flux

    # Build diffusion matrix (Standard 7-point stencil)
    logger.info("Building 3D Diffusion sparse matrix...")
    
    # The grid's linear index is z-fastest, idx = z + y*nz + x*nz*ny, as get_cell_index
    # defines it. A C-order reshape makes the LAST axis fastest, so the index array is
    # shaped (nx, ny, nz) and steps along its last axis are steps in z.
    idx = np.arange(N).reshape((nx, ny, nz))

    # z-direction edges (last axis)
    back = idx[:, :, :-1].flatten()
    front = idx[:, :, 1:].flatten()
    rows.extend(back); cols.extend(front); data.extend([-D_z] * len(back))
    rows.extend(front); cols.extend(back); data.extend([-D_z] * len(front))
    np.add.at(diag_A, back, D_z)
    np.add.at(diag_A, front, D_z)
    
    # y-direction edges
    bottom = idx[:, :-1, :].flatten()
    top = idx[:, 1:, :].flatten()
    rows.extend(bottom); cols.extend(top); data.extend([-D_y] * len(bottom))
    rows.extend(top); cols.extend(bottom); data.extend([-D_y] * len(top))
    np.add.at(diag_A, bottom, D_y)
    np.add.at(diag_A, top, D_y)
    
    # z-direction edges
    # x-direction edges (first axis)
    lo = idx[:-1, :, :].flatten()
    hi = idx[1:, :, :].flatten()
    rows.extend(lo); cols.extend(hi); data.extend([-D_x] * len(lo))
    rows.extend(hi); cols.extend(lo); data.extend([-D_x] * len(hi))
    np.add.at(diag_A, lo, D_x)
    np.add.at(diag_A, hi, D_x)
    
    # Add diagonal elements
    all_indices = np.arange(N)
    rows.extend(all_indices)
    cols.extend(all_indices)
    data.extend(diag_A)
    
    # We add a tiny regularization factor to the diagonal to prevent the matrix from being 
    # perfectly singular (since Neumann BCs mean pure diffusion has a null space).
    # But since we no longer have Q on the diagonal, we must add a tiny sink to stabilize CG.
    tiny_sink = 1e-12
    data = np.array(data)
    diag_mask = (np.array(rows) == np.array(cols))
    data[diag_mask] += tiny_sink
    
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    logger.info(f"Diffusion Matrix constructed. Shape: {A.shape}, Non-zeros: {A.nnz}")
    
    return A, q_total, s_incoming


def solve_perfusion_steady_state(grid: PerfusionGrid, A: Any, q_total: np.ndarray, s_incoming: np.ndarray, perf_config) -> np.ndarray:
    """
    Step 5: Solve the Non-Linear Steady-State Perfusion system using Picard Iteration.
    Solves for tissue PO2 (mmHg).
    """
    import scipy.sparse.linalg as splinalg
    
    N = grid.n_cells
    PO2 = np.zeros(N, dtype=np.float64) # Initial guess (0.0 mmHg everywhere)
    
    M_max = perf_config.M_max
    k_reduce = perf_config.k_reduce
    V_cell = grid.cell_volume
    
    # The venous washout is evaluated at systemic haematocrit, not at each cell's local
    # haematocrit, so in Tier 1 the washout is decoupled from phase separation. Read from the
    # config rather than written out here, so it cannot drift from the systemic value.
    h_baseline = perf_config.systemic_hematocrit
    
    max_iter = 50
    tolerance = 1e-5
    
    logger.info("Initializing ILU preconditioner for steady-state solver...")
    # Add a tiny diagonal regularizer to A to ensure ILU succeeds if entirely disconnected
    A_reg = A.copy()
    A_reg.setdiag(A_reg.diagonal() + 1e-6)
    
    # NUMERICAL STABILIZATION:
    # Because A is purely diffusion, its rows sum to 0. Solving A*x = b fails if sum(b) != 0.
    # The non-linear advective washout acts as a sink on the RHS, which is highly unstable for CG.
    # We apply a mathematical trick: Add a linear pseudo-washout to the LHS diagonal,
    # and add the exact same term to the RHS. The true steady-state roots remain identical,
    # but the LHS matrix becomes strictly diagonally dominant and highly invertible.
    # Increasing gamma_relax dampens the Picard step size, preventing sigmoidal oscillations.
    gamma_relax = 0.5 # Effective linearized slope
    pseudo_washout_diag = q_total * gamma_relax
    A_stable = A_reg.copy()
    A_stable.setdiag(A_stable.diagonal() + pseudo_washout_diag)
    
    M_pre = _jacobi_preconditioner(A_stable)

    logger.info("Starting Non-Linear Picard Iteration loop solving for PO2...")
    for iteration in range(max_iter):
        PO2_clamped = np.maximum(PO2, 0.0)
        
        # 1. Compute non-linear metabolic sink based on current PO2
        # M(PO2) = M_max * (1 - exp(-k * PO2))
        M_reduced = M_max * (1.0 - np.exp(-k_reduce * PO2_clamped))
        
        # 2. Compute dynamic Advective Washout
        # Voxel loses oxygen based on blood leaving at local tissue PO2
        s_washout = np.zeros(N, dtype=np.float64)
        for i in range(N):
            if q_total[i] > 0:
                c_venous = calculate_blood_oxygen_content(PO2_clamped[i], h_baseline)
                s_washout[i] = q_total[i] * c_venous
                
        # 3. Construct the full RHS: b = Advection_In - Advection_Out - Metabolic_Sink + Pseudo_Washout
        b = s_incoming - s_washout - (M_reduced * V_cell) + (pseudo_washout_diag * PO2_clamped)
        
        # 4. Solve the linear system A_stable * PO2_new = b
        PO2_new, info = splinalg.cg(A_stable, b, M=M_pre, x0=PO2, rtol=1e-6, maxiter=1000)
        
        if info != 0:
            logger.warning(f"CG Solver did not converge perfectly at iteration {iteration} (info={info})")
            
        # Prevent non-physical negative pressures which cause Picard oscillation
        PO2_new = np.maximum(PO2_new, 0.0)
        
        # 5. Check convergence
        diff = np.linalg.norm(PO2_new - PO2) / (np.linalg.norm(PO2_new) + 1e-12)
        logger.debug(f"  Iteration {iteration+1}: Relative change = {diff:.6e}")
        
        PO2 = PO2_new
        
        if diff < tolerance:
            logger.info(f"Steady-state perfusion converged successfully after {iteration+1} iterations.")
            break
    else:
        logger.warning(f"Picard iteration hit max_iter ({max_iter}) without reaching tolerance {tolerance}.")

    return PO2

#: Past iterates the Tier 3 Anderson step combines (open item 23).
ANDERSON_DEPTH = 5


def _blood_response_conductance(content_of_p, p_out: float, k: float, q: float) -> float:
    """
    How much one vessel's wall flux into a cell falls per mmHg rise in tissue pressure.

    The implicit step (``_implicit_cell_outlet_pressure``) solves C(P_out) + g (P_out - P_t) =
    C_in with g = k / q, so dP_out/dP_t = g / (C' + g) and the flux k (P_out - P_t) has slope
    -k C' / (C' + g). At high flow this is k; at low flow, where the blood follows the tissue,
    it is about q C', much less. C' is a central difference at P_out.
    """
    h = 1e-6 * max(abs(p_out), 1.0)
    lo = max(p_out - h, 1e-12)
    slope = (content_of_p(p_out + h) - content_of_p(lo)) / (p_out + h - lo)
    return k * slope / (slope + k / q)


def _relative_residual(A_diff, P: np.ndarray, net_source: np.ndarray, scale: np.ndarray) -> float:
    """
    Largest per-cell imbalance of the tissue balance, in mmHg, relative to the field.

    The balance is A_diff P = net_source (wall flux plus production, minus consumption). Each
    cell's imbalance is divided by ``scale``, the cell's linearised response to its own
    pressure (diffusion diagonal plus the blood's response), which turns mmol/s into the
    pressure correction the cell needs. Scaling by the full wall conductance instead would make
    low-flow cells, where the blood follows the tissue, look converged long before they are.
    """
    delta = np.abs(A_diff @ P - net_source) / scale
    return float(delta.max() / (np.abs(P).max() + 1e-12)) if delta.size else 0.0


class _AndersonMixer:
    """
    Anderson acceleration of a fixed-point iteration x -> G(x) (Walker & Ni 2011, type II).

    Keeps the last ``depth`` differences of x and of G(x) - x and returns the combination of
    past G values whose residual is smallest in least squares. The result is clipped at 0
    (pressures). If the measured residual rose since the last call, the history is dropped and
    this step is a plain one: unguarded, Anderson on top of the Newton-like update diverged on a
    stiff two-cell chain.
    """

    def __init__(self, depth: int):
        self.depth = depth
        self.xs, self.fs = [], []
        self.last = np.inf

    def update(self, x: np.ndarray, gx: np.ndarray, residual: float) -> np.ndarray:
        if residual > self.last:
            self.xs, self.fs = [], []
        self.last = residual
        self.xs.append(x)
        self.fs.append(gx - x)
        self.xs, self.fs = self.xs[-(self.depth + 1):], self.fs[-(self.depth + 1):]
        if len(self.fs) == 1:
            return np.maximum(gx, 0.0)
        dF = np.stack([b - a for a, b in zip(self.fs[:-1], self.fs[1:])], axis=1)
        dG = np.stack([(xb + fb) - (xa + fa) for xa, xb, fa, fb
                       in zip(self.xs[:-1], self.xs[1:], self.fs[:-1], self.fs[1:])], axis=1)
        gamma, *_ = np.linalg.lstsq(dF, self.fs[-1], rcond=None)
        return np.maximum(gx - dG @ gamma, 0.0)


def solve_multi_species_perfusion(grid: PerfusionGrid, G: nx.MultiGraph, starting_nodes: list, cell_to_vessels: Dict, perf_config,
                                  flow_to_um3_per_s: float = POISEUILLE_FLOW_TO_UM3_PER_S,
                                  return_info: bool = False):
    """
    Solve the Multi-Species (O2, CO2, pH) Coupled 1D-3D Steady-State system.
    Solves for tissue PO2, PCO2, and pH using Bohr/Haldane effects and Henderson-Hasselbalch.

    The march along each edge uses edge flow converted to um^3/s by ``flow_to_um3_per_s``, which
    must be the factor ``cell_to_vessels`` was built with (``_edge_flows_um3_per_s`` checks it).

    Each cell's exchange is implicit (``_implicit_cell_outlet_pressure``): the wall flux is taken
    at the blood's outlet pressure, which lies between the tissue's and the inlet's, so one cell
    can at most bring blood to equilibrium with its tissue.

    The Picard loop (open item 23) linearises the wall flux through the blood's actual response
    (``_blood_response_conductance``) and consumption through its slope, solves each tissue
    update exactly with a sparse LU, and speeds up with guarded Anderson acceleration
    (``_AndersonMixer``). It stops when the
    nonlinear residual of both species, scaled to mmHg by the blood's actual response, is below
    ``perf_config.picard_tolerance`` relative to the field (``_relative_residual``). It used to
    stop on the relative change between iterates, with a warm-started CG at rtol 1e-5: the loop
    moved slowly at capillary flow, and once a step fell below the CG tolerance the change read
    zero and it reported convergence short of the answer.

    Returns PO2, PCO2 and pH. With ``return_info`` it also returns a dict with ``converged``,
    ``iterations`` (Picard updates made) and ``residual_o2``/``residual_co2`` of the returned field.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as splinalg
    import networkx as nx

    N = grid.n_cells
    # grid.dims is (nz, ny, nx). See build_adr_matrix for why this was reversed.
    nz_dim, ny_dim, nx_dim = grid.dims
    res = grid.res

    PO2_tissue = np.zeros(N, dtype=np.float64) # mmHg
    PCO2_tissue = np.full(N, 40.0, dtype=np.float64) # mmHg (baseline arterial)
    pH_tissue = np.full(N, 7.4, dtype=np.float64)

    # Every field is read without a default. These were getattr fallbacks, and the M_max one
    # was 0.05 against PerfusionConfig's 0.005, so a config that lacked it ran at 10x the rate.
    M_max = perf_config.M_max
    k_reduce = perf_config.k_reduce
    RQ = perf_config.respiratory_quotient
    hco3_tissue = perf_config.hco3_tissue

    V_cell = grid.cell_volume
    P_perm_o2 = perf_config.permeability_o2_cm_s * 1e4 # um/s
    P_perm_co2 = perf_config.permeability_co2_cm_s * 1e4 # um/s
    po2_art = perf_config.po2_arterial_mmHg
    pco2_art = perf_config.pco2_arterial
    max_iter = perf_config.picard_max_iterations
    tolerance = perf_config.picard_tolerance

    logger.info("Initializing Multi-Species 1D-3D Picard Loop...")

    edge_to_cells = {}
    q_total = np.zeros(N)
    for cell_idx, vessels in cell_to_vessels.items():
        q_total[cell_idx] = sum(v['flow'] for v in vessels)
        for v in vessels:
            edge = v['edge']
            if edge not in edge_to_cells: edge_to_cells[edge] = []
            edge_to_cells[edge].append({'cell_idx': cell_idx, 'surface_area': v['surface_area'], 'flow': v['flow']})

    _raise_on_direction_size_mismatch(G)
    # Flow at the pressure solve's rounding level is not flow. In near-stagnant pockets it can
    # leave both edges of a node flowing outward, and the node then sends blood it never
    # receives (7 such nodes on WKY-A, net outflow ~1e-14 of the largest flow). Such edges
    # carry no blood in the march, the same as an exact zero.
    q_scale = max((abs(float(d.get("flow_signed", 0.0))) for _, _, d in G.edges(data=True)),
                  default=0.0)
    stagnant_tol = STAGNANT_FLOW_FRACTION * q_scale
    n_stagnant = 0
    DAG = nx.MultiDiGraph()
    for u, v, key, e_data in G.edges(keys=True, data=True):
        f = e_data.get("flow_signed", 0.0)
        if 0.0 < abs(f) <= stagnant_tol:
            n_stagnant += 1
        elif f > 0: DAG.add_edge(u, v, key=key, **e_data)
        elif f < 0: DAG.add_edge(v, u, key=key, **e_data)
    if n_stagnant:
        logger.info(f"{n_stagnant} of {G.number_of_edges()} edges have flow at or below "
                    f"{STAGNANT_FLOW_FRACTION:g} of the largest and carry no blood in the march.")

    # Flow runs from high to low pressure, so the directed graph has no cycle. If it had one,
    # the march would have no order to follow; it used to fall back to node order silently.
    try:
        topo_order = list(nx.topological_sort(DAG))
    except nx.NetworkXUnfeasible as exc:
        raise ValueError(
            "The flow direction graph has a cycle, so the blood-content march has no upstream "
            "order. Check flow_signed on the edges.") from exc

    q_by_edge = _edge_flows_um3_per_s(DAG, cell_to_vessels, flow_to_um3_per_s)

    starting_set = set(starting_nodes)
    # Arterial blood at pH 7.4.
    arterial_state = {"po2": po2_art, "pco2": pco2_art, "o2_pco2": pco2_art, "o2_ph": 7.4}

    alpha_o2 = 1.34e-3 # mmol/L per mmHg
    alpha_co2 = 0.03 # mmol/L per mmHg

    def build_diffusion_matrix(sigma_diff, alpha):
        sigma_um2_s = sigma_diff * 1e12
        # Scale D by alpha so the matrix solves for PO2 but outputs mmol/s
        # res is (rz, ry, rx); the z conductance uses the y-x face. The index array is
        # shaped (nx, ny, nz) so its last axis is z, matching the grid's z-fastest linear
        # index. See build_adr_matrix for the full account.
        D_z = sigma_um2_s * alpha * (res[1] * res[2]) / res[0]
        D_y = sigma_um2_s * alpha * (res[0] * res[2]) / res[1]
        D_x = sigma_um2_s * alpha * (res[0] * res[1]) / res[2]

        rows, cols, data = [], [], []
        diag_A = np.zeros(N, dtype=np.float64)

        idx = np.arange(N).reshape((nx_dim, ny_dim, nz_dim))
        back = idx[:, :, :-1].flatten(); front = idx[:, :, 1:].flatten()
        rows.extend(back); cols.extend(front); data.extend([-D_z] * len(back))
        rows.extend(front); cols.extend(back); data.extend([-D_z] * len(front))
        np.add.at(diag_A, back, D_z); np.add.at(diag_A, front, D_z)

        bottom = idx[:, :-1, :].flatten(); top = idx[:, 1:, :].flatten()
        rows.extend(bottom); cols.extend(top); data.extend([-D_y] * len(bottom))
        rows.extend(top); cols.extend(bottom); data.extend([-D_y] * len(top))
        np.add.at(diag_A, bottom, D_y); np.add.at(diag_A, top, D_y)

        lo = idx[:-1, :, :].flatten(); hi = idx[1:, :, :].flatten()
        rows.extend(lo); cols.extend(hi); data.extend([-D_x] * len(lo))
        rows.extend(hi); cols.extend(lo); data.extend([-D_x] * len(hi))
        np.add.at(diag_A, lo, D_x); np.add.at(diag_A, hi, D_x)

        all_indices = np.arange(N)
        rows.extend(all_indices); cols.extend(all_indices); data.extend(diag_A)
        data_arr = np.array(data)
        diag_mask = (np.array(rows) == np.array(cols))
        data_arr[diag_mask] += 1e-12 

        A = sp.coo_matrix((data_arr, (rows, cols)), shape=(N, N)).tocsr()
        return A

    A_diff_o2 = build_diffusion_matrix(perf_config.sigma_diff, alpha_o2)
    A_diff_co2 = build_diffusion_matrix(perf_config.sigma_diff_co2, alpha_co2)

    def exact_solve(A_diff, diagonal, b):
        # Sparse LU with a symmetric ordering (the matrix is symmetric): exact, and about 1 s
        # on a 30k-cell grid against 3 s with the default ordering. The CG used before had a
        # hard-coded rtol of 1e-5 and was warm-started, so a Picard step smaller than that came
        # back as no change and the loop reported convergence short of the answer (item 23).
        A = (A_diff + sp.diags(diagonal)).tocsc()
        lu = splinalg.splu(A, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})
        return lu.solve(b)

    mixer = _AndersonMixer(ANDERSON_DEPTH)
    converged = False
    residual_o2 = residual_co2 = np.inf
    # One pass more than max_iter updates: the last pass only measures the returned field.
    for iteration in range(max_iter + 1):
        PO2_clamped = np.maximum(PO2_tissue, 0.0)
        PCO2_clamped = np.maximum(PCO2_tissue, 0.0)

        # Coupled Metabolism
        M_o2_red = M_max * (1.0 - np.exp(-k_reduce * PO2_clamped))
        M_co2_prod = M_o2_red * RQ

        # Henderson-Hasselbalch
        pH_tissue = calculate_ph_from_pco2(PCO2_clamped, hco3_tissue)

        # Blood state at each node, as pressures (open item 24). Content per litre of blood is
        # not carried across a node, because phase separation gives each daughter a different
        # haematocrit: arterial content at H 0.45 can exceed anything a plasma-skimmed daughter
        # can hold at any PCO2. Pressures are shared, and each daughter's content is evaluated
        # at its own H with the same arguments, so O2 and CO2 are conserved (both curves are
        # affine in H, and the rheology conserves red cell and plasma flux).
        inflows = {n: [] for n in DAG.nodes()}

        transmural_o2 = np.zeros(N, dtype=np.float64)
        transmural_co2 = np.zeros(N, dtype=np.float64)
        # d(wall flux)/d(tissue pressure) through the blood, for the residual's scale.
        response_o2 = np.zeros(N, dtype=np.float64)
        response_co2 = np.zeros(N, dtype=np.float64)

        for node in topo_order:
            if node in starting_set:
                state = arterial_state
            elif len(inflows[node]) == 1:
                state = inflows[node][0]["state"]
            elif inflows[node]:
                state = _mixed_blood_state(inflows[node])
            elif DAG.out_degree(node) > 0:
                raise ValueError(
                    f"Node {node} sends blood out but receives none and is not a starting node. "
                    f"It used to be given arterial blood at systemic haematocrit.")
            else:
                continue

            for _, v, k, e_data in DAG.out_edges(node, data=True, keys=True):
                edge_key = (node, v, k)
                if edge_key not in edge_to_cells: edge_key = (v, node, k)
                q = q_by_edge[(node, v, k)]
                h = e_data["hematocrit"]

                po2_curr, pco2_curr = state["po2"], state["pco2"]
                o2_pco2_arg, o2_ph_arg = state["o2_pco2"], state["o2_ph"]
                c_o2_curr = calculate_blood_oxygen_content(po2_curr, h, o2_pco2_arg, o2_ph_arg)
                c_co2_curr = calculate_blood_co2_content(pco2_curr, h, po2_curr)

                for cell in edge_to_cells.get(edge_key, []):
                    idx = cell['cell_idx']
                    area = cell['surface_area']
                    ph_local = pH_tissue[idx]

                    k_o2 = P_perm_o2 * area * alpha_o2
                    k_co2 = P_perm_co2 * area * alpha_co2

                    if q > 0:
                        # Implicit step: flux at the outlet pressure (open item 21). O2 first at
                        # the inlet PCO2 (Bohr), then CO2 at the new PO2 (Haldane).
                        pco2_in_cell = pco2_curr
                        po2_curr = _implicit_cell_outlet_pressure(
                            lambda p: calculate_blood_oxygen_content(p, h, pco2_in_cell, ph_local),
                            c_o2_curr, k_o2 / q, PO2_clamped[idx], "O2")
                        po2_out_cell = po2_curr
                        pco2_curr = _implicit_cell_outlet_pressure(
                            lambda p: calculate_blood_co2_content(p, h, po2_out_cell),
                            c_co2_curr, k_co2 / q, PCO2_clamped[idx], "CO2")
                        o2_pco2_arg, o2_ph_arg = pco2_in_cell, ph_local
                        c_o2_curr = calculate_blood_oxygen_content(po2_curr, h, o2_pco2_arg, o2_ph_arg)
                        c_co2_curr = calculate_blood_co2_content(pco2_curr, h, po2_curr)
                        response_o2[idx] += _blood_response_conductance(
                            lambda p: calculate_blood_oxygen_content(p, h, pco2_in_cell, ph_local),
                            po2_curr, k_o2, q)
                        response_co2[idx] += _blood_response_conductance(
                            lambda p: calculate_blood_co2_content(p, h, po2_out_cell),
                            pco2_curr, k_co2, q)

                    flux_o2 = k_o2 * (po2_curr - PO2_clamped[idx])
                    flux_co2 = k_co2 * (pco2_curr - PCO2_clamped[idx])

                    transmural_o2[idx] += flux_o2
                    transmural_co2[idx] += flux_co2

                inflows[v].append({
                    "q": q, "h": h, "c_o2": c_o2_curr, "c_co2": c_co2_curr,
                    "state": {"po2": po2_curr, "pco2": pco2_curr,
                              "o2_pco2": o2_pco2_arg, "o2_ph": o2_ph_arg},
                })

        # The residual of this field, scaled by the linearised response of each cell to its own
        # pressure: diffusion, the blood's response through the implicit step, and (O2) the
        # metabolic slope, since consumption falls as PO2 does.
        metabolic_slope = M_max * k_reduce * np.exp(-k_reduce * PO2_clamped) * V_cell
        residual_o2 = _relative_residual(
            A_diff_o2, PO2_clamped, transmural_o2 - M_o2_red * V_cell,
            A_diff_o2.diagonal() + response_o2 + metabolic_slope)
        residual_co2 = _relative_residual(
            A_diff_co2, PCO2_clamped, transmural_co2 + M_co2_prod * V_cell,
            A_diff_co2.diagonal() + response_co2)
        if residual_o2 < tolerance and residual_co2 < tolerance:
            converged = True
            logger.info(f"Multi-Species solver converged after {iteration} iterations "
                        f"(residual O2 {residual_o2:.2e}, CO2 {residual_co2:.2e}).")
            break
        if iteration == max_iter:
            break

        # Newton-like update with the same linearisation on the diagonal. The wall term used to
        # be the full conductance P A alpha (a pseudo-washout, gamma 1). At low flow the blood
        # follows the tissue, the true slope is only q C', and each iteration moved the tissue
        # C' / (C' + P A alpha / q) of the way: thousands of iterations in plasma-skimmed or
        # slow vessels (open item 23).
        diag_o2 = response_o2 + metabolic_slope
        diag_co2 = response_co2
        b_o2 = transmural_o2 - (M_o2_red * V_cell) + diag_o2 * PO2_clamped
        b_co2 = transmural_co2 + (M_co2_prod * V_cell) + diag_co2 * PCO2_clamped

        if iteration == 0:
            logger.info(f"DEBUG Iteration 0: max(transmural_o2) = {np.max(transmural_o2)}")
            logger.info(f"DEBUG Iteration 0: max(M_o2_red * V_cell) = {np.max(M_o2_red * V_cell)}")
            logger.info(f"DEBUG Iteration 0: max(b_o2) = {np.max(b_o2)}")
            o2_flux_in = [f["c_o2"] * f["q"] for fs in inflows.values() for f in fs]
            logger.info(f"DEBUG Iteration 0: max(node O2 flux in) = {max(o2_flux_in, default=0.0)}")

        picard = np.concatenate([np.maximum(exact_solve(A_diff_o2, diag_o2, b_o2), 0.0),
                                 np.maximum(exact_solve(A_diff_co2, diag_co2, b_co2), 0.0)])
        mixed = mixer.update(np.concatenate([PO2_clamped, PCO2_clamped]), picard,
                             max(residual_o2, residual_co2))
        PO2_tissue, PCO2_tissue = mixed[:N], mixed[N:]

    PO2_tissue = np.maximum(PO2_tissue, 0.0)
    PCO2_tissue = np.maximum(PCO2_tissue, 0.0)
    if not converged:
        logger.warning(
            f"Multi-Species Picard iteration hit max_iter ({max_iter}) without reaching tolerance "
            f"{tolerance}: residual O2 {residual_o2:.2e}, CO2 {residual_co2:.2e}.")

    pH_tissue = calculate_ph_from_pco2(PCO2_tissue, hco3_tissue)
    if return_info:
        info = {"converged": converged, "iterations": int(iteration),
                "residual_o2": float(residual_o2), "residual_co2": float(residual_co2)}
        return PO2_tissue, PCO2_tissue, pH_tissue, info
    return PO2_tissue, PCO2_tissue, pH_tissue


def solve_coupled_1d3d_perfusion(grid: PerfusionGrid, G: nx.MultiGraph, starting_nodes: list, cell_to_vessels: Dict, perf_config,
                                 flow_to_um3_per_s: float = POISEUILLE_FLOW_TO_UM3_PER_S) -> np.ndarray:
    """
    Solve the Fully Coupled 1D-3D Steady-State Perfusion system using Picard Iteration.
    Solves for tissue PO2 (mmHg) and Blood PO2 simultaneously using an endothelial barrier model.

    Edge flow is converted to um^3/s as in ``solve_multi_species_perfusion``. The wall flux and
    the diffusion matrix still leave out O2 solubility (open item 22).
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as splinalg
    from scipy.optimize import brentq
    import networkx as nx
    
    N = grid.n_cells
    nz_dim, ny_dim, nx_dim = grid.dims
    res = grid.res
    
    sigma_diff_um2_s = perf_config.sigma_diff * 1e12
    # res is (rz, ry, rx); the z conductance uses the y-x face. See build_adr_matrix.
    D_z = sigma_diff_um2_s * (res[1] * res[2]) / res[0]
    D_y = sigma_diff_um2_s * (res[0] * res[2]) / res[1]
    D_x = sigma_diff_um2_s * (res[0] * res[1]) / res[2]
    
    rows, cols, data = [], [], []
    diag_A = np.zeros(N, dtype=np.float64)
    
    # Shaped (nx, ny, nz) so the last axis is z, matching the z-fastest linear index.
    idx = np.arange(N).reshape((nx_dim, ny_dim, nz_dim))
    back = idx[:, :, :-1].flatten(); front = idx[:, :, 1:].flatten()
    rows.extend(back); cols.extend(front); data.extend([-D_z] * len(back))
    rows.extend(front); cols.extend(back); data.extend([-D_z] * len(front))
    np.add.at(diag_A, back, D_z); np.add.at(diag_A, front, D_z)
    
    bottom = idx[:, :-1, :].flatten(); top = idx[:, 1:, :].flatten()
    rows.extend(bottom); cols.extend(top); data.extend([-D_y] * len(bottom))
    rows.extend(top); cols.extend(bottom); data.extend([-D_y] * len(top))
    np.add.at(diag_A, bottom, D_y); np.add.at(diag_A, top, D_y)
    
    lo = idx[:-1, :, :].flatten(); hi = idx[1:, :, :].flatten()
    rows.extend(lo); cols.extend(hi); data.extend([-D_x] * len(lo))
    rows.extend(hi); cols.extend(lo); data.extend([-D_x] * len(hi))
    np.add.at(diag_A, lo, D_x); np.add.at(diag_A, hi, D_x)
    
    all_indices = np.arange(N)
    rows.extend(all_indices); cols.extend(all_indices); data.extend(diag_A)
    data = np.array(data)
    diag_mask = (np.array(rows) == np.array(cols))
    data[diag_mask] += 1e-12 
    
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    
    PO2_tissue = np.zeros(N, dtype=np.float64)
    M_max = perf_config.M_max
    k_reduce = perf_config.k_reduce
    V_cell = grid.cell_volume
    P_perm = perf_config.permeability_o2_cm_s * 1e4 # um/s
    po2_arterial = perf_config.po2_arterial_mmHg
    systemic_h = perf_config.systemic_hematocrit
    
    edge_to_cells = {}
    q_total = np.zeros(N)
    for cell_idx, vessels in cell_to_vessels.items():
        q_total[cell_idx] = sum(v['flow'] for v in vessels)
        for v in vessels:
            edge = v['edge']
            if edge not in edge_to_cells: edge_to_cells[edge] = []
            edge_to_cells[edge].append({'cell_idx': cell_idx, 'surface_area': v['surface_area'], 'flow': v['flow']})
            
    _raise_on_direction_size_mismatch(G)
    DAG = nx.MultiDiGraph()
    for u, v, key, e_data in G.edges(keys=True, data=True):
        f = e_data.get("flow_signed", 0.0)
        if f > 0: DAG.add_edge(u, v, key=key, **e_data)
        elif f < 0: DAG.add_edge(v, u, key=key, **e_data)
            
    try:
        topo_order = list(nx.topological_sort(DAG))
    except nx.NetworkXUnfeasible:
        topo_order = list(G.nodes())

    q_by_edge = _edge_flows_um3_per_s(DAG, cell_to_vessels, flow_to_um3_per_s)
        
    area_total = np.zeros(N)
    for cell_idx, vessels in cell_to_vessels.items():
        area_total[cell_idx] = sum(v['surface_area'] for v in vessels)
        
    A_stable = A.copy()
    # The true linear sink of Tissue PO2 is the trans-mural flux: -P_perm * Area * Tissue_PO2.
    # By placing this exactly on the diagonal, the matrix is strictly diagonally dominant.
    # We add 1.0 to gamma_relax to ensure aggressive dampening against the non-linear Hill inversion.
    gamma_relax = 1.0
    pseudo_washout_diag = P_perm * area_total * gamma_relax
    A_stable.setdiag(A_stable.diagonal() + pseudo_washout_diag)
    
    M_pre = _jacobi_preconditioner(A_stable)

    logger.info("Starting Fully Coupled 1D-3D Picard Loop...")
    for iteration in range(50):
        PO2_clamped = np.maximum(PO2_tissue, 0.0)
        M_red = M_max * (1.0 - np.exp(-k_reduce * PO2_clamped))
        
        node_o2_flux_in = {n: 0.0 for n in DAG.nodes()}
        node_q_in = {n: 0.0 for n in DAG.nodes()}
        for n in starting_nodes:
            if n in DAG.nodes:
                for succ in DAG.successors(n):
                    for k, d in DAG[n][succ].items():
                        h = d["hematocrit"]
                        q = q_by_edge[(n, succ, k)]
                        node_o2_flux_in[n] += calculate_blood_oxygen_content(po2_arterial, h) * q
                        node_q_in[n] += q
        
        cell_transmural_flux = np.zeros(N, dtype=np.float64)
        for node in topo_order:
            c_mix = node_o2_flux_in[node] / node_q_in[node] if node_q_in[node] > 0 else calculate_blood_oxygen_content(po2_arterial, systemic_h)
            for _, v, k, e_data in DAG.out_edges(node, data=True, keys=True):
                edge_key = (node, v, k)
                if edge_key not in edge_to_cells: edge_key = (v, node, k)
                q = q_by_edge[(node, v, k)]
                h = e_data["hematocrit"]
                try:
                    po2_current = brentq(lambda p: calculate_blood_oxygen_content(p, h) - c_mix, 0.0, 150.0)
                except ValueError: po2_current = po2_arterial if c_mix > 0 else 0.0
                
                c_current = c_mix
                for cell in edge_to_cells.get(edge_key, []):
                    flux = P_perm * cell['surface_area'] * max(0.0, po2_current - PO2_clamped[cell['cell_idx']])
                    if q > 0:
                        c_current = max(0.0, c_current - (flux / q))
                        try:
                            po2_current = brentq(lambda p: calculate_blood_oxygen_content(p, h) - c_current, 0.0, 150.0)
                        except ValueError: po2_current = 0.0
                    cell_transmural_flux[cell['cell_idx']] += flux
                node_o2_flux_in[v] += c_current * q; node_q_in[v] += q
                
        b = cell_transmural_flux - (M_red * V_cell) + (pseudo_washout_diag * PO2_clamped)
        PO2_new, info = splinalg.cg(A_stable, b, M=M_pre, x0=PO2_tissue, rtol=1e-6, maxiter=1000)
        PO2_new = np.maximum(PO2_new, 0.0)
        diff = np.linalg.norm(PO2_new - PO2_tissue) / (np.linalg.norm(PO2_new) + 1e-12)
        PO2_tissue = PO2_new
        if diff < 1e-4:
            logger.info(f"Coupled solver converged after {iteration+1} iterations.")
            break
    return PO2_tissue
