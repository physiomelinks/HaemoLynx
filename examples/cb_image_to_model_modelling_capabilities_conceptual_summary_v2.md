# Physiological Modelling Documentation: ImageLynx `carotid_image_to_model.py` — Conceptual Edition

> **Purpose**: This document is the conceptually enriched companion to the Detailed Edition. It preserves every equation, algorithm, and parameter from the detailed summary, and wraps each section with physiological intuition, physical reasoning, and step-by-step commentary explaining *why* each modelling decision was made. A researcher reading this document should be able to understand both the mathematical machinery and the biological story behind it.

---

## 1. Scope and Domain

The pipeline takes 3D microscopy volumes of blood vessel networks (e.g., micro-CT or light-sheet images of the carotid body vasculature), extracts a mathematical graph of the vessel centrelines, and then solves for:

1. **Steady-state blood flow** (pressure, volumetric flow rate, velocity) using Poiseuille's Law.
2. **Spatially varying blood rheology** (viscosity and hematocrit distribution) using empirical in-vivo models.
3. **Tissue oxygen perfusion** ($PO_2$) by coupling the 1D vascular network to a 3D tissue diffusion grid.
4. **Multi-species gas transport** ($O_2$, $CO_2$, and $pH$) with Bohr–Haldane coupling.

The biological system under study is the **carotid body microvasculature** — a highly vascularised chemoreceptor organ supplied by branches of the external carotid artery.

> **Physiological Context: Why the Carotid Body?**
> The carotid body is one of the most highly perfused organs per unit mass in the entire body — it receives blood flow roughly 5–10× greater than the brain per gram of tissue. This extreme vascularity is essential for its role as the body's primary oxygen chemoreceptor: glomus (Type I) cells within the organ detect drops in arterial $PO_2$ and trigger the hypoxic ventilatory response. Modelling oxygen delivery within this organ is therefore directly relevant to understanding how the body senses and responds to hypoxia. Because the organ is so small (~1 mm³) and densely vascularised, the microvascular architecture — capillary diameters, branching patterns, and local flow distributions — has an outsized impact on the oxygen field experienced by individual glomus cells.

---

## 2. Governing Equations

### 2.1 Hagen–Poiseuille Flow (Steady-State, Laminar, Incompressible)

> **Physical Intuition: What Does Poiseuille's Law Describe?**
> Imagine pushing honey through a drinking straw versus through a garden hose. The straw offers enormously more resistance because the fluid near the walls is essentially stationary (the "no-slip" condition), and the proportion of fluid near the wall versus in the fast-flowing centre is much larger in a narrow tube. Poiseuille's Law quantifies this: resistance scales as $d^{-4}$, meaning that halving a vessel's diameter increases its resistance **16-fold**. This $d^4$ dependence is the single most important scaling law in microvascular physiology — it explains why small changes in vessel tone (e.g., pericyte contraction) can produce dramatic changes in local blood flow.

The central haemodynamic law used throughout the pipeline is the Hagen–Poiseuille equation for pressure-driven flow through a rigid cylindrical tube:

$$Q = \frac{\pi \, d^4 \, \Delta P}{128 \, \mu \, L}$$

where:
- $Q$ is volumetric flow rate ($\mu m^3/s$),
- $d$ is vessel diameter ($\mu m$),
- $\Delta P$ is the pressure drop across the segment (milliPascals, mPa),
- $\mu$ is apparent dynamic viscosity (mPa·s, equivalent to cP),
- $L$ is vessel centreline length ($\mu m$).

The hydraulic resistance of a single vessel segment is therefore:

$$R = \frac{128 \, \mu \, L}{\pi \, d^4}$$

This is implemented in `poiseuille.py` as:
```python
resistance = (128.0 * viscosity * length) / (PI * diameter**4)
```

And the flow through any edge is computed from the conductance (inverse resistance) and pressure drop:

$$Q_{ij} = \frac{1}{R_{ij}} \cdot (P_i - P_j)$$

> **Key Assumptions Embedded in This Law:**
> 1. **Rigid vessel walls** — no compliance, no Windkessel effect, no pulsatility.
> 2. **Fully developed, laminar flow** — Reynolds number is assumed to be low enough that entrance effects and turbulence are negligible.
> 3. **Newtonian fluid** (in the initial pass) — blood is treated as having a single effective viscosity at a given diameter and hematocrit. This is partially corrected by the Pries–Secomb model (§3.1), but the fundamental tube-flow profile remains parabolic.
> 4. **Steady-state** — no time-varying pulsatile waveform; flow is constant.
> 5. **Circular cross-section** — vessels are assumed to have a perfectly circular lumen.
> 6. **No-slip boundary condition** at the vessel wall.
> 7. **Incompressible fluid** — blood density is constant.

> **Physiological Context: Why Are These Assumptions Reasonable for the Microvasculature?**
> In large arteries (e.g., the aorta), pulsatility, turbulence, and wall compliance dominate the flow physics. But in the microvasculature (capillaries with diameters of 4–10 $\mu m$), the Reynolds number is typically $Re < 0.01$, making the flow extremely laminar. Pulsatile pressure oscillations are almost completely damped out by the time blood reaches the capillary bed. The main limitation is the Newtonian assumption: blood is a suspension of deformable cells, not a homogeneous fluid. This is addressed (imperfectly) by the Pries–Secomb rheology model in §3.

> **Verification:** The resistance formula $R = 128\mu L / \pi d^4$ is checked by recomputing it independently from measured diameter and comparing to the value `set_poiseuille_resistances()` assigns: `test_set_poiseuille_resistances_prefers_fwhm_optional()` (`tests/test_haemodynamics_automated_fwhm.py`) matches to `atol=1e-6`, and `test_measure_edge_diameters_fwhm_from_raw_tiff_cylinder()` (same file) matches to within 5% on a real FWHM-measured synthetic cylinder.

### 2.2 Network Flow: Graph Laplacian System

> **Physical Intuition: The Electrical Circuit Analogy**
> The vascular network is mathematically identical to an electrical resistor network. Each vessel segment is a "resistor," each junction (bifurcation or convergence point) is a "node," pressure is analogous to voltage, and volumetric flow rate is analogous to electric current. Just as Kirchhoff's current law states that the sum of currents entering any node must equal zero (charge conservation), the sum of flows entering any vascular node must also equal zero (mass conservation for an incompressible fluid). This analogy lets us use powerful linear algebra tools (sparse matrix solvers) developed for circuit simulation to solve for pressures and flows throughout the entire network simultaneously.

The entire vascular network is modelled as a resistor network (analogous to Kirchhoff's circuit laws). Flow conservation at each interior node $i$ is:

$$\sum_{j \in \text{neighbours}(i)} \frac{P_i - P_j}{R_{ij}} = 0$$

This is assembled into a sparse **Graph Laplacian** matrix $\mathbf{L}$ derived from the **Conductance matrix** $\mathbf{C}$:

**Step 1 — Build the Conductance Matrix:**

For each edge $(i, j)$ with resistance $R_{ij}$, the conductance is:

$$C_{ij} = \frac{1}{R_{ij}}$$

The matrix $\mathbf{C}$ is symmetric with $C_{ij} = C_{ji}$. For multigraphs with parallel edges, conductances are summed.

> **Numerical Rationale:** Conductances (rather than resistances) are stored in the matrix because parallel resistances combine by addition of conductances ($C_{\text{total}} = C_1 + C_2$), which is naturally handled by the sparse matrix COO format where duplicate entries are summed.

> **Guard Rail:** Edges with invalid resistances (`None` or $\leq 0$) are silently excluded from the conductance matrix by `build_conductance_matrix_from_graph()`. This prevents division-by-zero when computing $C_{ij} = 1/R_{ij}$ and ensures only physically meaningful vessel segments contribute to the flow solution.

**Step 2 — Compute the Graph Laplacian:**

$$\mathbf{L} = \text{diag}\left(\sum_j C_{ij}\right) - \mathbf{C}$$

This is implemented in `resistance.py` as:
```python
diag = np.array(C.sum(axis=1)).flatten()
L = sp.diags(diag) - C
```

> **Numerical Rationale:** The Graph Laplacian is always symmetric positive semi-definite. Its null space contains the constant vector (if all nodes were at the same pressure, no flow would occur). The Dirichlet boundary conditions (fixed pressures at inlets/outlets) remove this null space, making the reduced system $\mathbf{L}_{UU}$ strictly positive definite and uniquely solvable.

**Step 3 — Apply Dirichlet Boundary Conditions and Solve:**

Partition nodes into **known** (boundary) $K$ and **unknown** (interior) $U$ sets:

$$\mathbf{L} = \begin{bmatrix} \mathbf{L}_{UU} & \mathbf{L}_{UK} \\ \mathbf{L}_{KU} & \mathbf{L}_{KK} \end{bmatrix}$$

The unknown interior pressures are solved via:

$$\mathbf{L}_{UU} \, \mathbf{P}_U = -\mathbf{L}_{UK} \, \mathbf{P}_K$$

where $\mathbf{P}_K$ contains the prescribed Dirichlet pressures at inlet and outlet nodes. This is implemented as:

```python
l_uu = laplacian[unknown_idx, :][:, unknown_idx]
l_uk = laplacian[unknown_idx, :][:, known_idx]
p_k  = pressure[known_idx]
rhs  = -l_uk.dot(p_k)
pressure[unknown_idx] = _solve_system_smart(l_uu, rhs)
```

> **Implementation Note:** Steps 1–3 above (conductance matrix construction, Laplacian assembly, boundary condition application, pressure solve, and flow computation) are orchestrated by the function `solve_flow_from_conductance_matrix()` in `resistance.py`. This function takes the conductance matrix, boundary node lists, and Dirichlet pressures as inputs, and returns dictionaries of nodal pressures and signed edge flows. It also exports flow arrays to VTK for visualization.

> **Assumption**: Conservation of mass at every node — no leakage through vessel walls is modelled at this stage (perfusion leakage is handled separately in the tissue diffusion model, §5).

> **Verification:** The Laplacian network solve is benchmarked against exact series and parallel resistor-network formulas in `test_analytical_poiseuille_series()` ($R_{eq}=R_1+R_2$, `atol=1e-10`) and `test_analytical_poiseuille_parallel()` ($1/R_{eq}=1/R_1+1/R_2$, `atol=1e-10`), both in `tests/test_haemodynamics_analytical.py`. The conductance-matrix assembly itself — symmetry, parallel-edge conductance summing, and exclusion of non-positive resistances — is unit-tested directly in `test_build_conductance_matrix_from_graph()`, and the Laplacian's symmetry and zero row-sum property in `test_calc_laplacian_from_conductance_matrix()` (both `tests/test_haemodynamics.py`).

### 2.3 Effective Two-Point Resistance

> **Physical Intuition:** The effective resistance between two points in a network is the "total resistance the network presents" to flow between those points, accounting for all possible parallel and series pathways. It is the network-level analogue of measuring the resistance of a complex circuit with a multimeter. A low effective resistance means the network provides many parallel pathways (high redundancy), while a high value means flow is funnelled through a bottleneck.

The pipeline calculates the **effective resistance** between a specific inlet–outlet pair using the Laplacian pseudoinverse method:

**Algorithm:**
1. Construct a current injection vector $\mathbf{b}$ with $b_{\text{source}} = 1.0$ and all other entries zero.
2. Ground the target node by zeroing the corresponding row and column of $\mathbf{L}$, then setting the diagonal element to 1.0.
3. Solve $\mathbf{L}_{\text{mod}} \cdot \mathbf{x} = \mathbf{b}$.
4. The effective resistance is the voltage at the source node: $R_{\text{eff}} = x_{\text{source}}$.

This is implemented in `resistance.py` (`calc_two_point_from_laplacian_matrix_nodeID`):
```python
b[node_idx1] = 1.0
L_lil[node_idx2, :] = 0
L_lil[:, node_idx2] = 0
L_lil[node_idx2, node_idx2] = 1.0
x = _solve_system_smart(L_csr, b)
return float(x[node_idx1])
```

> **Verification:** Covered structurally by `test_calc_two_point_from_laplacian_matrix_nodeID()` (`tests/test_haemodynamics.py`), which asserts only that the returned resistance is positive — there is no closed-form two-point-resistance benchmark in the suite. The function is also exercised directionally in `tests/test_pericyte_mask_integration.py`, where a pericyte-induced constriction is asserted to strictly increase the computed effective resistance between two nodes. No test asserts an exact numerical value against a hand-derived network reduction.

---

## 3. Blood Rheology

> **Physiological Context: Why Is Blood Rheology So Complex?**
> Blood is not a simple fluid like water — it is a dense suspension of deformable biconcave discs (red blood cells, ~45% by volume) in plasma. This creates three phenomena that are absent in Newtonian fluids:
>
> 1. **The Fåhræus Effect**: In narrow tubes, the average hematocrit *inside the tube* is lower than the hematocrit of the blood *feeding* the tube. This occurs because RBCs travel faster than plasma (they concentrate in the fast-flowing centre), so fewer RBCs are needed at any instant to sustain a given RBC flux.
>
> 2. **The Fåhræus–Lindqvist Effect**: The apparent viscosity of blood *decreases* as tube diameter shrinks from ~300 $\mu m$ down to ~7 $\mu m$. This counterintuitive result arises because a cell-free plasma layer forms near the wall, acting as a lubricant. Below ~7 $\mu m$, viscosity rises sharply again because RBCs must physically deform to squeeze through.
>
> 3. **Plasma Skimming**: At bifurcations, RBCs do not split proportionally to flow — the branch receiving more flow disproportionately "skims" more RBCs from the central core, leaving the low-flow branch relatively cell-depleted.
>
> All three phenomena are captured by the models in this section.

### 3.1 Pries–Secomb In-Vivo Viscosity Model (Fåhræus–Lindqvist Effect)

Blood viscosity in microvessels is **not constant** — it varies dramatically with vessel diameter and local hematocrit. The pipeline implements the empirical Pries–Secomb (1992, 1994) model in the function `calculate_pries_secomb_viscosity()` in `rheology.py`.

**Step 1 — Diameter Clamping:**

To prevent mathematical singularities for vessels approaching the diameter of a single RBC:

$$D = \max(D_{\text{raw}}, 3.0 \, \mu m)$$

> **Physiological Context:** Human RBCs have a resting diameter of ~7.5 $\mu m$ but can deform to pass through capillaries as narrow as ~3 $\mu m$. Below this, RBCs physically cannot transit without lysing (rupturing). The 3.0 $\mu m$ clamp represents this hard physical limit.

**Step 2 — Relative Apparent Viscosity at Reference Hematocrit ($H_D = 0.45$):**

$$\mu_{45} = 220 \, e^{-1.3 D} + 3.2 - 2.44 \, e^{-0.06 D^{0.645}}$$

where:
- $\mu_{45}$ is the relative apparent viscosity (dimensionless) at reference hematocrit $H_D = 0.45$,
- $D$ is vessel diameter in $\mu m$.

> **Physical Intuition:** This equation has three distinct terms, each capturing a different physical mechanism:
>
> | Term | Expression | Physical Meaning |
> |---|---|---|
> | **RBC deformation resistance** | $220 \, e^{-1.3D}$ | Dominates at very small diameters ($D < 7 \, \mu m$). Captures the dramatic viscosity increase when RBCs must physically deform (elongate) to squeeze through vessels approaching their own resting diameter (~7.5 $\mu m$). The exponential decay means this term is negligible for $D > 15 \, \mu m$. |
> | **Asymptotic bulk viscosity** | $3.2$ | The constant baseline — at large diameters ($D > 100 \, \mu m$), blood behaves as a bulk suspension with viscosity ~3.2× plasma. |
> | **Cell-free layer lubrication** | $-2.44 \, e^{-0.06 D^{0.645}}$ | A *negative* correction that lowers viscosity at intermediate diameters (~7–50 $\mu m$). This captures the Fåhræus–Lindqvist effect: a cell-free plasma layer forms near the wall, acting as a lubricant that reduces apparent viscosity below the bulk value. The unusual exponent $D^{0.645}$ ensures the transition is gradual. |
>
> Together, these three terms produce the characteristic U-shaped viscosity curve: viscosity decreases from the bulk value as diameter shrinks (cell-free layer lubrication), reaches a minimum around 7 $\mu m$, then rises steeply as vessels approach the RBC diameter limit (deformation resistance).

**Step 3 — Shape Parameter $C$ (Hematocrit Dependence):**

$$C = \left(0.8 + e^{-0.075 D}\right) \left(-1 + \frac{1}{1 + 10^{-11} D^{12}}\right) + \frac{1}{1 + 10^{-11} D^{12}}$$

> **Physical Intuition:** The shape parameter $C$ controls how sensitively viscosity responds to hematocrit changes at different vessel sizes. In large vessels, viscosity scales strongly with hematocrit (more cells = more viscous). In very small capillaries, the relationship is weaker because single-file RBC flow dominates regardless of bulk hematocrit. The $10^{-11} D^{12}$ terms act as a smooth sigmoid switch between small-vessel and large-vessel behaviour.

**Step 4 — Relative Apparent Viscosity at Actual Hematocrit $H_D$:**

$$\mu_{\text{rel}} = 1 + (\mu_{45} - 1) \cdot \frac{(1 - H_D)^C - 1}{(1 - 0.45)^C - 1}$$

> **Physical Intuition:** This rescales the reference viscosity ($\mu_{45}$, which was computed at $H_D = 0.45$) to the actual local hematocrit. At $H_D = 0$, the blood is pure plasma, so $\mu_{\text{rel}} = 1$. At $H_D = 0.45$, we recover $\mu_{45}$.

**Step 5 — In-Vivo Correction for Cell-Free (Glycocalyx) Layer:**

$$\mu_{\text{app}} = \mu_{\text{rel}} \cdot \left(\frac{D}{D - 1.1}\right)^2$$

> **Physical Intuition:** In living vessels (as opposed to glass tubes used in laboratory experiments), the endothelial glycocalyx — a carbohydrate-rich layer coating the vessel wall — effectively narrows the lumen by about 1.1 $\mu m$. This correction accounts for the fact that the "effective diameter" available for blood flow is slightly smaller than the anatomical diameter. The squared ratio reflects the Poiseuille $d^4$ dependence applied as a correction factor (since resistance depends on $d^{-4}$, the effective viscosity increase from a reduced effective diameter scales approximately as $(D/(D-1.1))^2$).

**Step 6 — Final Apparent Viscosity in Physical Units:**

$$\mu = \mu_{\text{app}} \times \mu_{\text{plasma}}$$

> **Default parameter**: $\mu_{\text{plasma}} = 1.2$ mPa·s (cP).

> **Guard Rails:**
> - If $D \leq 0$ or $H_D \leq 0$, the function returns $\mu_{\text{plasma}}$ directly.
> - Maximum hematocrit is capped at 0.95 to prevent non-physical values.

> **Assumptions:**
> - The empirical correlations were derived from *in vivo* measurements in rat mesentery. Their direct applicability to the carotid body microvasculature (a glomus organ with unique perfusion characteristics) is assumed but not validated.
> - Minimum diameter cap: vessels smaller than 3.0 $\mu m$ are clamped to 3.0 $\mu m$ to avoid mathematical singularities (the cell-free-layer correction diverges at $D = 1.1 \, \mu m$).

> **Verification:** `test_rheology_fahraeus_lindqvist_curve()` (`tests/test_haemodynamics_analytical.py`) checks the qualitative U-shaped curve shape — monotonic viscosity decrease from 100 µm down to the ~7–10 µm minimum, then a monotonic rise from 10 µm down to 3 µm — via direct inequality assertions ($\mu_{100}>\mu_{30}>\mu_{10}$ and $\mu_{10}<\mu_6<\mu_3$). This confirms the curve's shape but does not assert against a specific published numerical value.

### 3.2 Initial (Pre-Rheology) Viscosity Approximation

Before the iterative rheology solver runs, the pipeline uses a simpler **power-law viscosity** for the initial Poiseuille resistance calculation:

$$\mu_{\text{initial}} = \frac{1}{d^{1.647}}$$

> **Numerical Rationale:** The full Pries–Secomb model requires knowing the hematocrit in each vessel, which itself depends on the flow distribution, which depends on the resistances. This circular dependency is broken by first computing an approximate viscosity (this power-law) that captures the essential trend (smaller vessels are more viscous) without needing hematocrit information. The exponent 1.647 was chosen as a heuristic fit to approximate the Pries–Secomb curve at $H_D = 0.45$. This initial estimate is then replaced by the full model during the coupled iteration (§3.4).

This is implemented in `poiseuille.py`:

```python
@staticmethod
def calculate_viscosity(diameter: float) -> float:
    return 1.0 / (diameter ** 1.647)
```

> **Verification:** `test_calculate_viscosity()` (`tests/test_haemodynamics.py`) checks the two defining properties of the power law directly: $\mu(1.0) = 1.0$ exactly, and $\mu(2.0) < 1.0$ (viscosity decreases as diameter increases). It is also exercised as the reference curve inside `test_analytical_sphincter_resistance_calculus()` (§4.4.4), where the same $1/d^{1.647}$ law is used to derive the exact-calculus resistance target.

### 3.3 Plasma Skimming (Phase Separation at Bifurcations)

> **Physiological Context: Why Don't RBCs Split Evenly?**
> In a parent vessel, RBCs are not uniformly distributed across the cross-section. Because of the parabolic velocity profile (Poiseuille flow), RBCs are carried by the faster-flowing central core, while a cell-free plasma layer exists near the wall. When this vessel splits at a bifurcation, the branch that draws more total flow also draws more from the central high-velocity core where the RBCs are concentrated. The branch drawing less flow preferentially receives the cell-poor plasma from near the wall. The result: the high-flow branch gets disproportionately more RBCs (higher hematocrit) than you would expect from simple proportional splitting, and the low-flow branch is relatively anaemic. This effect is called **plasma skimming** and has major consequences for oxygen delivery in the microcirculation — some capillary beds can become severely RBC-depleted even when overall perfusion is adequate.

At diverging bifurcations, red blood cells (RBCs) do **not** distribute proportionally to flow. The pipeline implements the Pries–Secomb empirical logistic skimming model in `calculate_phase_separation_hematocrit()` in `rheology.py`.

**Complete Step-by-Step Algorithm:**

Given:
- Total inflow $Q_{\text{in}}$ with hematocrit $H_{\text{in}}$
- Two daughter branches with flows $Q_1, Q_2$ and diameters $d_1, d_2$

**Step 1 — Guard Rails:**
```
If Q_in ≤ 1e-12 or H_in ≤ 0.0:
    Return (0.0, 0.0)
```
> **Numerical Rationale:** Prevents division-by-zero when there is no meaningful flow entering the bifurcation.

**Step 2 — Flow Fractions:**

$$f_{Q_1} = \frac{Q_1}{Q_{\text{in}}}, \quad f_{Q_2} = \frac{Q_2}{Q_{\text{in}}}$$

> **What this represents:** The fraction of the total parent flow that goes into each daughter branch.

**Step 3 — Edge-Case Handling:**
```
If f_Q1 < 1e-6:  Return (0.0, H_in × Q_in / max(Q_2, 1e-12))
If f_Q2 < 1e-6:  Return (H_in × Q_in / max(Q_1, 1e-12), 0.0)
```
> **What this means:** If almost all flow goes to one branch, all RBCs follow. The near-zero-flow branch gets zero hematocrit. The `max(..., 1e-12)` guard prevents division-by-zero in the surviving branch’s hematocrit calculation.

**Step 4 — Skimming Threshold:**

$$x_0 = 0.05$$

Branches receiving less than 5% of flow get zero RBCs.

> **Physiological Context:** Experimentally, branches drawing very little flow (\<5%) are observed to receive essentially no RBCs. This threshold represents the critical flow fraction below which the cell-free layer near the wall is the sole source of fluid entering the branch.

**Step 5 — Threshold-Based Bypass:**
```
If f_Q1 ≤ x_0:    f_E1 = 0.0
If f_Q1 ≥ 1 - x_0: f_E1 = 1.0
Otherwise → continue to Step 6.
```

**Step 6 — Asymmetry Parameter $A$:**

$$A = -13.29 \cdot \frac{d_1^2/d_2^2 - 1}{d_1^2/d_2^2 + 1} \cdot \frac{1 - H_{\text{in}}}{d_1}$$

> **Physical Intuition:** $A$ encodes how the *geometry* of the bifurcation (the relative diameters of the two daughter branches) biases RBC distribution. If both daughters have the same diameter, $d_1^2/d_2^2 = 1$, so $A = 0$ (no geometric bias). If one daughter is much larger, $A$ shifts the skimming curve to favour that daughter. The factor $(1 - H_{\text{in}})/d_1$ accounts for the fact that the cell-free layer width (and therefore the skimming effect) depends on both the parent hematocrit and diameter.

**Step 7 — Steepness Parameter $B$:**

$$B = 1 + 6.98 \cdot \frac{1 - H_{\text{in}}}{d_1}$$

> **Physical Intuition:** $B$ controls how sharply the RBC fraction responds to changes in the flow fraction. Higher $B$ means a steeper sigmoidal transition — small changes in flow split produce large changes in hematocrit split. This steepness increases when the parent vessel has lower hematocrit or smaller diameter, because the cell-free layer is proportionally larger and more influential.

**Step 8 — Logit Transformation of Flow Fraction:**

$$\text{logit}(f_{Q_1}) = \ln\!\left(\frac{f_{Q_1} - x_0}{1 - f_{Q_1} - x_0}\right)$$

> **Numerical Rationale:** The logit transformation maps the bounded flow fraction $(x_0, 1 - x_0)$ to the unbounded real line $(-\infty, +\infty)$. This allows the phase separation model to be expressed as a simple linear function in logit space (Step 9), then mapped back to the bounded $[0, 1]$ range via the sigmoid (Step 10). This is mathematically equivalent to fitting a logistic regression curve to the experimental data.

**Step 9 — Logit of RBC Flux Fraction:**

$$\text{logit}(f_{E_1}) = A + B \cdot \text{logit}(f_{Q_1})$$

> **What this represents:** In logit space, the RBC flux fraction is a linear function of the flow fraction, shifted by $A$ (geometric bias) and scaled by $B$ (steepness). When $A = 0$ and $B = 1$, RBCs split proportionally to flow (no skimming).

**Step 10 — Sigmoid (Inverse Logit) to Recover $f_{E_1}$:**

$$f_{E_1} = \frac{1}{1 + e^{-\text{logit}(f_{E_1})}}$$

> **What this represents:** Maps the logit back to a probability-like fraction in $[0, 1]$: the fraction of total RBC flux entering branch 1.

**Step 11 — Mass Conservation:**

$$f_{E_2} = 1 - f_{E_1}$$

> **What this ensures:** Every RBC that enters the parent must exit through one of the two daughters. No RBCs are created or destroyed at the bifurcation.

**Step 12 — Daughter Hematocrits:**

$$H_1 = H_{\text{in}} \cdot \frac{f_{E_1}}{f_{Q_1}}, \quad H_2 = H_{\text{in}} \cdot \frac{f_{E_2}}{f_{Q_2}}$$

> **Physical Intuition:** Hematocrit is the *concentration* of RBCs. If a branch receives a larger fraction of RBCs ($f_E$) than its fraction of flow ($f_Q$), the hematocrit in that branch is higher than the parent. Conversely, if it receives fewer RBCs relative to flow, it becomes diluted. This is the Fåhræus effect in action at the network level.

**Step 13 — Physical Bounds Clamping:**

$$H_1 = \min(\max(H_1, 0.0), 0.95), \quad H_2 = \min(\max(H_2, 0.0), 0.95)$$

> **Numerical Rationale:** Prevents hematocrit from becoming negative (non-physical) or exceeding 0.95 (at which point blood would be essentially a solid plug of RBCs, which cannot flow through a capillary).

> **Assumptions:**
> - The phase separation model is only applied at **binary bifurcations** (degree-2 splits). For trifurcations and higher, RBCs are distributed proportionally to flow (simple mixing: each daughter receives $H_{\text{mix}}$).
> - The empirical constants (−13.29, 6.98, $x_0 = 0.05$) were derived from glass tube experiments and *in vivo* rat cremaster observations.

> **Verification:** RBC mass conservation across a bifurcation is checked exactly in `test_rheology_hematocrit_mass_conservation()` (`tests/test_haemodynamics_analytical.py`): for a symmetric 50/50 split, flux in equals flux out to `atol=1e-8` and both daughters receive identical hematocrit. The skimming direction (the higher-flow, larger-diameter branch concentrates RBCs while the low-flow branch is depleted) is checked in `test_rheology_plasma_skimming_effect()`, which also re-verifies mass conservation (`atol=1e-8`) under an asymmetric 9:1 flow split. Neither test asserts the exact logit-sigmoid output value against a hand-derived number — both are conservation/directional oracles rather than closed-form benchmarks.

### 3.4 Coupled Flow–Hematocrit–Viscosity Iteration

> **Physical Intuition: Why Is This Iterative?**
> The core challenge is a three-way circular dependency:
>
> - **Flow** depends on **resistance** (Poiseuille's law).
> - **Resistance** depends on **viscosity** (which depends on diameter and local hematocrit).
> - **Hematocrit distribution** depends on **flow** (plasma skimming at each bifurcation).
>
> You cannot solve any one of these without knowing the other two. The pipeline breaks this chicken-and-egg problem with **Picard iteration** (also known as successive substitution): start with a guess, compute each quantity in turn assuming the others are fixed, then update and repeat until the system converges to a self-consistent solution. This is conceptually identical to how iterative methods work in computational fluid dynamics — linearize, solve, update, repeat.

The full non-linear coupling between flow, hematocrit distribution, and viscosity is solved iteratively (Picard-style fixed-point iteration) in `solve_coupled_flow_and_hematocrit()` in `rheology.py`.

**Complete Algorithm:**

```
  For each edge (u,v,k):
    hematocrit ← systemic_hematocrit (0.45)
    diameter ← assigned_diameter_um (or fwhm_diameter_um, or 5.0 μm fallback)
    If diameter is None or diameter ≤ 0: diameter ← 5.0 μm  [fallback to default capillary diameter]
    If length is None or length ≤ 0: length ← 10.0 μm  [fallback to default segment length]
    viscosity ← calculate_pries_secomb_viscosity(diameter, hematocrit)
    resistance ← (128 × viscosity × length) / (π × diameter⁴)
```
> **What happens here:** Every vessel starts with a uniform hematocrit (0.45, normal adult value). The Pries–Secomb model assigns diameter-appropriate viscosities, and initial resistances are computed. This is the "zeroth iterate" — an informed starting point. Vessels with missing or invalid diameter/length attributes are assigned physiologically reasonable defaults (5.0 μm diameter — a typical capillary — and 10.0 μm length) rather than being silently excluded.

```
  iteration ← 0
  max_flow_diff ← ∞
  previous_flows ← {}

WHILE iteration < max_iterations AND max_flow_diff > tolerance:
```
> **What happens here:** The outer loop repeats until either (a) flow values stop changing between iterations (convergence), or (b) the maximum number of iterations is reached. Convergence is measured by the largest absolute change in flow across all edges.

```
  STEP 1: Build Conductance and Laplacian
    conductance, node_list ← build_conductance_matrix_from_graph(G)
    laplacian ← calc_laplacian_from_conductance_matrix(conductance)
```
> **What happens here:** The resistances (which may have been updated from the previous iteration) are converted to conductances and assembled into the Graph Laplacian matrix — the linear system that encodes flow conservation at every node.

```
  STEP 2: Apply Dirichlet Boundary Conditions
    For each starting_node: pressure[node] ← input_p_bc
    For each output_node:   pressure[node] ← output_p_bc
```
> **What happens here:** We fix the pressure at inlets (100 mmHg = MAP) and outlets (2 mmHg = CVP). These are the driving forces that push blood through the network.

```
  STEP 3: Solve Pressure System
    L_UU × P_U = -L_UK × P_K
    pressure[unknown_idx] ← _solve_system_smart(L_UU, rhs)
```
> **What happens here:** The sparse linear system is solved to find pressures at every interior node. This is the most computationally expensive step per iteration.

```
  STEP 4: Calculate Flows & Build Directed Acyclic Graph (DAG)
    For each edge (u,v,k):
      flow_signed ← (1/resistance) × (P_u - P_v)
      flow_abs ← |flow_signed|
      If flow_signed > 0: DAG.add_edge(u → v)
      Else:               DAG.add_edge(v → u)
```
> **What happens here:** From the pressures, we compute the flow through every vessel using Ohm's law. The sign of the flow tells us the direction — blood flows from high pressure to low pressure. A directed graph (DAG) is built where every edge points "downstream." This DAG is essential for Step 6: we need to know which direction blood is flowing so we can correctly propagate hematocrit from inlets to outlets.

```
  STEP 5: Check Convergence (skip on first iteration)
    max_flow_diff ← max(|current_flows[k] - previous_flows[k]| for all k)
    If max_flow_diff ≤ tolerance: BREAK ("Converged!")
```
> **What happens here:** If all flow values have stabilised (changed by less than $10^{-4}$), the iterative loop has converged and we stop.

```
  STEP 6: Topologically Traverse DAG and Distribute Hematocrit
    Try: topological_order ← nx.topological_sort(DAG)
    Except NetworkXUnfeasible (cycle detected): BREAK
```
> **What happens here:** We sort the network nodes from inlets (upstream) to outlets (downstream) in **topological order**. This ensures that when we compute the hematocrit at any node, we have already computed the hematocrit at all of its upstream predecessors. It is like calculating the flow of water through a branching river system: you must know what enters each fork before you can determine what leaves. If the flow field contains cycles (which shouldn't happen in a steady-state flow but can occur due to pressure ties), the topological sort fails and the iteration terminates early.

```
    For each node in topological_order:
      h_mix ← node_h_in[node] / node_q_in[node]  (or systemic_hematocrit if q=0)
      out_edges ← DAG.out_edges(node)
      
      Case |out_edges| = 0: continue (leaf/outlet node)
      Case |out_edges| = 1: pass-through, daughter gets h_mix
      Case |out_edges| = 2: PLASMA SKIMMING (§3.3)
        h1, h2 ← calculate_phase_separation_hematocrit(q1+q2, h_mix, q1, d1, q2, d2)
      Case |out_edges| ≥ 3: PROPORTIONAL MIXING
        All daughters get h_mix
```
> **What happens here:** At each node, we compute the "mixed" hematocrit from all upstream contributions (flow-weighted average). Then, depending on how many daughter branches leave this node, we distribute the hematocrit:
> - **1 branch (pass-through):** Hematocrit is unchanged — blood just continues.
> - **2 branches (bifurcation):** The full plasma skimming model (§3.3) is applied, unequally distributing RBCs.
> - **3+ branches (trifurcation or higher):** The skimming model is only defined for binary splits, so we fall back to simple proportional mixing (each branch gets the same hematocrit as the parent). This is a known limitation.

```
  STEP 7: Update Viscosities and Resistances
    For each edge (u,v,k):
      mu_app ← calculate_pries_secomb_viscosity(d, h)
      resistance_new ← original_resistance × (mu_app / mu_old)
      WSS_Pa ← (32 × mu_app × Q_abs) / (π × d³) / 1000
```
> **What happens here:** With the new hematocrit distribution, we recompute the in-vivo viscosity for every vessel. Rather than recomputing resistance from scratch (which would destroy the sphincter/constriction geometry), we *scale* the original resistance by the ratio of new-to-old viscosity (see §3.5). Wall shear stress is also computed at this stage.

> **Default parameters:**
> - Maximum iterations: 15
> - Convergence tolerance: $10^{-4}$ (maximum absolute flow difference)
> - Systemic hematocrit: 0.45

> **Assumption**: Convergence is not guaranteed for all network topologies. If cycles are detected in the DAG (which can occur due to pressure ties or numerical precision), the iteration terminates early.

### 3.5 Resistance Scaling During Rheology Updates

> **Numerical Rationale: Why Scale Rather Than Recompute?**
> During Phase 4 of the pipeline, resistances were computed using the full spatial integration of sphincter and pericyte constriction profiles (§4.4.4), which involved numerically integrating 1000 sample points along each vessel to capture the precise diameter variations. If the rheology solver simply overwrote these resistances with a straight-tube $128 \mu L / \pi d^4$ formula, all the constriction geometry would be lost. Instead, the solver preserves the "geometric signature" by storing the original resistance and scaling it multiplicatively. This is equivalent to saying: "the vessel's shape hasn't changed, only the fluid viscosity within it has."

To preserve the complex geometric integration of sphincters and pericyte constrictions computed in the initial Poiseuille pass, the rheology solver does **not** overwrite resistances with a simple $128\mu L/\pi d^4$ formula. Instead, it scales the previously computed resistance by the ratio of the new in-vivo viscosity to the old power-law viscosity:

$$R_{\text{new}} = R_{\text{original}} \times \frac{\mu_{\text{Pries-Secomb}}(d, H_D)}{\mu_{\text{power-law}}(d)}$$

where:

$$\mu_{\text{power-law}}(d) = \frac{1}{d^{1.647}}$$

The `original_resistance` is saved on the first rheology iteration and never modified thereafter, ensuring the geometric constriction profile is embedded as a permanent scaling factor.

> **Verification:** No test targets this scaling rule directly by name. It is exercised only indirectly, as a step inside `test_coupled_solver_convergence()` (`tests/test_haemodynamics_rheology_integration.py`), which checks the downstream viscosity/hematocrit outputs but does not isolate or assert on the resistance-scaling ratio $\mu_{\text{Pries-Secomb}}/\mu_{\text{power-law}}$ itself. This is a coverage gap (see §11.3).

### 3.6 Wall Shear Stress

> **Physiological Context: Why Does WSS Matter?**
> Wall shear stress (WSS) is the tangential frictional force that flowing blood exerts on the endothelial cells lining the vessel wall. It is one of the most important biomechanical signals in vascular biology: endothelial cells sense WSS through mechanoreceptors and respond by releasing vasodilators (e.g., nitric oxide at high WSS) or pro-inflammatory signals (at low or oscillatory WSS). In the carotid body, WSS may influence the local microenvironment of glomus cells and potentially modulate chemosensory signalling.

Wall shear stress (WSS) is calculated for each edge after the rheology solver converges:

$$\tau_w = \frac{32 \, \mu \, Q}{{\pi \, d^3}}$$

where:
- $\mu$ is the Pries–Secomb apparent viscosity (mPa·s),
- $Q$ is the absolute volumetric flow rate ($\mu m^3/s$),
- $d$ is the vessel diameter ($\mu m$).

Units: The raw calculation yields WSS in mPa (since $\frac{\text{mPa·s} \times \mu m^3/s}{\mu m^3} = \text{mPa}$). The stored value is converted to Pa:

$$\tau_{w,\text{Pa}} = \frac{\tau_{w,\text{mPa}}}{1000}$$

> **Assumption**: This assumes a fully developed parabolic flow profile (Newtonian approximation), which may underestimate WSS in vessels where the Fåhræus–Lindqvist effect creates a significant cell-free layer.

> **Verification:** `test_analytical_wall_shear_stress()` (`tests/test_haemodynamics_analytical.py`) runs one iteration of the coupled solver on a two-node vessel, then recomputes $\tau_w = 32\mu Q/(\pi d^3)$ independently from the solver's own viscosity and flow outputs. The two values match to `atol=1e-10` — an exact formula check, not an independent physiological benchmark, since both sides of the comparison use the solver's own $\mu$ and $Q$.

---

## 4. Vessel Geometry and Constriction Models

### 4.1 Vessel Diameter Assignment

> **Physiological Context:** Vessel diameter is the single most influential parameter in the entire model — because of the $d^4$ dependence in Poiseuille's law, even small measurement errors in diameter translate to large errors in computed flow and resistance. Getting diameters right is therefore critical to the fidelity of the simulation.

Vessel diameters can be assigned via three modes, controlled by `radius_assignment_mode`:

| Mode | Description |
|---|---|
| `fwhm_radius` | **Default.** Per-edge diameters are measured directly from the raw 3D image using FWHM (Full Width at Half Maximum) Gaussian fitting of transverse intensity profiles along the vessel centreline. The edge attribute `fwhm_diameter_um` is read. |
| `edt_radius` | Diameters are derived from the Euclidean Distance Transform of the binary vessel mask. The edge attribute `edt_diameter_um` is read. |
| `constant_radius` | A uniform radius is applied to all vessels: $d = 2 \times \text{constant\_radius\_um}$. Default: $\text{constant\_radius\_um} = 5.0 \, \mu m$ (diameter = 10.0 $\mu m$). |

**Fallback Logic (implemented in `set_poiseuille_resistances()`):**

When using `fwhm_radius` or `edt_radius`, if the per-edge measurement attribute is `None` or $\leq 0$, the diameter falls back to the `diameter_by_branch_order` dictionary lookup. If the edge's branch order key is not found, it further falls back to the `"DEFAULT"` key in the dictionary.

> **Verification:** The `edt_radius` mode is checked in `test_poiseuille_edt_mode()` (`tests/test_edt_diameter.py`): a pre-assigned `edt_diameter_um` of 12.0 is confirmed to be read and used verbatim as `assigned_diameter_um` by both `set_poiseuille_resistances()` and `set_poiseuille_resistances_with_constrictions()`. The `fwhm_radius` mode and its precedence over the branch-order dictionary fallback is checked in `test_set_poiseuille_resistances_prefers_fwhm_optional()` (`tests/test_haemodynamics_automated_fwhm.py`).

### 4.2 FWHM Ray-Casting Diameter Measurement

> **Physical Intuition: Why FWHM?**
> In fluorescence or light-sheet microscopy, vessel cross-sections appear as bright spots against a darker background. The intensity profile across a vessel approximates a Gaussian (bell curve) because of optical blurring (the point spread function) and the cylindrical geometry. The Full Width at Half Maximum (FWHM) of this Gaussian is a robust estimator of the true vessel diameter, less sensitive to noise and background intensity than threshold-based methods.

For each edge, the algorithm:
1. Samples points along the 3D centreline at intervals of `fwhm_sample_spacing_along_edge_um` (default: 2.0 $\mu m$).
2. At each sample point, casts transverse rays perpendicular to the centreline.
3. Extracts the intensity profile from the raw image along each ray.
4. Fits a Gaussian curve to the profile using least-squares.
5. Reports the FWHM of the fitted Gaussian as the vessel diameter at that point.
6. The per-edge diameter is the median of all valid FWHM measurements along that edge.

> **Default parameters:**
> - Transverse profile step: 0.5 $\mu m$
> - Transverse half-extent: 15.0 $\mu m$
> - Minimum fit $R^2$: 0.85
> - Maximum centre offset: 1.5 $\mu m$

> **Verification:** The core $FWHM=2\sqrt{2\ln 2}\,\sigma$ relationship is checked against an ideal 1D Gaussian profile in `test_fwhm_from_profile_gaussian_fit()` (`tests/test_haemodynamics_automated_fwhm.py`, error `<0.2` µm), and end-to-end on a synthetic 3D Gaussian-intensity cylinder written to a real TIFF in `test_measure_edge_diameters_fwhm_from_raw_tiff_cylinder()` (error `<0.35` µm). The two baseline estimators (`wings` vs `percentile`) are cross-checked for agreement in `test_fwhm_percentile_and_wings_modes_symmetric_gaussian()`. Full pipeline robustness is further tested on three adversarial synthetic phantoms — clean multi-diameter tubes, a noisy X-junction with 30% off-centre graph misregistration, and a tight zig-zag — in `tests/test_integration_synthetic_vessel_fwhm.py`, with tolerances scaled to the target diameter (e.g. `abs(got-exp) < max(0.9, 0.18*exp)`).

### 4.3 Exponential Diameter Scaling by Branch Order

When per-edge measurements are unavailable, diameters are assigned based on topological **branch order** (number of bifurcations from the inlet). The pipeline uses a **3-point boundary-fitted exponential scaling** model:

Three biological anchor points are defined:
- **Arterial inlet** (B01): $D_{\text{start}} = 15.0 \, \mu m$
- **Capillary bed** (midpoint): $D_{\text{mid}} = 4.0 \, \mu m$
- **Venous outlet** (maximum branch order): $D_{\text{end}} = 20.0 \, \mu m$

For the arterial side ($n \leq n_{\text{mid}}$):

$$D(n) = D_{\text{mid}} \times \left(\frac{D_{\text{start}}}{D_{\text{mid}}}\right)^{(n_{\text{mid}} - n) / (n_{\text{mid}} - n_{\text{start}})}$$

For the venous side ($n > n_{\text{mid}}$):

$$D(n) = D_{\text{mid}} \times \left(\frac{D_{\text{end}}}{D_{\text{mid}}}\right)^{(n - n_{\text{mid}}) / (n_{\text{end}} - n_{\text{mid}})}$$

> **Physiological Context:** The arterial tree narrows exponentially from large arterioles (~15 $\mu m$) down to capillaries (~4 $\mu m$), then widens again as capillaries merge into venules (~20 $\mu m$). Venules are typically wider than the corresponding arterioles because venous blood is at lower pressure and veins have thinner, more compliant walls. The U-shaped diameter profile (large → small → large) reflects the fundamental architecture of all microvascular beds.

> **Assumption**: The arterial-to-capillary and capillary-to-venous transitions follow exponential scaling laws. This is a simplification; Murray's Law (cubic branching ratio) is not used.

> **Implementation Note:** The complete diameter dictionary is built by the function `build_diameter_by_branch_order()` in `poiseuille.py`. This function accepts the three anchor points above and generates entries for all branch orders up to `max_branch_order` (default: 51). It supports three override dictionaries (`manual_capillary_diameter_by_branch_order`, `manual_arteriole_diameter_by_branch_order`, `manual_venule_diameter_by_branch_order`) that allow the user to replace any automatically computed diameter with a manually specified value. An `all_diams_const` flag can also set all diameters to a uniform `default_diameter` (default: 4.0 $\mu m$), bypassing the exponential scaling entirely.

> **Verification:** No test exercises `build_diameter_by_branch_order()` or the exponential scaling formula directly by name — this is a coverage gap (see §11.3). The downstream consumer of branch-order labels, `assign_hierarchical_branch_orders()`, is thoroughly tested in `test_hierarchical_branch_order_pipeline_flow()` (`tests/test_branch_order_hierarchy.py`), which confirms arteriole (`Art*`), capillary (`B*`), and venule (`Ven*`) labels are assigned along the correct topological paths with monotonically increasing order numbers — but that test checks label correctness, not the diameter values the exponential formula would subsequently assign to those labels.

### 4.4 Sphincter and Pericyte Constriction Models

> **Implementation Note: Two Distinct Code Paths**
> The pipeline provides two functions for setting Poiseuille resistances, with significantly different error-handling behaviours:
>
> | Function | Behaviour on Invalid Edges |
> |---|---|
> | `set_poiseuille_resistances()` | **Silently skips** edges with missing or invalid attributes (diameter, branch order, length), logging them to warning lists. Used for initial bulk resistance assignment. |
> | `set_poiseuille_resistances_with_constrictions()` | **Raises `ValueError`** on any missing or invalid attribute. Used when constriction profiles require complete geometric data. |
>
> A third function, `set_poiseuille_edge_resistances()`, allows setting resistances on a specified subset of custom edges (e.g., for manually modifying specific vessel segments after the initial bulk assignment).

> **Physiological Context: What Are Pericytes and Sphincters?**
> **Pericytes** are contractile cells wrapped around capillaries and small venules. They can actively constrict the vessel lumen, regulating local blood flow at the capillary level — a process called "capillary-level flow control." This is increasingly recognised as a major mechanism for matching oxygen delivery to local metabolic demand.
>
> **Pre-capillary sphincters** are smooth muscle constrictions at the junction between terminal arterioles and capillaries. They act as "gatekeepers," controlling which capillaries are open and receiving blood at any given moment. In the carotid body, **intimal cushions** at the branch origin serve a similar flow-throttling function.
>
> Modelling these constrictions is critical because a localised 50% diameter reduction increases the local resistance by $1/(0.5)^4 = 16\times$, dramatically redirecting flow through the network.

The pipeline models localised vessel constrictions to simulate the effect of pericytes and vascular sphincters. Two constriction modes are available, controlled by `constriction_mode`.

#### 4.4.1 Sphincter Mode (Default: `"sphincter"`)

A single constriction is placed at the **origin** (proximal end) of each vessel segment. The constriction zone has physical length $L_s$ (default: 5.0 $\mu m$).

**Exact Spatial Diameter Profile** (implemented in `get_diameter_at_position()`):

Let $x$ be the position along the vessel (starting from 0), $d_1$ be the unconstricted diameter, and $d_2$ be the constricted diameter:

$$d(x) = \begin{cases}
d_1 + (d_2 - d_1) \cdot \dfrac{x}{0.25 \, L_s} & \text{if } 0 \leq x < 0.25 \, L_s \quad \text{(ramp down)} \\[8pt]
d_2 & \text{if } 0.25 \, L_s \leq x < 0.75 \, L_s \quad \text{(hold)} \\[8pt]
d_2 + (d_1 - d_2) \cdot \dfrac{x - 0.75 \, L_s}{0.25 \, L_s} & \text{if } 0.75 \, L_s \leq x \leq L_s \quad \text{(ramp up)} \\[8pt]
d_1 & \text{if } x > L_s \quad \text{(unconstricted)}
\end{cases}$$

> **Physical Intuition:** The trapezoidal ramp profile prevents discontinuous jumps in diameter, which would create unphysical infinite-gradient pressure drops. The ramp-hold-ramp shape approximates the smooth constriction observed in electron microscopy images of pericyte-wrapped capillaries.

> **Verification:** `test_get_diameter_at_position()` (`tests/test_haemodynamics.py`) checks that the profile stays bounded between $d_1$ and $d_2$ at an interior position, and the full ramp-hold-ramp shape (all four segments) is exercised as the geometry under test in `test_analytical_sphincter_resistance_calculus()` (§4.4.4), which integrates it and compares to exact calculus.

#### 4.4.2 Periodic Mode (`"periodic"`)

Constrictions repeat at regular intervals (`constriction_spacing`, default: 100 $\mu m$) along the vessel. Let $\phi = x \mod \text{constriction\_spacing}$ be the phase position:

$$d(\phi) = \begin{cases}
d_1 + (d_2 - d_1) \cdot \dfrac{\phi}{10} & \text{if } 0 \leq \phi < 10 \, \mu m \quad \text{(ramp down)} \\[8pt]
d_2 & \text{if } 10 \leq \phi < 30 \, \mu m \quad \text{(hold)} \\[8pt]
d_2 + (d_1 - d_2) \cdot \dfrac{\phi - 30}{10} & \text{if } 30 \leq \phi < 40 \, \mu m \quad \text{(ramp up)} \\[8pt]
d_1 & \text{if } \phi \geq 40 \, \mu m \quad \text{(unconstricted)}
\end{cases}$$

> **Physiological Context:** Pericytes are spaced approximately every 50–100 $\mu m$ along capillaries. The periodic mode captures this distributed contraction pattern, modelling each pericyte as creating a local constriction zone of ~40 $\mu m$.

> **Verification:** The periodic profile is the exact geometry benchmarked in `test_analytical_sphincter_resistance_calculus()` (`tests/test_haemodynamics_analytical.py`), which constructs a `PoiseuilleModel(mode="periodic")`, integrates the position-dependent resistance over one full period (ramp-hold-ramp-hold), and confirms the trapezoidal numerical integral matches the exact closed-form calculus to `rtol=1e-3`.

#### 4.4.3 Constriction Ratios

Two physiological constriction types are modelled:

| Constriction Type | Location | Default Ratio ($d_2/d_1$) | Physiological Basis |
|---|---|---|---|
| **Intimal cushion** | Branch order B01 (carotid origin) | 0.60 | Intimal cushions at the origin of the carotid body vessels reduce the lumen to ~60% of its unconstricted diameter. |
| **Pre-capillary sphincter** | At the topological midpoint (capillary bed transition), offset by `pre_capillary_topological_offset` (default: 1 branch order) | 0.50 | Pre-capillary sphincters constrict vessels to ~50% of their resting diameter at the arteriole-capillary junction. |

> **Minimum constriction ratio**: Clamped to 0.01 to prevent infinite resistance / matrix singularities.

**FWHM/EDT interaction with constrictions:**

When using measured diameters (`fwhm_radius` or `edt_radius`), the measured diameter becomes $d_1$, and $d_2$ is computed by preserving the constriction ratio from the branch-order dictionary:

$$d_2 = d_{1,\text{measured}} \times \frac{d_{2,\text{dict}}}{d_{1,\text{dict}}}$$

> **Verification:** Constriction ratios are checked directionally rather than against exact target values: `test_synthetic_pericyte_mask_constriction_integration()` and `test_synthetic_pericyte_mask_constriction_integration_ten_um_away()` (`tests/test_pericyte_mask_integration.py`) confirm that pericytes located near a vessel edge (constriction factor 0.8) strictly increase the two-point effective resistance versus baseline, while pericytes 10 µm away leave resistance unchanged (`np.isclose`). Neither test asserts a specific resistance ratio.

#### 4.4.4 Integrated Resistance with Variable Diameter

> **Numerical Rationale: Why Numerical Integration?**
> When the vessel diameter varies along its length (due to constrictions), we cannot use the simple $R = 128 \mu L / \pi d^4$ formula because $d$ is not constant. Instead, we treat the vessel as a series of infinitesimal cylindrical slices, each with its own diameter and viscosity, and sum their resistances. This is mathematically equivalent to integrating the "resistance per unit length" along the vessel — the trapezoidal rule with 1000 points provides a highly accurate approximation of this integral.

When constrictions are active, the total resistance is computed by **numerical integration** (trapezoidal rule, 1000 sample points) of the position-dependent resistance per unit length:

$$R_{\text{total}} = \int_0^L \frac{128 \, \mu(x)}{\pi \, d(x)^4} \, dx$$

where $d(x)$ varies along the vessel according to the constriction profile (§4.4.1 or §4.4.2) and $\mu(x) = 1/d(x)^{1.647}$ is the power-law viscosity at that position.

This is implemented in `calculate_integrated_resistance()`:
```python
positions = np.linspace(0, length, num_points)  # num_points = 1000
resistances = [resistance_integrand(pos, length, d1, d2) for pos in positions]
dx = length / (num_points - 1)
R_total = np.trapezoid(resistances, dx=dx)
```

> **Guard Rail:** If `length <= 0`, `calculate_integrated_resistance()` returns `float("inf")` — effectively blocking all flow through the segment.

> **Backward Compatibility:** The code first attempts `numpy.trapezoid` (NumPy ≥ 2.0); if unavailable, it falls back to the deprecated `numpy.trapz` for older NumPy versions.

Where the integrand at each position is:
```python
def resistance_integrand(position, length, d1, d2):
    diameter = get_diameter_at_position(position, length, d1, d2)
    viscosity = 1.0 / (diameter ** 1.647)
    return (128.0 * viscosity) / (np.pi * diameter ** 4)
```

> **Verification:** `test_analytical_sphincter_resistance_calculus()` (`tests/test_haemodynamics_analytical.py`) is the flagship benchmark for this section: it analytically integrates $\int (128/\pi) \cdot d(x)^{-5.647}\,dx$ term-by-term over all four segments of the periodic ramp-hold-ramp-hold profile using the closed-form power-rule antiderivative, and confirms the trapezoidal `calculate_integrated_resistance()` output matches to `rtol=1e-3` (0.1%). `test_calculate_integrated_resistance()` and `test_resistance_integrand()` (`tests/test_haemodynamics.py`) additionally check the guard rail (`length<=0` → `float("inf")`) and basic positivity/finiteness.

---

## 5. Tissue Perfusion Modelling

> **Physiological Context: From Vessels to Tissue**
> Up to this point, the pipeline has modelled blood flow *within* the vascular network — pressures, flows, viscosities, and hematocrit distributions. But the biological purpose of the vasculature is to deliver oxygen *to the surrounding tissue*. This section models the critical last step: oxygen leaving the blood, crossing the vessel wall, diffusing through the tissue, and being consumed by metabolically active cells. This is the oxygen "last mile delivery problem" — and it determines whether the glomus cells in the carotid body actually receive enough $O_2$ to function.

### 5.1 Perfusion Grid

A structured 3D Cartesian grid is overlaid on the vascular network. Each grid cell represents a tissue block.

> **Default resolution**: $10 \times 10 \times 10 \, \mu m$ per cell (configured as `grid_resolution_xyz = (10.0, 10.0, 10.0)`).

> **Physical Intuition:** Each grid cell represents a ~1000 $\mu m^3$ block of tissue. At this resolution, a typical capillary (4–8 $\mu m$ diameter) fits within a single cell, which means the model can resolve tissue $PO_2$ gradients at the scale of individual capillary spacings — the physiologically relevant length scale for oxygen delivery.

**Grid Construction (implemented in `PerfusionGrid.__init__()`):**

1. Extract all node positions from graph (in ZYX physical coordinates, $\mu m$).
2. Resolution is input as $(x, y, z)$ and internally flipped to $(z, y, x)$ to match ImageLynx conventions.
3. Grid bounds are padded by half-resolution:
   - $\text{min} = \min(\text{nodes}) - \frac{\text{res}}{2}$
   - $\text{max} = \max(\text{nodes}) + \frac{\text{res}}{2}$
4. Grid dimensions: $\text{dims} = \lceil (\text{max} - \text{min}) / \text{res} \rceil$
5. Cell volume: $V_{\text{cell}} = \Delta z \times \Delta y \times \Delta x$

**Linear Indexing Convention:**

$$\text{index} = i_z + i_y \cdot n_z + i_x \cdot n_z \cdot n_y$$

where $i_z$ is the fastest-varying index.

**Vessel-to-Grid Mapping (implemented in `map_vessels_to_grid()`):**

For each edge, the centreline voxels are point-sampled. The vessel surface area within each cell is accumulated as:

$$A_{\text{surface}} = 2\pi r \cdot L_{\text{segment}}$$

where $r$ is the vessel radius and $L_{\text{segment}} = L_{\text{edge}} / (N_{\text{voxels}} - 1)$ is the length per voxel segment.

> **Physical Intuition:** The surface area determines how much "membrane" is available for gas exchange in each tissue block. More vessel surface area in a cell means more oxygen can be delivered to that region — it's analogous to the alveolar surface area in the lungs.

> **Verification:** Grid construction is checked exactly in `test_perfusion_grid_dimensions()` (`tests/test_haemodynamics_perfusion.py`): a graph spanning [0,20]³ µm with 10 µm resolution is confirmed to produce exactly a 3×3×3 grid (27 cells) with bounds padded to [-5,25]³. Bidirectional index↔coordinate mapping is checked in `test_grid_index_bidirectional_mapping()` (round-trip recovers the original point to within half a cell), and out-of-bounds coordinates are confirmed to safely return `-1` in `test_grid_out_of_bounds_handling()`. Vessel-to-grid deposition is checked in `test_map_vessels_to_grid_straight_line()`, which confirms a straight line segment injects positive flow into the grid, and hematocrit-weighted deposition specifically in `test_advective_source_hematocrit_weighting()`, which confirms a pure-plasma vessel ($H_D=0$) delivers under 5% of the oxygen a normal-hematocrit vessel does for the same flow.

### 5.2 Advection–Diffusion–Reaction (ADR) Equation

> **Physical Intuition: The Three Competing Processes**
> The steady-state oxygen field in tissue is governed by a balance of three processes:
> 1. **Diffusion** ($\nabla \cdot (\sigma \nabla C)$): Oxygen spreads from regions of high concentration to low concentration, like heat diffusing through a metal rod.
> 2. **Advection/Source** ($S_{\text{advection}}$): Blood vessels deliver oxygen to the tissue (source) and carry depleted blood away (sink).
> 3. **Reaction/Metabolism** ($M(C)$): Cells consume oxygen for aerobic respiration, creating a local sink.
>
> At steady state, these three processes are in perfect balance everywhere: what diffuses in + what blood delivers = what cells consume + what diffuses out.

The steady-state tissue oxygen concentration field is governed by:

$$\nabla \cdot (\sigma \nabla C) + S_{\text{advection}} - M(C) = 0$$

where:
- $\sigma$ is the tissue oxygen diffusion coefficient (default: $1.5 \times 10^{-9}$ $m^2/s$). Internally, this is converted to $\mu m^2/s$ by multiplying by $10^{12}$, giving $\sigma = 1500 \, \mu m^2/s$. This conversion is applied when building the diffusion matrix (§7.1).
- $C$ is tissue oxygen concentration (mmol/L, but the solver works in $PO_2$ space, mmHg),
- $S_{\text{advection}}$ represents oxygen delivered and removed by blood flow,
- $M(C)$ is the metabolic consumption rate.

> **Verification:** The pure-diffusion limit ($S_{\text{advection}}=M=0$) is benchmarked against a straight-line analytical gradient in `test_analytical_1d_pure_diffusion()` (`tests/test_haemodynamics_analytical.py`, `atol=1e-10`). The pure reaction-diffusion limit (constant metabolic sink, no advection) is benchmarked against the exact parabola $C(x)=C_0-(M/2D)x(L-x)$ in `test_analytical_zero_order_metabolism()` (`atol=1e-10`). The pure advection-diffusion limit (point source, no metabolism) is checked against the qualitative $1/r$ radial falloff in `test_analytical_radial_point_source()` — strict monotonic decay is asserted exactly, while the $1/r$ ratio itself is checked only loosely ($1.5 < C(1)/C(2) < 2.5$) because of discretisation error near the source. No single test benchmarks all three terms acting together against a closed-form solution.

### 5.3 Oxygen–Haemoglobin Dissociation (Hill Equation)

> **Physiological Context: Why Is This Equation So Important?**
> The Hill equation describes how haemoglobin binds and releases oxygen. It is the central equation connecting the *partial pressure* of oxygen (the "driving force" for diffusion) to the *total oxygen content* in blood (the "carrying capacity"). The sigmoidal shape of the curve is critical: at high $PO_2$ (lungs), haemoglobin is nearly fully saturated and small increases in $PO_2$ add little extra $O_2$. At the steep part of the curve (~26 mmHg), small drops in $PO_2$ cause large amounts of $O_2$ to be released — this is exactly the range where tissue oxygen extraction occurs, making delivery highly efficient.
>
> The parameter $P_{50}$ (the $PO_2$ at which haemoglobin is 50% saturated) is the "set point" of this curve. The Bohr effect (§5.4) shifts $P_{50}$ based on local $CO_2$ and $pH$, creating an elegant feedback loop: metabolically active tissue (high $CO_2$, low $pH$) shifts the curve rightward, making haemoglobin release oxygen more readily precisely where it is needed most.

Blood oxygen content is calculated using the Hill equation for the oxygen–haemoglobin dissociation curve, implemented in `calculate_blood_oxygen_content()`:

**Step 1 — Bohr Effect (§5.4) — Dynamic $P_{50}$ Shift:**

$$\log_{10}(P_{50}) = \log_{10}(26.0) - 0.4 \cdot (pH - 7.4) + 0.06 \cdot \log_{10}\!\left(\frac{\max(PCO_2, 1.0)}{40}\right)$$

$$P_{50} = 10^{\log_{10}(P_{50})}$$

**Step 2 — Dissolved Oxygen (Henry's Law):**

$$C_{\text{dissolved}} = \alpha_{O_2} \cdot PO_2$$

where $\alpha_{O_2} = 1.34 \times 10^{-3}$ mmol/L per mmHg.

> **Physical Intuition:** Henry's Law states that the concentration of a gas dissolved in a liquid is proportional to its partial pressure. Dissolved $O_2$ is a small but important fraction — it is the *only* form of oxygen that can directly diffuse across cell membranes. Haemoglobin-bound $O_2$ must first be released (desaturation) and become dissolved before it can reach mitochondria.

**Step 3 — Haemoglobin Saturation (Hill Equation):**

$$S_{O_2} = \frac{PO_2^n}{PO_2^n + P_{50}^n}$$

where $n = 2.7$ (Hill coefficient).

> **Physical Intuition:** The Hill coefficient $n$ quantifies the *cooperativity* of oxygen binding. Haemoglobin has four binding sites, and binding of the first $O_2$ molecule makes the remaining sites bind more easily (positive cooperativity). The Hill coefficient of 2.7 (rather than 4.0) reflects the fact that cooperativity is not perfect. This cooperativity creates the sigmoidal shape of the dissociation curve that makes haemoglobin such an efficient oxygen transporter.

**Step 4 — Bound Oxygen:**

$$C_{\text{bound}} = H_D \cdot C_{Hb,\text{max}} \cdot S_{O_2}$$

where:
- $H_D$ is discharge hematocrit (dimensionless),
- $C_{Hb,\text{max}} = \frac{0.446 \times 20.4}{0.45}$ mmol/L (maximal haemoglobin O₂ binding capacity, scaled to pure RBC content).

> **Numerical Rationale: Origin of $C_{Hb,\text{max}}$ Constants:**
> - $0.446$ mL O₂/g Hb — the Hüfner number, representing the maximum volume of O₂ that can bind to 1 gram of haemoglobin.
> - $20.4$ — a conversion factor that transforms the volumetric capacity into mmol/L units at standard conditions (accounting for haemoglobin concentration in normal blood, ~150 g/L, and the molar volume of O₂).
> - $0.45$ — the reference whole-blood hematocrit. Dividing by 0.45 normalises the capacity from a whole-blood basis to a per-RBC basis, so that when multiplied by the local $H_D$ in Step 4, the result correctly reflects the actual local RBC concentration.

**Step 5 — Total Oxygen Content:**

$$C_{O_2} = C_{\text{dissolved}} + C_{\text{bound}} = \alpha_{O_2} \cdot PO_2 + H_D \cdot C_{Hb,\text{max}} \cdot S_{O_2}$$

> **Guard Rail**: If $PO_2 \leq 0$, the function returns 0.0 immediately.

> **Assumption**: The Hill equation provides a sigmoidal approximation to the full Adair equation for cooperative oxygen binding. The Hill coefficient $n = 2.7$ is appropriate for adult human haemoglobin but may differ for other species.

> **Verification:** `test_hill_equation_sigmoidal_curve()` (`tests/test_haemodynamics_perfusion.py`) checks the sigmoidal shape at three named points ($PO_2$ = 10, 26, 100 mmHg increase monotonically) and, critically, confirms the defining property of $P_{50}$: after subtracting the dissolved-oxygen term, the bound-haemoglobin fraction at $PO_2=26$ mmHg is exactly 50% of the maximum bound capacity (`atol=1e-5`), and exceeds 95% at $PO_2=100$ mmHg.

### 5.4 Bohr Effect

> **Physiological Context: Nature's Oxygen Delivery Optimiser**
> The Bohr effect is one of the most elegant feedback mechanisms in physiology. When tissue is metabolically active, it produces $CO_2$ (which lowers local $pH$). These changes shift the haemoglobin dissociation curve to the *right* — increasing $P_{50}$ — which means haemoglobin releases oxygen more readily at any given $PO_2$. The result: oxygen is preferentially unloaded precisely where demand is highest. In the carotid body, where metabolism is exceptionally high relative to tissue volume, the Bohr effect significantly enhances oxygen delivery to glomus cells.

The $P_{50}$ value shifts dynamically based on local $PCO_2$ and $pH$:

$$\log_{10}(P_{50}) = \log_{10}(26.0) - 0.4 \cdot (pH - 7.4) + 0.06 \cdot \log_{10}\left(\frac{PCO_2}{40}\right)$$

This empirical formulation is based on Kelman (1966) and Severinghaus (1979). Higher $PCO_2$ and lower $pH$ shift the curve rightward (decreased oxygen affinity), facilitating oxygen unloading in metabolically active tissue.

At baseline conditions ($pH = 7.4$, $PCO_2 = 40$ mmHg):
$$P_{50} = 26.0 \text{ mmHg}$$

> **Verification:** `test_bohr_haldane_atomic_curves()` (`tests/test_haemodynamics_analytical.py`) confirms the Bohr shift direction only: blood oxygen content at fixed $PO_2=26$ mmHg is strictly lower under severe acidosis (pH 7.0) than at normal pH (7.4). No test asserts the shifted $P_{50}$ against a specific numerical target.

### 5.5 Carbon Dioxide Transport and Haldane Effect

> **Physiological Context: The Bohr Effect's Mirror Image**
> The Haldane effect is the reciprocal of the Bohr effect: just as $CO_2$ influences $O_2$ binding (Bohr), $O_2$ influences $CO_2$ carrying capacity (Haldane). Deoxygenated haemoglobin binds $CO_2$ more readily than oxygenated haemoglobin. This means that as blood drops off $O_2$ in the tissues, it simultaneously becomes better at picking up $CO_2$ — a beautiful evolutionary optimisation that makes the same molecule (haemoglobin) serve as both the $O_2$ delivery truck and the $CO_2$ waste removal truck, with loading and unloading naturally coordinated.

CO₂ content in blood is modelled as the sum of dissolved and bound fractions, implemented in `calculate_blood_co2_content()`:

**Step 1 — Approximate O₂ Saturation (for Haldane Shift):**

$$S_{O_2} \approx \frac{PO_2^{2.7}}{PO_2^{2.7} + 26.0^{2.7}}$$

Note: This uses a simplified Hill equation with fixed $P_{50} = 26.0$ mmHg (no Bohr shift feedback here) for the sole purpose of estimating the Haldane effect magnitude.

> **Practical Impact of This Simplification:** In regions of high metabolic activity (low $PO_2$, high $PCO_2$), the Bohr effect would shift $P_{50}$ rightward (increasing it beyond 26 mmHg), which would lower the actual $S_{O_2}$ at a given $PO_2$ compared to the fixed-$P_{50}$ estimate used here. This means the Haldane effect is slightly *underestimated* in metabolically active tissue — the code treats haemoglobin as more oxygenated than it actually is, so it slightly underestimates the CO₂ carrying enhancement from deoxygenation. For the moderate $PO_2$ ranges encountered in the carotid body, this error is small ($<5\%$ in $C_{CO_2}$), but it represents a minor inconsistency in the coupled Bohr–Haldane system.

**Step 2 — Base CO₂ Carrying Capacity:**

$$C_{CO_2,\text{base}} = 11.02 \cdot PCO_2^{0.396}$$

**Step 3 — Haldane Shift:**

$$\text{Haldane shift} = (0.15 - 0.05 \cdot S_{O_2}) \cdot PCO_2$$

> **Physical Intuition:** When $S_{O_2}$ is low (deoxygenated blood), the Haldane shift is large ($(0.15 - 0.05 \times 0) \times PCO_2 = 0.15 \times PCO_2$), meaning the blood can carry much more $CO_2$. When $S_{O_2}$ is high (oxygenated blood), the shift is smaller ($(0.15 - 0.05 \times 1) \times PCO_2 = 0.10 \times PCO_2$). This differential is what drives efficient $CO_2$ loading in tissues and unloading in the lungs.

**Step 4 — Total CO₂ Content:**

$$C_{CO_2} = \alpha_{CO_2} \cdot PCO_2 + H_D \cdot (C_{CO_2,\text{base}} + \text{Haldane shift})$$

where:
- $\alpha_{CO_2} = 0.03$ mmol/L per mmHg (Henry's law solubility).

> **Guard Rail**: If $PCO_2 \leq 0$, the function returns 0.0 immediately.

> **Verification:** The same test, `test_bohr_haldane_atomic_curves()`, confirms the Haldane shift direction: at fixed $PCO_2=40$ mmHg, hyperoxic blood ($PO_2=100$) carries strictly less $CO_2$ than hypoxic blood ($PO_2=20$). As with the Bohr effect, only the direction of the shift is asserted, not a numerical magnitude.

### 5.6 Henderson–Hasselbalch pH Equation

> **Physiological Context: The CO₂–pH Link**
> When $CO_2$ dissolves in water, it forms carbonic acid ($H_2CO_3$), which rapidly dissociates into $H^+$ and $HCO_3^-$. This is why high $CO_2$ makes blood (and tissue) more acidic. The Henderson–Hasselbalch equation quantifies this relationship, linking the measurable $PCO_2$ to the resulting $pH$ given a known bicarbonate buffer concentration. In the model, this equation closes the loop: $CO_2$ production by metabolism → local $PCO_2$ rise → local $pH$ drop → Bohr shift → enhanced $O_2$ unloading.

Tissue $pH$ is calculated from the local $PCO_2$ using the Henderson–Hasselbalch equation, implemented in `calculate_ph_from_pco2()`:

$$pH = pK_a + \log_{10}\left(\frac{[HCO_3^-]}{\alpha_{CO_2} \cdot PCO_2}\right)$$

where:
- $pK_a = 6.1$ (carbonic acid dissociation constant),
- $[HCO_3^-] = 24.0$ mmol/L (tissue bicarbonate buffer concentration, assumed constant),
- $\alpha_{CO_2} = 0.03$ mmol/L per mmHg.

> **Guard Rail**: $PCO_2$ is clamped to $\geq 10^{-12}$ to prevent $\log_{10}(0)$.

> **Assumption**: Bicarbonate concentration is held constant (open buffer system). In reality, $[HCO_3^-]$ is regulated by renal compensation and varies with acid-base disturbances.

> **Verification:** `test_henderson_hasselbalch_equilibrium()` (`tests/test_haemodynamics_analytical.py`) checks two physiological anchor points exactly: normal conditions ($PCO_2=40$) recover $pH=7.4$ to `atol=1e-2`, and severe hypercapnia ($PCO_2=80$) drives $pH$ below 7.15, consistent with clinical severe acidosis.

### 5.7 Metabolic Oxygen Consumption

> **Physiological Context: Why a Saturating Function?**
> Cells cannot consume oxygen infinitely fast — mitochondrial cytochrome c oxidase has a maximum turnover rate. At high $PO_2$, metabolism runs at full speed ($M_{\text{max}}$). As $PO_2$ drops below a critical threshold, mitochondria begin to "starve" for oxygen and metabolism slows. At $PO_2 = 0$, consumption is zero (no oxygen to consume). This saturating behaviour is biologically universal and is a critical feature of the model: it creates the "tissue hypoxia" phenomenon where regions far from capillaries can have dangerously low $PO_2$.

Tissue metabolic consumption follows a saturating exponential:

$$M(PO_2) = M_{\text{max}} \cdot \left(1 - e^{-k \cdot PO_2}\right)$$

where:
- $M_{\text{max}} = 0.005$ mmol/L/s (maximum metabolic rate),
- $k = 0.1$ per mmol (reduction constant for hypoxic zones),
- $PO_2$ is clamped to $\geq 0$ before evaluation.

> **Assumption**: This is a phenomenological model, not a Michaelis–Menten kinetic model. The exponential form ensures consumption approaches zero as $PO_2$ → 0 and saturates at $M_{\text{max}}$ for high $PO_2$. A Michaelis–Menten form ($M = M_{\text{max}} \cdot PO_2 / (K_m + PO_2)$) is more commonly used in the literature for mitochondrial oxygen consumption.

> **Verification:** Verified indirectly as the reaction term inside `test_analytical_zero_order_metabolism()` (§5.2), which forces $k$ large enough to make $M(PO_2)$ behave as a constant zero-order sink and confirms the resulting parabolic profile to `atol=1e-10`. The saturating (non-zero-order) regime of the exponential itself — i.e. metabolism actually varying continuously with $PO_2$ rather than being pinned to $M_{\text{max}}$ — is not directly benchmarked against a closed-form solution.

### 5.8 Respiratory Quotient

> **Physiological Context:** The respiratory quotient (RQ) is the ratio of $CO_2$ molecules produced to $O_2$ molecules consumed during metabolism. It depends on the metabolic substrate: pure carbohydrate oxidation gives $RQ = 1.0$, pure fat oxidation gives $RQ \approx 0.7$, and protein oxidation gives $RQ \approx 0.8$. The default value of 0.82 represents a typical mixed diet.

The coupling between $O_2$ consumption and $CO_2$ production uses a fixed respiratory quotient:

$$M_{CO_2} = RQ \times M_{O_2}$$

> **Default**: $RQ = 0.82$ (typical for a mixed metabolic substrate of carbohydrates and fats).

> **Verification:** Exercised as one term inside `test_multi_species_0d_fick_mass_balance()` (§6.3), where $M_{CO_2}=RQ\times M_{O_2}$ is used to derive the analytical target $PCO_2$ that the multi-species Picard solver is checked against (`atol=1e-2`). No test isolates the $RQ$ coupling on its own.

---

## 6. Endothelial Barrier and 1D–3D Coupling

> **Physiological Context: The Endothelial Barrier**
> The endothelium — a single layer of cells lining every blood vessel — is not just a passive container. It is a semi-permeable membrane that controls what crosses from blood to tissue. For small gases like $O_2$ and $CO_2$, the endothelium presents relatively little resistance compared to the diffusion distance through tissue, but it is not negligible, especially in the carotid body where fenestrated (perforated) capillaries have unusually high permeability. Modelling this barrier explicitly (rather than assuming instantaneous equilibrium) allows the pipeline to capture scenarios where oxygen delivery is "barrier-limited" rather than "diffusion-limited."

The pipeline implements three solver tiers of increasing complexity. The tier is selected based on configuration flags.

### 6.1 Tier 1: Simple ADR Solver (`solve_perfusion_steady_state`)

Used when `use_endothelial_barrier_model = False` and `use_multi_species_model = False`.

Oxygen delivery is modelled as a bulk advective source and sink (assembled into a matrix by `build_adr_matrix()` in `perfusion.py`):
- **Source** (oxygen in): $S_{\text{in},i} = \sum_{\text{vessels}} Q_v \cdot C_{O_2}(PO_{2,\text{arterial}}, H_D)$
- **Washout** (oxygen out): $S_{\text{out},i} = Q_{\text{total},i} \cdot C_{O_2}(PO_{2,\text{tissue},i}, H_{\text{baseline}})$

> **Physical Intuition:** In this simplified model, blood enters each tissue block at arterial $PO_2$ and leaves at the local tissue $PO_2$ — as if the blood instantly equilibrates with the surrounding tissue. The net oxygen delivered is the difference between what enters (arterial) and what leaves (venous). This is the "well-stirred" or "Krogh cylinder" approximation applied at the voxel level.

The washout term is non-linear (depends on $PO_2$ through the Hill equation), solved via Picard iteration.

> **Verification:** Physical bounds are checked in `test_perfusion_solver_zero_flow()` (`tests/test_haemodynamics_perfusion.py`): with zero flow through every vessel, steady-state tissue concentration is confirmed to be exactly 0.0 everywhere (`atol=1e-10`). `test_perfusion_solver_no_metabolism()` confirms the complementary case — with metabolism switched off, tissue concentration is strictly positive. The full non-linear washout is benchmarked against an exact Fick-principle root in `test_analytical_0d_fick_principle_mass_balance()`: a single isolated voxel's steady-state concentration is inverted via `scipy.optimize.brentq` to an analytical $PO_2$ target, and the Picard solver's output matches to `atol=1e-2`.

### 6.2 Tier 2: Coupled 1D–3D with Endothelial Barrier (`solve_coupled_1d3d_perfusion`)

Used when `use_endothelial_barrier_model = True` and `use_multi_species_model = False`.

**Endothelial Permeability Model:**

Oxygen transport across the vessel wall is governed by a **permeability-limited flux**:

$$J_{O_2} = P_{O_2} \cdot A_{\text{surface}} \cdot \alpha_{O_2} \cdot (PO_{2,\text{blood}} - PO_{2,\text{tissue}})$$

where:
- $P_{O_2} = 1.0 \times 10^{-4}$ cm/s (endothelial permeability coefficient for $O_2$; internally converted to $\mu m/s$ by multiplying by $10^4$, giving $P_{O_2} = 1.0 \, \mu m/s$),
- $\alpha_{O_2} = 1.34 \times 10^{-3}$ mmol/L per mmHg (the oxygen solubility coefficient from Henry's Law; required here to convert the $\Delta PO_2$ driving force from mmHg into a concentration difference for dimensional consistency with the permeability coefficient),
- $A_{\text{surface}}$ is the vessel surface area in the grid cell ($\mu m^2$),
- The driving force is the partial pressure difference across the endothelium (mmHg).

> **Physical Intuition:** This is Fick's first law applied to the endothelial membrane: flux = permeability × area × concentration gradient. The key insight is that oxygen delivery is now *limited by the rate at which it can cross the vessel wall*, not just by how much is available in the blood. This is more realistic because it accounts for the finite permeability of the endothelium.

> **Verification — caveat:** `test_analytical_transmural_exponential_decay()` (`tests/test_haemodynamics_perfusion.py`) computes an analytical target $PO_{2,\text{out}} = PO_{2,\text{in}} \cdot e^{-P_{perm}A/\alpha Q}$, but the test's own assertions do not compare the solver's output to that target — the metabolic sink is deliberately set to $10^9$ to force *tissue* $PO_2$ to ~0, and the only assertions are `len(po2_num) == 1` and `po2_num[0] < 1e-3` on the tissue array. The analytical value is computed but never checked against a solver output. This is a structural-completion test, not an exponential-decay benchmark, despite the docstring's framing — the same pattern as the Krogh cylinder test (§11.3).

### 6.3 Tier 3: Multi-Species 1D–3D Solver (`solve_multi_species_perfusion`)

Used when `use_multi_species_model = True`.

Solves for tissue $PO_2$, $PCO_2$, and $pH$ simultaneously with Bohr/Haldane coupling.

> **Physical Intuition:** This is the most biologically complete model. By tracking $O_2$, $CO_2$, and $pH$ simultaneously and allowing them to interact through the Bohr and Haldane effects, the solver captures the physiological feedback loops that regulate oxygen delivery. In metabolically active tissue: metabolism consumes $O_2$ → local $PO_2$ drops → metabolism produces $CO_2$ → local $PCO_2$ rises → local $pH$ drops → Bohr shift increases $P_{50}$ → haemoglobin releases $O_2$ more readily → local $PO_2$ partially recovers. This feedback loop is computed at every grid cell and every vessel segment.

**For $CO_2$ transmural flux:**
$$J_{CO_2} = P_{CO_2} \cdot A_{\text{surface}} \cdot \alpha_{CO_2} \cdot (PCO_{2,\text{blood}} - PCO_{2,\text{tissue}})$$

where $P_{CO_2} = 2.0 \times 10^{-3}$ cm/s ($= 20.0 \, \mu m/s$).

> **Physical Intuition:** $CO_2$ permeability is ~20× higher than $O_2$ because $CO_2$ is more soluble in the lipid bilayer of cell membranes. This means $CO_2$ exchange is rarely a bottleneck — the endothelium is almost transparent to $CO_2$.

> **Verification:** `test_multi_species_0d_fick_mass_balance()` (`tests/test_haemodynamics_analytical.py`) is the flagship benchmark for the coupled Tier 3 solver: a single isolated voxel's steady-state $PO_2$, $PCO_2$, and $pH$ are derived analytically from Fick's principle and the Henderson–Hasselbalch equation, and the fully coupled Bohr/Haldane Picard solver's output matches to `atol=1e-2` ($PO_2$, $PCO_2$) and `atol=1e-3` ($pH$). This is the only test in the suite that validates the interaction of all three species simultaneously against independently derived numbers.

### 6.4 1D Blood Oxygen Tracking Along Vessels

> **Physical Intuition: Why Track Blood Along Each Vessel?**
> In the simple model (Tier 1), blood enters each tissue block at arterial $PO_2$ regardless of how far it has already travelled through the network. In reality, blood progressively loses oxygen as it flows from arterioles through capillaries. By the time blood reaches the venous end of a capillary, it may have delivered 30–50% of its oxygen. Tracking this progressive desaturation along each vessel is essential for accurately predicting the oxygen gradient from arteriolar to venous ends of capillaries — the "longitudinal gradient" that determines whether downstream tissue cells receive adequate oxygen.

In the coupled 1D–3D models (Tiers 2 and 3), blood oxygen content is tracked **along each vessel** as it traverses tissue grid cells.

**Complete Algorithm:**

```
STEP 1: Build DAG from flow directions
  For each edge (u,v,k):
    If flow_signed > 0: DAG.add_edge(u → v)
    If flow_signed < 0: DAG.add_edge(v → u)
```
> **What happens here:** We create a directed graph where every edge points in the direction blood actually flows (from high pressure to low pressure). This ensures we always track oxygen content in the correct direction.

```
STEP 2: Topological Sort
  Try: topo_order ← nx.topological_sort(DAG)
  Except: topo_order ← list(G.nodes())  [fallback]
```
> **What happens here:** We sort all nodes from upstream (inlets, arteries) to downstream (outlets, veins). This guarantees that when we process any node, we have already computed the blood oxygen content at all of its upstream predecessors — ensuring a causal, physics-consistent propagation of oxygen from arteries to veins.

```
STEP 3: Initialize Inlet Blood Gas Content
  For each inlet node n:
    For each outgoing edge from n:
      node_o2_flux_in[n] += C_O2(PO2_arterial, H_edge, PCO2_arterial, pH=7.4) × Q
      node_co2_flux_in[n] += C_CO2(PCO2_arterial, H_edge, PO2_arterial) × Q
      node_q_in[n] += Q
```
> **What happens here:** At the network inlets, fresh arterial blood enters at $PO_2 = 100$ mmHg, $PCO_2 = 40$ mmHg, $pH = 7.4$. These are the "boundary conditions" for the 1D blood tracking.

```
STEP 4: Traverse DAG Topologically
  For each node in topo_order:
    c_o2_mix ← node_o2_flux_in[node] / node_q_in[node]
    c_co2_mix ← node_co2_flux_in[node] / node_q_in[node]
```
> **What happens here:** At converging junctions (where multiple vessels merge), blood from different upstream paths mixes. The mixed concentration is a flow-weighted average — branches carrying more blood contribute proportionally more to the mixture.

```
    For each outgoing edge (node → v, key k):
      # STEP 4a: Invert Hill equation to get blood PO2 from concentration
      po2_in ← brentq(λ p: C_O2(p, H, 40.0, 7.4) - c_o2_mix, 0.0, 150.0)
      pco2_in ← brentq(λ p: C_CO2(p, H, po2_in) - c_co2_mix, 0.0, 150.0)
```
> **What happens here:** We know the *total oxygen content* (concentration) in the blood at this node, but the transmural flux equations (§6.2) need the *partial pressure*. Converting from concentration back to partial pressure requires "inverting" the Hill equation — but because the Hill equation is a non-linear function involving the exponent $n = 2.7$, there is no closed-form algebraic inverse. **Brent's root-finding method** is used instead: it searches for the unique $PO_2$ value where $C_{O_2}(PO_2) = C_{\text{known}}$. Brent's method is chosen because it is guaranteed to converge (bracket-based) and is highly efficient for smooth, monotonic functions like the Hill equation.

```
      # STEP 4b: Walk along each grid cell this edge passes through
      For each cell traversed by this edge:
        flux_o2 ← P_O2 × A_surface × α_O2 × (po2_curr − PO2_tissue[cell_idx])
        c_o2_curr ← max(0, c_o2_curr − flux_o2 / Q)
        po2_curr ← brentq(...)  # Re-invert Hill equation
```
> **What happens here:** As blood flows through each tissue grid cell, oxygen leaks out through the vessel wall at a rate proportional to the $PO_2$ difference between blood and tissue (§6.2). This flux is subtracted from the blood's oxygen content, reducing $c_{O_2}$. The Hill equation is then re-inverted to find the new, lower blood $PO_2$. This process repeats for every grid cell the vessel passes through, creating a progressive desaturation profile along the vessel length.

```
      # STEP 4c: Pass remaining blood content to downstream node
      node_o2_flux_in[v] += c_o2_curr × Q
      node_q_in[v] += Q
```
> **What happens here:** Whatever oxygen remains in the blood after traversing this vessel is passed to the downstream node, where it will contribute to the mixed concentration for the next generation of vessels.

**Brent's Root-Finding for Hill Equation Inversion:**

The objective function passed to `scipy.optimize.brentq` is:

$$f(p) = C_{O_2}(p, H_D, PCO_2, pH) - C_{\text{target}}$$

where $C_{O_2}(p, H_D, PCO_2, pH)$ is the full `calculate_blood_oxygen_content()` function (§5.3). The root $p^*$ satisfying $f(p^*) = 0$ is the blood $PO_2$ that corresponds to the target concentration $C_{\text{target}}$.

- Search bracket: $[0.0, 150.0]$ mmHg.
- **Fallback for $O_2$:** If `brentq` raises `ValueError` (no root in bracket), the solver returns $PO_{2,\text{arterial}}$ if the target concentration is positive, or $0.0$ otherwise.
- **Fallback for $CO_2$:** If `brentq` raises `ValueError`, the solver returns the arterial $PCO_2$ value if target concentration is positive, or $0.0$ otherwise.

> **Numerical Rationale:** Brent's method combines the guaranteed convergence of the bisection method with the speed of the secant method. For the Hill equation, which is strictly monotonically increasing in $PO_2$, there is always exactly one root in any valid bracket, making Brent's method ideal.

Similarly for $CO_2$:
$$g(p) = C_{CO_2}(p, H_D, PO_2) - C_{\text{target,CO_2}}$$

> **Assumption**: Blood is perfectly mixed within each vessel cross-section (plug-flow approximation). There is no radial $PO_2$ gradient within the vessel lumen.

> **Verification:** No test in the suite exercises the 1D blood-tracking DAG traversal or the `brentq` Hill-equation inversion in isolation — both are only reached transitively inside the Tier 2/3 solver calls of §6.2 and §6.3 (and in the structural-only Krogh test, §11.3). A dedicated unit test that isolates `brentq(...)` — feeding it a known concentration and asserting the recovered $PO_2$ matches the value used to generate that concentration — does not exist. This is a coverage gap (§11.3).

---

## 7. Numerical Methods and Solver Details

### 7.1 Diffusion Matrix (7-Point Stencil)

> **Numerical Rationale: Why a 7-Point Stencil?**
> The tissue diffusion operator uses a standard **7-point finite-difference stencil** on the 3D Cartesian grid, yielding a symmetric positive semi-definite sparse matrix. Each interior cell connects to its 6 face-sharing neighbours (±x, ±y, ±z) plus itself (the diagonal term). This produces a sparse, symmetric, diagonally dominant matrix that is ideal for iterative solvers like Conjugate Gradient. Higher-order stencils (19-point or 27-point) could reduce numerical diffusion but would significantly increase matrix bandwidth and computational cost for minimal gain at the grid resolutions used here (~10 $\mu m$).

The tissue diffusion operator uses a standard **7-point finite-difference stencil** on the 3D Cartesian grid, yielding a symmetric positive semi-definite sparse matrix.

**Diffusive conductance between adjacent cells:**

$$D_z = \sigma \cdot \frac{\Delta y \cdot \Delta x}{\Delta z}, \quad D_y = \sigma \cdot \frac{\Delta z \cdot \Delta x}{\Delta y}, \quad D_x = \sigma \cdot \frac{\Delta z \cdot \Delta y}{\Delta x}$$

> **Physical Intuition:** Each diffusive conductance is the product of the diffusion coefficient ($\sigma$) and the cross-sectional area of the face through which diffusion occurs, divided by the distance between cell centres. This is directly analogous to thermal conductance in heat transfer: $G = kA/L$, where $k$ is thermal conductivity, $A$ is the face area, and $L$ is the distance. Note that the diffusion coefficient must first be converted from SI units ($m^2/s$) to mesh units ($\mu m^2/s$) by multiplying by $10^{12}$ before these conductances are computed (see §5.2).

Note: In the multi-species solver, the diffusion matrix is pre-scaled by the solubility coefficient $\alpha$ so that the system solves directly for partial pressure (mmHg) rather than concentration:

$$D_x^{(\text{scaled})} = \sigma \cdot \alpha \cdot \frac{\Delta y \cdot \Delta x}{\Delta z}$$

**Sparse Matrix Assembly Algorithm:**

```
N ← total number of grid cells (nx × ny × nz)
diag_A ← zeros(N)

Reshape index array: idx = arange(N).reshape(nz, ny, nx)

# X-direction connections (left-right neighbours)
left  ← idx[:, :, :-1].flatten()
right ← idx[:, :, 1:].flatten()
For each pair (left[i], right[i]):
  A[left[i], right[i]] += -D_x
  A[right[i], left[i]] += -D_x
  diag_A[left[i]]  += D_x
  diag_A[right[i]] += D_x
```
> **What happens here:** For every pair of adjacent cells in the x-direction, we add off-diagonal entries representing the diffusive "pipe" between them (with a negative sign, because flux *leaving* one cell *enters* the other). The diagonal accumulates the sum of all outgoing conductances — this is the discrete analogue of the divergence operator.

```
# Y-direction and Z-direction connections follow the same pattern...

# Regularization to prevent singularity under Neumann BCs
A[i, i] += 1e-12  for all i
```
> **Numerical Rationale:** Under pure Neumann (zero-flux) boundary conditions, the diffusion matrix is singular — the constant function is in the null space (if there are no sources or sinks, any uniform concentration is a valid steady-state). Adding a tiny diagonal perturbation ($10^{-12}$) regularizes the matrix without measurably affecting the solution, allowing iterative solvers like CG to converge.

> **Boundary condition**: Neumann (zero-flux) at the tissue grid boundaries — no oxygen escapes through the tissue surface. This is implicit in the stencil: boundary cells simply have fewer neighbours, resulting in fewer off-diagonal entries.

> **Verification:** `test_build_adr_matrix_structure()` (`tests/test_haemodynamics_perfusion.py`) checks the stencil's structural integrity exactly on a 3×3×3 grid: the central interior cell (index 13) has exactly 7 non-zero entries (itself + 6 neighbours), and a corner cell (index 0) has exactly 4 (itself + 3 neighbours, confirming Neumann boundary truncation). `test_advective_source_vector()` additionally confirms that non-zero source entries correspond exactly to grid cells containing vessels.

### 7.2 Picard Iteration for Non-Linear Perfusion

> **Numerical Rationale: Why Picard Instead of Newton?**
> The perfusion system is non-linear because: (a) the metabolic consumption $M(PO_2)$ is a non-linear function of the unknown $PO_2$, and (b) the advective washout involves the Hill equation (also non-linear in $PO_2$). Newton's method would converge faster (quadratically vs. linearly) but requires computing the Jacobian of these non-linear terms at each iteration — which is expensive and complicated for the Hill equation. Picard iteration simply "freezes" the non-linear terms at the previous iteration's solution, solves the resulting linear system, and updates. It is simpler, more robust, and for the mild non-linearities encountered here, converges in 10–30 iterations.

The non-linear steady-state perfusion system is solved using **Picard (fixed-point) iteration**.

**Numerical Stabilization (Pseudo-Washout Trick):**

> **Numerical Rationale: Why Is This Trick Necessary?**
> Without the pseudo-washout, the LHS matrix $\mathbf{A}$ is pure diffusion — its rows sum to zero (Neumann boundary conditions). This means $\mathbf{A}$ is singular or nearly singular, and the CG solver either fails outright or converges extremely slowly. Meanwhile, the actual "sink" in the system (advective washout + metabolism) lives entirely on the RHS, where it depends non-linearly on the unknown $PO_2$. This creates oscillatory instability: a large $PO_2$ guess produces a large washout sink on the RHS, which causes the next iterate to overshoot to low $PO_2$, which produces a small washout, causing overshoot in the other direction.
>
> The trick: take a *linearized portion* of the washout (the $\gamma \cdot q \cdot PO_2$ term) and move it to the LHS diagonal. Since the same term is added to both sides, the true solution is unchanged. But now the LHS matrix has positive diagonal entries everywhere that flow exists, making it strictly diagonally dominant and perfectly conditioned for CG. The parameter $\gamma$ controls the "aggressiveness" of this linearization — higher values produce more stable but slower convergence.

The pure diffusion matrix $\mathbf{A}$ has rows summing to zero (Neumann BCs). The non-linear advective washout acts as a sink on the RHS, which is highly unstable for CG. The pipeline adds a linear pseudo-washout to the LHS diagonal, and the exact same term to the RHS. The true steady-state roots remain identical, but the LHS matrix becomes strictly diagonally dominant:

$$\mathbf{A}_{\text{stable}} = \mathbf{A} + \text{diag}(\gamma \cdot \mathbf{q}_{\text{total}})$$

where:
- $\gamma = 0.5$ (for the simple ADR solver) or $\gamma = 1.0$ (for the coupled 1D-3D and multi-species solvers),
- $\mathbf{q}_{\text{total}}$ is the total bulk flow through each voxel ($\mu m^3/s$).

For the endothelial barrier models, the pseudo-washout is based on permeability:
$$\text{pseudo\_washout}_i = P_{O_2} \cdot A_{\text{surface},i} \cdot \alpha_{O_2} \cdot \gamma$$

**Complete Picard Loop (Simple ADR Solver):**

```
PO2 ← zeros(N)  [initial guess: 0 mmHg everywhere]

A_stable ← A + diag(pseudo_washout)  [add regularizer + pseudo-washout]

ILU ← spilu(A_stable, drop_tol=1e-4, fill_factor=10)
M_pre ← LinearOperator from ILU
```
> **What happens here:** Before entering the iteration loop, we pre-compute an Incomplete LU (ILU) factorization of the stable matrix. This serves as a **preconditioner** for the CG solver — it provides an approximate inverse of $\mathbf{A}_{\text{stable}}$ that dramatically accelerates convergence. Think of it as giving CG a "good starting guess" for each inner solve.

```
For iteration = 0, 1, ..., max_iter-1:
  PO2_clamped ← max(PO2, 0.0)
```
> **What happens here:** Negative $PO_2$ values are clamped to zero. Physically, oxygen partial pressure cannot be negative. Numerically, negative values can arise from oscillations in early iterations and would cause the exponential in the metabolic function to blow up.

```
  # 1. Metabolic Sink
  M_red ← M_max × (1 − exp(−k × PO2_clamped))
```
> **What happens here:** We compute how fast each tissue cell is consuming oxygen at the current $PO_2$. Hypoxic cells (low $PO_2$) consume less; well-oxygenated cells consume at the maximum rate.

```
  # 2. Advective Washout (non-linear)
  For each voxel i with q_total[i] > 0:
    c_venous[i] ← C_O2(PO2_clamped[i], H_baseline)  [Hill equation]
    s_washout[i] ← q_total[i] × c_venous[i]
```
> **What happens here:** Blood leaving each tissue cell carries away oxygen. The amount carried away depends on the local tissue $PO_2$ through the Hill equation — this is the primary source of non-linearity in the system.

```
  # 3. RHS Construction
  b ← s_incoming − s_washout − (M_red × V_cell) + (pseudo_washout × PO2_clamped)
```
> **What happens here:** We assemble the right-hand side: oxygen delivered by arteries, minus oxygen carried away by veins, minus oxygen consumed by metabolism, plus the pseudo-washout correction term that balances the corresponding LHS addition.

```
  # 4. Solve Linear System
  PO2_new, info ← CG(A_stable, b, M=M_pre, x0=PO2, rtol=1e-6, maxiter=1000)
  
  # 5. Physical Clamping
  PO2_new ← max(PO2_new, 0.0)
  
  # 6. Convergence Check (Relative L2-norm)
  diff ← ||PO2_new − PO2||₂ / (||PO2_new||₂ + 1e-12)
  If diff < tolerance: BREAK
  
  PO2 ← PO2_new
```
> **What happens here:** The linear system is solved by preconditioned CG, negative values are clamped, and we check whether the solution has stabilised. If the relative change is below the Picard convergence tolerance ($10^{-5}$ for simple ADR, $10^{-4}$ for multi-species), the system has converged and we stop.

**Multi-Species Picard Loop:**

```
For iteration = 0, 1, ..., max_iter-1:
  PO2_clamped ← max(PO2_tissue, 0.0)
  PCO2_clamped ← max(PCO2_tissue, 0.0)
  
  # Coupled Metabolism
  M_o2 ← M_max × (1 − exp(−k × PO2_clamped))
  M_co2 ← M_o2 × RQ
```
> **What happens here:** Metabolism consumes $O_2$ and produces $CO_2$ at a fixed ratio (RQ = 0.82). The $CO_2$ production rate is directly proportional to $O_2$ consumption.

```
  # Henderson-Hasselbalch
  pH_tissue ← calculate_ph_from_pco2(PCO2_clamped, hco3_tissue)
```
> **What happens here:** The local $pH$ is updated from the current $PCO_2$ field. This $pH$ feeds back into the Bohr shift in the 1D blood tracking step, closing the feedback loop.

```
  # 1D Blood Tracking with Bohr/Haldane Coupling (see §6.4)
  [... compute transmural_o2, transmural_co2 ...]
  
  # RHS Construction
  b_o2  ← transmural_o2 − (M_o2 × V_cell) + (pseudo_washout_o2 × PO2_clamped)
  b_co2 ← transmural_co2 + (M_co2 × V_cell) + (pseudo_washout_co2 × PCO2_clamped)
```
> **What happens here:** Note the sign difference: for $O_2$, metabolism is a *sink* (subtracted). For $CO_2$, metabolism is a *source* (added).

```
  # Solve O2 and CO2 independently
  PO2_new ← CG(A_o2, b_o2, ...)
  PCO2_new ← CG(A_co2, b_co2, ...)
  
  # Convergence (both species must converge)
  diff_o2  ← ||PO2_new − PO2_tissue||₂ / (||PO2_new||₂ + 1e-12)
  diff_co2 ← ||PCO2_new − PCO2_tissue||₂ / (||PCO2_new||₂ + 1e-12)
  If diff_o2 < tolerance AND diff_co2 < tolerance: BREAK
```
> **What happens here:** Both the $O_2$ and $CO_2$ fields must independently converge before the solver declares success. This ensures the coupled Bohr–Haldane system has reached a self-consistent steady state.

> **Default parameters:**
> - Maximum Picard iterations: 50
> - Convergence tolerance: $10^{-4}$ (relative $L^2$-norm) for multi-species; $10^{-5}$ for simple ADR
> - ILU drop tolerance: $10^{-4}$, fill factor: 10
> - CG tolerance: $10^{-6}$ (simple) or $10^{-5}$ (multi-species), max 1000 or 500 iterations

> **Verification:** The simple-ADR Picard loop is bounds-tested in `test_perfusion_solver_zero_flow()` and `test_perfusion_solver_no_metabolism()`, and benchmarked against the exact Fick-principle root in `test_analytical_0d_fick_principle_mass_balance()` (all `tests/test_haemodynamics_perfusion.py`). The multi-species Picard loop is benchmarked in `test_multi_species_0d_fick_mass_balance()` (§6.4). No test isolates the ILU-preconditioning step or the pseudo-washout stabilisation trick on its own — both are only verified transitively through the convergence of the tests above.

### 7.3 Linear Solver Strategy

> **Numerical Rationale: Direct vs. Iterative**
> Direct solvers (like LU factorization via `spsolve`) compute the exact solution in a single pass but consume memory proportional to the number of non-zero fill-in elements — for large 3D grids, this can be prohibitive. Iterative solvers (like CG) use only matrix-vector products, requiring memory proportional to the number of non-zeros in the original matrix, but converge only approximately over many iterations. The pipeline automatically selects the appropriate solver based on the problem size.

The pipeline selects between direct and iterative solvers based on system size (implemented in `_solve_system_smart()`):

| System Size | Solver | Details |
|---|---|---|
| < 50,000 unknowns | **Direct** (SciPy `spsolve`) | Exact solution; fast for small/medium networks. |
| ≥ 50,000 unknowns | **Iterative** (CG with ILU preconditioner) | Memory-efficient for massive networks; tolerance $10^{-8}$, max 1000 iterations. |

**Fallback Chain:**
1. Direct solver (`spsolve`) — if it raises an exception (singular matrix):
2. LSQR (least-squares) — `splinalg.lsqr(A, b)[0]`

For iterative solving:
1. ILU preconditioning (`spilu` with `drop_tol=1e-4, fill_factor=10`) + CG — if info ≠ 0 (didn't converge):
2. LSQR fallback
3. If ILU itself fails (Exception): LSQR fallback directly.

> **Numerical Rationale:** The LSQR fallback is a least-squares solver that can handle rank-deficient or poorly conditioned systems. It won't give the exact pressure solution but will find the best approximation in the least-squares sense — a graceful degradation rather than a crash.

> **Verification:** `test_solve_system_smart_routing()` (`tests/test_haemodynamics.py`) confirms both the direct (`spsolve`) and iterative (CG) branches return the identical correct solution for the same well-conditioned system, by forcing `iterative_threshold` above and below the problem size. `test_solve_system_smart_singular_fallback()` confirms a singular matrix does not crash either branch. `test_solve_system_smart_preconditioner_failure()` mocks `spilu` to raise, and confirms the solver falls all the way through to the LSQR fallback and still returns the correct answer.

---

## 8. Boundary Conditions

### 8.1 Pressure Boundary Conditions (Dirichlet)

| Boundary | Default Value | Physical Basis |
|---|---|---|
| **Inlet pressure** ($P_{\text{in}}$) | 13.332 × 10⁶ mPa (= 100 mmHg) | Mean arterial pressure (MAP) — the average pressure in the systemic arterial circulation. |
| **Outlet pressure** ($P_{\text{out}}$) | 0.27 × 10⁶ mPa (= 2 mmHg) | Central venous pressure (CVP) — the pressure in the systemic venous circulation near the right atrium. |

> **Physiological Context:** MAP (100 mmHg) and CVP (2 mmHg) are the two endpoints of the systemic circulation. In reality, most of the pressure drop from MAP to CVP occurs in the arterioles (the "resistance vessels"), and the actual pressure at the entrance to a micro-organ like the carotid body would be substantially lower than full MAP. This simplification means the model likely overestimates the perfusion pressure and therefore the flow through the network. More accurate boundary conditions would require knowledge of the upstream arterial tree, which is typically not available from organ-level imaging.

> **Assumption**: The pressure drop from 100 mmHg (MAP) to 2 mmHg (CVP) across a micro-organ is a significant simplification. In reality, the carotid body is perfused at high flow rates relative to its mass, and the upstream resistance of the feeding artery and downstream venous drainage significantly modulate the actual pressures at the organ boundary. The effective perfusion pressure across the carotid body is likely substantially less than the full MAP–CVP gradient.

> **Verification:** The Dirichlet pressure solve itself (fixed inlet/outlet pressures, solved via the Laplacian) is the mechanism benchmarked in `test_analytical_poiseuille_series()` and `test_analytical_poiseuille_parallel()` (§2.2). No test uses the specific MAP/CVP default values (100/2 mmHg) — all analytical tests use arbitrary pressure pairs to isolate the resistance-network mathematics from the specific physiological boundary values.

### 8.2 Inlet/Outlet Node Selection

Boundary nodes are auto-selected by finding **dead-end nodes** (degree-1) located within a configurable percentage band at the spatial extremes of the image volume along a specified axis.

| Parameter | Default | Description |
|---|---|---|
| `edge_percent` | 25% | Nodes in the top 25% of the chosen axis are candidates for inlet (starting) nodes. |
| `end_percent` | 25% | Nodes in the bottom 25% are candidates for outlet nodes. |
| `node_edge_axis` | 0 (Z-axis) | The spatial axis along which the network is oriented for boundary selection. |

> **Physical Intuition:** Dead-end nodes (degree-1) represent vessels that were "cut" by the imaging volume boundary. In the intact organism, these vessels connect to upstream arteries or downstream veins outside the field of view. The pipeline assigns them as inlets or outlets based on their spatial position (arterial supply typically enters from one side and venous drainage exits from the other).

> **Verification:** `test_boundary_mode_universal_sink()` (`tests/test_haemodynamics.py`) checks the exact default parameters described here ($edge\_percent=25\%$, $end\_percent=25\%$, axis 0) on a synthetic 4-node graph, confirming both Z-axis dead-ends are selected as inlet/outlet and — specific to `universal_sink` mode — that a Y-edge dead-end is also swept up as an outlet. The related but distinct manual-selection function `select_boundary_nodes_by_method()` (coordinate-based, volume-based, and degree-1-from-starting-node methods) is separately unit-tested in `tests/test_graph.py`.

### 8.3 Boundary Permeability Modes

| Mode | Description |
|---|---|
| `caged` (default) | Only the Z-axis boundaries allow vessels to enter/exit. X and Y boundaries are sealed. Virtual padding (10 voxels) is applied to the Z faces only. |
| `universal_sink` | All six faces are permeable. Dead-ends at any boundary face can be assigned as outlets. Virtual padding (10 voxels) is applied to all faces. |
| `robin_resistance` | Dead-end capillaries at boundaries are connected to a virtual "Robin Ghost Node" with a resistance equal to `robin_distal_resistance_multiplier` × average resistance of connected edges (default: 10×). This simulates flow bleeding out through severed capillaries. |

> **Physical Intuition: Robin Ghost Node**
> In reality, capillaries that appear to be "dead ends" in the image are actually connected to downstream vasculature outside the imaging field of view. Simply capping them (zero flow) is unrealistic — it creates an artificial flow bottleneck. The Robin Ghost Node provides a more physiological alternative: it connects each cut capillary to a virtual downstream vascular bed with a configurable resistance. The 10× multiplier means the "escape" resistance is high (the severed capillary can't easily dump all its blood), which prevents the ghost node from dominating the flow pattern while still allowing realistic drainage.

**Robin Ghost Node Implementation (from `build_conductance_matrix_from_graph()`):**

For each node tagged with `is_robin_boundary=True`:
1. Compute the average resistance of all edges connected to this node: $R_{\text{avg}} = \frac{1}{N} \sum R_i$
2. Compute the ghost resistance: $R_{\text{ghost}} = R_{\text{avg}} \times \text{robin\_multiplier}$ (default multiplier: 10.0)
3. Compute the ghost conductance: $C_{\text{ghost}} = 1/R_{\text{ghost}}$
4. Add symmetric off-diagonal entries connecting the boundary node to the ghost node.
5. The ghost node is added to the `output_nodes` list and receives $P_{\text{out}}$ as its Dirichlet BC.

> **Verification:** `universal_sink` mode is checked in `test_boundary_mode_universal_sink()` above. `robin_resistance` mode is checked exactly in `test_robin_matrix_ghost_node_generation()` (`tests/test_haemodynamics.py`): a 3-node graph with one Robin-tagged boundary node produces a 4×4 matrix (including the ghost node), and the ghost conductance is confirmed to be exactly $1/(2.0\times10.0)=0.05$. Kirchhoff current conservation between `robin_resistance` and `universal_sink` behaviour is checked in `test_robin_vs_sink_flow_conservation()`: flow into a hub node is confirmed to equal flow out (direct edge flow plus ghost-node flow) to `atol=1e-8`. The default `caged` mode has no dedicated test — a coverage gap (§11.3).

### 8.4 Blood Gas Boundary Conditions

| Parameter | Default | Units | Description |
|---|---|---|---|
| Arterial $PO_2$ | 100.0 | mmHg | Oxygen partial pressure in arterial blood entering the network. |
| Arterial $PCO_2$ | 40.0 | mmHg | Carbon dioxide partial pressure in arterial blood. |
| Systemic hematocrit | 0.45 | dimensionless | Volume fraction of red blood cells in systemic blood. |
| Tissue bicarbonate | 24.0 | mmol/L | $[HCO_3^-]$ buffer concentration for Henderson-Hasselbalch pH calculation. |

> **Physiological Context:** These are normal adult human resting values. Arterial $PO_2 = 100$ mmHg corresponds to normal lungs at sea level. $PCO_2 = 40$ mmHg is the normal arterial carbon dioxide tension. $H_D = 0.45$ is the midpoint of the normal hematocrit range (42–50% for males). To model hypoxemia (e.g., high altitude, lung disease), the user would lower $PO_2$; to model anaemia, lower $H_D$; to model hypercapnia, raise $PCO_2$.

> **Verification:** These default values (arterial $PO_2$ 100 mmHg, $PCO_2$ 40 mmHg, $H_D$ 0.45) are the exact inputs used throughout the analytical benchmarks in §5–§6 (e.g. `test_multi_species_0d_fick_mass_balance()`, `test_henderson_hasselbalch_equilibrium()`), so their internal consistency is continuously exercised, but no test asserts that these specific numbers are the correct physiological defaults — that is a modelling choice, not a testable claim.

---

## 9. Complete Parameter Reference Table

### 9.1 Haemodynamic Parameters

| Parameter | Symbol | Default Value | Units | Source/Basis |
|---|---|---|---|---|
| Plasma viscosity | $\mu_{\text{plasma}}$ | 1.2 | mPa·s (cP) | Standard value for human blood plasma at 37°C |
| Systemic hematocrit | $H_D$ | 0.45 | dimensionless | Normal adult hematocrit |
| Inlet pressure (MAP) | $P_{\text{in}}$ | 13.332 × 10⁶ | mPa | 100 mmHg |
| Outlet pressure (CVP) | $P_{\text{out}}$ | 0.27 × 10⁶ | mPa | 2 mmHg |
| Minimum vessel diameter | $D_{\text{min}}$ | 3.0 | $\mu m$ | RBC minimum traversal diameter |
| Maximum hematocrit clamp | — | 0.95 | dimensionless | Physical upper bound |
| Skimming threshold | $x_0$ | 0.05 | dimensionless | Pries & Secomb empirical |
| Skimming asymmetry constant | — | −13.29 | dimensionless | Pries & Secomb empirical |
| Skimming steepness constant | — | 6.98 | dimensionless | Pries & Secomb empirical |
| Cell-free layer width | — | 1.1 | $\mu m$ | Pries & Secomb empirical |
| Power-law viscosity exponent | — | 1.647 | dimensionless | Heuristic fit |
| Rheology max iterations | — | 15 | — | Convergence criterion |
| Rheology tolerance | — | $10^{-4}$ | — | Max absolute flow change |

### 9.2 Constriction Parameters

| Parameter | Default | Units | Description |
|---|---|---|---|
| Constriction mode | `sphincter` | — | Single proximal constriction per vessel |
| Sphincter length | 5.0 | $\mu m$ | Physical length of the constriction zone |
| Intimal cushion ratio | 0.60 | dimensionless | $d_2/d_1$ at carotid origin (B01) |
| Pre-capillary sphincter ratio | 0.50 | dimensionless | $d_2/d_1$ at capillary bed transition |
| Pre-capillary topological offset | 1 | branch orders | Shift from midpoint |
| Minimum constriction ratio | 0.01 | dimensionless | Clamped floor to prevent singularities |
| Integration sample points | 1000 | — | For trapezoidal resistance integration |
| Periodic constriction spacing | 100.0 | $\mu m$ | Interval between repeated constrictions |
| Periodic ramp length | 10.0 | $\mu m$ | Length of ramp-up/ramp-down in periodic mode |
| Periodic hold length | 20.0 | $\mu m$ | Length of constricted hold in periodic mode |

### 9.3 Perfusion / Gas Transport Parameters

| Parameter | Symbol | Default Value | Units | Source/Basis |
|---|---|---|---|---|
| $O_2$ tissue diffusion coefficient | $\sigma_{\text{diff}}$ | $1.5 \times 10^{-9}$ | $m^2/s$ | Typical mammalian tissue |
| $CO_2$ tissue diffusion coefficient | $\sigma_{\text{diff,CO2}}$ | $3.0 \times 10^{-8}$ | $m^2/s$ | ~20× $O_2$ diffusivity |
| $O_2$ endothelial permeability | $P_{O_2}$ | $1.0 \times 10^{-4}$ | cm/s | Estimated from CellML models |
| $CO_2$ endothelial permeability | $P_{CO_2}$ | $2.0 \times 10^{-3}$ | cm/s | Estimated from CellML models |
| Hill coefficient | $n$ | 2.7 | dimensionless | Adult human haemoglobin |
| $P_{50}$ (baseline) | $P_{50}$ | 26.0 | mmHg | At pH 7.4, $PCO_2$ 40 mmHg |
| $O_2$ plasma solubility | $\alpha_{O_2}$ | $1.34 \times 10^{-3}$ | mmol/L per mmHg | Henry's Law |
| $CO_2$ plasma solubility | $\alpha_{CO_2}$ | 0.03 | mmol/L per mmHg | Henry's Law |
| Max haemoglobin O₂ capacity | $C_{Hb,\text{max}}$ | $0.446 \times 20.4 / 0.45$ | mmol/L | Scaled to pure RBC |
| Base CO₂ capacity coefficient | — | 11.02 | — | Spencer (1979) empirical |
| Base CO₂ capacity exponent | — | 0.396 | — | Spencer (1979) empirical |
| Haldane coefficient (deoxy) | — | 0.15 | — | Spencer (1979) empirical |
| Haldane coefficient (oxy) | — | 0.05 | — | Spencer (1979) empirical |
| Bohr shift coefficient (pH) | — | −0.4 | per pH unit | Kelman (1966) / Severinghaus (1979) |
| Bohr shift coefficient ($PCO_2$) | — | 0.06 | per log unit | Kelman (1966) / Severinghaus (1979) |
| Henderson-Hasselbalch $pK_a$ | $pK_a$ | 6.1 | dimensionless | Carbonic acid |
| Max metabolic rate | $M_{\text{max}}$ | 0.005 | mmol/L/s | Phenomenological |
| Metabolic reduction constant | $k$ | 0.1 | per mmol | Phenomenological |
| Respiratory quotient | $RQ$ | 0.82 | dimensionless | Mixed substrate metabolism |
| Tissue bicarbonate | $[HCO_3^-]$ | 24.0 | mmol/L | Normal plasma |
| Grid resolution | — | $10 \times 10 \times 10$ | $\mu m$ | User-configurable |
| Picard max iterations | — | 50 | — | Convergence criterion |
| Picard tolerance | — | $10^{-4}$ | — | Relative $L^2$-norm |
| Brent's root bracket | — | $[0.0, 150.0]$ | mmHg | Search range for Hill inversion |
| Pseudo-washout $\gamma$ (simple) | — | 0.5 | dimensionless | Damping factor |
| Pseudo-washout $\gamma$ (coupled) | — | 1.0 | dimensionless | Damping factor |
| Grid regularization | — | $10^{-12}$ | — | Tiny diagonal sink for CG stability |
| Arterial O₂ content ($C_{\text{arterial}}$) | — | Computed from $PO_{2,\text{arterial}}$ and $H_D$ | mmol/L | Used as source term in simple ADR solver |

### 9.4 Solver Parameters

| Parameter | Default | Context |
|---|---|---|
| Direct/Iterative threshold | 50,000 unknowns | `_solve_system_smart()` |
| CG tolerance (flow solve) | $10^{-8}$ | Large network flow solver |
| CG max iterations (flow) | 1000 | Large network flow solver |
| CG tolerance (perfusion simple) | $10^{-6}$ | Simple ADR Picard inner loop |
| CG tolerance (perfusion multi) | $10^{-5}$ | Multi-species Picard inner loop |
| CG max iterations (perfusion) | 1000 (simple), 500 (multi) | Picard inner loop |
| ILU drop tolerance | $10^{-4}$ | Preconditioning |
| ILU fill factor | 10 | Preconditioning |
| Matrix regularization (diffusion grid) | $10^{-12}$ | Added to diffusion matrix diagonal for CG stability (§7.1) |
| Matrix regularization (perfusion `A_reg`) | $10^{-6}$ | Separate regularization added to the perfusion matrix diagonal |

---

## 10. Summary of Key Assumptions for Literature Comparison

The following is a consolidated list of the most significant physiological and modelling assumptions, each suitable for targeted critique against published literature:

1. **Rigid vessel walls**: No vascular compliance, distensibility, or autoregulation.
2. **Steady-state flow**: No cardiac pulsatility, no time-varying boundary conditions.
3. **Newtonian fluid (initial)**: Blood treated as Newtonian in the Poiseuille formula; partially corrected by Pries–Secomb empirical viscosity.
4. **Circular cross-section**: All vessels assumed perfectly cylindrical.
5. **No-slip condition**: Assumed at the vessel wall (standard for Poiseuille flow).
6. **Empirical rheology from rat mesentery**: Pries–Secomb correlations applied to carotid body vasculature without organ-specific validation.
7. **Phase separation only at bifurcations**: Trifurcations and higher use simple mixing.
8. **Exponential diameter scaling**: Murray's Law is not used; a heuristic 3-point exponential fit is used instead.
9. **Full MAP-to-CVP pressure drop**: The entire systemic pressure gradient (100 mmHg → 2 mmHg) is applied across the micro-organ, without accounting for upstream arterial resistance or downstream venous back-pressure.
10. **Plug-flow approximation**: No radial concentration gradients within vessel lumens.
11. **Constant bicarbonate buffer**: $[HCO_3^-]$ is fixed at 24 mmol/L; no renal regulation.
12. **Phenomenological metabolism**: Saturating exponential rather than Michaelis–Menten kinetics.
13. **Homogeneous tissue**: Uniform diffusion coefficient and metabolic rate throughout the tissue grid; no cell-type heterogeneity.
14. **Neumann boundary on tissue**: Zero-flux at tissue grid edges — no oxygen exchange with surrounding tissue outside the imaged volume.
15. **Human haemoglobin parameters**: Hill coefficient ($n = 2.7$) and $P_{50}$ are set for adult human Hb; species-specific adjustments are not made.
16. **No lymphatic drainage or interstitial fluid flow**: Only vascular perfusion is modelled.
17. **No active vasoregulation**: Constriction ratios are static — no myogenic response, no metabolic feedback, no shear-dependent vasodilation.
18. **Simplified Haldane approximation**: O₂ saturation for the Haldane shift uses a fixed $P_{50} = 26.0$ mmHg without Bohr feedback, creating a minor inconsistency in the coupled Bohr-Haldane system.
19. **Point-sampling vessel mapping**: Vessels are mapped to grid cells by point-sampling centreline voxels rather than exact line-plane intersection.
20. **Single baseline hematocrit for washout**: The simple ADR solver uses a fixed $H_D = 0.45$ for venous washout calculations rather than per-voxel flow-weighted hematocrit.

---

## 11. Verification and Testing

> **Purpose:** Every preceding section describes what a capability *computes*. This section describes how the pipeline confirms each capability *behaves correctly* — the automated test suite that checks the mathematics against closed-form solutions, conservation laws, and known synthetic targets. Verification claims are anchored to specific test functions so they can be checked against the source directly; §11.3 states plainly where verification is thin or absent, rather than leaving the reader to assume uniform coverage.

### 11.0 Test Suite Overview

The pipeline's mathematical and computational claims are checked by an automated `pytest` suite of 38 modules and 210 test functions under `tests/`, run via `pytest -s` from the repository root (see `README.md`). Configuration lives in `pyproject.toml`'s `[tool.pytest.ini_options]`: `testpaths = ["tests"]`, and three custom markers — `slow`, `plotting`, and `integration` — allow selective runs (e.g. `pytest -m "not slow"`). Continuous integration (`.github/workflows/pytest-pr.yml`) runs the full suite on every pull request against Python 3.10.

### 11.1 Verification Methodologies

Six distinct verification strategies recur across the suite:

1. **Analytical closed-form comparison** — the flagship methodology, concentrated in `tests/test_haemodynamics_analytical.py` (14 tests) and the Fick-principle tests in `tests/test_haemodynamics_perfusion.py`. A numerical solver's output is compared to an independently derived exact mathematical solution (e.g. Poiseuille series/parallel resistance, §2.2; the parabolic reaction-diffusion profile, §5.2; the Henderson–Hasselbalch pH equilibrium, §5.6).
2. **Conservation / invariant checks** — mass and flux balance are asserted exactly rather than compared to a target value: RBC flux conservation across bifurcations (§3.3), Kirchhoff current conservation at network hubs (§8.3).
3. **Synthetic phantoms with analytically-known targets** — dedicated builder functions across the suite construct volumes, graphs, and masks with a mathematically prescribed correct answer (e.g. a Gaussian-intensity cylinder with a known FWHM, §4.2).
4. **Equivalence oracles** — two independent code paths are checked for agreement rather than against a formula. (Most instances — Dask vs NumPy, mocked vs real PyVista — sit in preprocessing/export modules outside this document's scope.)
5. **Graceful-degradation / negative testing** — deliberately pathological inputs (singular matrices, DAG cycles, missing boundary conditions) are checked to fail safely rather than crash or hang (§3.4, §7.3).
6. **Physical-bounds testing** — extreme configurations are checked to respect known physical limits (e.g. zero flow must yield exactly zero concentration, §6.1).

### 11.2 Coverage Matrix

| § | Capability | Verifying test(s) | Oracle type | Tolerance |
|---|---|---|---|---|
| 2.1 | Hagen–Poiseuille resistance formula | `test_set_poiseuille_resistances_prefers_fwhm_optional`, `test_measure_edge_diameters_fwhm_from_raw_tiff_cylinder` | Formula recomputation | `atol=1e-6`; <5% |
| 2.2 | Graph Laplacian network solve | `test_analytical_poiseuille_series`, `test_analytical_poiseuille_parallel`, `test_build_conductance_matrix_from_graph`, `test_calc_laplacian_from_conductance_matrix` | Closed-form + structural | `atol=1e-10` |
| 2.3 | Two-point effective resistance | `test_calc_two_point_from_laplacian_matrix_nodeID` | Structural only ($R>0$) | — |
| 3.1 | Pries–Secomb viscosity | `test_rheology_fahraeus_lindqvist_curve` | Curve-shape (inequalities) | — |
| 3.2 | Power-law initial viscosity | `test_calculate_viscosity` | Exact value + inequality | — |
| 3.3 | Plasma skimming | `test_rheology_hematocrit_mass_conservation`, `test_rheology_plasma_skimming_effect` | Mass conservation + direction | `atol=1e-8` |
| 3.4 | Coupled flow–Hct–viscosity Picard | `test_coupled_solver_convergence`, `_dag_cycle_handling`, `_matrix_singularity_safety` | Convergence + directional + graceful degradation | `atol=1e-5` (Hct) |
| 3.5 | Resistance rescaling | *(indirect only)* | — | — |
| 3.6 | Wall shear stress | `test_analytical_wall_shear_stress` | Closed form | `atol=1e-10` |
| 4.1 | Diameter assignment modes | `test_poiseuille_edt_mode`, `test_set_poiseuille_resistances_prefers_fwhm_optional` | Mode precedence | exact |
| 4.2 | FWHM diameter measurement | `test_fwhm_from_profile_gaussian_fit`, `test_measure_edge_diameters_fwhm_from_raw_tiff_cylinder`, 3 phantom tests in `test_integration_synthetic_vessel_fwhm.py` | Analytical Gaussian target | <0.2–0.35 µm; scaled tolerance |
| 4.3 | Branch-order diameter scaling | *(no direct test)* | — | — |
| 4.4.1 | Sphincter constriction profile | `test_get_diameter_at_position` | Bounds check | — |
| 4.4.2 | Periodic constriction profile | `test_analytical_sphincter_resistance_calculus` | Closed-form integral | `rtol=1e-3` |
| 4.4.3 | Constriction ratios | 2 tests in `test_pericyte_mask_integration.py` | Directional (before/after) | — |
| 4.4.4 | Integrated variable-diameter resistance | `test_analytical_sphincter_resistance_calculus`, `test_calculate_integrated_resistance` | Closed-form integral | `rtol=1e-3` |
| 5.1 | Perfusion grid + vessel mapping | `test_perfusion_grid_dimensions`, `_grid_index_bidirectional_mapping`, `_grid_out_of_bounds_handling`, `_map_vessels_to_grid_straight_line`, `_advective_source_hematocrit_weighting` | Exact geometry + bijective mapping | exact |
| 5.2 | ADR equation | `test_analytical_1d_pure_diffusion`, `test_analytical_zero_order_metabolism`, `test_analytical_radial_point_source` | Closed form (linear/parabolic); qualitative ($1/r$) | `atol=1e-10`; loose bracket |
| 5.3 | Hill equation O₂ content | `test_hill_equation_sigmoidal_curve` | $P_{50}$ exact + shape | `atol=1e-5` |
| 5.4 | Bohr effect | `test_bohr_haldane_atomic_curves` | Direction only | — |
| 5.5 | Haldane effect | `test_bohr_haldane_atomic_curves` | Direction only | — |
| 5.6 | Henderson–Hasselbalch pH | `test_henderson_hasselbalch_equilibrium` | Closed form (2 anchor points) | `atol=1e-2` |
| 5.7 | Metabolic consumption | `test_analytical_zero_order_metabolism` (zero-order limit only) | Closed form | `atol=1e-10` |
| 5.8 | Respiratory quotient | `test_multi_species_0d_fick_mass_balance` (indirect) | Closed form (coupled) | `atol=1e-2` |
| 6.1 | Tier 1 ADR solver | `test_perfusion_solver_zero_flow`, `_no_metabolism`, `test_analytical_0d_fick_principle_mass_balance` | Physical bounds + closed form | `atol=1e-10`; `atol=1e-2` |
| 6.2 | Tier 2 coupled 1D–3D | `test_analytical_transmural_exponential_decay` | **Structural only — see caveat** | — |
| 6.3 | Tier 3 multi-species | `test_multi_species_0d_fick_mass_balance` | Closed form (coupled root) | `atol=1e-2`, `atol=1e-3` (pH) |
| 6.4 | 1D blood gas tracking / Brent inversion | *(no direct unit test)* | — | — |
| 7.1 | 7-point stencil diffusion matrix | `test_build_adr_matrix_structure`, `test_advective_source_vector` | Structural (exact nnz count) | exact |
| 7.2 | Picard iteration | `test_perfusion_solver_zero_flow`, `test_multi_species_0d_fick_mass_balance` | Physical bounds + closed form | as above |
| 7.3 | `_solve_system_smart` fallback ladder | `test_solve_system_smart_routing`, `_singular_fallback`, `_preconditioner_failure` | Dispatch + graceful degradation | exact |
| 8.1 | Pressure Dirichlet BCs | `test_analytical_poiseuille_series`/`_parallel` (mechanism only) | Closed form | `atol=1e-10` |
| 8.2 | Inlet/outlet node selection | `test_boundary_mode_universal_sink`, 3× `test_select_boundary_nodes_by_method_*` | Selection correctness | exact |
| 8.3 | Boundary permeability modes | `test_boundary_mode_universal_sink`, `test_robin_matrix_ghost_node_generation`, `test_robin_vs_sink_flow_conservation` | Structural + Kirchhoff conservation | `atol=1e-8` |
| 8.4 | Blood gas BCs | *(indirect only, via §6.3)* | — | — |

### 11.3 Known Verification Gaps

Stated plainly, so the verification claims above are not read as broader than they are:

1. **Two tests compute an analytical target but never assert against it.** `test_krogh_cylinder_radial_diffusion()` (`tests/test_haemodynamics_analytical.py:393`) is docstring-billed as proving the solver "perfectly traces... Krogh's... analytical cylinder equation," but its only assertion is `len(po2_num) == grid.n_cells` — a length check, not a value check. `test_analytical_transmural_exponential_decay()` (`tests/test_haemodynamics_perfusion.py:276`, §6.2) computes `analytical_po2_out` but never compares the solver's output to it — the test instead forces a giant metabolic sink and checks only that tissue $PO_2$ stays near zero. Both tests verify that the solver runs without crashing on the given geometry ("structural completion"), not that it reproduces the named closed-form solution. `modelling_and_hypothesis_testing_documentation.md` §4.2 previously repeated the Krogh overstatement; it has been corrected to match.
2. **No mesh-refinement or order-of-accuracy study exists for any PDE solver.** Every ADR/diffusion test in §5.2, §6, and §7 runs at a single fixed grid resolution. Spatial discretisation error as a function of grid spacing is never measured.
3. **`brentq` Hill-equation inversion (§6.4) has no isolated unit test.** It is only reached transitively inside the Tier 2/3 solver tests.
4. **Three capabilities are covered only transitively, through integration-style tests, not directly:** resistance rescaling during rheology updates (§3.5), the branch-order exponential diameter scaling formula (§4.3), and the `caged` boundary permeability mode (§8.3 — the pipeline's *default* mode).
5. **No determinism/seeding test exists for stochastic pericyte placement** (`ImageLynx.haemodynamics.probability`, outside this document's scope but adjacent to §4.4).
6. **No `@pytest.mark.parametrize` use anywhere in the 210-test suite.** Near-duplicate test bodies (e.g. the near/far pericyte-distance pair in §4.4.3) stand in for parameter sweeps.
7. **Several capabilities are checked only directionally or qualitatively, not against an exact closed-form target:** plasma skimming (§3.3 — mass conservation and direction are exact, but the logit-sigmoid output value itself is not benchmarked), the Pries–Secomb viscosity curve (§3.1 — an inequality chain, not a specific literature value), and the Bohr/Haldane shifts (§5.4–5.5 — direction only).
