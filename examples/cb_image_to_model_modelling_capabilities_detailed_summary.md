# Physiological Modelling Documentation: ImageLynx `carotid_image_to_model.py` — Detailed Edition

> **Purpose**: This document catalogues **every** physical law, physiological assumption, empirical model, parameter value, numerical algorithm, and boundary condition implemented in the ImageLynx vascular modelling pipeline. It is intended for uploading to NotebookLM alongside published literature so that the notebook can systematically cross-reference and critique each modelling decision. Unlike the companion summary document, this version provides fully expanded equations, step-by-step algorithmic pseudo-code, and explicit variable definitions for every symbol used.

---

## 1. Scope and Domain

The pipeline takes 3D microscopy volumes of blood vessel networks (e.g., micro-CT or light-sheet images of the carotid body vasculature), extracts a mathematical graph of the vessel centrelines, and then solves for:

1. **Steady-state blood flow** (pressure, volumetric flow rate, velocity) using Poiseuille's Law.
2. **Spatially varying blood rheology** (viscosity and hematocrit distribution) using empirical in-vivo models.
3. **Tissue oxygen perfusion** ($PO_2$) by coupling the 1D vascular network to a 3D tissue diffusion grid.
4. **Multi-species gas transport** ($O_2$, $CO_2$, and $pH$) with Bohr–Haldane coupling.

The biological system under study is the **carotid body microvasculature** — a highly vascularised chemoreceptor organ supplied by branches of the external carotid artery.

---

## 2. Governing Equations

### 2.1 Hagen–Poiseuille Flow (Steady-State, Laminar, Incompressible)

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

### 2.2 Network Flow: Graph Laplacian System

The entire vascular network is modelled as a resistor network (analogous to Kirchhoff's circuit laws). Flow conservation at each interior node $i$ is:

$$\sum_{j \in \text{neighbours}(i)} \frac{P_i - P_j}{R_{ij}} = 0$$

This is assembled into a sparse **Graph Laplacian** matrix $\mathbf{L}$ derived from the **Conductance matrix** $\mathbf{C}$:

**Step 1 — Build the Conductance Matrix:**

For each edge $(i, j)$ with resistance $R_{ij}$, the conductance is:

$$C_{ij} = \frac{1}{R_{ij}}$$

The matrix $\mathbf{C}$ is symmetric with $C_{ij} = C_{ji}$. For multigraphs with parallel edges, conductances are summed.

**Step 2 — Compute the Graph Laplacian:**

$$\mathbf{L} = \text{diag}\left(\sum_j C_{ij}\right) - \mathbf{C}$$

This is implemented in `resistance.py` as:
```python
diag = np.array(C.sum(axis=1)).flatten()
L = sp.diags(diag) - C
```

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

> **Assumption**: Conservation of mass at every node — no leakage through vessel walls is modelled at this stage (perfusion leakage is handled separately in the tissue diffusion model, §5).

### 2.3 Effective Two-Point Resistance

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

---

## 3. Blood Rheology

### 3.1 Pries–Secomb In-Vivo Viscosity Model (Fåhræus–Lindqvist Effect)

Blood viscosity in microvessels is **not constant** — it varies dramatically with vessel diameter and local hematocrit. The pipeline implements the empirical Pries–Secomb (1992, 1994) model in the function `calculate_pries_secomb_viscosity()` in `rheology.py`.

**Step 1 — Diameter Clamping:**

To prevent mathematical singularities for vessels approaching the diameter of a single RBC:

$$D = \max(D_{\text{raw}}, 3.0 \, \mu m)$$

**Step 2 — Relative Apparent Viscosity at Reference Hematocrit ($H_D = 0.45$):**

$$\mu_{45} = 220 \, e^{-1.3 D} + 3.2 - 2.44 \, e^{-0.06 D^{0.645}}$$

where:
- $\mu_{45}$ is the relative apparent viscosity (dimensionless) at reference hematocrit $H_D = 0.45$,
- $D$ is vessel diameter in $\mu m$.

**Step 3 — Shape Parameter $C$ (Hematocrit Dependence):**

$$C = \left(0.8 + e^{-0.075 D}\right) \left(-1 + \frac{1}{1 + 10^{-11} D^{12}}\right) + \frac{1}{1 + 10^{-11} D^{12}}$$

**Step 4 — Relative Apparent Viscosity at Actual Hematocrit $H_D$:**

$$\mu_{\text{rel}} = 1 + (\mu_{45} - 1) \cdot \frac{(1 - H_D)^C - 1}{(1 - 0.45)^C - 1}$$

**Step 5 — In-Vivo Correction for Cell-Free (Glycocalyx) Layer:**

$$\mu_{\text{app}} = \mu_{\text{rel}} \cdot \left(\frac{D}{D - 1.1}\right)^2$$

**Step 6 — Final Apparent Viscosity in Physical Units:**

$$\mu = \mu_{\text{app}} \times \mu_{\text{plasma}}$$

> **Default parameter**: $\mu_{\text{plasma}} = 1.2$ mPa·s (cP).

> **Guard Rails:**
> - If $D \leq 0$ or $H_D \leq 0$, the function returns $\mu_{\text{plasma}}$ directly.
> - Maximum hematocrit is capped at 0.95 to prevent non-physical values.

> **Assumptions:**
> - The empirical correlations were derived from *in vivo* measurements in rat mesentery. Their direct applicability to the carotid body microvasculature (a glomus organ with unique perfusion characteristics) is assumed but not validated.
> - Minimum diameter cap: vessels smaller than 3.0 $\mu m$ are clamped to 3.0 $\mu m$ to avoid mathematical singularities (the cell-free-layer correction diverges at $D = 1.1 \, \mu m$).

### 3.2 Initial (Pre-Rheology) Viscosity Approximation

Before the iterative rheology solver runs, the pipeline uses a simpler **power-law viscosity** for the initial Poiseuille resistance calculation:

$$\mu_{\text{initial}} = \frac{1}{d^{1.647}}$$

This is a heuristic approximation to give smaller vessels higher viscosity. It is replaced by the full Pries–Secomb model during the coupled iteration (§3.4). This is implemented in `poiseuille.py`:

```python
@staticmethod
def calculate_viscosity(diameter: float) -> float:
    return 1.0 / (diameter ** 1.647)
```

### 3.3 Plasma Skimming (Phase Separation at Bifurcations)

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

**Step 2 — Flow Fractions:**

$$f_{Q_1} = \frac{Q_1}{Q_{\text{in}}}, \quad f_{Q_2} = \frac{Q_2}{Q_{\text{in}}}$$

**Step 3 — Edge-Case Handling:**
```
If f_Q1 < 1e-6:  Return (0.0, H_in × Q_in / Q_2)
If f_Q2 < 1e-6:  Return (H_in × Q_in / Q_1, 0.0)
```

**Step 4 — Skimming Threshold:**

$$x_0 = 0.05$$

Branches receiving less than 5% of flow get zero RBCs.

**Step 5 — Threshold-Based Bypass:**
```
If f_Q1 ≤ x_0:    f_E1 = 0.0
If f_Q1 ≥ 1 - x_0: f_E1 = 1.0
Otherwise → continue to Step 6.
```

**Step 6 — Asymmetry Parameter $A$:**

$$A = -13.29 \cdot \frac{d_1^2/d_2^2 - 1}{d_1^2/d_2^2 + 1} \cdot \frac{1 - H_{\text{in}}}{d_1}$$

**Step 7 — Steepness Parameter $B$:**

$$B = 1 + 6.98 \cdot \frac{1 - H_{\text{in}}}{d_1}$$

**Step 8 — Logit Transformation of Flow Fraction:**

$$\text{logit}(f_{Q_1}) = \ln\!\left(\frac{f_{Q_1} - x_0}{1 - f_{Q_1} - x_0}\right)$$

**Step 9 — Logit of RBC Flux Fraction:**

$$\text{logit}(f_{E_1}) = A + B \cdot \text{logit}(f_{Q_1})$$

**Step 10 — Sigmoid (Inverse Logit) to Recover $f_{E_1}$:**

$$f_{E_1} = \frac{1}{1 + e^{-\text{logit}(f_{E_1})}}$$

**Step 11 — Mass Conservation:**

$$f_{E_2} = 1 - f_{E_1}$$

**Step 12 — Daughter Hematocrits:**

$$H_1 = H_{\text{in}} \cdot \frac{f_{E_1}}{f_{Q_1}}, \quad H_2 = H_{\text{in}} \cdot \frac{f_{E_2}}{f_{Q_2}}$$

**Step 13 — Physical Bounds Clamping:**

$$H_1 = \min(\max(H_1, 0.0), 0.95), \quad H_2 = \min(\max(H_2, 0.0), 0.95)$$

> **Assumptions:**
> - The phase separation model is only applied at **binary bifurcations** (degree-2 splits). For trifurcations and higher, RBCs are distributed proportionally to flow (simple mixing: each daughter receives $H_{\text{mix}}$).
> - The empirical constants (−13.29, 6.98, $x_0 = 0.05$) were derived from glass tube experiments and *in vivo* rat cremaster observations.

### 3.4 Coupled Flow–Hematocrit–Viscosity Iteration

The full non-linear coupling between flow, hematocrit distribution, and viscosity is solved iteratively (Picard-style fixed-point iteration) in `solve_coupled_flow_and_hematocrit()` in `rheology.py`.

**Complete Algorithm:**

```
INITIALIZE:
  For each edge (u,v,k):
    hematocrit ← systemic_hematocrit (0.45)
    diameter ← assigned_diameter_um (or fwhm_diameter_um, or 5.0 μm fallback)
    diameter ← max(diameter, 0.0) [clamp to positive]
    viscosity ← calculate_pries_secomb_viscosity(diameter, hematocrit)
    resistance ← (128 × viscosity × length) / (π × diameter⁴)
  
  iteration ← 0
  max_flow_diff ← ∞
  previous_flows ← {}

WHILE iteration < max_iterations AND max_flow_diff > tolerance:
  
  STEP 1: Build Conductance and Laplacian
    conductance, node_list ← build_conductance_matrix_from_graph(G)
    laplacian ← calc_laplacian_from_conductance_matrix(conductance)
  
  STEP 2: Apply Dirichlet Boundary Conditions
    For each starting_node: pressure[node] ← input_p_bc
    For each output_node:   pressure[node] ← output_p_bc
  
  STEP 3: Solve Pressure System
    L_UU × P_U = -L_UK × P_K
    pressure[unknown_idx] ← _solve_system_smart(L_UU, rhs)
  
  STEP 4: Calculate Flows & Build Directed Acyclic Graph (DAG)
    For each edge (u,v,k):
      flow_signed ← (1/resistance) × (P_u - P_v)
      flow_abs ← |flow_signed|
      If flow_signed > 0: DAG.add_edge(u → v)
      Else:               DAG.add_edge(v → u)
  
  STEP 5: Check Convergence (skip on first iteration)
    max_flow_diff ← max(|current_flows[k] - previous_flows[k]| for all k)
    If max_flow_diff ≤ tolerance: BREAK ("Converged!")
  
  STEP 6: Topologically Traverse DAG and Distribute Hematocrit
    Try: topological_order ← nx.topological_sort(DAG)
    Except NetworkXUnfeasible (cycle detected): BREAK
    
    Initialize node_h_in[n] = 0.0 for all nodes
    Initialize node_q_in[n] = 0.0 for all nodes
    
    Force Inlets: node_h_in[inlet] ← systemic_hematocrit
                  node_q_in[inlet] ← 1.0 (prevents div-by-zero)
    
    For each node in topological_order:
      h_mix ← node_h_in[node] / node_q_in[node]  (or systemic_hematocrit if q=0)
      out_edges ← DAG.out_edges(node)
      
      Case |out_edges| = 0: continue (leaf/outlet node)
      Case |out_edges| = 1: pass-through, daughter gets h_mix
      Case |out_edges| = 2: PLASMA SKIMMING (§3.3)
        h1, h2 ← calculate_phase_separation_hematocrit(q1+q2, h_mix, q1, d1, q2, d2)
      Case |out_edges| ≥ 3: PROPORTIONAL MIXING
        All daughters get h_mix
      
      Accumulate into downstream nodes:
        node_h_in[v] += h_daughter × q_daughter
        node_q_in[v] += q_daughter
  
  STEP 7: Update Viscosities and Resistances
    For each edge (u,v,k):
      h ← edge.hematocrit
      d ← edge.assigned_diameter_um
      mu_app ← calculate_pries_secomb_viscosity(d, h)
      
      # Preserve geometric integration of sphincters/pericytes
      If "original_resistance" not stored yet:
        original_resistance ← current resistance
      
      mu_old ← 1.0 / d^1.647  (the power-law viscosity from Phase 4)
      resistance_new ← original_resistance × (mu_app / mu_old)
      
      # Wall Shear Stress
      WSS_mPa ← (32 × mu_app × Q_abs) / (π × d³)
      WSS_Pa ← WSS_mPa / 1000
  
  iteration ← iteration + 1
```

> **Default parameters:**
> - Maximum iterations: 15
> - Convergence tolerance: $10^{-4}$ (maximum absolute flow difference)
> - Systemic hematocrit: 0.45

> **Assumption**: Convergence is not guaranteed for all network topologies. If cycles are detected in the DAG (which can occur due to pressure ties or numerical precision), the iteration terminates early.

### 3.5 Resistance Scaling During Rheology Updates

To preserve the complex geometric integration of sphincters and pericyte constrictions computed in the initial Poiseuille pass, the rheology solver does **not** overwrite resistances with a simple $128\mu L/\pi d^4$ formula. Instead, it scales the previously computed resistance by the ratio of the new in-vivo viscosity to the old power-law viscosity:

$$R_{\text{new}} = R_{\text{original}} \times \frac{\mu_{\text{Pries-Secomb}}(d, H_D)}{\mu_{\text{power-law}}(d)}$$

where:

$$\mu_{\text{power-law}}(d) = \frac{1}{d^{1.647}}$$

The `original_resistance` is saved on the first rheology iteration and never modified thereafter, ensuring the geometric constriction profile is embedded as a permanent scaling factor.

### 3.6 Wall Shear Stress

Wall shear stress (WSS) is calculated for each edge after the rheology solver converges:

$$\tau_w = \frac{32 \, \mu \, Q}{{\pi \, d^3}}$$

where:
- $\mu$ is the Pries–Secomb apparent viscosity (mPa·s),
- $Q$ is the absolute volumetric flow rate ($\mu m^3/s$),
- $d$ is the vessel diameter ($\mu m$).

Units: The raw calculation yields WSS in mPa (since $\frac{\text{mPa·s} \times \mu m^3/s}{\mu m^3} = \text{mPa}$). The stored value is converted to Pa:

$$\tau_{w,\text{Pa}} = \frac{\tau_{w,\text{mPa}}}{1000}$$

> **Assumption**: This assumes a fully developed parabolic flow profile (Newtonian approximation), which may underestimate WSS in vessels where the Fåhræus–Lindqvist effect creates a significant cell-free layer.

---

## 4. Vessel Geometry and Constriction Models

### 4.1 Vessel Diameter Assignment

Vessel diameters can be assigned via three modes, controlled by `radius_assignment_mode`:

| Mode | Description |
|---|---|
| `fwhm_radius` | **Default.** Per-edge diameters are measured directly from the raw 3D image using FWHM (Full Width at Half Maximum) Gaussian fitting of transverse intensity profiles along the vessel centreline. The edge attribute `fwhm_diameter_um` is read. |
| `edt_radius` | Diameters are derived from the Euclidean Distance Transform of the binary vessel mask. The edge attribute `edt_diameter_um` is read. |
| `constant_radius` | A uniform radius is applied to all vessels: $d = 2 \times \text{constant\_radius\_um}$. Default: $\text{constant\_radius\_um} = 5.0 \, \mu m$ (diameter = 10.0 $\mu m$). |

**Fallback Logic (implemented in `set_poiseuille_resistances()`):**

When using `fwhm_radius` or `edt_radius`, if the per-edge measurement attribute is `None` or $\leq 0$, the diameter falls back to the `diameter_by_branch_order` dictionary lookup. If the edge's branch order key is not found, it further falls back to the `"DEFAULT"` key in the dictionary.

### 4.2 FWHM Ray-Casting Diameter Measurement

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

> **Assumption**: The arterial-to-capillary and capillary-to-venous transitions follow exponential scaling laws. This is a simplification; Murray's Law (cubic branching ratio) is not used.

### 4.4 Sphincter and Pericyte Constriction Models

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

#### 4.4.2 Periodic Mode (`"periodic"`)

Constrictions repeat at regular intervals (`constriction_spacing`, default: 100 $\mu m$) along the vessel. Let $\phi = x \mod \text{constriction\_spacing}$ be the phase position:

$$d(\phi) = \begin{cases}
d_1 + (d_2 - d_1) \cdot \dfrac{\phi}{10} & \text{if } 0 \leq \phi < 10 \, \mu m \quad \text{(ramp down)} \\[8pt]
d_2 & \text{if } 10 \leq \phi < 30 \, \mu m \quad \text{(hold)} \\[8pt]
d_2 + (d_1 - d_2) \cdot \dfrac{\phi - 30}{10} & \text{if } 30 \leq \phi < 40 \, \mu m \quad \text{(ramp up)} \\[8pt]
d_1 & \text{if } \phi \geq 40 \, \mu m \quad \text{(unconstricted)}
\end{cases}$$

Note: The constriction length in periodic mode is hard-coded to 40 $\mu m$ (with ramp zones of 10 $\mu m$ and a hold zone of 20 $\mu m$).

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

#### 4.4.4 Integrated Resistance with Variable Diameter

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

Where the integrand at each position is:
```python
def resistance_integrand(position, length, d1, d2):
    diameter = get_diameter_at_position(position, length, d1, d2)
    viscosity = 1.0 / (diameter ** 1.647)
    return (128.0 * viscosity) / (np.pi * diameter ** 4)
```

---

## 5. Tissue Perfusion Modelling

### 5.1 Perfusion Grid

A structured 3D Cartesian grid is overlaid on the vascular network. Each grid cell represents a tissue block.

> **Default resolution**: $10 \times 10 \times 10 \, \mu m$ per cell (configured as `grid_resolution_xyz = (10.0, 10.0, 10.0)`).

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

### 5.2 Advection–Diffusion–Reaction (ADR) Equation

The steady-state tissue oxygen concentration field is governed by:

$$\nabla \cdot (\sigma \nabla C) + S_{\text{advection}} - M(C) = 0$$

where:
- $\sigma$ is the tissue oxygen diffusion coefficient (default: $1.5 \times 10^{-9}$ $m^2/s$, internally converted to $1.5 \times 10^{3}$ $\mu m^2/s$),
- $C$ is tissue oxygen concentration (mmol/L, but the solver works in $PO_2$ space, mmHg),
- $S_{\text{advection}}$ represents oxygen delivered and removed by blood flow,
- $M(C)$ is the metabolic consumption rate.

### 5.3 Oxygen–Haemoglobin Dissociation (Hill Equation)

Blood oxygen content is calculated using the Hill equation for the oxygen–haemoglobin dissociation curve, implemented in `calculate_blood_oxygen_content()`:

**Step 1 — Bohr Effect (§5.4) — Dynamic $P_{50}$ Shift:**

$$\log_{10}(P_{50}) = \log_{10}(26.0) - 0.4 \cdot (pH - 7.4) + 0.06 \cdot \log_{10}\!\left(\frac{\max(PCO_2, 10^{-12})}{40}\right)$$

$$P_{50} = 10^{\log_{10}(P_{50})}$$

**Step 2 — Dissolved Oxygen (Henry's Law):**

$$C_{\text{dissolved}} = \alpha_{O_2} \cdot PO_2$$

where $\alpha_{O_2} = 1.34 \times 10^{-3}$ mmol/L per mmHg.

**Step 3 — Haemoglobin Saturation (Hill Equation):**

$$S_{O_2} = \frac{PO_2^n}{PO_2^n + P_{50}^n}$$

where $n = 2.7$ (Hill coefficient).

**Step 4 — Bound Oxygen:**

$$C_{\text{bound}} = H_D \cdot C_{Hb,\text{max}} \cdot S_{O_2}$$

where:
- $H_D$ is discharge hematocrit (dimensionless),
- $C_{Hb,\text{max}} = \frac{0.446 \times 20.4}{0.45}$ mmol/L (maximal haemoglobin O₂ binding capacity, scaled to pure RBC content).

**Step 5 — Total Oxygen Content:**

$$C_{O_2} = C_{\text{dissolved}} + C_{\text{bound}} = \alpha_{O_2} \cdot PO_2 + H_D \cdot C_{Hb,\text{max}} \cdot S_{O_2}$$

> **Guard Rail**: If $PO_2 \leq 0$, the function returns 0.0 immediately.

> **Assumption**: The Hill equation provides a sigmoidal approximation to the full Adair equation for cooperative oxygen binding. The Hill coefficient $n = 2.7$ is appropriate for adult human haemoglobin but may differ for other species.

### 5.4 Bohr Effect

The $P_{50}$ value shifts dynamically based on local $PCO_2$ and $pH$:

$$\log_{10}(P_{50}) = \log_{10}(26.0) - 0.4 \cdot (pH - 7.4) + 0.06 \cdot \log_{10}\left(\frac{PCO_2}{40}\right)$$

This empirical formulation is based on Kelman (1966) and Severinghaus (1979). Higher $PCO_2$ and lower $pH$ shift the curve rightward (decreased oxygen affinity), facilitating oxygen unloading in metabolically active tissue.

At baseline conditions ($pH = 7.4$, $PCO_2 = 40$ mmHg):
$$P_{50} = 26.0 \text{ mmHg}$$

### 5.5 Carbon Dioxide Transport and Haldane Effect

CO₂ content in blood is modelled as the sum of dissolved and bound fractions, implemented in `calculate_blood_co2_content()`:

**Step 1 — Approximate O₂ Saturation (for Haldane Shift):**

$$S_{O_2} \approx \frac{PO_2^{2.7}}{PO_2^{2.7} + 26.0^{2.7}}$$

Note: This uses a simplified Hill equation with fixed $P_{50} = 26.0$ mmHg (no Bohr shift feedback here) for the sole purpose of estimating the Haldane effect magnitude.

**Step 2 — Base CO₂ Carrying Capacity:**

$$C_{CO_2,\text{base}} = 11.02 \cdot PCO_2^{0.396}$$

**Step 3 — Haldane Shift:**

$$\text{Haldane shift} = (0.15 - 0.05 \cdot S_{O_2}) \cdot PCO_2$$

The Haldane effect means that deoxygenated blood carries more $CO_2$ (lower $S_{O_2}$ → larger Haldane shift). This is based on the Spencer (1979) empirical formulation.

**Step 4 — Total CO₂ Content:**

$$C_{CO_2} = \alpha_{CO_2} \cdot PCO_2 + H_D \cdot (C_{CO_2,\text{base}} + \text{Haldane shift})$$

where:
- $\alpha_{CO_2} = 0.03$ mmol/L per mmHg (Henry's law solubility).

> **Guard Rail**: If $PCO_2 \leq 0$, the function returns 0.0 immediately.

### 5.6 Henderson–Hasselbalch pH Equation

Tissue $pH$ is calculated from the local $PCO_2$ using the Henderson–Hasselbalch equation, implemented in `calculate_ph_from_pco2()`:

$$pH = pK_a + \log_{10}\left(\frac{[HCO_3^-]}{\alpha_{CO_2} \cdot PCO_2}\right)$$

where:
- $pK_a = 6.1$ (carbonic acid dissociation constant),
- $[HCO_3^-] = 24.0$ mmol/L (tissue bicarbonate buffer concentration, assumed constant),
- $\alpha_{CO_2} = 0.03$ mmol/L per mmHg.

> **Guard Rail**: $PCO_2$ is clamped to $\geq 10^{-12}$ to prevent $\log_{10}(0)$.

> **Assumption**: Bicarbonate concentration is held constant (open buffer system). In reality, $[HCO_3^-]$ is regulated by renal compensation and varies with acid-base disturbances.

### 5.7 Metabolic Oxygen Consumption

Tissue metabolic consumption follows a saturating exponential:

$$M(PO_2) = M_{\text{max}} \cdot \left(1 - e^{-k \cdot PO_2}\right)$$

where:
- $M_{\text{max}} = 0.005$ mmol/L/s (maximum metabolic rate),
- $k = 0.1$ per mmol (reduction constant for hypoxic zones),
- $PO_2$ is clamped to $\geq 0$ before evaluation.

> **Assumption**: This is a phenomenological model, not a Michaelis–Menten kinetic model. The exponential form ensures consumption approaches zero as $PO_2$ → 0 and saturates at $M_{\text{max}}$ for high $PO_2$. A Michaelis–Menten form ($M = M_{\text{max}} \cdot PO_2 / (K_m + PO_2)$) is more commonly used in the literature for mitochondrial oxygen consumption.

### 5.8 Respiratory Quotient

The coupling between $O_2$ consumption and $CO_2$ production uses a fixed respiratory quotient:

$$M_{CO_2} = RQ \times M_{O_2}$$

> **Default**: $RQ = 0.82$ (typical for a mixed metabolic substrate of carbohydrates and fats).

---

## 6. Endothelial Barrier and 1D–3D Coupling

The pipeline implements three solver tiers of increasing complexity. The tier is selected based on configuration flags.

### 6.1 Tier 1: Simple ADR Solver (`solve_perfusion_steady_state`)

Used when `use_endothelial_barrier_model = False` and `use_multi_species_model = False`.

Oxygen delivery is modelled as a bulk advective source and sink:
- **Source** (oxygen in): $S_{\text{in},i} = \sum_{\text{vessels}} Q_v \cdot C_{O_2}(PO_{2,\text{arterial}}, H_D)$
- **Washout** (oxygen out): $S_{\text{out},i} = Q_{\text{total},i} \cdot C_{O_2}(PO_{2,\text{tissue},i}, H_{\text{baseline}})$

The washout term is non-linear (depends on $PO_2$ through the Hill equation), solved via Picard iteration.

### 6.2 Tier 2: Coupled 1D–3D with Endothelial Barrier (`solve_coupled_1d3d_perfusion`)

Used when `use_endothelial_barrier_model = True` and `use_multi_species_model = False`.

**Endothelial Permeability Model:**

Oxygen transport across the vessel wall is governed by a **permeability-limited flux**:

$$J_{O_2} = P_{O_2} \cdot A_{\text{surface}} \cdot (PO_{2,\text{blood}} - PO_{2,\text{tissue}})$$

where:
- $P_{O_2} = 1.0 \times 10^{-4}$ cm/s (endothelial permeability coefficient for $O_2$; internally converted to $\mu m/s$ by multiplying by $10^4$, giving $P_{O_2} = 1.0 \, \mu m/s$),
- $A_{\text{surface}}$ is the vessel surface area in the grid cell ($\mu m^2$),
- The driving force is the partial pressure difference across the endothelium (mmHg).

Note: In the simple solver, this flux uses $\alpha_{O_2}$ as a scaling factor: $J = P \cdot A \cdot \alpha_{O_2} \cdot \Delta PO_2$. In the coupled 1D-3D solver (Tier 2), the transmural flux uses partial pressures directly without $\alpha_{O_2}$:
$$J_{O_2} = P_{O_2} \cdot A_{\text{surface}} \cdot \max(0, PO_{2,\text{blood}} - PO_{2,\text{tissue}})$$

### 6.3 Tier 3: Multi-Species 1D–3D Solver (`solve_multi_species_perfusion`)

Used when `use_multi_species_model = True`.

Solves for tissue $PO_2$, $PCO_2$, and $pH$ simultaneously with Bohr/Haldane coupling.

**For $CO_2$ transmural flux:**
$$J_{CO_2} = P_{CO_2} \cdot A_{\text{surface}} \cdot \alpha_{CO_2} \cdot (PCO_{2,\text{blood}} - PCO_{2,\text{tissue}})$$

where $P_{CO_2} = 2.0 \times 10^{-3}$ cm/s ($= 20.0 \, \mu m/s$).

### 6.4 1D Blood Oxygen Tracking Along Vessels

In the coupled 1D–3D models (Tiers 2 and 3), blood oxygen content is tracked **along each vessel** as it traverses tissue grid cells.

**Complete Algorithm:**

```
STEP 1: Build DAG from flow directions
  For each edge (u,v,k):
    If flow_signed > 0: DAG.add_edge(u → v)
    If flow_signed < 0: DAG.add_edge(v → u)

STEP 2: Topological Sort
  Try: topo_order ← nx.topological_sort(DAG)
  Except: topo_order ← list(G.nodes())  [fallback]

STEP 3: Initialize Inlet Blood Gas Content
  For each inlet node n:
    For each outgoing edge from n:
      node_o2_flux_in[n] += C_O2(PO2_arterial, H_edge, PCO2_arterial, pH=7.4) × Q
      node_co2_flux_in[n] += C_CO2(PCO2_arterial, H_edge, PO2_arterial) × Q
      node_q_in[n] += Q

STEP 4: Traverse DAG Topologically
  For each node in topo_order:
    c_o2_mix ← node_o2_flux_in[node] / node_q_in[node]
    c_co2_mix ← node_co2_flux_in[node] / node_q_in[node]
    
    For each outgoing edge (node → v, key k):
      Q ← flow_abs
      H ← hematocrit
      
      # STEP 4a: Invert Hill equation to get blood PO2 from concentration
      po2_in ← brentq(λ p: C_O2(p, H, 40.0, 7.4) - c_o2_mix, 0.0, 150.0)
      pco2_in ← brentq(λ p: C_CO2(p, H, po2_in) - c_co2_mix, 0.0, 150.0)
      
      c_o2_curr ← c_o2_mix
      c_co2_curr ← c_co2_mix
      po2_curr ← po2_in
      pco2_curr ← pco2_in
      
      # STEP 4b: Walk along each grid cell this edge passes through
      For each cell traversed by this edge:
        ph_local ← pH_tissue[cell_idx]
        
        # Transmural O2 flux
        flux_o2 ← P_O2 × A_surface × α_O2 × (po2_curr − PO2_tissue[cell_idx])
        flux_co2 ← P_CO2 × A_surface × α_CO2 × (pco2_curr − PCO2_tissue[cell_idx])
        
        # Subtract flux from blood content
        If Q > 0:
          c_o2_curr ← max(0, c_o2_curr − flux_o2 / Q)
          c_co2_curr ← max(0, c_co2_curr − flux_co2 / Q)
          
          # Re-invert Hill equation with Bohr/Haldane coupling
          po2_curr ← brentq(λ p: C_O2(p, H, pco2_curr, ph_local) − c_o2_curr, 0.0, 150.0)
          pco2_curr ← brentq(λ p: C_CO2(p, H, po2_curr) − c_co2_curr, 0.0, 150.0)
        
        # Accumulate flux to tissue
        transmural_o2[cell_idx] += flux_o2
        transmural_co2[cell_idx] += flux_co2
      
      # STEP 4c: Pass remaining blood content to downstream node
      node_o2_flux_in[v] += c_o2_curr × Q
      node_co2_flux_in[v] += c_co2_curr × Q
      node_q_in[v] += Q
```

**Brent's Root-Finding for Hill Equation Inversion:**

The objective function passed to `scipy.optimize.brentq` is:

$$f(p) = C_{O_2}(p, H_D, PCO_2, pH) - C_{\text{target}}$$

where $C_{O_2}(p, H_D, PCO_2, pH)$ is the full `calculate_blood_oxygen_content()` function (§5.3). The root $p^*$ satisfying $f(p^*) = 0$ is the blood $PO_2$ that corresponds to the target concentration $C_{\text{target}}$.

- Search bracket: $[0.0, 150.0]$ mmHg.
- If `brentq` raises `ValueError` (no root in bracket), the solver uses a fallback: $PO_{2,\text{arterial}}$ if concentration is positive, 0.0 otherwise.

Similarly for $CO_2$:
$$g(p) = C_{CO_2}(p, H_D, PO_2) - C_{\text{target,CO_2}}$$

> **Assumption**: Blood is perfectly mixed within each vessel cross-section (plug-flow approximation). There is no radial $PO_2$ gradient within the vessel lumen.

---

## 7. Numerical Methods and Solver Details

### 7.1 Diffusion Matrix (7-Point Stencil)

The tissue diffusion operator uses a standard **7-point finite-difference stencil** on the 3D Cartesian grid, yielding a symmetric positive semi-definite sparse matrix.

**Diffusive conductance between adjacent cells:**

$$D_z = \sigma \cdot \frac{\Delta y \cdot \Delta x}{\Delta z}, \quad D_y = \sigma \cdot \frac{\Delta z \cdot \Delta x}{\Delta y}, \quad D_x = \sigma \cdot \frac{\Delta z \cdot \Delta y}{\Delta x}$$

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

# Y-direction connections (bottom-top neighbours)
bottom ← idx[:, :-1, :].flatten()
top    ← idx[:, 1:, :].flatten()
For each pair (bottom[i], top[i]):
  A[bottom[i], top[i]] += -D_y
  A[top[i], bottom[i]] += -D_y
  diag_A[bottom[i]] += D_y
  diag_A[top[i]]    += D_y

# Z-direction connections (back-front neighbours)
back  ← idx[:-1, :, :].flatten()
front ← idx[1:, :, :].flatten()
For each pair (back[i], front[i]):
  A[back[i], front[i]] += -D_z
  A[front[i], back[i]] += -D_z
  diag_A[back[i]]  += D_z
  diag_A[front[i]] += D_z

# Add accumulated diagonal
A[i, i] += diag_A[i]  for all i

# Regularization to prevent singularity under Neumann BCs
A[i, i] += 1e-12  for all i
```

> **Boundary condition**: Neumann (zero-flux) at the tissue grid boundaries — no oxygen escapes through the tissue surface. This is implicit in the stencil: boundary cells simply have fewer neighbours, resulting in fewer off-diagonal entries.

### 7.2 Picard Iteration for Non-Linear Perfusion

The non-linear steady-state perfusion system is solved using **Picard (fixed-point) iteration**.

**Numerical Stabilization (Pseudo-Washout Trick):**

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

For iteration = 0, 1, ..., max_iter-1:
  PO2_clamped ← max(PO2, 0.0)
  
  # 1. Metabolic Sink
  M_red ← M_max × (1 − exp(−k × PO2_clamped))
  
  # 2. Advective Washout (non-linear)
  For each voxel i with q_total[i] > 0:
    c_venous[i] ← C_O2(PO2_clamped[i], H_baseline)  [Hill equation]
    s_washout[i] ← q_total[i] × c_venous[i]
  
  # 3. RHS Construction
  b ← s_incoming − s_washout − (M_red × V_cell) + (pseudo_washout × PO2_clamped)
  
  # 4. Solve Linear System
  PO2_new, info ← CG(A_stable, b, M=M_pre, x0=PO2, rtol=1e-6, maxiter=1000)
  
  # 5. Physical Clamping
  PO2_new ← max(PO2_new, 0.0)
  
  # 6. Convergence Check (Relative L2-norm)
  diff ← ||PO2_new − PO2||₂ / (||PO2_new||₂ + 1e-12)
  If diff < tolerance: BREAK
  
  PO2 ← PO2_new
```

**Multi-Species Picard Loop:**

```
For iteration = 0, 1, ..., max_iter-1:
  PO2_clamped ← max(PO2_tissue, 0.0)
  PCO2_clamped ← max(PCO2_tissue, 0.0)
  
  # Coupled Metabolism
  M_o2 ← M_max × (1 − exp(−k × PO2_clamped))
  M_co2 ← M_o2 × RQ
  
  # Henderson-Hasselbalch
  pH_tissue ← calculate_ph_from_pco2(PCO2_clamped, hco3_tissue)
  
  # 1D Blood Tracking with Bohr/Haldane Coupling (see §6.4)
  [... compute transmural_o2, transmural_co2 ...]
  
  # RHS Construction
  b_o2  ← transmural_o2 − (M_o2 × V_cell) + (pseudo_washout_o2 × PO2_clamped)
  b_co2 ← transmural_co2 + (M_co2 × V_cell) + (pseudo_washout_co2 × PCO2_clamped)
  
  # Solve O2 and CO2 independently
  PO2_new ← CG(A_o2, b_o2, ...)
  PCO2_new ← CG(A_co2, b_co2, ...)
  
  # Convergence (both species must converge)
  diff_o2  ← ||PO2_new − PO2_tissue||₂ / (||PO2_new||₂ + 1e-12)
  diff_co2 ← ||PCO2_new − PCO2_tissue||₂ / (||PCO2_new||₂ + 1e-12)
  If diff_o2 < tolerance AND diff_co2 < tolerance: BREAK
```

> **Default parameters:**
> - Maximum Picard iterations: 50
> - Convergence tolerance: $10^{-4}$ (relative $L^2$-norm) for multi-species; $10^{-5}$ for simple ADR
> - ILU drop tolerance: $10^{-4}$, fill factor: 10
> - CG tolerance: $10^{-6}$ (simple) or $10^{-5}$ (multi-species), max 1000 or 500 iterations

### 7.3 Linear Solver Strategy

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

---

## 8. Boundary Conditions

### 8.1 Pressure Boundary Conditions (Dirichlet)

| Boundary | Default Value | Physical Basis |
|---|---|---|
| **Inlet pressure** ($P_{\text{in}}$) | 13.332 × 10⁶ mPa (= 100 mmHg) | Mean arterial pressure (MAP) — the average pressure in the systemic arterial circulation. |
| **Outlet pressure** ($P_{\text{out}}$) | 0.27 × 10⁶ mPa (= 2 mmHg) | Central venous pressure (CVP) — the pressure in the systemic venous circulation near the right atrium. |

> **Assumption**: The pressure drop from 100 mmHg (MAP) to 2 mmHg (CVP) across a micro-organ is a significant simplification. In reality, the carotid body is perfused at high flow rates relative to its mass, and the upstream resistance of the feeding artery and downstream venous drainage significantly modulate the actual pressures at the organ boundary. The effective perfusion pressure across the carotid body is likely substantially less than the full MAP–CVP gradient.

### 8.2 Inlet/Outlet Node Selection

Boundary nodes are auto-selected by finding **dead-end nodes** (degree-1) located within a configurable percentage band at the spatial extremes of the image volume along a specified axis.

| Parameter | Default | Description |
|---|---|---|
| `edge_percent` | 25% | Nodes in the top 25% of the chosen axis are candidates for inlet (starting) nodes. |
| `end_percent` | 25% | Nodes in the bottom 25% are candidates for outlet nodes. |
| `node_edge_axis` | 0 (Z-axis) | The spatial axis along which the network is oriented for boundary selection. |

### 8.3 Boundary Permeability Modes

| Mode | Description |
|---|---|
| `caged` (default) | Only the Z-axis boundaries allow vessels to enter/exit. X and Y boundaries are sealed. Virtual padding (10 voxels) is applied to the Z faces only. |
| `universal_sink` | All six faces are permeable. Dead-ends at any boundary face can be assigned as outlets. Virtual padding (10 voxels) is applied to all faces. |
| `robin_resistance` | Dead-end capillaries at boundaries are connected to a virtual "Robin Ghost Node" with a resistance equal to `robin_distal_resistance_multiplier` × average resistance of connected edges (default: 10×). This simulates flow bleeding out through severed capillaries. |

**Robin Ghost Node Implementation (from `build_conductance_matrix_from_graph()`):**

For each node tagged with `is_robin_boundary=True`:
1. Compute the average resistance of all edges connected to this node: $R_{\text{avg}} = \frac{1}{N} \sum R_i$
2. Compute the ghost resistance: $R_{\text{ghost}} = R_{\text{avg}} \times \text{robin\_multiplier}$ (default multiplier: 10.0)
3. Compute the ghost conductance: $C_{\text{ghost}} = 1/R_{\text{ghost}}$
4. Add symmetric off-diagonal entries connecting the boundary node to the ghost node.
5. The ghost node is added to the `output_nodes` list and receives $P_{\text{out}}$ as its Dirichlet BC.

### 8.4 Blood Gas Boundary Conditions

| Parameter | Default | Units | Description |
|---|---|---|---|
| Arterial $PO_2$ | 100.0 | mmHg | Oxygen partial pressure in arterial blood entering the network. |
| Arterial $PCO_2$ | 40.0 | mmHg | Carbon dioxide partial pressure in arterial blood. |
| Systemic hematocrit | 0.45 | dimensionless | Volume fraction of red blood cells in systemic blood. |
| Tissue bicarbonate | 24.0 | mmol/L | $[HCO_3^-]$ buffer concentration for Henderson-Hasselbalch pH calculation. |

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
| Matrix regularization | $10^{-6}$ | Added to perfusion A_reg diagonal |

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
