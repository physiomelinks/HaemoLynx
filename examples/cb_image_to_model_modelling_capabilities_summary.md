# Physiological Modelling Documentation: ImageLynx `carotid_image_to_model.py`

> **Purpose**: This document catalogues every physical law, physiological assumption, empirical model, parameter value, and boundary condition implemented in the ImageLynx vascular modelling pipeline. It is intended to be uploaded to NotebookLM alongside published literature so that the notebook can systematically cross-reference and critique each modelling decision.

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

$$\mathbf{L} = \text{diag}\left(\sum_j C_{ij}\right) - \mathbf{C}$$

where $C_{ij} = 1/R_{ij}$ is the conductance (inverse resistance) of the edge between nodes $i$ and $j$.

**Dirichlet boundary conditions** (fixed pressures) are applied at inlet and outlet nodes. The unknown interior pressures are solved via:

$$\mathbf{L}_{UU} \, \mathbf{P}_U = -\mathbf{L}_{UK} \, \mathbf{P}_K$$

where subscript $U$ denotes unknown (interior) nodes and $K$ denotes known (boundary) nodes.

> **Assumption**: Conservation of mass at every node — no leakage through vessel walls is modelled at this stage (perfusion leakage is handled separately in the tissue diffusion model, §5).

### 2.3 Effective Two-Point Resistance

The pipeline also calculates the **effective resistance** between a specific inlet–outlet pair using the Laplacian pseudoinverse method. A unit current is injected at the source node, the target node is grounded (row/column zeroed with a 1 on the diagonal), and the resulting voltage at the source equals the effective resistance.

---

## 3. Blood Rheology

### 3.1 Pries–Secomb In-Vivo Viscosity Model (Fåhræus–Lindqvist Effect)

Blood viscosity in microvessels is **not constant** — it varies dramatically with vessel diameter and local hematocrit. The pipeline implements the empirical Pries–Secomb (1992, 1994) model:

$$\mu_{45} = 220 \, e^{-1.3 D} + 3.2 - 2.44 \, e^{-0.06 D^{0.645}}$$

where $\mu_{45}$ is the relative apparent viscosity at a reference hematocrit of $H_D = 0.45$, and $D$ is the vessel diameter in $\mu m$.

A shape parameter $C$ describes the hematocrit dependence:

$$C = \left(0.8 + e^{-0.075 D}\right) \left(-1 + \frac{1}{1 + 10^{-11} D^{12}}\right) + \frac{1}{1 + 10^{-11} D^{12}}$$

The relative apparent viscosity at actual hematocrit $H_D$ is:

$$\mu_{\text{rel}} = 1 + (\mu_{45} - 1) \cdot \frac{(1 - H_D)^C - 1}{(1 - 0.45)^C - 1}$$

An in-vivo correction for the cell-free (glycocalyx) layer is applied:

$$\mu_{\text{app}} = \mu_{\text{rel}} \cdot \left(\frac{D}{D - 1.1}\right)^2$$

The final apparent viscosity in physical units is:

$$\mu = \mu_{\text{app}} \times \mu_{\text{plasma}}$$

> **Default parameter**: $\mu_{\text{plasma}} = 1.2$ mPa·s (cP).

> **Assumptions:**
> - The empirical correlations were derived from *in vivo* measurements in rat mesentery. Their direct applicability to the carotid body microvasculature (a glomus organ with unique perfusion characteristics) is assumed but not validated.
> - Minimum diameter cap: vessels smaller than 3.0 $\mu m$ are clamped to 3.0 $\mu m$ to avoid mathematical singularities (the cell-free-layer correction diverges at $D = 1.1 \, \mu m$).
> - Maximum hematocrit is capped at 0.95 to prevent non-physical values.

### 3.2 Initial (Pre-Rheology) Viscosity Approximation

Before the iterative rheology solver runs, the pipeline uses a simpler **power-law viscosity** for the initial Poiseuille resistance calculation:

$$\mu_{\text{initial}} = \frac{1}{d^{1.647}}$$

This is a heuristic approximation to give smaller vessels higher viscosity. It is replaced by the full Pries–Secomb model during the coupled iteration (§3.4).

### 3.3 Plasma Skimming (Phase Separation at Bifurcations)

At diverging bifurcations, red blood cells (RBCs) do **not** distribute proportionally to flow. The pipeline implements the Pries–Secomb empirical logistic skimming model:

Given total inflow $Q_{\text{in}}$ with hematocrit $H_{\text{in}}$, and two daughter branches with flows $Q_1, Q_2$ and diameters $d_1, d_2$:

1. Flow fraction: $f_{Q_1} = Q_1 / Q_{\text{in}}$

2. Skimming threshold: $x_0 = 0.05$ (branches receiving less than 5% of flow get zero RBCs).

3. Asymmetry parameter:

$$A = -13.29 \cdot \frac{d_1^2/d_2^2 - 1}{d_1^2/d_2^2 + 1} \cdot \frac{1 - H_{\text{in}}}{d_1}$$

4. Steepness parameter:

$$B = 1 + 6.98 \cdot \frac{1 - H_{\text{in}}}{d_1}$$

5. Logit transformation:

$$\text{logit}(f_{E_1}) = A + B \cdot \ln\!\left(\frac{f_{Q_1} - x_0}{1 - f_{Q_1} - x_0}\right)$$

6. RBC flux fraction: $f_{E_1} = \text{sigmoid}(\text{logit}(f_{E_1}))$

7. Daughter hematocrit: $H_1 = H_{\text{in}} \cdot f_{E_1} / f_{Q_1}$

> **Assumptions:**
> - The phase separation model is only applied at **binary bifurcations** (degree-2 splits). For trifurcations and higher, RBCs are distributed proportionally to flow (simple mixing).
> - The empirical constants (−13.29, 6.98, $x_0 = 0.05$) were derived from glass tube experiments and *in vivo* rat cremaster observations.

### 3.4 Coupled Flow–Hematocrit–Viscosity Iteration

The full non-linear coupling between flow, hematocrit distribution, and viscosity is solved iteratively (Picard-style fixed-point iteration):

1. Initialize all edges with systemic hematocrit ($H_D = 0.45$) and compute Pries–Secomb viscosities.
2. Solve the linear Laplacian pressure/flow system.
3. Build a Directed Acyclic Graph (DAG) from pressure gradients.
4. Topologically traverse the DAG from inlets to outlets, applying plasma skimming at each bifurcation.
5. Update edge viscosities and resistances based on the new hematocrit distribution.
6. Repeat until the maximum absolute flow change between iterations falls below a tolerance.

> **Default parameters:**
> - Maximum iterations: 15
> - Convergence tolerance: $10^{-4}$ (maximum absolute flow difference)
> - Systemic hematocrit: 0.45

> **Assumption**: Convergence is not guaranteed for all network topologies. If cycles are detected in the DAG (which can occur due to pressure ties or numerical precision), the iteration terminates early.

### 3.5 Resistance Scaling During Rheology Updates

To preserve the complex geometric integration of sphincters and pericyte constrictions computed in the initial Poiseuille pass, the rheology solver does **not** overwrite resistances with a simple $128\mu L/\pi d^4$ formula. Instead, it scales the previously computed resistance by the ratio of the new in-vivo viscosity to the old power-law viscosity:

$$R_{\text{new}} = R_{\text{original}} \times \frac{\mu_{\text{Pries-Secomb}}}{\mu_{\text{power-law}}}$$

### 3.6 Wall Shear Stress

Wall shear stress (WSS) is calculated for each edge after the rheology solver converges:

$$\tau_w = \frac{32 \, \mu \, Q}{\pi \, d^3}$$

Units: computed in mPa, then converted to Pa by dividing by 1000.

> **Assumption**: This assumes a fully developed parabolic flow profile (Newtonian approximation), which may underestimate WSS in vessels where the Fåhræus–Lindqvist effect creates a significant cell-free layer.

---

## 4. Vessel Geometry and Constriction Models

### 4.1 Vessel Diameter Assignment

Vessel diameters can be assigned via three modes:

| Mode | Description |
|---|---|
| `fwhm_radius` | **Default.** Per-edge diameters are measured directly from the raw 3D image using FWHM (Full Width at Half Maximum) Gaussian fitting of transverse intensity profiles along the vessel centreline. |
| `edt_radius` | Diameters are derived from the Euclidean Distance Transform of the binary vessel mask. |
| `constant_radius` | A uniform radius is applied to all vessels. Default: 5.0 $\mu m$. |

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

The pipeline models localised vessel constrictions to simulate the effect of pericytes and vascular sphincters. Two constriction modes are available:

#### 4.4.1 Sphincter Mode (Default)

A single constriction is placed at the **origin** (proximal end) of each vessel segment. Within the constriction zone of length $L_s$ (default: 5.0 $\mu m$):
- 0 to $0.25 L_s$: linear ramp from $d_1$ (unconstricted) down to $d_2$ (constricted).
- $0.25 L_s$ to $0.75 L_s$: held at $d_2$.
- $0.75 L_s$ to $L_s$: linear ramp back from $d_2$ to $d_1$.

Beyond $L_s$, the diameter returns to $d_1$ for the remainder of the vessel.

#### 4.4.2 Periodic Mode

Constrictions repeat at regular intervals (`constriction_spacing`, default: 100 $\mu m$) along the vessel:
- 0–10 $\mu m$: ramp from $d_1$ to $d_2$
- 10–30 $\mu m$: held at $d_2$
- 30–40 $\mu m$: ramp from $d_2$ back to $d_1$

#### 4.4.3 Constriction Ratios

Two physiological constriction types are modelled:

| Constriction Type | Location | Default Ratio ($d_2/d_1$) | Physiological Basis |
|---|---|---|---|
| **Intimal cushion** | Branch order B01 (carotid origin) | 0.60 | Intimal cushions at the origin of the carotid body vessels reduce the lumen to ~60% of its unconstricted diameter. |
| **Pre-capillary sphincter** | At the topological midpoint (capillary bed transition) | 0.50 | Pre-capillary sphincters constrict vessels to ~50% of their resting diameter at the arteriole-capillary junction. |

> **Minimum constriction ratio**: Clamped to 0.01 to prevent infinite resistance / matrix singularities.

#### 4.4.4 Integrated Resistance with Variable Diameter

When constrictions are active, the total resistance is computed by **numerical integration** (trapezoidal rule, 1000 sample points) of the position-dependent resistance per unit length:

$$R_{\text{total}} = \int_0^L \frac{128 \, \mu(x)}{\pi \, d(x)^4} \, dx$$

where $d(x)$ and $\mu(x) = 1/d(x)^{1.647}$ vary along the vessel according to the constriction profile.

---

## 5. Tissue Perfusion Modelling

### 5.1 Perfusion Grid

A structured 3D Cartesian grid is overlaid on the vascular network. Each grid cell represents a tissue block.

> **Default resolution**: $10 \times 10 \times 10 \, \mu m$ per cell.

Vessel segments (graph edges) are mapped to grid cells by point-sampling the centreline voxels. The vessel surface area within each cell is calculated as:

$$A_{\text{surface}} = 2\pi r \cdot L_{\text{segment}}$$

where $r$ is the vessel radius and $L_{\text{segment}}$ is the length of the vessel segment passing through that cell.

### 5.2 Advection–Diffusion–Reaction (ADR) Equation

The steady-state tissue oxygen concentration field is governed by:

$$\nabla \cdot (\sigma \nabla C) + S_{\text{advection}} - M(C) = 0$$

where:
- $\sigma$ is the tissue oxygen diffusion coefficient (default: $1.5 \times 10^{-9}$ $m^2/s$),
- $C$ is tissue oxygen concentration,
- $S_{\text{advection}}$ represents oxygen delivered and removed by blood flow,
- $M(C)$ is the metabolic consumption rate.

### 5.3 Oxygen–Haemoglobin Dissociation (Hill Equation)

Blood oxygen content is calculated using the Hill equation for the oxygen–haemoglobin dissociation curve:

$$S_{O_2} = \frac{PO_2^n}{PO_2^n + P_{50}^n}$$

$$C_{O_2} = \alpha_{O_2} \cdot PO_2 + H_D \cdot C_{Hb,max} \cdot S_{O_2}$$

where:
- $n = 2.7$ (Hill coefficient),
- $P_{50} = 26.0$ mmHg (at pH 7.4, $PCO_2$ = 40 mmHg),
- $\alpha_{O_2} = 1.34 \times 10^{-3}$ mmol/L per mmHg (Henry's law solubility of $O_2$ in plasma),
- $C_{Hb,max} = 0.446 \times 20.4 / 0.45$ mmol/L (maximal haemoglobin O₂ binding capacity, scaled to pure RBC).

> **Assumption**: The Hill equation provides a sigmoidal approximation to the full Adair equation for cooperative oxygen binding. The Hill coefficient $n = 2.7$ is appropriate for adult human haemoglobin but may differ for other species.

### 5.4 Bohr Effect

The $P_{50}$ value shifts dynamically based on local $PCO_2$ and $pH$:

$$\log_{10}(P_{50}) = \log_{10}(26.0) - 0.4 \cdot (pH - 7.4) + 0.06 \cdot \log_{10}\left(\frac{PCO_2}{40}\right)$$

This empirical formulation is based on Kelman (1966) and Severinghaus (1979). Higher $PCO_2$ and lower $pH$ shift the curve rightward (decreased oxygen affinity), facilitating oxygen unloading in metabolically active tissue.

### 5.5 Carbon Dioxide Transport and Haldane Effect

CO₂ content in blood is modelled as the sum of dissolved and bound fractions:

$$C_{CO_2} = \alpha_{CO_2} \cdot PCO_2 + H_D \cdot (C_{CO_2,\text{base}} + \text{Haldane shift})$$

where:
- $\alpha_{CO_2} = 0.03$ mmol/L per mmHg (Henry's law solubility),
- $C_{CO_2,\text{base}} = 11.02 \cdot PCO_2^{0.396}$ (empirical CO₂ dissociation curve),
- Haldane shift $= (0.15 - 0.05 \cdot S_{O_2}) \cdot PCO_2$.

The Haldane effect means that deoxygenated blood carries more $CO_2$ (lower $S_{O_2}$ → larger Haldane shift). This is based on the Spencer (1979) empirical formulation.

### 5.6 Henderson–Hasselbalch pH Equation

Tissue $pH$ is calculated from the local $PCO_2$ using the Henderson–Hasselbalch equation:

$$pH = pK_a + \log_{10}\left(\frac{[HCO_3^-]}{\alpha_{CO_2} \cdot PCO_2}\right)$$

where:
- $pK_a = 6.1$ (carbonic acid dissociation constant),
- $[HCO_3^-] = 24.0$ mmol/L (tissue bicarbonate buffer concentration, assumed constant),
- $\alpha_{CO_2} = 0.03$ mmol/L per mmHg.

> **Assumption**: Bicarbonate concentration is held constant (open buffer system). In reality, $[HCO_3^-]$ is regulated by renal compensation and varies with acid-base disturbances.

### 5.7 Metabolic Oxygen Consumption

Tissue metabolic consumption follows a saturating exponential:

$$M(PO_2) = M_{\text{max}} \cdot \left(1 - e^{-k \cdot PO_2}\right)$$

where:
- $M_{\text{max}} = 0.005$ mmol/L/s (maximum metabolic rate),
- $k = 0.1$ per mmol (reduction constant for hypoxic zones).

> **Assumption**: This is a phenomenological model, not a Michaelis–Menten kinetic model. The exponential form ensures consumption approaches zero as $PO_2$ → 0 and saturates at $M_{\text{max}}$ for high $PO_2$. A Michaelis–Menten form ($M = M_{\text{max}} \cdot PO_2 / (K_m + PO_2)$) is more commonly used in the literature for mitochondrial oxygen consumption.

### 5.8 Respiratory Quotient

The coupling between $O_2$ consumption and $CO_2$ production uses a fixed respiratory quotient:

$$M_{CO_2} = RQ \times M_{O_2}$$

> **Default**: $RQ = 0.82$ (typical for a mixed metabolic substrate of carbohydrates and fats).

---

## 6. Endothelial Barrier and 1D–3D Coupling

### 6.1 Endothelial Permeability Model

When the endothelial barrier model is enabled, oxygen transport across the vessel wall is governed by a **permeability-limited flux** rather than instantaneous equilibrium:

$$J_{O_2} = P_{O_2} \cdot A_{\text{surface}} \cdot \alpha_{O_2} \cdot (PO_{2,\text{blood}} - PO_{2,\text{tissue}})$$

where:
- $P_{O_2} = 1.0 \times 10^{-4}$ cm/s (endothelial permeability coefficient for $O_2$; internally converted to $\mu m/s$),
- $A_{\text{surface}}$ is the vessel surface area in the grid cell,
- The driving force is the partial pressure difference across the endothelium.

For $CO_2$:
- $P_{CO_2} = 2.0 \times 10^{-3}$ cm/s (20× higher than $O_2$ permeability, reflecting $CO_2$'s higher membrane solubility).

### 6.2 1D Blood Oxygen Tracking Along Vessels

In the coupled 1D–3D model, blood oxygen content is tracked **along each vessel** as it traverses tissue grid cells. The algorithm:

1. Topologically sorts the flow-directed network (DAG).
2. At each inlet node, blood enters with arterial $PO_2$ (default: 100 mmHg) and arterial $PCO_2$ (default: 40 mmHg).
3. As blood flows through each grid cell, the transmural flux $J$ is subtracted from the blood oxygen content.
4. The new blood $PO_2$ is computed by **inverting the Hill equation** using Brent's root-finding method.
5. At converging nodes, blood from multiple upstream edges is mixed by flow-weighted averaging.
6. The transmural flux is added to the tissue side as a source term.

> **Assumption**: Blood is perfectly mixed within each vessel cross-section (plug-flow approximation). There is no radial $PO_2$ gradient within the vessel lumen.

---

## 7. Numerical Methods and Solver Details

### 7.1 Diffusion Matrix (7-Point Stencil)

The tissue diffusion operator uses a standard **7-point finite-difference stencil** on the Cartesian grid, yielding a symmetric positive semi-definite sparse matrix.

Diffusive conductance between adjacent cells:

$$D_x = \sigma \cdot \frac{\Delta y \cdot \Delta z}{\Delta x}, \quad D_y = \sigma \cdot \frac{\Delta x \cdot \Delta z}{\Delta y}, \quad D_z = \sigma \cdot \frac{\Delta x \cdot \Delta y}{\Delta z}$$

> **Boundary condition**: Neumann (zero-flux) at the tissue grid boundaries — no oxygen escapes through the tissue surface.

### 7.2 Picard Iteration for Non-Linear Perfusion

The non-linear steady-state perfusion system is solved using **Picard (fixed-point) iteration**:

1. Linearize the non-linear terms (metabolic consumption, advective washout) around the current solution.
2. Solve the resulting linear system using preconditioned Conjugate Gradient (CG) with Incomplete LU (ILU) preconditioning.
3. Update the solution and repeat until the relative $L^2$-norm change falls below tolerance.

> **Numerical stabilization**: A pseudo-washout term ($\gamma \cdot Q \cdot PO_2$) is added to both the LHS diagonal and the RHS to make the matrix strictly diagonally dominant, preventing oscillation and ensuring CG convergence.

> **Default parameters:**
> - Maximum Picard iterations: 50
> - Convergence tolerance: $10^{-4}$ (relative $L^2$-norm)
> - ILU drop tolerance: $10^{-4}$, fill factor: 10

### 7.3 Linear Solver Strategy

The pipeline selects between direct and iterative solvers based on system size:

| System Size | Solver | Details |
|---|---|---|
| < 50,000 unknowns | **Direct** (SciPy `spsolve`) | Exact solution; fast for small/medium networks. |
| ≥ 50,000 unknowns | **Iterative** (CG with ILU preconditioner) | Memory-efficient for massive networks; tolerance $10^{-8}$, max 1000 iterations. |

Fallback: If the direct solver or ILU preconditioning fails (singular matrix), the system falls back to LSQR (least-squares).

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

### 8.4 Blood Gas Boundary Conditions

| Parameter | Default | Units | Description |
|---|---|---|---|
| Arterial $PO_2$ | 100.0 | mmHg | Oxygen partial pressure in arterial blood entering the network. |
| Arterial $PCO_2$ | 40.0 | mmHg | Carbon dioxide partial pressure in arterial blood. |
| Systemic hematocrit | 0.45 | dimensionless | Volume fraction of red blood cells in systemic blood. |

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
| Skimming threshold | $x_0$ | 0.05 | dimensionless | Pries & Secomb empirical |
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
| Max metabolic rate | $M_{\text{max}}$ | 0.005 | mmol/L/s | Phenomenological |
| Metabolic reduction constant | $k$ | 0.1 | per mmol | Phenomenological |
| Respiratory quotient | $RQ$ | 0.82 | dimensionless | Mixed substrate metabolism |
| Tissue bicarbonate | $[HCO_3^-]$ | 24.0 | mmol/L | Normal plasma |
| Grid resolution | — | $10 \times 10 \times 10$ | $\mu m$ | User-configurable |
| Picard max iterations | — | 50 | — | Convergence criterion |
| Picard tolerance | — | $10^{-4}$ | — | Relative $L^2$-norm |

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
