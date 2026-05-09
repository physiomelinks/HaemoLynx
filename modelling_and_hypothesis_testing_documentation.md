# ImageLynx: Computational Modeling and Hypothesis Testing Documentation

This document serves as the comprehensive architectural reference for the computational models, mathematical frameworks, and analytical methodologies implemented within the `carotid_image_to_model` pipeline to study Carotid Body (CB) morphology and physiology.

---

## 1. Current Mathematical & Computational Framework

The pipeline utilizes a two-stage sequential physics engine. Stage 1 solves the internal fluid dynamics of the vascular network. Stage 2 utilizes those flow results to simulate the advection and diffusion of oxygen into the surrounding tissue.

### 1.1 1D Hemodynamics (Flow & Pressure)

**Vessel Geometry & Resistance Calculation:**
Rather than relying on theoretical branching hierarchies (e.g., Murray's Law), the model measures the true physical biological radius ($r$) directly from the source image. The algorithm casts orthogonal 3D rays from the vessel centerline through the machine-learning generated continuous probability field (Ilastik). By fitting a 1D Gaussian curve to this noise-free, normalized gradient ($0.0 - 1.0$), the algorithm extracts the Full-Width-at-Half-Maximum (FWHM) diameter with sub-voxel precision. These direct physical measurements are exclusively used to determine the Hagen-Poiseuille resistance.

**Physiological Pre-Capillary Sphincter Modeling:**
To simulate localized vasoconstriction (such as sympathetic tone in SHR models), the pipeline employs a highly targeted "Sphincter" integration logic rather than uniform or periodic constrictions. Based on topological branch-order generation:
1.  **Intimal Cushion (Systemic Inflow):** A mathematically defined localized pinch is applied exclusively to `B01` (Inlet) nodes to restrict total system perfusion prior to capillary distribution.
2.  **Pre-Capillary Sphincters:** By dynamically calculating the topological center (`n_mid`) of the network (representing the capillary bed), the algorithm applies localized $5.0 \mu m$ pinches exclusively at the transitional arterioles just prior to capillary distribution.
3.  **Unrestricted Capillary/Venous Shunting:** Deep capillaries and collecting veins remain entirely unconstricted (utilizing the pure FWHM measurements). This forces the mathematical flow solver to naturally shunt blood through large-diameter AVAs when pre-capillary sphincters are constricted.

**The System of Governing Equations:**
The vascular network is modeled as a 1D directed graph (hydraulic circuit).
*   **Hagen-Poiseuille Resistance:** $R_{ij} = \frac{8 \mu L_{ij}}{\pi r_{ij}^4}$
*   **Conductance:** $G_{ij} = \frac{1}{R_{ij}}$
*   **Ohm's Law for Fluids:** $Q_{ij} = G_{ij} (P_i - P_j)$
*   **Kirchhoff's Current Law (Mass Conservation):** $\sum_{j \in \mathcal{N}(i)} G_{ij} (P_i - P_j) = 0$

**Computational Solver:**
The pipeline constructs a global, sparse Laplacian conductance matrix ($A\mathbf{P} = \mathbf{b}$). The linear system is solved dynamically:
*   **Small Networks ($N < 50k$):** UMFPACK Direct Matrix Inversion (`scipy.sparse.linalg.spsolve`).
*   **Massive Networks ($N \geq 50k$):** Preconditioned Conjugate Gradient (`cg`) utilizing an Incomplete LU (ILU) preconditioner to dramatically reduce RAM footprint and ensure convergence.

**Inputs, Outputs, and Boundary Conditions:**
*   **Inputs:** Vascular graph topology, physical segment lengths ($L$), measured radii ($r$), and blood dynamic viscosity ($\mu$).
*   **Boundary Conditions (Dirichlet):** Fixed physiological pressure constraints are applied at mathematically identified root nodes. Default configurations map Arterial Inlets to Mean Arterial Pressure (MAP $\approx 13.3$ kPa) and Venous Outlets to Central Venous Pressure (CVP $\approx 0.27$ kPa).
*   **Outputs:** Nodal pressures ($P$), Volumetric flow rates ($Q$), and Wall Shear Stress ($\tau$).

**Associated Assumptions:**
1.  **Newtonian Fluid:** Blood viscosity ($\mu$) is assumed constant regardless of vessel diameter or shear rate (ignoring the Fåhræus–Lindqvist effect).
2.  **Laminar Flow:** Fluid mechanics are governed entirely by viscous forces (Reynolds number $\ll 1$); inertial forces and pulsatility are ignored (steady-state flow).
3.  **Rigid Cylinders:** Vessel walls are assumed perfectly inelastic and perfectly circular.

### 1.2 3D Tissue Perfusion (Oxygen Transport)

**The System of Governing Equations:**
Oxygen delivery is modeled using the steady-state Advection-Diffusion-Reaction (ADR) equation mathematically mapped onto a 3D Cartesian grid.
*   **The ADR Equation:** $D \nabla^2 C(\mathbf{x}) + S_{adv}(\mathbf{x}) - M(C(\mathbf{x})) = 0$
*   **Advective Source:** $S_{adv} = \frac{Q \cdot C_{arterial}}{V_{cell}}$ (Oxygen injected by the 1D vessels).
*   **Metabolic Sink:** $M(C) = M_{max} \left( 1 - e^{-k_{reduce} \cdot C} \right)$ (Non-linear cellular consumption).

**Computational Solver:**
The continuous spatial gradient ($D \nabla^2 C$) is discretized using a 7-point central finite difference stencil. Because the metabolic sink $M(C)$ introduces non-linearity, the system cannot be solved directly. The pipeline employs **Picard Iteration**, solving sequential sparse matrix updates ($A_{diff} \mathbf{C}^{(n+1)} = \mathbf{b}^{(n)}$) via Conjugate Gradient until the spatial concentration gradients reach steady-state convergence (tolerance $\leq 10^{-5}$).

**Inputs, Outputs, and Boundary Conditions:**
*   **Inputs:** Solved vessel flow rates ($Q$), Arterial baseline concentration ($C_{arterial}$), Tissue diffusivity coefficient ($D$), Maximum metabolic rate ($M_{max}$), and the exponential decay constant ($k_{reduce}$).
*   **Boundary Conditions:** Neumann Zero-Flux boundary conditions ($\frac{\partial C}{\partial n} = 0$) are assumed at the external edges of the bounding box. The 1D flow segments act as internal localized Dirichlet-style volumetric sources.
*   **Outputs:** A complete 3D discrete grid of steady-state oxygen concentrations ($C$), exportable as a `.vti` heatmap.

**Associated Assumptions:**
1.  **Homogeneous Diffusion:** The tissue diffusion coefficient ($D$) is identical in all spatial directions and tissue types.
2.  **Dissolved Oxygen:** Oxygen transport assumes the gas is purely dissolved in plasma; the non-linear release curve of hemoglobin bound $O_2$ is simplified.
3.  **Zero-Resistance Permeability:** Oxygen transfers from the vessel lumen into the tissue voxel instantly, assuming the endothelial wall presents no diffusion barrier.

---

## 2. Hypothesis Testing Methodologies

### Hypothesis 1: Does the CB morphology change under hypertension?
To quantitatively answer this hypothesis by comparing WKY (normotensive) and SHR (hypertensive) cohorts, the 1D vascular network data must be combined with the 3D Tyrosine Hydroxylase (TH) glomus cell mask. This combination shifts the analysis from basic anatomy to functional structural capacity.

**Combined Analyses:**
1.  **Exact Vascular Density of the Glomus Tissue:**
    *   *Concept:* Calculate the vessel density *exclusively* inside the TH-positive regions (`Total Vessel Volume inside TH Mask / Total TH Volume`).
    *   *Rationale:* If the SHR cohort shows glomus cell hyperplasia (massive tissue growth) without proportional capillary expansion, this density metric drops, physically proving the hypertensive tissue is structurally under-vascularized.
2.  **3D Diffusion Distance Mapping (The "Hypoxic Core" Test):**
    *   *Concept:* Run a 3D Euclidean Distance Transform on the vascular mask to calculate the physical distance from every TH-positive voxel to the nearest blood vessel wall.
    *   *Rationale:* If SHR glomus clusters are larger but possess fewer penetrating capillaries, the mean diffusion distance increases. This structurally proves that the center of SHR glomus clusters are chronically hypoxic, explaining hyperactive chemosensitivity.
3.  **Spatial Coupling (Capillary Proximity to Constrictions):**
    *   *Concept:* Map the coordinates of highly constricted vessels (or modeled pericytes) against the 3D bounding box of the TH clusters.
    *   *Rationale:* Evaluates whether hypertensive constrictions (decreased cross-sectional area) occur directly inside the glomus tissue or in feeding arterioles upstream, addressing arguments regarding sympathetic vs. vascular control of chemosensitivity.
4.  **Typical/Effective Vessel Cross-Sectional Area by Branch Order:**
    *   *Concept:* Calculate the mean cross-sectional area ($\pi r^2$) and stratify these measurements by topological branch order (e.g., distinguishing deep capillaries from feeding arteries).
    *   *Rationale:* Directly addresses the hypothesis of sympathetic vasoconstriction. By comparing specific branch orders, it proves whether hypertensive constriction is localized to the microvasculature.
5.  **Vessel Network Length Across Branch Points:**
    *   *Concept:* Calculate the total physical length of the vascular network and the average physical segment lengths between bifurcations.
    *   *Rationale:* Quantifies vascular expansion or pruning. An increase in total capillary length alongside a decrease in cross-sectional area indicates a specific type of structural reconfiguration under hypertension.
6.  **Vessel Tortuosity & True Biological Length:**
    *   *Concept:* Calculate the tortuosity index (ratio of actual physical path length to the straight-line Euclidean distance between nodes) for all vessel segments.
    *   *Rationale:* Hypertension often induces tortuous, "corkscrew" vessel remodeling due to sustained high pressure and endothelial stress. Measuring this provides a direct metric of structural degradation. To ensure mathematical accuracy and prevent "stair-stepping" artifacts from artificially inflating the measured lengths of rasterized voxels, the pipeline utilizes a multi-core continuous B-spline smoothing algorithm on all centerlines prior to measuring. This guarantees that tortuosity calculations trace realistic biological curvature.

### Hypothesis 2: Does the perfusion profile of the CB change due to changes involved in hypertension?
To answer this hypothesis, static morphology is bypassed to utilize the pipeline's fluid dynamics and mass transport modeling capabilities.

**Targeted Analyses and Required Features:**
1.  **Targeted Steady-State Perfusion Fields:** By combining blood flow results with TH-channel data, the 3D Picard Iteration solver generates a heatmap of oxygen concentration. Comparing the mean steady-state $O_2$ concentrations of the SHR and WKY glomus clusters definitively shows if morphological remodeling causes functional deficits in oxygen delivery.
2.  **TH-Masked Metabolic Grid Integration:** The `PerfusionGrid` assumes uniform metabolic consumption ($M_{max}$) across the bounding volume. The grid must map the 3D TH-binary mask, setting $M_{max} = 0$ for non-glomus cells, restricting oxygen consumption strictly to the functional parenchyma.
3.  **Hypoxic Tissue Fraction Statistic:** A feature to calculate the exact percentage (volume fraction) of TH-positive grid cells where the steady-state $O_2$ concentration falls below a defined "hypoxic threshold".
4.  **Shunt vs. Perfusion Flow Ratio Analysis:** Flow rates must be categorized to differentiate vessels bypassing TH clusters (Arterial-Venous Anastomoses) from those penetrating TH clusters (Capillaries). Comparing the total blood flow distributed to each pathway tests whether hypertension alters CB perfusion by decreasing the proportion of shunted blood flow.
5.  **Targeted Pressure Distribution and Gradients:** The 1D pressure fields computed across the network must be extracted and mapped to the structural data to calculate the exact pressure drop ($\Delta P$) across the functional capillary beds inside the glomus tissue. This maps the systemic hypertensive state into the micro-environment, determining if systemic high pressure is transmitted directly into the glomus tissue capillaries or if upstream constriction shields them.

---

## 3. Future Work: Advanced Physics Enhancements

To better address the presented hypotheses and elevate the model from a basic approximation to a highly rigorous, biologically representative simulation, the following areas of improvement should be targeted to address the baseline assumptions outlined in Section 1.

### 3.1 Resolving the Newtonian Assumption (Plasma Skimming)
*   **The Limitation:** Assuming constant viscosity ignores blood hematocrit distribution. At bifurcations (especially AVAs), Red Blood Cells (RBCs) disproportionately favor the wider, faster branch.
*   **The Improvement:** Integrate a Non-Newtonian Rheology Model (e.g., Pries-Secomb) into the fluid dynamics solver to calculate unequal hematocrit splitting at junctions.
*   **Hypothesis Impact:** Validates whether plasma skimming structurally starves hyperplastic capillary beds of actual oxygen delivery, despite increased physical vascularization.

### 3.2 Resolving the Dissolved Oxygen Assumption (The Bohr Effect)
*   **The Limitation:** The current ADR matrix assumes dissolved $O_2$. In reality, $O_2$ release from hemoglobin follows a highly non-linear, S-shaped curve dependent on partial pressure ($PO_2$).
*   **The Improvement:** Update the Picard Iteration solver to convert concentration to partial pressure. The advective source term from the vessels must release oxygen governed by the Hill equation for hemoglobin saturation.
*   **Hypothesis Impact:** Ensures oxygen is delivered realistically based on the localized hypoxic gradients of the glomus clusters, rather than linearly dumping into the tissue.

### 3.3 Resolving the Permeability Assumption (Endothelial Barriers)
*   **The Limitation:** Oxygen flux is currently unrestricted by the physical barrier of the endothelial wall.
*   **The Improvement:** Implement the permeability ($perm_{O2}$) and surface area ($2\pi r L$) variables. Oxygen flux must be calculated as $Permeability \times Area \times (PO_{2\_vessel} - PO_{2\_tissue})$.
*   **Hypothesis Impact:** Directly tests if endothelial dysfunction or wall thickening (common in hypertension) acts as a physical diffusion barrier contributing to the hypoxic state of the CB.

### 3.4 Multi-Species Coupling ($CO_2$ and pH)
*   **The Limitation:** The model currently only solves for Oxygen transport.
*   **The Improvement:** Expand the ADR matrix builder to simultaneously solve for three coupled fields: $O_2$ (consumption), $CO_2$ (production), and $H^+$ ions.
*   **Hypothesis Impact:** Glomus cells are stimulated by hypoxia, hypercapnia, and acidity. Because $CO_2$ diffuses roughly 20 times faster than $O_2$, the "Hypoxic Core" of a hyperplastic cluster may also act as a highly acidic "Hypercapnic Core". Modeling all three provides the complete chemosensory stimulus profile.

### 3.5 Modeling Capillary Collapse (Structural vs. Functional Vasculature)
*   **The Limitation:** CFM imaging only captures physically patent vessels. Hypertension and sympathetic tone cause micro-capillaries to collapse entirely, meaning anatomical capacity does not equal functional capacity.
*   **The Improvement:** Add a "Virtual Pruning" step that systematically removes network edges from the control graph that fall below a specific diameter/pressure threshold.
*   **Hypothesis Impact:** Simulates the functional morphology under high sympathetic tone, allowing a direct comparison between maximum anatomical capacity and restricted functional reality.