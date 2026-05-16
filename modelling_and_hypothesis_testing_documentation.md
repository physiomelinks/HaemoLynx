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

**Non-Newtonian In-Vivo Rheology (Plasma Skimming):**
The pipeline explicitly resolves the Newtonian assumption by simulating blood as a biphasic suspension (Red Blood Cells + Plasma). The engine utilizes the empirical **Pries-Secomb Model**:
1.  **Fåhræus–Lindqvist Effect:** Blood viscosity ($\mu$) is dynamically calculated per-vessel based on its physical diameter and local hematocrit, simulating the drop in viscosity in micro-capillaries and the extreme spike when diameters approach RBC dimensions ($<7 \mu m$).
2.  **Plasma Skimming (Phase Separation):** The pipeline employs a highly iterative Flow-Hematocrit solver. It builds a Directed Acyclic Graph (DAG) based on solved pressure gradients, traverses the network, and applies logistic skimming equations at every bifurcation. RBCs disproportionately favor faster/larger branches (AVAs), leaving slower capillaries with near-pure plasma. Flow and hematocrit are solved iteratively until steady-state convergence.

**The System of Governing Equations:**
The vascular network is modeled as a 1D directed graph (hydraulic circuit).
*   **Hagen-Poiseuille Resistance:** $R_{ij} = \frac{8 \mu_{app}(d, H_D) L_{ij}}{\pi r_{ij}^4}$ (where $\mu_{app}$ is the dynamic apparent viscosity).
*   **Conductance:** $G_{ij} = \frac{1}{R_{ij}}$
*   **Ohm's Law for Fluids:** $Q_{ij} = G_{ij} (P_i - P_j)$
*   **Kirchhoff's Current Law (Mass Conservation):** $\sum_{j \in \mathcal{N}(i)} G_{ij} (P_i - P_j) = 0$

**Computational Solver:**
The pipeline constructs a global, sparse Laplacian conductance matrix ($A\mathbf{P} = \mathbf{b}$). The linear system is solved dynamically:
*   **Small Networks ($N < 50k$):** UMFPACK Direct Matrix Inversion (`scipy.sparse.linalg.spsolve`).
*   **Massive Networks ($N \geq 50k$):** Preconditioned Conjugate Gradient (`cg`) utilizing an Incomplete LU (ILU) preconditioner to dramatically reduce RAM footprint and ensure convergence.
*   **Iterative Coupling:** The flow matrix is wrapped in a non-linear `while` loop that recalculates hematocrit ($H_D$) and viscosity ($\mu_{app}$) until the maximum flow delta drops below $1\times10^{-4}$.

**Inputs, Outputs, and Boundary Conditions:**
*   **Inputs:** Vascular graph topology, physical segment lengths ($L$), measured FWHM radii ($r$), and systemic baseline hematocrit ($H_{sys} \approx 0.45$).
*   **Boundary Conditions (Dirichlet):** Fixed physiological pressure constraints are applied at mathematically identified root nodes. Default configurations map Arterial Inlets to Mean Arterial Pressure (MAP $\approx 13.3$ kPa) and Venous Outlets to Central Venous Pressure (CVP $\approx 0.27$ kPa).
*   **Outputs:** Nodal pressures ($P$), Volumetric flow rates ($Q$), Wall Shear Stress ($\tau$), Local Hematocrit ($H_D$), and Apparent Viscosity ($\mu_{app}$).

**Associated Assumptions:**
1.  **Laminar Flow:** Fluid mechanics are governed entirely by viscous forces (Reynolds number $\ll 1$); inertial forces and pulsatility are ignored (steady-state flow).
2.  **Rigid Cylinders:** Vessel walls are assumed perfectly inelastic and perfectly circular.

### 1.2 3D Tissue Perfusion (Oxygen Transport)

**The System of Governing Equations:**
Oxygen delivery is mathematically mapped onto a 3D Cartesian grid. To accurately simulate physiological oxygen unloading, the pipeline explicitly resolves the "Dissolved Oxygen Assumption" by shifting the state variable from raw Concentration ($C$) to true **Partial Pressure ($PO_2$)**, leveraging the highly non-linear Bohr Effect.
*   **The ADR Equation:** $D \nabla^2 PO_2(\mathbf{x}) + S_{adv}(PO_2(\mathbf{x})) - M(PO_2(\mathbf{x})) = 0$
*   **The Hill Equation (O2 Content):** $C_{blood}(PO_2) = (\alpha_{plasma} \cdot PO_2) + \left( H_D \cdot C_{Hb\_max} \cdot \frac{PO_2^n}{PO_2^n + P_{50}^n} \right)$
*   **Dynamic Advective Source:** $S_{adv} = \frac{Q}{V_{cell}} \times \Big( C_{blood}(PO_{2\_arterial}) - C_{blood}(PO_2^{(n)}) \Big)$. Oxygen injected by the 1D vessels dynamically scales based on localized tissue hypoxia, perfectly simulating hemoglobin's sigmoidal "dumping" curve.
*   **Metabolic Sink:** $M(PO_2) = M_{max} \left( 1 - e^{-k_{reduce} \cdot PO_2} \right)$ (Non-linear cellular consumption).

**Computational Solver:**
The continuous spatial gradient ($D \nabla^2 PO_2$) is discretized using a 7-point central finite difference stencil. Because both the metabolic sink $M(PO_2)$ and the advective washout source $S_{adv}(PO_2)$ introduce severe non-linearities, the system cannot be solved directly. 
*   **Picard Iteration:** The pipeline employs a non-linear Picard loop, updating the dynamic source terms at each step and solving the sequential sparse matrices via Conjugate Gradient until $PO_2$ gradients reach steady-state convergence.
*   **Numerical Stabilization:** To prevent the non-linear advective washout from destroying the diagonal dominance of the sparse matrix (which causes violent Picard oscillation), the solver employs mathematical relaxation. It injects a linearized pseudo-washout term onto the LHS diagonal and balances it identically on the RHS, guaranteeing solver stability while preserving the exact theoretical steady-state roots.

**Inputs, Outputs, and Boundary Conditions:**
*   **Inputs:** Solved vessel flow rates ($Q$), local hematocrit ($H_D$), Arterial $PO_2$ baseline, Tissue diffusivity coefficient ($D$), Maximum metabolic rate ($M_{max}$), and Hemoglobin binding constants ($P_{50}, n$).
*   **Boundary Conditions:** Neumann Zero-Flux boundary conditions ($\frac{\partial PO_2}{\partial n} = 0$) are assumed at the external edges of the bounding box. 
*   **Outputs:** A complete 3D discrete grid of steady-state tissue Partial Pressures ($PO_2$ in mmHg), exportable as a `.vti` heatmap.

**Associated Assumptions:**
1.  **Homogeneous Diffusion:** The tissue diffusion coefficient ($D$) is identical in all spatial directions and tissue types.
2.  **Zero-Resistance Permeability:** Oxygen transfers from the vessel lumen into the tissue voxel instantly, assuming the endothelial wall presents no diffusion barrier.

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

### 3.1 Resolving the Permeability Assumption (Endothelial Barriers)
*   **The Limitation:** Oxygen flux is currently unrestricted by the physical barrier of the endothelial wall.
*   **The Improvement:** Implement the permeability ($perm_{O2}$) and surface area ($2\pi r L$) variables. Oxygen flux must be calculated as $Permeability \times Area \times (PO_{2\_vessel} - PO_{2\_tissue})$.
*   **Hypothesis Impact:** Directly tests if endothelial dysfunction or wall thickening (common in hypertension) acts as a physical diffusion barrier contributing to the hypoxic state of the CB.

### 3.2 Multi-Species Coupling ($CO_2$ and pH)
*   **The Limitation:** The model currently only solves for Oxygen transport.
*   **The Improvement:** Expand the ADR matrix builder to simultaneously solve for three coupled fields: $O_2$ (consumption), $CO_2$ (production), and $H^+$ ions.
*   **Hypothesis Impact:** Glomus cells are stimulated by hypoxia, hypercapnia, and acidity. Because $CO_2$ diffuses roughly 20 times faster than $O_2$, the "Hypoxic Core" of a hyperplastic cluster may also act as a highly acidic "Hypercapnic Core". Modeling all three provides the complete chemosensory stimulus profile.

### 3.3 Modeling Capillary Collapse (Structural vs. Functional Vasculature)
*   **The Limitation:** CFM imaging only captures physically patent vessels. Hypertension and sympathetic tone cause micro-capillaries to collapse entirely, meaning anatomical capacity does not equal functional capacity.
*   **The Improvement:** Add a "Virtual Pruning" step that systematically removes network edges from the control graph that fall below a specific diameter/pressure threshold.
*   **Hypothesis Impact:** Simulates the functional morphology under high sympathetic tone, allowing a direct comparison between maximum anatomical capacity and restricted functional reality.

---

## 4. Model Validation & Testing Strategies

To guarantee mathematical accuracy, the pipeline implements a rigorous suite of automated integration and analytical physics tests (via `pytest`). These validate the complex hemodynamics and perfusion engines against known theoretical benchmarks:

### 4.1 Structural & Mechanic Validation
These tests ensure the 3D grid, matrix builders, and non-linear solver bounds function correctly without catastrophic mathematical failure.
*   **Geometric Mapping:** Proves that physical `ZYX` coordinates map flawlessly to the linear discrete `PerfusionGrid` arrays, and that 1D physical line segments deposit their advective flow specifically into the exact 3D voxels they intersect.
*   **ADR Sparse Matrix Integrity:** Verifies the physical structure of the 7-Point Stencil Laplacian. It asserts that central tissue nodes contain exactly 7 non-zero connectivity elements (itself + 6 neighbors), preventing isolated grid blocks.
*   **Picard Iteration Bounds:** Subjects the non-linear solver to extreme biological boundaries. It proves that zero inlet flow results in exactly $0.0$ steady-state tissue concentration, and that ridiculously massive metabolic sinks mathematically cannot force tissue oxygen into non-physical negative concentrations.
*   **Matrix Singularity Bounds:** Intentionally strips a network of all pressure boundary conditions. Proves the iterative solver safely catches the missing bounds and gracefully aborts rather than crashing the C++ sparse matrix factorization libraries.
*   **Dataclass Validation:** Subjects the `HaemodynamicsConfig` to physically impossible configurations (e.g., negative sphincter lengths, 0.0 vessel diameters). Asserts that the `__post_init__` functions actively catch, warn, and clamp these values to prevent $1/r^4$ divide-by-zero singularities from destroying the mathematical engines.
*   **End-to-End Pipeline Integrity:** A full Phase 1 through 6 smoke test validating that the top-level execution script correctly orchestrates the continuous FWHM measurements, B-spline smoothing, and Iterative Rheology solvers in unison without internal data structure mismatches.

### 4.2 Analytical Physics Benchmarking
These tests compare the numerical sparse-matrix solvers against exact mathematical formulas for simplified physical geometries.
*   **Poiseuille Flow (Series & Parallel):** Creates test networks of differing radii and forces boundary pressures. Proves the numerical flow solver perfectly matches the analytical series ($R_{eq} = R_1 + R_2$) and parallel ($1/R_{eq} = 1/R_1 + 1/R_2$) conductance formulas.
*   **Sphincter Resistance Calculus:** Defines a complex periodic constriction (ramp down, hold, ramp up) and integrates the non-linear continuous radius equation $\int (1/r(x)^4) dx$ analytically. Asserts that the mathematical solver's trapezoidal numerical integrator perfectly matches the exact calculus output (to within 0.1%), proving no precision is lost across steep pre-capillary sphincters.
*   **Wall Shear Stress (WSS):** Extracts flow and non-Newtonian viscosity from a simulated network edge and asserts that the resulting WSS exactly matches the analytical mathematical formula ($\tau = 32\mu Q / \pi d^3$).
*   **0D Fick Principle Mass Balance (The Bohr Effect):** Isolates a single zero-diffusion voxel with continuous blood flow and a metabolic sink. Uses a high-precision root-finder to mathematically invert the non-linear Hill Equation to find the exact theoretical $PO_2$ target. Proves the massive 3D Picard matrix solver perfectly navigates the sigmoidal oxygen unloading curves to converge to the exact analytical answer without numerical drift.
*   **1D Pure Diffusion:** Turns off metabolism and advection. Proves the 3D finite difference Laplacian matrix produces a flawless, straight-line linear concentration gradient along a single axis.
*   **Parabolic Reaction-Diffusion:** Forces the non-linear metabolic sink into a constant zero-order rate. Proves the steady-state solver's spatial output perfectly traces the exact theoretical mathematical parabola.
*   **Radial Point Source:** Places a single advective source in the center of the grid. Proves the spatial diffusion radiating outward strictly conforms to the expected inverse-radius curve ($C(r) \propto 1/r$).

### 4.3 Non-Newtonian Rheology Validation
These tests validate the Pries-Secomb empirical math and its integration into the iterative flow solver.
*   **Fåhræus–Lindqvist Curve:** Empirically tests the viscosity equations across massive arteries ($100 \mu m$) down to extreme capillaries ($3 \mu m$), confirming that viscosity correctly drops as vessels shrink, and then correctly spikes at the $8 \mu m$ inversion point where RBCs must deform.
*   **Plasma Skimming Mechanics & Mass Conservation:** Forces an asymmetric bifurcation (e.g., a $20 \mu m$ AVA vs a $5 \mu m$ capillary). Asserts that the AVA mathematically "steals" the RBCs, driving capillary hematocrit near zero, while strictly proving that total RBC flux is perfectly conserved across the node.
*   **Coupled Solver Convergence & Safety:** Builds a mock Direct Acyclic Graph (DAG) and intentionally introduces an infinite fluid loop via impossible pressures. Asserts that the topological sorter safely catches the cycle and prevents a fatal crash.
*   **Hematocrit-Weighted Perfusion:** Creates two identical tissue cells with equal volumetric blood flow, assigning one to receive pure skimmed plasma ($H=0.0$). Asserts that the Advective Source mathematically starves the plasma-filled cell of oxygen, proving the downstream physiological impact of upstream plasma skimming.