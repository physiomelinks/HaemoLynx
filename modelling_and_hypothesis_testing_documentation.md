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

### 1.2 3D Tissue Perfusion (Oxygen Transport)

**The System of Governing Equations:**
Oxygen delivery is mathematically mapped onto a 3D Cartesian grid. To accurately simulate physiological oxygen unloading, the pipeline explicitly resolves both the "Dissolved Oxygen Assumption" and the "Permeability Assumption" by decoupling Blood $PO_2$ from Tissue $PO_2$ and treating the vessel wall as a physical barrier. Furthermore, the pipeline utilizes a **Multi-Species ($O_2, CO_2,$ pH) Coupled Model** to perfectly simulate the polymodal chemosensory environment of the Carotid Body.
*   **The ADR Equations:** $D \nabla^2 PO_{2\_tissue}(\mathbf{x}) + S_{adv\_o2} - M_{O2} = 0$ and $D_{co2} \nabla^2 PCO_{2\_tissue}(\mathbf{x}) + S_{adv\_co2} + M_{CO2} = 0$. The solver builds two distinct 3D sparse matrices, explicitly reflecting that $CO_2$ diffuses roughly $20\times$ faster through tissue than $O_2$.
*   **The Bohr Effect (O2 Content):** $C_{blood}(PO_2) = (\alpha_{plasma} \cdot PO_2) + \left( H_D \cdot C_{Hb\_max} \cdot \frac{PO_2^n}{PO_2^n + P_{50}^n} \right)$. The $P_{50}$ (hemoglobin affinity) dynamically shifts rightward based on local $PCO_2$ gradients and acidosis (pH) via the Kelman/Severinghaus equations, aggressively "dumping" oxygen in hypoxic/hypercapnic cores.
*   **The Haldane Effect (CO2 Content):** $CO_2$ carrying capacity is explicitly coupled to local $PO_2$ via Spencer's empirical dissociation curve. As blood gives up oxygen to the tissue, its capacity to carry $CO_2$ mathematically increases.
*   **Henderson-Hasselbalch Equilibration:** Tissue $pH$ is dynamically calculated per-voxel based on the local steady-state $PCO_2$ gradient and the standard tissue bicarbonate buffer ($[HCO_3^-]$).
*   **Trans-Mural Oxygen Flux:** $Flux = P_{perm} \times Area \times \left(PO_{2\_blood} - PO_{2\_tissue}^{(n)}\right)$. Oxygen leakage is strictly regulated by the physical surface area ($2\pi r L$) of the vessel segment and its endothelial permeability coefficient ($P_{perm}$).
*   **Dynamic Advective Source:** $S_{adv} = \frac{Flux}{V_{cell}}$. The total trans-mural flux deposited into the voxel becomes the dynamic right-hand-side source term for the 3D diffusion matrix.
*   **Coupled Metabolic Sinks:** $M_{O2} = M_{max} \left( 1 - e^{-k_{reduce} \cdot PO_{2\_tissue}} \right)$. $CO_2$ production is mathematically linked via the Respiratory Quotient: $M_{CO2} = M_{O2} \times RQ$.

**Computational Solver:**
Because Blood $PO_2$, Tissue $PO_2$, $PCO_2$, and pH are inextricably interacting state variables governed by non-linear sigmoidal curves, the pipeline employs a **Triple-Coupled 1D-3D Picard Iteration Solver**. 
*   **1D Blood Traversal:** The algorithm traces blood down a Directed Acyclic Graph (DAG). At every voxel, it calculates the $Flux$ leaving the vessel and uses high-precision multi-variate numerical root-finders (`brentq`) to invert the coupled Bohr/Haldane equations, simultaneously determining the new depleted downstream Blood $PO_2$ and $PCO_2$.
*   **3D Tissue Matrices:** The leaked $Fluxes$ are deposited into the two distinct 3D sparse diffusion matrices, which are solved side-by-side via Preconditioned Conjugate Gradient (`cg`). The 1D blood tracing and 3D tissue solving alternate iteratively until the entire multi-species environment reaches steady-state thermodynamic convergence.

**Inputs, Outputs, and Boundary Conditions:**
*   **Inputs:** Solved vessel flow rates ($Q$), local hematocrit ($H_D$), Arterial $PO_2$ & $PCO_2$ baselines, Tissue diffusivity coefficients ($D_{O2}, D_{CO2}$), Endothelial Permeabilities ($P_{perm\_O2}, P_{perm\_CO2}$), Maximum metabolic rate ($M_{max}$), Respiratory Quotient ($RQ$), and Tissue Bicarbonate.
*   **Boundary Conditions:** Neumann Zero-Flux boundary conditions ($\frac{\partial P}{\partial n} = 0$) are assumed at the external edges of the bounding box. 
*   **Outputs:** Three complete 3D discrete grids of steady-state tissue Partial Pressures ($PO_2$ and $PCO_2$ in mmHg) and $pH$, exportable as a `.vti` heatmap.

### 1.3 Key Assumptions and Limitations
While the pipeline utilizes state-of-the-art non-linear thermodynamics, empirical in-vivo rheology, and multi-species coupling, it remains a mathematical abstraction bound by several key assumptions:
1.  **Laminar, Steady-State Flow:** Fluid mechanics are governed entirely by viscous forces (Reynolds number $\ll 1$). Inertial forces, pulsatility (heartbeats), and vasomotion over time are ignored. The system represents a "snapshot" of steady-state flow.
2.  **Rigid Cylinders:** Vessel walls are assumed perfectly inelastic. While FWHM measurements capture true biological radii and pre-capillary sphincters are explicitly modeled, the vessels do not dynamically expand or contract in response to solved blood pressure (compliance is ignored).
3.  **Homogeneous Diffusion:** The tissue diffusion coefficients ($D_{O2}$ and $D_{CO2}$) are assumed identical in all spatial directions (isotropic) and across all tissue types (homogeneous). The model does not differentiate between glomus cell clusters and interstitial stroma regarding diffusion speed.
4.  **Endothelial Permeability Uniformity:** The endothelial barrier is modeled using constant permeability coefficients ($P_{perm\_O2}, P_{perm\_CO2}$). In reality, capillary fenestrations and continuous endothelium vary across the Carotid Body, which could lead to localized permeability differences not captured by a global constant.

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

### 3.1 Modeling Capillary Collapse (Structural vs. Functional Vasculature)
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
*   **0D Fick Principle Mass Balance (Multi-Species):** Isolates a single zero-diffusion voxel with continuous blood flow and a metabolic sink linked by the Respiratory Quotient. Uses a high-precision multi-variate root-finder (`scipy.optimize.fsolve`) to mathematically invert the coupled Bohr/Haldane sigmoidal equations to find the exact theoretical $PO_2$ and $PCO_2$ targets. Proves the massive 3D Picard matrix solver perfectly navigates the interacting sigmoidal oxygen/carbon-dioxide unloading curves to converge to the exact analytical multivariate roots without numerical drift.
*   **Henderson-Hasselbalch Equilibrium:** Mathematically verifies the conversion of $PCO_2$ arrays to 3D $pH$ heatmaps against precise physiological baselines ($pH = 7.4$ at $PCO_2 = 40$) and extreme acidic bounds.
*   **1D Pure Diffusion:** Turns off metabolism and advection. Proves the 3D finite difference Laplacian matrix produces a flawless, straight-line linear concentration gradient along a single axis.
*   **Parabolic Reaction-Diffusion:** Forces the non-linear metabolic sink into a constant zero-order rate. Proves the steady-state solver's spatial output perfectly traces the exact theoretical mathematical parabola.
*   **Radial Point Source:** Places a single advective source in the center of the grid. Proves the spatial diffusion radiating outward strictly conforms to the expected inverse-radius curve ($C(r) \propto 1/r$).
*   **Transmural Exponential Decay:** Isolates a single vessel passing through an infinite tissue vacuum ($PO_2 = 0$). Computes the exact mathematical exponential decay curve ($PO_{out} = PO_{in} \cdot e^{-P_{perm} \cdot Area / \alpha Q}$) as a reference, but the test's assertions check only that the forced tissue sink drives $PO_2$ below $10^{-3}$ and that the solver returns an array of the correct length — the decay curve itself is not compared against the solver's blood-side output.
*   **Krogh Cylinder Radial Diffusion:** Places a single capillary in a 3D grid with constant metabolism. Confirms the 3D diffusion solver runs to completion on this geometry (`len(po2_num) == grid.n_cells`); it does not assert the solver's output against August Krogh's analytical cylinder equation, so this is a structural-completion check rather than a value-level validation of the Krogh profile.

### 4.3 Non-Newtonian Rheology & Multi-Species Validation
These tests validate the empirical mathematical functions and their integration into the iterative flow solver.
*   **Atomic Bohr/Haldane Curves:** Passes artificially high/low pH and $PCO_2$ values into the blood content functions to mathematically verify the Bohr shift (low pH definitively lowers $O_2$ affinity at $P_{50}$) and the Haldane shift (high $PO_2$ definitively lowers $CO_2$ carrying capacity), proving the atomic coupling equations are flawless prior to matrix assembly.
*   **Fåhræus–Lindqvist Curve:** Empirically tests the viscosity equations across massive arteries ($100 \mu m$) down to extreme capillaries ($3 \mu m$), confirming that viscosity correctly drops as vessels shrink, and then correctly spikes at the $8 \mu m$ inversion point where RBCs must deform.
*   **Plasma Skimming Mechanics & Mass Conservation:** Forces an asymmetric bifurcation (e.g., a $20 \mu m$ AVA vs a $5 \mu m$ capillary). Asserts that the AVA mathematically "steals" the RBCs, driving capillary hematocrit near zero, while strictly proving that total RBC flux is perfectly conserved across the node.
*   **Coupled Solver Convergence & Safety:** Builds a mock Direct Acyclic Graph (DAG) and intentionally introduces an infinite fluid loop via impossible pressures. Asserts that the topological sorter safely catches the cycle and prevents a fatal crash.
*   **Hematocrit-Weighted Perfusion:** Creates two identical tissue cells with equal volumetric blood flow, assigning one to receive pure skimmed plasma ($H=0.0$). Asserts that the Advective Source mathematically starves the plasma-filled cell of oxygen, proving the downstream physiological impact of upstream plasma skimming.

### 4.4 Hyperparameter Optimization (HPO) & Benchmarking Validation
These tests ensure the automated Optuna Bayesian optimization algorithms are mathematically robust and incapable of being "gamed" by extreme edge cases.
*   **Mock Pipeline Rigging:** Replaces the computationally heavy 3D morphological algorithms with lightning-fast, procedurally generated mathematical functions. This allows the Pytest suite to simulate thousands of Bayesian trials in under a second without requiring real biological data.
*   **Volumetric & Centricity Baseline:** Generates perfect mathematical primitives (e.g., straight cylinders). Asserts the benchmarking suite perfectly evaluates these objects to exactly `DSC = 1.0` and exactly `0.0` Centricity Error.
*   **Topological Cycle Extraction:** Procedurally generates a closed 3D mathematical Torus. Asserts the Betti number/Euler evaluator correctly flags exactly 1 fundamental graph loop, penalizing the AI for spiderwebs.
*   **Anti-Cheating Mass Penalty:** Simulates a thresholding trial that generates a tiny 1-voxel mask to artificially achieve a 100% "Confidence" and "Dust" score. Asserts the `Probability Yield` metric safely catches the volume destruction, crushing the Loss score with a 10,000x penalty.
*   **Optuna TPE Discovery Test:** Runs the full Optuna engine against a rigged mock pipeline where a specific parameter (e.g., `min_branch_length = 15`) is mathematically enforced as the global minimum. Asserts that the Bayesian Tree-structured Parzen Estimator (TPE) successfully navigates the multidimensional space to discover this hidden sweet spot without crashing.
*   **Early Stopping Resilience:** Asserts that the custom `EarlyStoppingCallback` safely intercepts and terminates trials when the Loss function mathematically flatlines.

---

## 5. Appendix

### 5.1 Dimensional Analysis
To guarantee the mathematical validity of the coupled 1D-3D multiphysics engine, the pipeline maintains strict dimensional consistency across all interacting equations.

#### 1D Fluid Dynamics (Poiseuille & WSS)
*   **Viscosity ($\mu_{app}$):** Pries-Secomb equations output in centipoise ($cP$), which is physically identical to **milliPascal-seconds ($mPa \cdot s$)**.
*   **Vessel Dimensions ($r, L$):** Measured natively in **micrometers ($\mu m$)**.
*   **Resistance ($R$):** $R = \frac{128 \mu L}{\pi d^4} \rightarrow \frac{mPa \cdot s \cdot \mu m}{\mu m^4} = \mathbf{\frac{mPa \cdot s}{\mu m^3}}$.
*   **Pressure ($\Delta P$):** Set via boundary conditions in **milliPascals ($mPa$)**.
*   **Fluid Flow ($Q$):** $Q = \frac{\Delta P}{R} \rightarrow \frac{mPa}{\frac{mPa \cdot s}{\mu m^3}} = \mathbf{\frac{\mu m^3}{s}}$.
*   **Wall Shear Stress ($\tau$):** $\tau = \frac{32 \mu Q}{\pi d^3} \rightarrow \frac{(mPa \cdot s) \cdot (\mu m^3 / s)}{\mu m^3} = \mathbf{mPa}$. *(The engine automatically divides by 1000 to export VTK arrays in standard Pascals).*

#### Blood Gas Content & Non-Newtonian Rheology
*   **Hematocrit ($H_D$):** A volumetric fraction. **Dimensionless.**
*   **Plasma Skimming:** Operates on ratios of Flow ($Q_1 / Q_{in}$) and Diameters ($d_1 / d_2$). **Dimensionless.**
*   **Gas Content ($C_{blood}$):** The Bohr/Haldane Hill equations scale the Partial Pressure ($PO_2$ in **mmHg**) by the solubility coefficient $\alpha$ (measured in $\mathbf{\frac{mmol/L}{mmHg}}$). Therefore, blood oxygen content is strictly measured in **mmol/L**.

#### Fully Coupled 3D Tissue Perfusion
To mathematically add diffusion, metabolic consumption, and trans-mural leakage together in the Picard sparse matrix ($A \cdot x = b$), they must all resolve to the exact same unit. In this pipeline, they all flawlessly resolve to **Mass Flux**: $\mathbf{\frac{\mu m^3}{s} \cdot \frac{mmol}{L}}$.

*   **Metabolic Sink ($M$):** 
    *   Maximum consumption ($M_{max}$) is set in $\frac{mmol}{L \cdot s}$. Voxel Volume ($V_{cell}$) is in $\mu m^3$. 
    *   *Resolved Sink Unit:* $M_{max} \times V_{cell} \rightarrow \mathbf{\left(\frac{\mu m^3}{s}\right) \cdot \left(\frac{mmol}{L}\right)}$.
*   **Trans-Mural Leakage ($Flux$):** 
    *   Endothelial Permeability ($P_{perm}$) is converted from $cm/s$ to **$\mu m/s$**. Vessel Surface Area ($2\pi rL$) is in **$\mu m^2$**. Solubility ($\alpha$) is **$\frac{mmol/L}{mmHg}$**. Pressure gradient ($\Delta PO_2$) is **mmHg**.
    *   *Resolved Leakage Unit:* $(\mu m/s) \cdot (\mu m^2) \cdot \left(\frac{mmol/L}{mmHg}\right) \cdot (mmHg) = \mathbf{\left(\frac{\mu m^3}{s}\right) \cdot \left(\frac{mmol}{L}\right)}$.
*   **3D Tissue Diffusion ($A_{diff}$):**
    *   Diffusivity ($D_{tissue}$) is converted from $m^2/s$ to **$\mu m^2/s$**. The 7-point stencil conductance ($D \cdot \frac{Area}{Length}$) resolves to **$\frac{\mu m^3}{s}$**. The engine explicitly scales the diffusion matrix by the solubility coefficient $\alpha$ ($\frac{mmol/L}{mmHg}$).
    *   *Resolved Diffusion Unit:* $A_{diff} \cdot \Delta PO_2 \rightarrow \left(\frac{\mu m^3}{s} \cdot \frac{mmol/L}{mmHg}\right) \cdot (mmHg) = \mathbf{\left(\frac{\mu m^3}{s}\right) \cdot \left(\frac{mmol}{L}\right)}$.