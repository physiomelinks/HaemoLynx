# Methodology: Computational Haemodynamics and Mass Transport Modelling

This section details the mathematical and computational framework utilized to model the steady-state haemodynamics (pressure, flow, and hematocrit partitioning) and the subsequent mass transport (oxygen and carbon dioxide perfusion) within the reconstructed microvascular networks of the Carotid Body.

## 1. Network Extraction and Graph Representation

The continuous 3D binary segmentation mask of the microvasculature is abstracted into a discrete, spatially-embedded directed multigraph $\mathcal{G} = (V, E)$, where $V$ represents the set of vascular bifurcations and terminations (nodes) and $E$ represents the set of connecting vessel segments (edges). 

### 1.1 Skeletonization and Topological Mapping
The binary mask undergoes 3D morphological thinning (skeletonization) to extract the 1-voxel wide centerlines of the vascular network. The resulting skeleton is parsed into a graph structure where voxels with exactly one neighbor are classified as terminal nodes, voxels with exactly two neighbors form the internal segments of edges, and voxels with three or more neighbors are classified as junction nodes.

### 1.2 Physical Length and EDT-Derived Radius Assignment
For each edge $e \in E$ connecting node $i$ to node $j$, the true physical centerline length ($L_e$, units: $\mu m$) is calculated by integrating the Euclidean distances between sequential voxel coordinates along the extracted 3D spline:
$$ L_e = \sum_{k=1}^{N-1} || \mathbf{x}_{k+1} - \mathbf{x}_k ||_2 $$
Where:
*   $N$ is the total number of voxels constituting the centerline of edge $e$.
*   $\mathbf{x}_k = (x_k, y_k, z_k)$ is the 3D spatial coordinate vector of the $k$-th voxel in the vessel segment, scaled by the physical voxel dimensions ($\mu m$/voxel).
*   $|| \cdot ||_2$ denotes the $L^2$-norm (Euclidean distance).

To mitigate mathematical instability caused by artificial surface irregularities in the binary mask, the vessel radius ($r_e$, units: $\mu m$) is derived using the 3D Euclidean Distance Transform (EDT). The EDT computes the orthogonal distance from every centerline voxel $\mathbf{x}_k$ to the nearest anatomical boundary within the binary mask. The effective scalar radius $r_e$ assigned to the entire segment $e$ is defined as the median EDT value along the centerline, providing a robust, localized morphological measurement.

## 2. Mathematical Formulation of Blood Flow

Steady-state blood flow through the microvascular network is modelled using a 1D lumped-parameter electrical circuit analogy, governed by Poiseuille’s Law and Kirchhoff’s Current Law.

### 2.1 Hydrodynamic Resistance (Poiseuille's Law)
The hydrodynamic resistance ($R_e$, units: $mmHg \cdot s \cdot \mu m^{-3}$) of each vessel segment $e$ is calculated assuming fully developed, laminar flow of an incompressible Newtonian fluid in a rigid cylindrical tube:
$$ R_e = \frac{8 \mu_{app, e} L_e}{\pi r_e^4} $$
Where:
*   $\mu_{app, e}$ is the effective apparent viscosity of blood within segment $e$ (units: $mmHg \cdot s$).
*   $L_e$ is the physical length of the segment ($\mu m$).
*   $r_e$ is the effective radius of the segment ($\mu m$).

The volumetric flow rate ($Q_{ij}$, units: $\mu m^3 \cdot s^{-1}$) through edge $e$ directed from node $i$ to node $j$ is strictly proportional to the hydrostatic pressure drop:
$$ Q_{ij} = \frac{P_i - P_j}{R_e} = C_e (P_i - P_j) $$
Where:
*   $P_i$ and $P_j$ are the hydrostatic fluid pressures at nodes $i$ and $j$ respectively (units: $mmHg$).
*   $C_e = R_e^{-1}$ is the hydrodynamic conductance of the segment (units: $\mu m^3 \cdot s^{-1} \cdot mmHg^{-1}$).

### 2.2 Conservation of Mass (Kirchhoff’s Current Law)
At every internal junction node $i \in V$ (where no fluid is created or destroyed), the conservation of mass dictates that the net volumetric flow must equal zero:
$$ \sum_{j \in \mathcal{N}(i)} Q_{ij} = 0 \implies \sum_{j \in \mathcal{N}(i)} C_{ij} (P_i - P_j) = 0 $$
Where:
*   $\mathcal{N}(i)$ is the set of all nodes immediately adjacent to node $i$ via a connecting edge.
*   $C_{ij}$ is the hydrodynamic conductance of the specific edge connecting node $i$ to node $j$.

## 3. Boundary Conditions and System Assembly

To obtain a unique solution for the pressure field over the entire domain, Dirichlet boundary conditions are applied at the network's external interfaces.

### 3.1 Anatomical Boundary Identification
1.  **Arterial Inlets ($V_{in}$):** Terminal nodes connected to the largest identified feeding vessels (e.g., the primary Carotid artery branch) are assigned a defined, constant systemic Mean Arterial Pressure ($P_{in}$, units: $mmHg$).
2.  **Venous Outlets ($V_{out}$):** The remaining terminal nodes located at the periphery of the network are assigned a Central Venous Pressure ($P_{out}$, units: $mmHg$).

### 3.2 Matrix Formulation (The Graph Laplacian)
The system of coupled linear equations representing Kirchhoff's Current Law for all unknown nodal pressures is formulated as a sparse matrix equation:
$$ \mathbf{L} \mathbf{P} = \mathbf{B} $$
Where:
*   $\mathbf{P}$ is the column vector of size $|V| \times 1$ representing the unknown hydrostatic pressures at each node.
*   $\mathbf{L}$ is the heavily weighted, symmetric Graph Laplacian matrix of size $|V| \times |V|$. The elements of $\mathbf{L}$ are populated based on the hydrodynamic conductances:
    *   **Off-diagonal elements:** $L_{ij} = -C_{ij}$ if a direct vessel edge exists connecting node $i$ and node $j$, else $0$.
    *   **Diagonal elements:** $L_{ii} = \sum_{j \in \mathcal{N}(i)} C_{ij}$, representing the sum of conductances of all edges connected to node $i$.
*   $\mathbf{B}$ is the Right-Hand Side column vector of size $|V| \times 1$ containing the boundary constraints.

For nodes with defined Dirichlet boundary conditions ($i \in V_{in} \cup V_{out}$), the linear equation is explicitly modified to enforce the known pressure constraint. The corresponding row $i$ in matrix $\mathbf{L}$ is zeroed out, the diagonal element $L_{ii}$ is set to $1$, and the corresponding entry in the vector $\mathbf{B}_i$ is set to the known boundary pressure ($P_{in}$ or $P_{out}$). For all internal, unconstrained nodes, $\mathbf{B}_i = 0$.

This highly sparse, structurally symmetric system is solved computationally using a direct solver (e.g., `scipy.sparse.linalg.spsolve`). If the matrix becomes ill-conditioned due to massive geometric variance resulting in numerical instability, an iterative Least Squares (LSQR) solver is employed as a fallback to ensure the solution stringently adheres to the Maximum Principle (where no internal node pressure exceeds the bounds of $[P_{out}, P_{in}]$).

## 4. Non-Linear In Vivo Rheology

Blood in the microcirculation behaves as a non-Newtonian fluid. The apparent viscosity ($\mu_{app, e}$) within a specific edge $e$ depends dynamically on both the effective vessel diameter ($D_e = 2 r_e$, units: $\mu m$) and the local discharge hematocrit ($H_{D, e}$, the volume fraction of erythrocytes within the segment, unitless).

### 4.1 In Vivo Apparent Viscosity (Fåhræus–Lindqvist Effect)
The pipeline employs the empirical in vivo viscosity formulation derived by Pries et al. (1992):
$$ \mu_{app, e} = \mu_{plasma} \left[ 1 + (\mu_{0.45, e} - 1) \frac{(1 - H_{D, e})^C - 1}{(1 - 0.45)^C - 1} \left(\frac{D_e}{D_e - 1.1}\right)^2 \right] \left(\frac{D_e}{D_e - 1.1}\right)^2 $$
Where:
*   $\mu_{plasma}$ is the constant dynamic viscosity of blood plasma (typically $\approx 1.2 \ cP$ or $mmHg \cdot s$).
*   $\mu_{0.45, e}$ is the relative apparent viscosity of blood evaluated at a standardized systemic hematocrit of $0.45$ discharging through a glass tube of identical diameter $D_e$.
*   $C$ is a shape parameter characterizing the dependence of relative viscosity on hematocrit, which is formulated empirically as a function of the vessel diameter $D_e$.

### 4.2 Phase Separation and Plasma Skimming
At diverging vascular bifurcations, erythrocytes do not partition proportionately to the bulk fluid flow. Due to the presence of a cell-free plasma layer near the endothelial wall, a disproportionately higher fraction of erythrocytes enters the daughter branch receiving the higher bulk volumetric flow rate. 

For a bifurcation where a parent vessel branches into two daughter vessels ($\alpha$ and $\beta$), the fractional erythrocyte flux ($FQ_{E, \alpha}$) entering daughter branch $\alpha$ is modeled as a non-linear logit function of the fractional bulk blood flow ($FQ_{B, \alpha}$):
$$ \text{logit}(FQ_{E, \alpha}) = A + B \cdot \text{logit} \left( \frac{FQ_{B, \alpha} - X_0}{1 - 2X_0} \right) $$
Where:
*   $\text{logit}(x) = \ln(x / (1-x))$ is the log-odds function mapping probabilities $(0,1)$ to $(-\infty, \infty)$.
*   $FQ_{B, \alpha} = Q_\alpha / Q_P$ is the ratio of volumetric flow entering branch $\alpha$ relative to the total flow in the parent vessel.
*   $FQ_{E, \alpha} = (Q_\alpha H_{D, \alpha}) / (Q_P H_{D, P})$ is the resulting ratio of erythrocyte flux entering branch $\alpha$.

The empirical parameters $A$, $B$, and $X_0$ are formulated from the in vivo microvascular studies of Pries et al. (1989, 1990) and are calculated locally for every bifurcation based on the parent vessel diameter ($D_P$), the daughter vessel diameters ($D_\alpha, D_\beta$), and the parent discharge hematocrit ($H_{D,P}$):

*   **$A$ (Asymmetry Parameter):** Determines the baseline bias in erythrocyte distribution due to the unequal physical sizes of the daughter vessels. It is an explicit function of the diameter ratio of the daughter branches:
    $$ A = -13.29 \left( \frac{D_\alpha^2 / D_\beta^2 - 1}{D_\alpha^2 / D_\beta^2 + 1} \right) \frac{1 - H_{D,P}}{D_P} $$
*   **$B$ (Shape Parameter):** Characterizes the non-linearity of the phase separation curve, reflecting the physical exclusion of erythrocytes from the cell-free marginal plasma layer. It depends inversely on the parent diameter:
    $$ B = 1 + \frac{6.98 (1 - H_{D,P})}{D_P} $$
*   **$X_0$ (Minimum Fractional Flow):** Defines the critical threshold of fractional bulk flow ($FQ_{B}$) below which a daughter branch receives exclusively plasma (zero erythrocytes). It is primarily influenced by the physical width of the plasma layer relative to the parent branching geometry:
    $$ X_0 = \frac{0.964 (1 - H_{D,P})}{D_P} $$

### 4.3 Computational Integration: The Picard Iterative Solver
Because the apparent viscosity ($\mu_{app}$) dictates the segment resistance ($\mathbf{R}$), which dictates the bulk flow field ($\mathbf{Q}$), which drives phase separation determining local hematocrit ($\mathbf{H}_D$)—which subsequently re-defines the viscosity—the system is tightly mathematically coupled. It is resolved computationally using a fixed-point Picard solver:

1.  **Initialization ($k=0$):** A uniform systemic hematocrit ($H_D = 0.45$) is assumed across all edges. The initial linear system $\mathbf{L}^{(0)} \mathbf{P}^{(0)} = \mathbf{B}$ is formulated and solved.
2.  **Flow Update:** Directed edge volumetric flows $\mathbf{Q}^{(k)}$ are computed using the resulting pressure field $\mathbf{P}^{(k)}$.
3.  **Advection:** Erythrocytes are routed downstream via the phase separation logit equations to compute a new spatial discharge hematocrit distribution $\mathbf{H}_D^{(k+1)}$ for all edges.
4.  **Viscosity Update:** New apparent viscosities and hydrodynamic resistances $\mathbf{R}^{(k+1)}$ are computed using $\mathbf{H}_D^{(k+1)}$.
5.  **Iteration:** The Laplacian matrix $\mathbf{L}^{(k+1)}$ is rebuilt, and the new pressure field $\mathbf{P}^{(k+1)}$ is solved.

Convergence is achieved when the Chebyshev norm ($L^\infty$-norm) of the relative change in the nodal pressure field falls below a strict tolerance threshold ($\epsilon$, typically set to $10^{-4}$):
$$ \max_{i \in V} \left| \frac{P_i^{(k+1)} - P_i^{(k)}}{P_i^{(k)}} \right| < \epsilon $$

## 5. Perfusion and Mass Transport Modelling

Once the haemodynamic steady-state is computationally converged, the pipeline physically couples the 1D discrete vascular network multigraph to a 3D continuous tissue domain to model the spatio-temporal transport of metabolic gases (O$_2$ and CO$_2$).

### 5.1 The 3D Reaction-Diffusion Equation
The continuous extra-vascular parenchymal space is discretized into a 3D Cartesian grid. For a given chemical species $x \in \{\text{O}_2, \text{CO}_2\}$, the steady-state partial pressure field $P_x(\mathbf{r})$ (units: $mmHg$) is governed by the Reaction-Diffusion equation:
$$ D_x \alpha_x \nabla^2 P_x(\mathbf{r}) - M_x(P_{\text{O}_2}(\mathbf{r})) + \Phi_x(\mathbf{r}) = 0 $$
Where:
*   $\mathbf{r} = (x,y,z)$ is the 3D spatial coordinate vector within the continuous tissue grid.
*   $D_x$ is the effective diffusivity of species $x$ in the biological tissue (units: $\mu m^2 \cdot s^{-1}$).
*   $\alpha_x$ is the specific tissue solubility coefficient (units: $mmol \cdot L^{-1} \cdot mmHg^{-1}$), converting partial pressure to concentration via Henry's Law ($C_x = \alpha_x P_x$).
*   $\nabla^2$ is the Laplace operator representing spatial diffusion, discretized computationally using a standard 7-point 3D finite difference stencil.
*   $M_x(P_{\text{O}_2}(\mathbf{r}))$ is the local metabolic consumption (for O$_2$) or production (for CO$_2$) rate (units: $mmol \cdot L^{-1} \cdot s^{-1}$).
*   $\Phi_x(\mathbf{r})$ is the transvascular source flux representing gas delivered or cleared by the local capillary network (units: $mmol \cdot L^{-1} \cdot s^{-1}$).

### 5.2 Transvascular Flux and Hemoglobin Saturation
The source term $\Phi_x(\mathbf{r})$ explicitly couples the 1D graph edge properties to the 3D grid. The total intra-vascular oxygen concentration ($C_{\text{O}_2, e}$, units: $mmol \cdot L^{-1}$) inside a specific vessel edge $e$ is the sum of freely dissolved plasma oxygen and hemoglobin-bound oxygen, defined by the non-linear Hill equation:
$$ C_{\text{O}_2, e} = \alpha_{\text{O}_2} P_{\text{O}_2, e} + H_{D, e} \cdot C_{Hb} \cdot \left( \frac{P_{\text{O}_2, e}^n}{P_{\text{O}_2, e}^n + P_{50}^n} \right) $$
Where:
*   $P_{\text{O}_2, e}$ is the intra-vascular partial pressure of oxygen in segment $e$ ($mmHg$).
*   $H_{D,e}$ is the converged local discharge hematocrit solved in Section 4.3 (unitless).
*   $C_{Hb}$ is the maximum oxygen-binding capacity of pure erythrocytes (units: $mmol \cdot L^{-1}$).
*   $n$ is the empirical Hill coefficient determining the cooperativity of oxygen binding.
*   $P_{50}$ is the partial pressure at which hemoglobin is exactly 50% saturated ($mmHg$).

The physical mass flux across the endothelial barrier from vessel $e$ into the surrounding discrete tissue voxel $i$ is calculated as:
$$ \Phi_{x, i} = K_{x, e} S_e (P_{x, e} - P_{x, i}) $$
Where:
*   $P_{x, i}$ is the current extra-vascular partial pressure of species $x$ in voxel $i$.
*   $S_e$ is the physical surface area of the vessel segment geometrically embedded within voxel $i$ (units: $\mu m^2$).
*   $K_{x, e}$ is the effective mass transfer permeability coefficient accounting for the endothelial wall and boundary layers (units: $mmol \cdot L^{-1} \cdot s^{-1} \cdot \mu m^{-2} \cdot mmHg^{-1}$). 

Because $H_{D,e}$ mathematically limits the bounding capacity $C_{Hb}$, vessels experiencing severe plasma skimming (low $H_{D,e}$) natively deliver drastically reduced transvascular flux $\Phi_x$, directly driving local tissue hypoxia.

### 5.3 Non-Linear Metabolic Consumption and Multi-Species Coupling
Oxygen consumption by the glomus cells and parenchyma follows Michaelis-Menten-like kinetics, slowing down biologically as the local tissue becomes hypoxic:
$$ M_{\text{O}_2}(P_{\text{O}_2}) = M_{max} \left[ 1 - \exp(-k_{reduce} \cdot \alpha_{\text{O}_2} P_{\text{O}_2}) \right] $$
Where:
*   $M_{max}$ is the maximum, unconstrained metabolic oxygen consumption rate under perfect normoxia (units: $mmol \cdot L^{-1} \cdot s^{-1}$).
*   $k_{reduce}$ is a calibrated exponential reduction constant governing the physiological shutdown rate under hypoxic stress.

Carbon dioxide production is explicitly mathematically coupled to the local oxygen consumption via the dimensionless Respiratory Quotient ($RQ$):
$$ M_{\text{CO}_2} = RQ \cdot M_{\text{O}_2}(P_{\text{O}_2}) $$

### 5.4 Computational Resolution of the Perfusion Field
Because the metabolic consumption term $M_x(P_{\text{O}_2})$ is highly non-linear and the species are coupled (CO$_2$ production relies entirely on the solved O$_2$ field), the discretized 3D Reaction-Diffusion system cannot be solved as a singular linear matrix. 

Instead, it is resolved using an iterative finite-difference implicit solver (e.g., Picard iteration evaluated over the full 3D tissue grid). The computational solver updates the 3D tissue partial pressures $P_x(\mathbf{r})$ iteratively. In each iteration step, the transvascular source fluxes $\Phi_x$ are re-evaluated using the current tissue estimates $P_{x, i}$, and the local consumption rates $M_x$ are dynamically updated. The implicit iteration halts when the spatial residual norm (the difference between successive 3D pressure maps) falls below a predefined tolerance, yielding the final, high-resolution spatial gradients of tissue oxygenation and hypercapnia.