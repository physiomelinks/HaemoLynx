# Chapter X (Methods) — §X: Computational modelling of carotid body microvascular haemodynamics and tissue gas transport

> **Status:** Skeleton. Headings, equations, nomenclature and tables are complete and thesis-ready.
> Prose is left as `[STUB: …]` markers for the author to write.
>
> **Source:** Derived from `examples/cb_image_to_model_modelling_capabilities_conceptual_summary_v2.md`.
> Section numbers in the form *(doc §n.n)* refer back to that file so each stub can be written against its source.
>
> **Conventions adopted here:**
> - Equations numbered `(X.1)`…`(X.36)`; renumber to your chapter number on adoption.
> - Symbol collisions in the source document have been resolved — see §X.0 Nomenclature. **Do not revert to the source document's symbols**; four of them are triple-booked there.
> - No function names, file names, or source code. Boxed algorithms only.
> - UK spelling; past tense for procedure, present tense for what equations state.

---

## §X.0 Nomenclature

> **Note on resolved collisions.** The source document reuses `C` for hydraulic conductance, blood gas content, *and* the Pries–Secomb shape parameter; `A` for the skimming asymmetry parameter, vessel surface area, *and* the system matrix; `D` for both vessel diameter and diffusive conductance; `n` for the Hill coefficient, branch order, *and* grid dimension; and `L` for both segment length and the graph Laplacian. The scheme below resolves all five. Reproduce this table in the thesis (or in a front-matter nomenclature list).

### Geometry

| Symbol | Quantity | Units |
|---|---|---|
| $d$ | Vessel segment diameter | µm |
| $d_1, d_2$ | Unconstricted / constricted diameter | µm |
| $r$ | Vessel radius ($d/2$) | µm |
| $\ell$ | Vessel segment centreline length | µm |
| $s$ | Arc-length position along a vessel segment | µm |
| $\ell_s$ | Constriction zone length | µm |
| $\beta$ | Topological branch order | — |
| $a_v$ | Vessel surface area within a tissue grid cell | µm² |
| $V_c$ | Tissue grid cell volume | µm³ |
| $\Delta x, \Delta y, \Delta z$ | Grid cell dimensions | µm |
| $n_x, n_y, n_z$ | Grid dimensions (cell counts) | — |
| $N$ | Total number of grid cells | — |

### Haemodynamics

| Symbol | Quantity | Units |
|---|---|---|
| $p$ | Nodal pressure | mPa |
| $Q$ | Volumetric flow rate | µm³ s⁻¹ |
| $R$ | Hydraulic resistance | mPa s µm⁻³ |
| $g$ | Hydraulic conductance, $g = 1/R$ | µm³ mPa⁻¹ s⁻¹ |
| $\mathbf{G}$ | Conductance matrix | — |
| $\mathbf{L}$ | Graph Laplacian matrix | — |
| $\mu$ | Apparent dynamic viscosity | mPa s |
| $\mu_{45}$ | Relative apparent viscosity at $H_D = 0.45$ | — |
| $\mu_{\mathrm{rel}}$ | Relative apparent viscosity at local $H_D$ | — |
| $\mu_{\mathrm{pl}}$ | Plasma viscosity | mPa s |
| $\tau_w$ | Wall shear stress | Pa |

### Rheology

| Symbol | Quantity | Units |
|---|---|---|
| $H_D$ | Discharge haematocrit | — |
| $\kappa$ | Pries–Secomb haematocrit shape parameter | — |
| $\phi_Q$ | Fraction of parent flow entering a daughter branch | — |
| $\phi_E$ | Fraction of parent RBC flux entering a daughter branch | — |
| $\phi_0$ | Plasma-skimming flow-fraction threshold | — |
| $\lambda_A$ | Skimming asymmetry parameter | — |
| $\lambda_B$ | Skimming steepness parameter | — |

### Blood gas chemistry and tissue transport

| Symbol | Quantity | Units |
|---|---|---|
| $P_{\mathrm{O_2}}, P_{\mathrm{CO_2}}$ | Partial pressures (blood or tissue, subscripted) | mmHg |
| $P_{50}$ | Partial pressure at 50% haemoglobin saturation | mmHg |
| $S_{\mathrm{O_2}}$ | Haemoglobin oxygen saturation | — |
| $\nu$ | Hill coefficient | — |
| $C_{\mathrm{O_2}}, C_{\mathrm{CO_2}}$ | Blood gas content (dissolved + bound) | mmol L⁻¹ |
| $C_{\mathrm{Hb}}$ | Maximal haemoglobin O₂ binding capacity (per unit RBC) | mmol L⁻¹ |
| $c$ | Tissue gas concentration field | mmol L⁻¹ |
| $\alpha_{\mathrm{O_2}}, \alpha_{\mathrm{CO_2}}$ | Henry's-law solubility coefficients | mmol L⁻¹ mmHg⁻¹ |
| $\sigma_{\mathrm{O_2}}, \sigma_{\mathrm{CO_2}}$ | Tissue diffusion coefficients | µm² s⁻¹ |
| $\mathcal{P}_{\mathrm{O_2}}, \mathcal{P}_{\mathrm{CO_2}}$ | Endothelial permeability coefficients | µm s⁻¹ |
| $J$ | Transmural flux | mmol s⁻¹ |
| $\Sigma$ | Advective source/sink term | mmol L⁻¹ s⁻¹ |
| $M$ | Metabolic consumption rate | mmol L⁻¹ s⁻¹ |
| $M_{\max}$ | Maximum metabolic rate | mmol L⁻¹ s⁻¹ |
| $k_M$ | Metabolic saturation constant | mmHg⁻¹ |
| $\mathrm{RQ}$ | Respiratory quotient | — |
| $pK_a$ | Carbonic acid dissociation constant | — |
| $[\mathrm{HCO_3^-}]$ | Bicarbonate buffer concentration | mmol L⁻¹ |

### Numerical

| Symbol | Quantity | Units |
|---|---|---|
| $\mathbf{A}$ | Discretised system matrix | — |
| $\mathbf{b}$ | Right-hand-side vector | — |
| $D_x, D_y, D_z$ | Diffusive conductances between adjacent cells | µm³ s⁻¹ |
| $\gamma$ | Pseudo-washout stabilisation coefficient | — |
| $\varepsilon$ | Convergence tolerance | — |

---

## §X.1 Overview and modelling rationale

`[STUB: One page. State the problem — from a segmented 3D microscopy volume of carotid body vasculature, recover steady-state blood flow, spatially varying rheology, tissue PO2, and coupled multi-species gas transport. Name the four solved quantities. State that the model is modular in three respects: diameter assignment mode, constriction mode, and vascular–tissue coupling tier. Forward-reference §X.2–X.8. Close with a sentence on scope: this section covers the modelling and solution machinery only; upstream image processing and graph extraction are described in §[earlier section].]` (doc §1)

`[STUB: Justify the modelling approach in one paragraph — why a 1D network coupled to a 3D tissue grid, rather than full 3D CFD. Key arguments: Re < 0.01 in the microvasculature makes Poiseuille flow an excellent approximation; the network topology, not the intra-luminal velocity field, is what determines delivery heterogeneity; and full 3D CFD over ~10^4 segments is computationally intractable at the required tissue resolution.]`

> **FIGURE X.1** — Pipeline schematic. Segmented volume → vascular graph → diameter assignment → network flow solve → coupled rheology → tissue grid overlay → gas transport solve. Annotate with the section number that describes each stage.

---

## §X.2 Model domain and geometric representation

### §X.2.1 Vascular graph representation

`[STUB: Brief — one or two paragraphs. The network is represented as a multigraph in which nodes are junctions or vessel termini and edges are vessel segments carrying diameter, length and centreline geometry as attributes. Cross-reference the extraction method in §[earlier section] rather than repeating it. State the domain: carotid body microvasculature, approximate volume, approximate node and edge counts for the datasets used.]` (doc §1, §2.2)

### §X.2.2 Vessel diameter assignment

`[STUB: Diameter is the single most influential geometric parameter, entering resistance at fourth power (Eq. X.2). State the three available assignment modes and which was used for the results reported in this thesis.]` (doc §4.1)

| Mode | Basis |
|---|---|
| EDT (default) | Euclidean distance transform of the binary vessel mask, sampled at each centreline voxel, per-edge median |
| FWHM | Per-edge measurement from the raw intensity volume by Gaussian fitting of transverse profiles |
| Constant | Uniform diameter applied to all segments |

`[Default corrected: the pipeline default is EDT, not FWHM. On 1330 edges of the reference
subvolume EDT covered 100% with a median diameter of 6.37 µm; FWHM covered 76.5% with a median
of 8.20 µm and an unphysical maximum of 39.16 µm, the documented failure mode of Gaussian
fitting against the flat in-vessel plateau of a probability field. Pearson r between the two
was +0.245 — they do not measure the same thing, and that disagreement needs reporting.]`

`[STUB: State that EDT radii within 3.73 µm (two voxels) of a junction node are discarded before
the per-edge median, because the distance transform there returns the junction's inscribed
sphere rather than the vessel's. Measured effect on the reference subvolume: ~8% on segment
resistance, with 61% of segments too short to trim at all and therefore retaining the bias —
report that fraction. Segments that cannot be trimmed are tagged rather than discarded, since
dropping them would bias the distribution towards long vessels.]`

`[STUB: Describe the FWHM ray-casting procedure. Sample points along the centreline at fixed spacing; at each, cast transverse rays perpendicular to the local centreline tangent; extract the intensity profile; fit a Gaussian by least squares; take the FWHM of the fit as the local diameter; take the per-edge diameter as the median of all valid local measurements. State the acceptance criteria (minimum fit R², maximum centre offset) and what fraction of measurements were rejected in your data.]` (doc §4.2)

For a Gaussian intensity profile of standard deviation $\sigma_g$, the diameter estimate is

$$d = 2\sqrt{2\ln 2}\;\sigma_g \tag{X.1}$$

`[STUB: State the fallback chain explicitly — where a per-edge measurement is unavailable or non-positive, diameter falls back to the branch-order model of §X.2.3, and thence to a default. REPORT THE FRACTION OF EDGES THAT TOOK EACH PATH IN YOUR DATASET. This is an examiner question; answer it pre-emptively.]`

> **FIGURE X.2** — FWHM ray-casting geometry: centreline sample point, transverse ray fan, extracted intensity profile with fitted Gaussian and FWHM annotated.

### §X.2.3 Branch-order diameter model

`[STUB: Used where per-edge measurement is unavailable. Diameters are assigned from topological branch order using a three-point boundary-fitted exponential, anchored at arterial inlet, capillary bed, and venous outlet. Note that the venous anchor exceeds the arterial anchor, reflecting the lower pressure and greater compliance of venules.]` (doc §4.3)

For branch orders on the arterial side ($\beta \leq \beta_{\mathrm{mid}}$):

$$d(\beta) = d_{\mathrm{mid}} \left( \frac{d_{\mathrm{start}}}{d_{\mathrm{mid}}} \right)^{\frac{\beta_{\mathrm{mid}} - \beta}{\beta_{\mathrm{mid}} - \beta_{\mathrm{start}}}} \tag{X.2}$$

and on the venous side ($\beta > \beta_{\mathrm{mid}}$):

$$d(\beta) = d_{\mathrm{mid}} \left( \frac{d_{\mathrm{end}}}{d_{\mathrm{mid}}} \right)^{\frac{\beta - \beta_{\mathrm{mid}}}{\beta_{\mathrm{end}} - \beta_{\mathrm{mid}}}} \tag{X.3}$$

`[STUB: State explicitly that Murray's law (cubic branching ratio) was not used, and why the heuristic exponential was preferred. Note this is a fallback path only — its influence on results is bounded by the fraction of edges reported in §X.2.2.]`

### §X.2.4 Constriction geometry

`[STUB: Motivate — pericytes and pre-capillary sphincters impose localised diameter reductions; because resistance scales as d^-4, a 50% local constriction raises local resistance sixteenfold, making these structures a plausible mechanism of capillary-level flow control. In the carotid body, intimal cushions at branch origins serve an analogous throttling role.]` (doc §4.4)

Two constriction modes are available. In **sphincter mode**, a single constriction is placed at the proximal end of each segment, with a trapezoidal ramp–hold–ramp profile over a zone of length $\ell_s$:

$$d(s) = \begin{cases}
d_1 + (d_2 - d_1)\,\dfrac{s}{0.25\,\ell_s} & 0 \leq s < 0.25\,\ell_s \\[8pt]
d_2 & 0.25\,\ell_s \leq s < 0.75\,\ell_s \\[8pt]
d_2 + (d_1 - d_2)\,\dfrac{s - 0.75\,\ell_s}{0.25\,\ell_s} & 0.75\,\ell_s \leq s \leq \ell_s \\[8pt]
d_1 & s > \ell_s
\end{cases} \tag{X.4}$$

In **periodic mode**, constrictions repeat at fixed spacing $\Lambda$ along the segment. With phase $\varphi = s \bmod \Lambda$, ramp length $\ell_r$ and hold length $\ell_h$:

$$d(\varphi) = \begin{cases}
d_1 + (d_2 - d_1)\,\dfrac{\varphi}{\ell_r} & 0 \leq \varphi < \ell_r \\[8pt]
d_2 & \ell_r \leq \varphi < \ell_r + \ell_h \\[8pt]
d_2 + (d_1 - d_2)\,\dfrac{\varphi - \ell_r - \ell_h}{\ell_r} & \ell_r + \ell_h \leq \varphi < 2\ell_r + \ell_h \\[8pt]
d_1 & \varphi \geq 2\ell_r + \ell_h
\end{cases} \tag{X.5}$$

`[STUB: Justify the trapezoidal rather than step profile — a discontinuous diameter change would impose an unphysical infinite pressure gradient, and the ramp–hold–ramp form approximates constriction geometry observed in electron microscopy of pericyte-invested capillaries.]`

Two constriction types are applied, distinguished by location and ratio $d_2/d_1$:

| Type | Location | Ratio |
|---|---|---|
| Intimal cushion | Branch order $\beta_{01}$ (carotid origin) | 0.60 |
| Pre-capillary sphincter | Topological midpoint, offset by one branch order | 0.50 |

`[STUB: Where measured diameters are in use, the measured value becomes d1 and d2 is derived by preserving the constriction ratio from the branch-order model. State the minimum ratio clamp and its purpose.]`

> **FIGURE X.3** — Constriction diameter profiles $d(s)$ for sphincter and periodic modes, with the resulting resistance-per-unit-length $128\mu(s)/\pi d(s)^4$ overlaid on a secondary axis to show how sharply resistance concentrates in the constricted zone.

---

## §X.3 Haemodynamic model

### §X.3.1 Segment resistance

`[STUB: State the assumptions embedded in Hagen–Poiseuille flow — rigid walls, fully developed laminar flow, Newtonian fluid at the first pass, steady state, circular cross-section, no-slip, incompressible. Justify their applicability at Re < 0.01 in the microvasculature, and flag the Newtonian assumption as the substantive one, addressed in §X.4.]` (doc §2.1)

Volumetric flow through a rigid cylindrical segment under a pressure difference $\Delta p$ is

$$Q = \frac{\pi\, d^4\, \Delta p}{128\, \mu\, \ell} \tag{X.6}$$

giving a hydraulic resistance

$$R = \frac{128\, \mu\, \ell}{\pi\, d^4} \tag{X.7}$$

### §X.3.2 Variable-diameter segment resistance

`[STUB: Where a constriction profile is active, d varies along the segment and Eq. X.7 does not apply. The segment is treated as a series of infinitesimal cylindrical elements and the resistance per unit length integrated along the centreline. State the quadrature rule and sample count, and note that the local viscosity is evaluated at the local diameter.]` (doc §4.4.4)

$$R_{\mathrm{tot}} = \int_0^{\ell} \frac{128\,\mu(s)}{\pi\, d(s)^4}\, \mathrm{d}s \tag{X.8}$$

`[STUB: Note the guard: segments of non-positive length are assigned infinite resistance, excluding them from the flow solution.]`

### §X.3.3 Network flow: graph Laplacian formulation

`[STUB: Introduce the resistor-network analogy — pressure as potential, flow as current, mass conservation at each junction as Kirchhoff's current law. Justify why conductances rather than resistances are assembled: parallel segments combine additively in conductance, which the sparse assembly handles natively.]` (doc §2.2)

Mass conservation at each interior node $i$ requires

$$\sum_{j \in \mathcal{N}(i)} g_{ij}\,(p_i - p_j) = 0, \qquad g_{ij} = \frac{1}{R_{ij}} \tag{X.9}$$

Assembling the symmetric conductance matrix $\mathbf{G}$ and forming the graph Laplacian

$$\mathbf{L} = \mathrm{diag}\!\left( \sum_j g_{ij} \right) - \mathbf{G} \tag{X.10}$$

Partitioning nodes into a known (Dirichlet boundary) set $K$ and an unknown interior set $U$,

$$\mathbf{L} = \begin{bmatrix} \mathbf{L}_{UU} & \mathbf{L}_{UK} \\ \mathbf{L}_{KU} & \mathbf{L}_{KK} \end{bmatrix}, \qquad \mathbf{L}_{UU}\, \mathbf{p}_U = -\,\mathbf{L}_{UK}\, \mathbf{p}_K \tag{X.11}$$

`[STUB: State the mathematical properties that make this well-posed — L is symmetric positive semi-definite with the constant vector in its null space; the Dirichlet conditions remove that null space, rendering L_UU strictly positive definite and the system uniquely solvable. Note the guard: edges with invalid or non-positive resistance are excluded from assembly.]`

Signed flow through each edge follows from the recovered pressures:

$$Q_{ij} = g_{ij}\,(p_i - p_j) \tag{X.12}$$

### §X.3.4 Effective two-point resistance

> ⚠ **Include only if this quantity is reported in your results.** If not, delete this subsection.

`[STUB: Define the network-level effective resistance between an inlet–outlet pair as the total resistance the network presents between those points, accounting for all parallel and series pathways. State the algorithm: inject unit current at the source, ground the target by zeroing the corresponding row and column and setting the diagonal to unity, solve, and read the effective resistance as the potential at the source.]` (doc §2.3)

$$\mathbf{L}_{\mathrm{mod}}\,\mathbf{x} = \mathbf{b}, \quad b_{\mathrm{src}} = 1, \quad R_{\mathrm{eff}} = x_{\mathrm{src}} \tag{X.13}$$

### §X.3.5 Wall shear stress

> ⚠ **Include only if WSS is reported in your results.**

`[STUB: Motivate briefly — WSS is a principal mechanotransductive signal in vascular biology, with endothelial cells responding to it through nitric-oxide-mediated vasodilation at high WSS and pro-inflammatory signalling at low or oscillatory WSS.]` (doc §3.6)

$$\tau_w = \frac{32\,\mu\,Q}{\pi\, d^3} \tag{X.14}$$

`[STUB: State the unit conversion from mPa to Pa, and the caveat that the parabolic-profile assumption may underestimate WSS where the Fåhræus–Lindqvist cell-free layer is proportionally large.]`

---

## §X.4 Blood rheology

`[STUB: Motivate the section. Blood is a dense suspension of deformable cells, not a Newtonian fluid, giving rise to three phenomena absent in homogeneous fluids: the Fåhræus effect, the Fåhræus–Lindqvist effect, and plasma skimming at bifurcations. State that all three are represented, and that this necessitates an iterative solution because viscosity, flow and haematocrit are mutually dependent.]` (doc §3)

### §X.4.1 In vivo apparent viscosity

`[STUB: Introduce the Pries–Secomb empirical parameterisation. Note that it was derived from in vivo measurement in rat mesentery and that its transferability to the carotid body is assumed rather than demonstrated — this belongs in the assumptions table (§X.11, item 6).]` (doc §3.1)

Diameter is first clamped to a physical floor, $d \leftarrow \max(d, d_{\min})$, representing the minimum lumen an erythrocyte can traverse without lysis. The relative apparent viscosity at reference haematocrit $H_D = 0.45$ is

$$\mu_{45} = 220\,e^{-1.3 d} + 3.2 - 2.44\,e^{-0.06\, d^{0.645}} \tag{X.15}$$

`[STUB: Explain the three terms — the first captures erythrocyte deformation resistance and dominates below ~7 µm; the constant is the asymptotic bulk value; the third is a negative correction representing cell-free-layer lubrication, producing the Fåhræus–Lindqvist minimum near 7 µm. One short paragraph; the extended physiological discussion belongs in the Introduction.]`

The haematocrit shape parameter is

$$\kappa = \left(0.8 + e^{-0.075 d}\right)\left(-1 + \frac{1}{1 + 10^{-11} d^{12}}\right) + \frac{1}{1 + 10^{-11} d^{12}} \tag{X.16}$$

and the viscosity is rescaled to the local haematocrit by

$$\mu_{\mathrm{rel}} = 1 + (\mu_{45} - 1)\,\frac{(1 - H_D)^{\kappa} - 1}{(1 - 0.45)^{\kappa} - 1} \tag{X.17}$$

An in vivo correction accounts for the endothelial glycocalyx, which narrows the effective lumen by approximately 1.1 µm relative to the anatomical diameter:

$$\mu_{\mathrm{app}} = \mu_{\mathrm{rel}} \left( \frac{d}{d - 1.1} \right)^{2} \tag{X.18}$$

$$\mu = \mu_{\mathrm{app}}\, \mu_{\mathrm{pl}} \tag{X.19}$$

`[STUB: State the guards — non-positive diameter or haematocrit returns plasma viscosity; haematocrit is capped at 0.95.]`

**Initialisation.** `[STUB: One short paragraph. The full model requires local haematocrit, which depends on flow, which depends on resistance, which depends on viscosity. This circularity is broken by initialising resistances with a diameter-only power law that approximates the Pries–Secomb curve at reference haematocrit; the estimate is replaced during the coupled iteration of §X.4.3.]` (doc §3.2)

$$\mu_{\mathrm{init}}(d) = d^{-1.647} \tag{X.20}$$

> **FIGURE X.4** — Apparent viscosity against diameter (Eq. X.19) over 3–100 µm at several haematocrits, showing the Fåhræus–Lindqvist minimum. Overlay Eq. X.20 to show the quality of the initialisation approximation.

### §X.4.2 Phase separation at bifurcations

`[STUB: Motivate — erythrocytes concentrate in the fast-flowing core of the parabolic velocity profile, so the daughter branch drawing more flow disproportionately skims the cell-rich core while the low-flow branch preferentially receives cell-poor near-wall plasma. Consequence: capillary beds can be severely erythrocyte-depleted despite adequate volumetric perfusion.]` (doc §3.3)

For a parent segment of flow $Q_{\mathrm{in}}$ and haematocrit $H_{\mathrm{in}}$ dividing into daughters of flow $Q_1, Q_2$ and diameters $d_1, d_2$, the flow fractions are $\phi_{Q,i} = Q_i / Q_{\mathrm{in}}$. Geometric asymmetry and transition steepness are parameterised as

$$\lambda_A = -13.29 \cdot \frac{d_1^2/d_2^2 - 1}{d_1^2/d_2^2 + 1} \cdot \frac{1 - H_{\mathrm{in}}}{d_1} \tag{X.21}$$

$$\lambda_B = 1 + 6.98 \cdot \frac{1 - H_{\mathrm{in}}}{d_1} \tag{X.22}$$

The erythrocyte flux fraction is then linear in the logit of the flow fraction:

$$\mathrm{logit}(\phi_{E,1}) = \lambda_A + \lambda_B \cdot \ln\!\left( \frac{\phi_{Q,1} - \phi_0}{1 - \phi_{Q,1} - \phi_0} \right) \tag{X.23}$$

$$\phi_{E,1} = \frac{1}{1 + e^{-\mathrm{logit}(\phi_{E,1})}}, \qquad \phi_{E,2} = 1 - \phi_{E,1} \tag{X.24}$$

giving daughter haematocrits

$$H_{D,i} = H_{\mathrm{in}} \cdot \frac{\phi_{E,i}}{\phi_{Q,i}} \tag{X.25}$$

`[STUB: State the threshold behaviour — branches receiving less than the threshold fraction of flow are assigned zero haematocrit, reflecting the experimental observation that such branches draw almost exclusively from the cell-free wall layer. State the clamping of daughter haematocrits to physical bounds. State the limitation explicitly: the model is defined only for binary bifurcations; trifurcations and higher divisions receive proportional mixing (assumptions table item 7).]`

`[STUB: Justify the logit–sigmoid construction in one sentence — it maps the bounded flow fraction to the real line so that phase separation can be expressed as a linear relation, equivalent to a logistic regression on the underlying experimental data.]`

### §X.4.3 Coupled flow–haematocrit–viscosity solution

`[STUB: State the three-way circular dependency explicitly: flow depends on resistance; resistance depends on viscosity, which depends on diameter and local haematocrit; haematocrit distribution depends on flow through plasma skimming. State that this is resolved by fixed-point (Picard) iteration. Present the physics here; the numerical mechanics and convergence criteria are in §X.7.2.]` (doc §3.4)

> **ALGORITHM X.1 — Coupled rheological solution**
>
> Initialise every segment at systemic haematocrit; assign diameters (§X.2.2) and viscosities by Eq. X.19; compute resistances by Eq. X.7 or X.8.
>
> Repeat until converged or iteration limit reached:
> 1. Assemble $\mathbf{G}$ and $\mathbf{L}$ (Eqs. X.9–X.10) from current resistances.
> 2. Apply pressure boundary conditions (§X.8.1) and solve Eq. X.11 for nodal pressures.
> 3. Compute signed edge flows (Eq. X.12) and orient each edge downstream to form a directed acyclic graph.
> 4. Test convergence on the maximum absolute change in edge flow.
> 5. Traverse the DAG in topological order. At each node compute the flow-weighted mixed haematocrit of all upstream contributions, then distribute to daughters: unchanged for a single outlet; by Eqs. X.21–X.25 for a binary bifurcation; proportionally for three or more outlets.
> 6. Recompute viscosities (Eq. X.19) and rescale resistances (Eq. X.26).

`[STUB: Explain why topological ordering is required — it guarantees that every upstream contribution to a node has been computed before that node is processed, making haematocrit propagation causally consistent. State the failure mode: if the flow field contains a cycle, topological sorting is impossible and the iteration terminates early. Report whether this occurred in your data.]`

**Resistance rescaling.** `[STUB: This is a subtle and consequential modelling choice — explain it carefully. Resistances computed by spatial integration over a constriction profile (Eq. X.8) encode geometric information that recomputation by the straight-tube formula would destroy. The solver therefore preserves the geometric signature by rescaling the original resistance by the ratio of new to initialising viscosity, rather than recomputing it. Note that the reference resistance is stored on the first iteration and never subsequently modified.]` (doc §3.5)

$$R^{(k+1)} = R^{(0)} \cdot \frac{\mu\!\left(d, H_D^{(k)}\right)}{\mu_{\mathrm{init}}(d)} \tag{X.26}$$

---

## §X.5 Blood gas chemistry

`[STUB: Short introduction. The submodel below converts between partial pressure — the driving force for transmural diffusion — and total gas content, which is the conserved quantity transported along vessels. Oxygen and carbon dioxide carriage are coupled through haemoglobin by the reciprocal Bohr and Haldane effects, and to tissue pH through the bicarbonate equilibrium.]` (doc §5.3–5.6)

### §X.5.1 Oxygen content and the Hill equation

Dissolved oxygen follows Henry's law, $c_{\mathrm{diss}} = \alpha_{\mathrm{O_2}} P_{\mathrm{O_2}}$. Haemoglobin saturation follows the Hill relation

$$S_{\mathrm{O_2}} = \frac{P_{\mathrm{O_2}}^{\,\nu}}{P_{\mathrm{O_2}}^{\,\nu} + P_{50}^{\,\nu}} \tag{X.27}$$

and total blood oxygen content is the sum of dissolved and bound fractions:

$$C_{\mathrm{O_2}} = \alpha_{\mathrm{O_2}} P_{\mathrm{O_2}} + H_D\, C_{\mathrm{Hb}}\, S_{\mathrm{O_2}} \tag{X.28}$$

`[STUB: One paragraph on why the sigmoidal form matters physiologically — cooperative binding across four haem sites makes the curve steep in the 20–40 mmHg range where tissue extraction occurs, so small falls in PO2 release large quantities of oxygen. Note that the Hill coefficient of 2.7 rather than 4.0 reflects imperfect cooperativity, and that Eq. X.27 is an approximation to the full Adair formulation. Note the composition of C_Hb from the Hüfner number, normalised to a per-erythrocyte basis so that multiplication by local H_D recovers the correct local capacity.]`

### §X.5.2 Bohr effect

`[STUB: The half-saturation pressure is not constant; it shifts with local CO2 and pH. Explain the physiological consequence: metabolically active tissue, being hypercapnic and acidotic, shifts the dissociation curve rightward and so unloads oxygen preferentially where demand is greatest.]` (doc §5.4)

$$\log_{10} P_{50} = \log_{10}(26.0) - 0.4\,(\mathrm{pH} - 7.4) + 0.06 \log_{10}\!\left( \frac{P_{\mathrm{CO_2}}}{40} \right) \tag{X.29}$$

`[STUB: State that this recovers P50 = 26.0 mmHg at reference conditions (pH 7.4, PCO2 40 mmHg), and cite the empirical source.]`

### §X.5.3 Carbon dioxide content and the Haldane effect

`[STUB: The Haldane effect is the reciprocal of the Bohr effect — deoxygenated haemoglobin binds CO2 more readily than oxygenated haemoglobin, so blood becomes a better CO2 carrier at precisely the point it has delivered its oxygen.]` (doc §5.5)

$$C_{\mathrm{CO_2}} = \alpha_{\mathrm{CO_2}} P_{\mathrm{CO_2}} + H_D \left[ 11.02\, P_{\mathrm{CO_2}}^{\,0.396} + \left(0.15 - 0.05\, S_{\mathrm{O_2}}\right) P_{\mathrm{CO_2}} \right] \tag{X.30}$$

> ⚠ `[STUB — do not omit: the saturation term in Eq. X.30 is evaluated at a fixed P50 = 26.0 mmHg without Bohr feedback. State this plainly, state its direction (the Haldane enhancement is underestimated in hypoxic, hypercapnic tissue because haemoglobin is treated as more oxygenated than it is), and state the estimated magnitude (<5% in C_CO2 over the PO2 range encountered). This is assumptions table item 18. Naming it here costs a sentence; being caught on it costs more.]`

### §X.5.4 Tissue pH

`[STUB: Closes the feedback loop — metabolic CO2 production raises local PCO2, which lowers pH, which shifts P50 by Eq. X.29, which enhances oxygen unloading.]` (doc §5.6)

$$\mathrm{pH} = pK_a + \log_{10}\!\left( \frac{[\mathrm{HCO_3^-}]}{\alpha_{\mathrm{CO_2}} P_{\mathrm{CO_2}}} \right) \tag{X.31}$$

`[STUB: State that bicarbonate is held constant (open buffer, no renal compensation) — assumptions table item 11.]`

---

## §X.6 Tissue transport and vascular coupling

### §X.6.1 Tissue grid and vessel–grid mapping

`[STUB: A structured Cartesian grid is overlaid on the bounding volume of the vascular network, each cell representing a tissue block. Justify the resolution: at 10 µm a typical capillary is contained within a single cell, so the grid resolves PO2 gradients at the scale of inter-capillary spacing — the physiologically relevant length scale. State the bounds padding and the linear indexing convention.]` (doc §5.1)

Vessel centrelines are sampled and the exchange surface area accumulated per cell as

$$a_v = \sum 2\pi r\, \ell_{\mathrm{seg}} \tag{X.32}$$

`[STUB: State that surface area, not vessel volume, is the coupling quantity, because transmural flux is proportional to exchange area. Note the point-sampling approximation (assumptions table item 19).]`

### §X.6.2 Advection–diffusion–reaction formulation

`[STUB: State the steady-state balance in words before the equation — diffusive transport through tissue, plus delivery and removal by perfusion, minus metabolic consumption, sum to zero at every point.]` (doc §5.2)

$$\nabla \cdot \left( \sigma \nabla c \right) + \Sigma - M(c) = 0 \tag{X.33}$$

`[STUB: State the unit convention — the diffusion coefficient is specified in SI (m² s⁻¹) and converted to mesh units (µm² s⁻¹) during assembly; the solver works in partial-pressure space (mmHg) rather than concentration.]`

### §X.6.3 Metabolic consumption and respiratory quotient

`[STUB: Justify the saturating form — mitochondrial cytochrome c oxidase has a finite maximum turnover rate, so consumption saturates at high PO2 and falls to zero as PO2 approaches zero. The latter property is essential: it permits the model to develop genuinely hypoxic regions distant from capillaries rather than driving PO2 negative.]` (doc §5.7–5.8)

$$M(P_{\mathrm{O_2}}) = M_{\max}\left( 1 - e^{-k_M P_{\mathrm{O_2}}} \right) \tag{X.34}$$

$$M_{\mathrm{CO_2}} = \mathrm{RQ} \cdot M_{\mathrm{O_2}} \tag{X.35}$$

`[STUB: State plainly that Eq. X.34 is phenomenological rather than a Michaelis–Menten kinetic model, and that Michaelis–Menten is the more common choice in the literature. Assumptions table item 12.]`

### §X.6.4 Vascular–tissue coupling: three model tiers

> ⚠ **Structural note.** Written below as *progressive coupling* — each tier adds one physical mechanism to the last, sharing a single equation set. If you later restrict results to a subset of tiers, delete the corresponding paragraphs; no restructuring is required.

`[STUB: Frame the three tiers as a sequence of successively relaxed idealisations, and state which tier(s) produced the results reported in this thesis.]` (doc §6.1–6.3)

**Tier 1 — bulk advective exchange.** `[STUB: Blood is assumed to equilibrate instantaneously with the surrounding tissue block. Oxygen enters each cell at arterial content and leaves at the content corresponding to local tissue PO2; the net delivery is the difference. This is the well-stirred, Krogh-type approximation applied per voxel. Note the non-linearity: the washout term depends on tissue PO2 through Eq. X.28. Note also that Tier 1 uses a fixed baseline haematocrit for washout rather than a per-voxel flow-weighted value — assumptions table item 20.]`

**Tier 2 — permeability-limited endothelial barrier.** `[STUB: Relaxes instantaneous equilibration. Transmural transport is governed by Fick's first law across the endothelium, so delivery may be barrier-limited rather than diffusion-limited. Explain why the solubility coefficient appears: it converts the partial-pressure driving force into a concentration difference for dimensional consistency with the permeability coefficient.]`

$$J_{\mathrm{O_2}} = \mathcal{P}_{\mathrm{O_2}}\, a_v\, \alpha_{\mathrm{O_2}} \left( P_{\mathrm{O_2}}^{\mathrm{blood}} - P_{\mathrm{O_2}}^{\mathrm{tissue}} \right) \tag{X.36}$$

**Tier 3 — multi-species with Bohr–Haldane coupling.** `[STUB: Adds CO2 and pH as simultaneously solved species, closing the feedback loop of §X.5. Describe the loop concretely: metabolism consumes O2 and produces CO2 in ratio RQ; rising PCO2 lowers pH by Eq. X.31; falling pH raises P50 by Eq. X.29; raised P50 enhances oxygen unloading by Eq. X.27. State that CO2 transmural flux takes the same form as Eq. X.36 with its own permeability, roughly twentyfold higher than that of oxygen owing to greater lipid solubility, so that CO2 exchange is rarely rate-limiting.]`

> **FIGURE X.5** — Tier comparison schematic: three panels showing what each tier couples (bulk exchange / barrier-limited exchange / barrier-limited exchange with CO₂–pH feedback), with the governing equation numbers annotated on each.

### §X.6.5 Longitudinal blood gas tracking

`[STUB: Motivate — in Tier 1 blood enters every tissue block at arterial PO2 regardless of the distance already travelled. In reality blood desaturates progressively along its path, so the venous end of a capillary delivers into tissue at substantially lower driving pressure than the arteriolar end. Capturing this longitudinal gradient is essential for predicting which tissue regions are at risk of hypoxia.]` (doc §6.4)

> **ALGORITHM X.2 — Longitudinal blood gas tracking**
>
> 1. Orient every edge along its computed flow direction to form a directed acyclic graph.
> 2. Sort nodes in topological order.
> 3. Initialise inlet nodes at arterial $P_{\mathrm{O_2}}$, $P_{\mathrm{CO_2}}$ and pH; convert to content by Eqs. X.28 and X.30.
> 4. For each node in topological order:
>    a. Compute flow-weighted mixed content over all upstream contributions.
>    b. Invert Eqs. X.28 and X.30 numerically to recover blood partial pressures (§X.7.3).
>    c. For each downstream edge, walk the tissue cells it traverses; at each, compute transmural flux by Eq. X.36, decrement blood content, and re-invert to update blood partial pressure.
>    d. Pass the residual content to the downstream node.

`[STUB: State the plug-flow assumption — blood is perfectly mixed across each vessel cross-section with no radial gradient (assumptions table item 10). Explain why Eq. X.28 must be inverted numerically: the Hill relation has no closed-form algebraic inverse for non-integer ν. State the root-finding method and bracket in §X.7.3 rather than here.]`

---

## §X.7 Numerical implementation

### §X.7.1 Spatial discretisation

`[STUB: The tissue diffusion operator is discretised by a seven-point finite-difference stencil on the Cartesian grid, coupling each interior cell to its six face-sharing neighbours. Justify the choice: the resulting matrix is sparse, symmetric and diagonally dominant, well suited to preconditioned iterative solution, and higher-order stencils would broaden the bandwidth for negligible accuracy gain at 10 µm resolution.]` (doc §7.1)

Diffusive conductances between adjacent cells are the products of diffusivity and interface area divided by centre separation:

$$D_z = \sigma \frac{\Delta y\, \Delta x}{\Delta z}, \qquad D_y = \sigma \frac{\Delta z\, \Delta x}{\Delta y}, \qquad D_x = \sigma \frac{\Delta z\, \Delta y}{\Delta x} \tag{X.37}$$

`[STUB: Note that in the multi-species solver the conductances are pre-scaled by the solubility coefficient so that the system solves directly in partial-pressure units. Note that Neumann (zero-flux) boundaries are imposed implicitly by stencil truncation at the domain edge, and that a small diagonal regularisation is added to remove the resulting null space.]`

### §X.7.2 Picard iteration and pseudo-washout stabilisation

`[STUB: Justify Picard over Newton — the non-linearities (saturating metabolism, Hill-equation washout) would require Jacobian evaluation that is expensive and awkward for the Hill term, whereas fixed-point iteration freezes the non-linear terms at the previous iterate and solves the resulting linear system. For the mild non-linearities present, convergence is achieved in tens of iterations.]` (doc §7.2)

`[STUB: Explain the stabilisation carefully — this is a non-obvious numerical device and an examiner may probe it. Under pure Neumann conditions the diffusion matrix has zero row sums and is near-singular, while the entire sink lives on the right-hand side where it depends non-linearly on the unknown. This produces oscillatory divergence: a high PO2 iterate generates a large washout, overshooting to low PO2, which generates a small washout, overshooting back. Adding a linearised portion of the washout to the left-hand diagonal AND the identical term to the right-hand side leaves the fixed point unchanged but renders the matrix strictly diagonally dominant.]`

$$\mathbf{A}_{\mathrm{stab}} = \mathbf{A} + \mathrm{diag}\!\left( \gamma\, \mathbf{q}_{\mathrm{tot}} \right) \tag{X.38}$$

> **ALGORITHM X.3 — Picard loop (multi-species)**
>
> Assemble $\mathbf{A}_{\mathrm{stab}}$ and its incomplete-LU preconditioner once, outside the loop.
>
> Repeat until both species converge or the iteration limit is reached:
> 1. Clamp tissue partial pressures to non-negative values.
> 2. Evaluate metabolic terms by Eqs. X.34–X.35.
> 3. Update tissue pH by Eq. X.31.
> 4. Run longitudinal blood gas tracking (Algorithm X.2) to obtain transmural fluxes.
> 5. Assemble right-hand sides — note the sign asymmetry: metabolism is a sink for O₂ and a source for CO₂.
> 6. Solve each species by preconditioned conjugate gradients.
> 7. Clamp to non-negative values and test convergence on the relative $L^2$ change in each field.

`[STUB: State convergence criteria and iteration limits, and report actual iteration counts observed in your runs — a solver that reliably converges in 12 iterations against a limit of 50 is worth stating.]`

### §X.7.3 Linear solver selection and root finding

`[STUB: Two sentences on solver dispatch — direct factorisation below a size threshold, preconditioned conjugate gradients with incomplete-LU above it, with a least-squares fallback for rank-deficient or ill-conditioned systems. Do not reproduce the full fallback ladder; move solver tolerances to the appendix.]` (doc §7.3)

`[STUB: One paragraph on the Hill inversion of §X.6.5. Brent's method is used because the Hill relation is strictly monotonic in PO2, guaranteeing a unique root in the search bracket, and because the method combines the guaranteed convergence of bisection with superlinear speed. State the bracket and the fallback behaviour when no root exists within it.]`

### §X.7.4 Numerical safeguards and physical clamping

> ⚠ **Do not omit or scatter this subsection.** The source document distributes these guards across ten sections; presented individually each is trivial, but collectively they influence results, and scattered mention reads as concealment. Consolidate here and quantify.

`[STUB: Introduce as deliberate numerical policy, not incidental defensive coding.]`

| Safeguard | Value | Purpose | Frequency in this dataset |
|---|---|---|---|
| Minimum vessel diameter | 3.0 µm | Erythrocyte traversal limit; prevents divergence of Eq. X.18 at $d \to 1.1$ µm | `[REPORT]` |
| Missing/invalid diameter fallback | 5.0 µm | Retains segment in solution rather than excluding it | `[REPORT]` |
| Missing/invalid length fallback | 10.0 µm | As above | `[REPORT]` |
| Maximum haematocrit | 0.95 | Physical packing limit | `[REPORT]` |
| Minimum constriction ratio | 0.01 | Prevents singular resistance | `[REPORT]` |
| Non-positive segment length | $R \to \infty$ | Excludes degenerate segments from flow | `[REPORT]` |
| Invalid resistance | Edge dropped from assembly | Prevents division by zero in Eq. X.9 | `[REPORT]` |
| Negative partial pressure | Clamped to 0 | Physical bound; prevents blow-up in Eq. X.34 | `[REPORT]` |
| Diffusion matrix regularisation | $10^{-12}$ | Removes Neumann null space | n/a |

`[STUB: Close with a sentence stating the aggregate effect. If the fallback rates are low, say so and the section becomes a demonstration of rigour rather than an admission. If any rate is material, state it and carry it into the Discussion.]`

---

## §X.8 Boundary and initial conditions

### §X.8.1 Pressure boundary conditions

| Boundary | Value | Basis |
|---|---|---|
| Inlet | 100 mmHg | Mean arterial pressure |
| Outlet | 2 mmHg | Central venous pressure |

> ⚠ `[STUB — this is the assumption most exposed to challenge; state it here as well as in the assumptions table. The full systemic pressure gradient is applied across a micro-organ, neglecting upstream arterial and downstream venous resistance. The effective perfusion pressure across the carotid body in vivo is substantially lower, so absolute flow magnitudes are expected to be overestimated. State whether your conclusions depend on absolute flow or only on relative spatial distribution — if the latter, say so explicitly, because it substantially limits the damage.]` (doc §8.1)

### §X.8.2 Inlet and outlet identification

`[STUB: Boundary nodes are identified automatically as degree-one termini lying within a configurable band at the spatial extremes of the imaged volume along a chosen axis. Justify: degree-one nodes represent vessels truncated by the field of view, which in the intact organ continue to upstream arterial supply or downstream venous drainage. State the band width and axis used.]` (doc §8.2)

### §X.8.3 Domain truncation handling

`[STUB: Introduce the problem — the imaged volume truncates the vasculature, and how truncated vessels are terminated materially affects the computed flow field. Three modes are available; state which was used.]` (doc §8.3)

| Mode | Treatment |
|---|---|
| Caged (default) | Only the axial faces are permeable; lateral faces sealed |
| Universal sink | All six faces permeable |
| Robin resistance | Truncated termini connect to a virtual node through a finite resistance |

For the Robin condition, the ghost resistance at a truncated terminus is scaled from the mean resistance of its incident edges:

$$R_{\mathrm{ghost}} = \zeta \cdot \frac{1}{n}\sum_{i} R_i \tag{X.39}$$

`[STUB: Justify the Robin condition physically — capping truncated capillaries at zero flow imposes an artificial bottleneck, since in vivo they drain to vasculature outside the field of view. A finite escape resistance permits realistic drainage without allowing the virtual node to dominate the flow pattern. State the multiplier used and note it is a chosen rather than measured value (parameter class iii, §X.9).]`

### §X.8.4 Blood gas and tissue boundary conditions

| Quantity | Value |
|---|---|
| Arterial $P_{\mathrm{O_2}}$ | 100 mmHg |
| Arterial $P_{\mathrm{CO_2}}$ | 40 mmHg |
| Systemic haematocrit | 0.45 |
| Tissue $[\mathrm{HCO_3^-}]$ | 24 mmol L⁻¹ |
| Tissue grid boundary | Neumann (zero flux) |

`[STUB: State that these are normal adult resting values at sea level, and indicate how each would be modified to model hypoxaemia, anaemia or hypercapnia — this establishes the model's intended experimental range. State the Neumann tissue condition and its consequence: no exchange with tissue outside the imaged volume, so PO2 is expected to be overestimated near the domain boundary. Assumptions table item 14.]`

---

## §X.9 Model parameters

`[STUB: Short framing paragraph, and make this specific point: the parameters below fall into three classes of very different epistemic standing, and it matters which class a given value belongs to. Purely numerical solver settings — tolerances, iteration limits, preconditioner settings — are not model parameters at all and are tabulated separately in Appendix [n]. Making this separation explicit prevents the model appearing to carry ~40 free parameters when it carries roughly 25, of which only class (iii) is genuinely at the author's discretion.]`

### Class (i) — Literature-derived

| Parameter | Symbol | Value | Units | Basis |
|---|---|---|---|---|
| Plasma viscosity | $\mu_{\mathrm{pl}}$ | 1.2 | mPa s | Human plasma at 37 °C |
| Systemic haematocrit | $H_D$ | 0.45 | — | Normal adult |
| Hill coefficient | $\nu$ | 2.7 | — | Adult human haemoglobin |
| Half-saturation pressure | $P_{50}$ | 26.0 | mmHg | At pH 7.4, $P_{\mathrm{CO_2}}$ 40 mmHg |
| O₂ solubility | $\alpha_{\mathrm{O_2}}$ | $1.34\times10^{-3}$ | mmol L⁻¹ mmHg⁻¹ | Henry's law |
| CO₂ solubility | $\alpha_{\mathrm{CO_2}}$ | 0.03 | mmol L⁻¹ mmHg⁻¹ | Henry's law |
| Haemoglobin O₂ capacity | $C_{\mathrm{Hb}}$ | $0.446\times20.4/0.45$ | mmol L⁻¹ | Hüfner number, normalised per erythrocyte volume |
| Carbonic acid p$K_a$ | $pK_a$ | 6.1 | — | Standard |
| Tissue bicarbonate | $[\mathrm{HCO_3^-}]$ | 24.0 | mmol L⁻¹ | Normal plasma |
| Respiratory quotient | $\mathrm{RQ}$ | 0.82 | — | Mixed metabolic substrate |
| O₂ tissue diffusivity | $\sigma_{\mathrm{O_2}}$ | $1.5\times10^{-9}$ | m² s⁻¹ | Mammalian tissue |
| CO₂ tissue diffusivity | $\sigma_{\mathrm{CO_2}}$ | $3.0\times10^{-8}$ | m² s⁻¹ | Mammalian tissue |
| Mean arterial pressure | — | 100 | mmHg | Systemic |
| Central venous pressure | — | 2 | mmHg | Systemic |
| Arterial $P_{\mathrm{O_2}}$ / $P_{\mathrm{CO_2}}$ | — | 100 / 40 | mmHg | Normal adult, sea level |
| Minimum vessel diameter | $d_{\min}$ | 3.0 | µm | Erythrocyte traversal limit |

`[STUB: CITATIONS REQUIRED for every row. The source document gives bare attributions or none.]`

### Class (ii) — Empirical correlations transferred from other tissues or species

| Parameter group | Values | Origin | Eq. |
|---|---|---|---|
| Pries–Secomb $\mu_{45}$ coefficients | 220, −1.3, 3.2, −2.44, −0.06, 0.645 | In vivo rat mesentery | X.15 |
| Pries–Secomb shape coefficients | 0.8, −0.075, $10^{-11}$, 12 | As above | X.16 |
| Glycocalyx layer width | 1.1 µm | As above | X.18 |
| Skimming threshold | $\phi_0 = 0.05$ | Glass tube + rat cremaster | X.23 |
| Skimming asymmetry constant | −13.29 | As above | X.21 |
| Skimming steepness constant | 6.98 | As above | X.22 |
| CO₂ capacity coefficient / exponent | 11.02, 0.396 | Empirical human blood | X.30 |
| Haldane coefficients | 0.15 (deoxy), 0.05 (oxy) | As above | X.30 |
| Bohr coefficients | −0.4 per pH unit, 0.06 per log unit | Empirical human blood | X.29 |

> ⚠ `[STUB: State plainly that these correlations were derived in preparations other than the carotid body, and in the rheological case in a different species. Their transferability is assumed, not demonstrated. Assumptions table item 6. CITATIONS REQUIRED — Pries & Secomb (1992, 1994); Spencer (1979); Kelman (1966); Severinghaus (1979). Verify each against the primary source before submission.]`

### Class (iii) — Chosen, heuristic, or estimated

> ⚠ **This is the exposed class.** Each row needs a citation, a sensitivity analysis, or an explicit statement that results are qualitative with respect to it.

| Parameter | Value | Units | Current basis | Status |
|---|---|---|---|---|
| Initialising viscosity exponent | 1.647 | — | Heuristic fit to Eq. X.19 at reference $H_D$ | `[Low risk — initialisation only, replaced by Eq. X.26. Justify on that ground.]` |
| Maximum metabolic rate | $M_{\max} = 0.005$ | mmol L⁻¹ s⁻¹ | Phenomenological | `[HIGH RISK — directly sets tissue PO2. Needs literature anchor or sensitivity sweep.]` |
| Metabolic saturation constant | $k_M = 0.1$ | mmHg⁻¹ | Phenomenological | `[HIGH RISK — sets the hypoxic knee of Eq. X.34.]` |
| O₂ endothelial permeability | $1.0\times10^{-4}$ | cm s⁻¹ | Estimated from published cellular models | `[HIGH RISK — sole control on Tier 2/3 delivery rate.]` |
| CO₂ endothelial permeability | $2.0\times10^{-3}$ | cm s⁻¹ | As above | `[Moderate — CO2 exchange is not rate-limiting.]` |
| Intimal cushion ratio | 0.60 | — | Assumed | `[If constriction is a tested hypothesis, this must be swept.]` |
| Pre-capillary sphincter ratio | 0.50 | — | Assumed | `[As above.]` |
| Constriction zone length | 5.0 | µm | Assumed | `[As above.]` |
| Periodic spacing / ramp / hold | 100 / 10 / 20 | µm | Assumed from pericyte spacing | `[As above.]` |
| Robin resistance multiplier | $\zeta = 10$ | — | Chosen | `[Affects boundary flow. Sweep if caged mode not used.]` |
| Pseudo-washout coefficient | $\gamma = 0.5$ / $1.0$ | — | Chosen for stability | `[Low risk — does not alter the fixed point (Eq. X.38). Say so explicitly.]` |
| Grid resolution | $10^3$ | µm³ | Chosen | `[Addressed by the grid-convergence study, §X.10.]` |
| Branch-order anchors | 15 / 4 / 20 | µm | Assumed | `[Fallback path only — bound by the rate reported in §X.2.2.]` |
| Diameter / length fallbacks | 5.0 / 10.0 | µm | Assumed | `[As above — see §X.7.4.]` |
| Boundary band width | 25 | % | Chosen | `[Affects inlet/outlet count. Report sensitivity.]` |
| Capillary calibre window | 4–7 | µm | Asserted, uncited (segmentation handover §8, §10.1) | `[HIGH RISK — see note below. Needs a primary citation for rat carotid body capillary diameter, or demotion to a reported diagnostic.]` |

> ⚠ **The capillary calibre window is upstream of everything else in this table.** The 4–7 µm
> expected capillary diameter is not measured from these specimens. It enters as the objective
> that selects the segmentation probability threshold: the threshold is chosen as the highest
> value whose median inscribed diameter falls inside the window and whose skeleton has not begun
> to fragment (`ImageLynx.statistics.threshold_selection`). Every downstream geometric and
> haemodynamic quantity therefore inherits it, and resistance inherits it at the fourth power.
>
> It is used deliberately as an *external* target rather than an internal optimum, because a
> threshold chosen to optimise a property of the data has no independent standard to be right
> or wrong against — the alternative criterion in the handover, based on connected-component
> statistics, was tested on these data and returns no answer at all (the largest component's
> voxel share never falls, because a vascular mask percolates). The cost of that choice is that
> the window is doing real work while resting on an assertion.
>
> Two things this obliges. State the provenance plainly rather than presenting the threshold as
> data-derived. And report the selected threshold's sensitivity across the plausible width of
> the window — not across an arbitrary band around the chosen threshold, which tests the wrong
> quantity. If the group contrast survives the window being 3–8 µm as well as 4–7, say so; if it
> does not, the result is a statement about the assumed calibre, not about the tissue.

`[STUB: Solver settings — Picard and rheology iteration limits and tolerances, conjugate-gradient tolerances, incomplete-LU drop tolerance and fill factor, direct/iterative dispatch threshold, quadrature sample count, root-finding bracket, matrix regularisations — are tabulated in Appendix [n]. State here in one sentence that these govern solution accuracy, not model behaviour, and that convergence was verified independently of them (§X.10).]`

---

## §X.10 Model verification

`[STUB — OPENING PARAGRAPH, HIGH PRIORITY. Define the distinction before presenting any results: verification establishes that the governing equations are solved correctly; validation establishes that those equations describe the physical system. State that this section addresses verification. State that validation against experimental measurement of carotid body perfusion or tissue oxygenation was not undertaken, and identify it as the principal direction for further work. Cite the standard framing — Roache (1998), Oberkampf & Roy (2010), or ASME V&V 20; check departmental convention. Two sentences of explicit self-awareness here are worth an hour of viva defence.]`

### §X.10.1 Verification methodologies

`[STUB: Describe the six strategies applied. Keep to a paragraph each at most, or a single paragraph covering all six.]` (doc §11.1)

1. **Analytical closed-form comparison** — solver output compared against independently derived exact solutions.
2. **Conservation and invariant checks** — mass and flux balance asserted directly rather than against a target.
3. **Synthetic phantoms with prescribed answers** — volumes and graphs constructed with a mathematically known correct result.
4. **Equivalence oracles** — independent code paths checked for mutual agreement.
5. **Graceful degradation** — pathological inputs checked to fail safely rather than crash or hang.
6. **Physical bounds** — extreme configurations checked against known physical limits.

`[STUB: State the scale of the automated verification suite and that it runs under continuous integration. One or two sentences; do not name individual test functions.]`

### §X.10.2 Grid convergence

> ⚠ **This subsection does not yet exist and should be generated before submission.** No mesh-refinement or order-of-accuracy study currently exists for any PDE solver in the pipeline (source doc §11.3, item 2); every result runs at a single fixed resolution. This is the most standard expectation in computational verification and, given that no validation exists, closing it is disproportionately valuable.
>
> **Suggested procedure:** take the zero-order metabolism case, which has the exact closed-form solution $c(x) = c_0 - (M/2\sigma)\,x(L-x)$ and is already verified to `atol=1e-10` at a single resolution. Run at 20, 10, 5 and 2.5 µm. Plot $L^2$ error against grid spacing on log axes and fit the slope. A slope near 2 demonstrates the expected second-order convergence of the seven-point stencil.

`[STUB: Write once run. Report the observed order of accuracy and state whether the production resolution lies in the asymptotic range.]`

> **FIGURE X.6** — Grid convergence: $L^2$ error against grid spacing, log–log, with fitted slope and the production resolution marked.

### §X.10.3 Verification coverage

`[STUB: Introduce the table. State the oracle type and tolerance convention.]` (doc §11.2)

| Model component | Oracle type | Tolerance |
|---|---|---|
| Poiseuille segment resistance (Eq. X.7) | Independent formula recomputation | $10^{-6}$; <5% on measured phantom |
| Network Laplacian solve (Eq. X.11) | Closed-form series and parallel reduction | $10^{-10}$ |
| Variable-diameter resistance (Eq. X.8) | Closed-form term-by-term integration | $r = 10^{-3}$ |
| Wall shear stress (Eq. X.14) | Closed-form recomputation | $10^{-10}$ |
| Apparent viscosity (Eq. X.19) | Curve shape (inequality chain) | qualitative |
| Plasma skimming (Eqs. X.21–X.25) | Erythrocyte mass conservation; direction | $10^{-8}$ |
| FWHM diameter measurement (Eq. X.1) | Analytical Gaussian phantom | <0.2–0.35 µm |
| Perfusion grid and mapping (Eq. X.32) | Exact geometry; bijective index mapping | exact |
| Pure diffusion limit (Eq. X.33) | Closed-form linear gradient | $10^{-10}$ |
| Reaction–diffusion limit (Eqs. X.33–X.34) | Closed-form parabolic profile | $10^{-10}$ |
| Advection–diffusion limit | Qualitative $1/r$ radial decay | bracketed |
| Hill equation (Eqs. X.27–X.28) | Exact $P_{50}$ property; shape | $10^{-5}$ |
| Bohr and Haldane shifts (Eqs. X.29–X.30) | Direction only | qualitative |
| Henderson–Hasselbalch (Eq. X.31) | Closed form at two anchor points | $10^{-2}$ |
| Tier 1 solver | Physical bounds; Fick-principle root | $10^{-10}$; $10^{-2}$ |
| Tier 3 solver | Coupled Fick + Henderson–Hasselbalch root | $10^{-2}$; $10^{-3}$ (pH) |
| Seven-point stencil (Eq. X.37) | Exact non-zero structure | exact |
| Linear solver dispatch | Cross-branch agreement; graceful degradation | exact |
| Boundary permeability modes | Kirchhoff conservation at hubs | $10^{-8}$ |

### §X.10.4 Extent of verification

> ⚠ **Keep this factual.** State coverage limits as fact here; the implications belong in the Discussion. Do not upgrade any claim the source document qualifies.

`[STUB: State plainly which components are verified only transitively through integration-style tests rather than directly — the resistance rescaling rule (Eq. X.26), the branch-order diameter formulae (Eqs. X.2–X.3), the default boundary permeability mode, the numerical Hill inversion, and the blood gas boundary values. State which components are verified only directionally or qualitatively rather than against a closed-form target — the apparent viscosity curve, the skimming output value (as distinct from its mass conservation, which is exact), and the Bohr and Haldane shifts. State that two tests in the suite compute an analytical target without asserting against it and are therefore structural-completion checks rather than closed-form benchmarks.]` (doc §11.3)

---

## §X.11 Summary of assumptions

`[STUB: One framing sentence. The table states each assumption and the direction of its expected effect; the significance of each is assessed in §[Discussion].]` (doc §10)

| # | Assumption | Enters at | Expected direction of effect |
|---|---|---|---|
| 1 | Rigid vessel walls; no compliance or autoregulation | Eq. X.6 | Removes pressure-dependent flow redistribution; resistance is static |
| 2 | Steady state; no cardiac pulsatility | Eq. X.6 | Removes cyclic WSS variation; mean flow largely unaffected |
| 3 | Newtonian fluid at initialisation | Eq. X.20 | Biases initial resistances; largely relaxed by Eq. X.26 |
| 4 | Circular lumen cross-section | Eq. X.7 | Non-circular lumens have higher resistance at equal area → resistance underestimated |
| 5 | No-slip at the vessel wall | Eq. X.6 | Standard; negligible |
| 6 | Rheological correlations from rat mesentery | Eqs. X.15–X.18 | Transferability unknown; magnitude unquantified |
| 7 | Phase separation at binary bifurcations only | §X.4.2 | Higher-order divisions mix proportionally → haematocrit heterogeneity underestimated |
| 8 | Exponential branch-order diameter scaling; Murray's law not used | Eqs. X.2–X.3 | Active on fallback path only; bounded by fallback rate (§X.2.2) |
| 9 | Full MAP-to-CVP gradient across the organ | §X.8.1 | Perfusion pressure and absolute flow overestimated |
| 10 | Plug flow; no radial intraluminal gradient | Eq. X.36 | Transmural driving force slightly overestimated |
| 11 | Constant bicarbonate buffer; no renal compensation | Eq. X.31 | Fixes the pH response to $P_{\mathrm{CO_2}}$ |
| 12 | Phenomenological rather than Michaelis–Menten metabolism | Eq. X.34 | Differs from the literature standard in the low-$P_{\mathrm{O_2}}$ regime |
| 13 | Homogeneous tissue; uniform diffusivity and metabolic rate | Eq. X.33 | Smooths the tissue $P_{\mathrm{O_2}}$ field; no cell-type heterogeneity |
| 14 | Neumann tissue boundary; no exchange beyond the imaged volume | §X.7.1 | Tissue $P_{\mathrm{O_2}}$ overestimated near the domain boundary |
| 15 | Human haemoglobin parameters | Eqs. X.27, X.29 | Species mismatch if applied to non-human tissue |
| 16 | No lymphatic drainage or interstitial fluid flow | Eq. X.33 | Omits a minor transport pathway |
| 17 | Static constriction ratios; no active vasoregulation | §X.2.4 | No myogenic, metabolic or shear-mediated feedback |
| 18 | Haldane saturation evaluated at fixed $P_{50}$ | Eq. X.30 | CO₂ carriage underestimated in hypoxic tissue (<5%) |
| 19 | Point-sampled vessel-to-grid mapping | Eq. X.32 | Discretisation error in deposited exchange area |
| 20 | Fixed baseline haematocrit for Tier 1 washout | §X.6.4 | Decouples washout from local haematocrit in Tier 1 only |

---

## §X.12 Software and reproducibility

`[STUB: Not present in the source document; write from scratch. Cover: software name and version or commit hash; language and version; principal dependencies with versions; availability statement and repository or archive DOI; random seeds where any stochastic component is used; hardware; and representative runtime and peak memory for a typical network of the size analysed. State that the verification suite runs under continuous integration.]`

---

## Appendix [n] — Solver settings

`[STUB: Table of purely numerical settings, separated from the model parameters of §X.9: Picard iteration limit and tolerance (per tier); rheology iteration limit and tolerance; conjugate-gradient tolerances and iteration limits; incomplete-LU drop tolerance and fill factor; direct/iterative dispatch threshold; quadrature sample count for Eq. X.8; root-finding bracket for the Hill inversion; matrix regularisation constants. Source: doc §9.4 and scattered §7.]`

---

## Drafting order (delete before submission)

Write in this sequence, not the reading sequence:

1. **§X.9 parameter tables and the Appendix.** Mechanical, and it forces every provenance question to be resolved before prose is committed around a shaky value. Resolve every `[CITATIONS REQUIRED]` marker here first.
2. **§X.11 assumptions table.** Sets the honest scope that everything else must respect.
3. **§X.7.4 safeguard frequencies.** Run the counts. You need these numbers before you can write §X.2.2 or §X.2.3 honestly.
4. **§X.10.2 grid convergence study.** Run it. It is the cheapest available improvement to the verification argument.
5. **§X.2 → §X.6**, in order. The bulk of the prose.
6. **§X.7, §X.8.** Condense hard from the source document.
7. **§X.10**, remaining subsections.
8. **§X.1 overview**, last — once you know what you are overviewing.

## Open items

- Decide which coupling tiers appear in results (§X.6.4) and prune accordingly.
- Decide whether §X.3.4 (two-point resistance) and §X.3.5 (wall shear stress) are reported; delete if not.
- Resolve all `[CITATIONS REQUIRED]` markers in §X.9.
- Run and report the safeguard frequencies in §X.7.4.
- Run the grid convergence study for §X.10.2.
- Confirm the chapter number and renumber equations `(X.n)` → `(n.m)` throughout.
