# Equations executed after vessel radius assignment

Listed in execution order for the default configuration.

---

# Nomenclature

## Geometry

| Symbol | Description | Units |
|---|---|---|
| $D$ | Vessel segment diameter | µm |
| $r$ | Vessel segment radius | µm |
| $L$ | Vessel segment centreline length | µm |
| $D_1, D_2$ | Diameters of the two daughter branches at a bifurcation | µm |
| $S$ | Vessel wall surface area contributed to a grid cell | µm² |
| $\ell_{\text{cell}}$ | Length of an edge lying within one grid cell | µm |
| $\ell_{\text{total}}$ | Total centreline length of an edge | µm |
| $f_{\ell}$ | Fraction of an edge's length lying within one grid cell | — |
| $r_z, r_y, r_x$ | Perfusion grid cell dimensions | µm |
| $V_{\text{cell}}$ | Perfusion grid cell volume | µm³ |

## Rheology

| Symbol | Description | Units |
|---|---|---|
| $\mu_{\text{old}}$ | Legacy scaling factor used in the Poiseuille assignment phase | — |
| $\mu_{0.45}$ | Pries–Secomb relative apparent viscosity at $H = 0.45$ | — |
| $C(D)$ | Pries–Secomb haematocrit shape parameter | — |
| $\Phi$ | Pries–Secomb haematocrit dependence term | — |
| $W$ | Cell-depleted wall layer correction factor | — |
| $\mu_{\text{rel}}$ | Relative apparent viscosity | — |
| $\mu_{\text{app}}$ | Apparent blood viscosity | mPa·s |
| $\mu_{\text{plasma}}$ | Plasma viscosity | mPa·s |

## Network flow

| Symbol | Description | Units |
|---|---|---|
| $R$ | Hydraulic resistance of an edge | mPa·s·µm⁻³ |
| $R_{\text{eff}}$ | Two-point effective resistance between two nodes | mPa·s·µm⁻³ |
| $C_{ij}$ | Hydraulic conductance of the edge joining nodes $i$ and $j$ | µm³·mPa⁻¹·s⁻¹ |
| $\mathbf{C}$ | Conductance matrix | — |
| $\mathbf{L}$ | Graph Laplacian | — |
| $\mathbf{L}^{+}$ | Moore–Penrose pseudoinverse of the Laplacian | — |
| $\mathbf{L}_{uu}, \mathbf{L}_{uk}$ | Unknown–unknown and unknown–known blocks of the Laplacian | — |
| $\mathbf{e}_i$ | Unit basis vector for node $i$ | — |
| $p_i$ | Pressure at node $i$ | mPa |
| $\mathbf{p}_u, \mathbf{p}_k$ | Unknown and Dirichlet-fixed nodal pressures | mPa |
| $Q_{ij}$ | Volumetric blood flow on the edge joining nodes $i$ and $j$ | µm³·s⁻¹ |
| $q_{\text{total},i}$ | Total bulk flow passing through grid cell $i$ | µm³·s⁻¹ |
| $\tau_w$ | Wall shear stress | mPa |

## Haematocrit and phase separation

| Symbol | Description | Units |
|---|---|---|
| $H$ | Discharge haematocrit | — |
| $H_{\text{in}}$ | Haematocrit entering a bifurcation | — |
| $H_1, H_2$ | Haematocrit in the two daughter branches | — |
| $FQ_1, FQ_2$ | Fraction of total bulk flow entering each daughter branch | — |
| $FQ_{E1}, FQ_{E2}$ | Fraction of total red cell flux entering each daughter branch | — |
| $A$ | Phase separation bifurcation asymmetry parameter | — |
| $B$ | Phase separation logistic steepness parameter | — |
| $X_0$ | Flow fraction below which a branch receives no red cells | — |
| $D_F$ | Diameter of the feeding (parent) vessel at a bifurcation | µm |

## Blood gas chemistry

| Symbol | Description | Units |
|---|---|---|
| $P_{\mathrm{O_2}}$ | Oxygen partial pressure | mmHg |
| $P_{\mathrm{CO_2}}$ | Carbon dioxide partial pressure | mmHg |
| $P_{50}$ | Oxygen partial pressure at 50% haemoglobin saturation | mmHg |
| $n$ | Hill coefficient | — |
| $S_{\mathrm{O_2}}$ | Haemoglobin oxygen saturation | — |
| $S_{\mathrm{O_2}}^{H}$ | Oxygen saturation term used in the Haldane shift | — |
| $\alpha_{\mathrm{O_2}}$ | Solubility of oxygen in plasma | mmol·L⁻¹·mmHg⁻¹ |
| $\alpha_{\mathrm{CO_2}}$ | Solubility of carbon dioxide in plasma | mmol·L⁻¹·mmHg⁻¹ |
| $c_{\mathrm{Hb,max}}$ | Maximum oxygen carrying capacity of pure red cells | mmol·L⁻¹ |
| $C_{\mathrm{O_2}}$ | Total blood oxygen content | mmol·L⁻¹ |
| $C_{\mathrm{CO_2}}$ | Total blood carbon dioxide content | mmol·L⁻¹ |
| $c_{\mathrm{CO_2}}^{\text{base}}$ | Base carbon dioxide carrying capacity | mmol·L⁻¹ |
| $\Delta_{\text{Haldane}}$ | Haldane shift in carbon dioxide carrying capacity | mmol·L⁻¹ |
| $C^{\text{mix}}$ | Flow-weighted mixed blood gas content at a node | mmol·L⁻¹ |
| $\mathrm{pH}$ | Tissue or blood pH | — |
| $[\mathrm{HCO_3^-}]$ | Tissue bicarbonate buffer concentration | mmol·L⁻¹ |

## Tissue transport

| Symbol | Description | Units |
|---|---|---|
| $\sigma$ | Tissue diffusion coefficient of the species | µm²·s⁻¹ |
| $D_z, D_y, D_x$ | Diffusive conductance across a grid cell face, per axis | µm³·s⁻¹ |
| $\mathbf{A}$ | Sparse diffusion operator matrix | — |
| $\mathcal{N}(i)$ | Set of grid cells face-adjacent to cell $i$ | — |
| $\varepsilon$ | Diagonal regularisation term | — |
| $P_{\text{perm}}$ | Endothelial permeability coefficient for the species | µm·s⁻¹ |
| $\Lambda$ | Pseudo-washout diagonal augmentation | — |
| $\gamma$ | Picard relaxation factor on the pseudo-washout | — |
| $\phi_{\mathrm{O_2}}, \phi_{\mathrm{CO_2}}$ | Transmural flux into a grid cell | mmol·s⁻¹ |
| $M_{\mathrm{O_2}}$ | Local metabolic oxygen consumption rate | mmol·L⁻¹·s⁻¹ |
| $M_{\mathrm{CO_2}}$ | Local metabolic carbon dioxide production rate | mmol·L⁻¹·s⁻¹ |
| $M_{\max}$ | Maximum metabolic consumption rate | mmol·L⁻¹·s⁻¹ |
| $k$ | Metabolic reduction constant for hypoxic regions | mmHg⁻¹ |
| $RQ$ | Respiratory quotient | — |
| $\mathbf{b}$ | Right-hand side vector of the linear system | — |
| $s_{\text{in},i}$ | Incoming arterial oxygen flux into grid cell $i$ | mmol·s⁻¹ |

## Symbols reused with different meanings

| Symbol | Meaning 1 | Meaning 2 | Meaning 3 |
|---|---|---|---|
| $A$ | Phase separation asymmetry parameter (E16) | Sparse diffusion operator matrix (E36) | — |
| $C$ | Hydraulic conductance (E12) | Pries–Secomb shape parameter (E5) | Blood gas content (E33) |
| $D$ | Vessel diameter (E1) | Diffusive face conductance (E35) | — |
| $L$ | Vessel segment length (E2) | Graph Laplacian (E13) | — |
| $P$ | Partial pressure (E31) | Permeability coefficient (E43) | — |
| $S$ | Vessel wall surface area (E28) | Haemoglobin saturation (E32) | — |

---

# Parameter and constant values

## Rheology and network flow

| Symbol | Parameter name | Value | Units | Source |
|---|---|---|---|---|
| $\mu_{\text{plasma}}$ | `mu_plasma` | 1.2 | mPa·s | [@kesmarky2008]: normal range 1.10–1.30 mPa·s at 37 °C (human) |
| — | Pries–Secomb law | `in_vivo` | — | — |
| — | Diameter floor for the Pries–Secomb relation | 3.0 | µm | — |
| $X_0$ | Phase separation skimming threshold | $0.964\,(1 - H_{\text{in}})/D_F$ (E18) | — | Derived from $H_{\text{in}}$ and $D_F$ (E18) [@pries1989; @rasmussen2018] |
| $H$ | `systemic_hematocrit` | 0.45 | — | [@dash2010]: standard haematocrit ≈ 0.45 (human) |
| $p_{\text{in}}$ | `input_p_bc` | $13.332\times10^{6}$ (100 mmHg) | mPa | [CITE — unconfirmed]. Measured resting MAP in conscious rats: WKY 116 ± 3, SHR 154 ± 3 mmHg [@li1997] |
| $p_{\text{out}}$ | `output_p_bc` | $0.27\times10^{6}$ (2 mmHg) | mPa | [@willenbrock1997]: central venous pressure 4 ± 3 mmHg in conscious control rats |
| — | `rheology_max_iterations` | 15 | — | — |
| — | `rheology_tolerance` | $1\times10^{-4}$ | — | — |
| — | `robin_distal_resistance_multiplier` | 10.0 | — | — |

## Blood gas chemistry

| Symbol | Parameter name | Value | Units | Source |
|---|---|---|---|---|
| $\alpha_{\mathrm{O_2}}$ | Oxygen plasma solubility | $1.34\times10^{-3}$ | mmol·L⁻¹·mmHg⁻¹ | [@dash2010]: $1.37\times10^{-3}$ in plasma at 37 °C (human) |
| $\alpha_{\mathrm{CO_2}}$ | Carbon dioxide plasma solubility | 0.03 | mmol·L⁻¹·mmHg⁻¹ | [@dash2010]: $3.07\times10^{-2}$ in plasma at 37 °C (human) |
| $n$ | Hill coefficient | 2.7 | — | [@dash2010]: $n = 2.7$ fits normal blood for saturations of 20–98% (human) |
| $c_{\mathrm{Hb,max}}$ | Pure red cell oxygen capacity | $0.446 \times 20.4 / 0.45 = 20.22$ | mmol·L⁻¹ | Derived (E33). Compare $4\,[\mathrm{Hb}]_{\text{rbc}} = 4 \times 5.18 = 20.7$ mmol·L⁻¹ [@dash2010] (human) |
| $P_{50}$ | Baseline P50 at pH 7.4, $P_{\mathrm{CO_2}}$ 40 mmHg | 26.0 | mmHg | [@dash2010]: about 26.8 mmHg (human); 26.0 is slightly lower |
| — | Bohr pH coefficient | $-0.4$ | — | — |
| — | Bohr $P_{\mathrm{CO_2}}$ coefficient | 0.06 | — | — |
| — | Carbon dioxide base capacity prefactor | 11.02 | — | — |
| — | Carbon dioxide base capacity exponent | 0.396 | — | — |
| — | Haldane shift coefficients | 0.15, 0.05 | — | — |
| $\mathrm{p}K_a$ | Henderson–Hasselbalch dissociation constant | 6.1 | — | [@severinghaus1956]: 6.10, revised to 6.09 at pH 7.4 and 37.5 °C (human and canine serum) |
| $[\mathrm{HCO_3^-}]$ | `hco3_tissue` | 24.0 | mmol·L⁻¹ | [@berend2014]: arterial reference $24 \pm 2$ mmol·L⁻¹ (human); used here for tissue |
| $P_{\mathrm{O_2}}^{\text{art}}$ | `po2_arterial_mmHg` | 100.0 | mmHg | [@dash2010]: standard arterial $P_{\mathrm{O_2}}$ = 100 mmHg (human) |
| $P_{\mathrm{CO_2}}^{\text{art}}$ | `pco2_arterial` | 40.0 | mmHg | [@dash2010; @berend2014]: arterial $P_{\mathrm{CO_2}}$ = 40 mmHg, reference $40 \pm 2$ (human) |

## Tissue transport

| Symbol | Parameter name | Value | Units | Source |
|---|---|---|---|---|
| — | `grid_resolution_xyz` | (10.0, 10.0, 10.0) | µm | — |
| $\sigma_{\mathrm{O_2}}$ | `sigma_diff` | $1.5\times10^{-9}$ ($=1.5\times10^{3}$ µm²·s⁻¹) | m²·s⁻¹ | Consistent with $K_{\mathrm{O_2}}/\alpha_{\mathrm{O_2}} \approx 1.6\times10^{-9}$ from rat skeletal muscle [@kawashiro1975] and $(1.04 \pm 0.78)\times10^{-9}$ in rat mesentery [@yaegashi1996] |
| $\sigma_{\mathrm{CO_2}}$ | `sigma_diff_co2` | $3.0\times10^{-8}$ ($=3.0\times10^{4}$ µm²·s⁻¹) | m²·s⁻¹ | [CITE — unconfirmed]. $K_{\mathrm{CO_2}}/\alpha_{\mathrm{CO_2}} \approx 1.6\times10^{-9}$ in rat skeletal muscle [@kawashiro1975], about 1/19 of this value. The 20× ratio holds for Krogh’s constant ($D\alpha$), not for $D$; see `cb_modelling_reference.md` open item 18 |
| $P^{\mathrm{O_2}}_{\text{perm}}$ | `permeability_o2_cm_s` | $1.0\times10^{-4}$ ($=1.0$ µm·s⁻¹) | cm·s⁻¹ | [CITE — unconfirmed] |
| $P^{\mathrm{CO_2}}_{\text{perm}}$ | `permeability_co2_cm_s` | $2.0\times10^{-3}$ ($=20.0$ µm·s⁻¹) | cm·s⁻¹ | [CITE — unconfirmed] |
| $M_{\max}$ | `M_max` | 0.005 | mmol·L⁻¹·s⁻¹ | Chosen; see `cb_modelling_reference.md` §10.9 and open item 8 |
| $k$ | `k_reduce` | 0.1 | mmHg⁻¹ | Chosen; phenomenological, see `cb_modelling_reference.md` §10.9 |
| $RQ$ | `respiratory_quotient` | 0.82 | — | [CITE — unconfirmed]. Measured 0.85 in rat skeletal muscle [@kawashiro1975] |
| $\gamma$ | Picard relaxation factor, multi-species | 1.0 | — | Numerical choice |
| $\varepsilon$ | Diagonal regularisation | $1\times10^{-12}$ | — | Numerical choice |
| — | `picard_max_iterations` | 50 | — | — |
| — | `picard_tolerance` | $1\times10^{-4}$ | — | — |
| — | Conjugate gradient relative tolerance | $1\times10^{-5}$ | — | — |
| — | Conjugate gradient maximum iterations | 500 | — | — |

## Upstream constants referenced by these equations

| Symbol | Parameter name | Value | Units |
|---|---|---|---|
| — | `PROCESSING_VOXEL_UM` (z, y, x) | (1.8639, 1.866, 1.866) | µm |
| — | `edt_junction_proximity_exclusion_um` | 3.73 | µm |
| — | `radius_assignment_mode` | `edt_radius` | — |

---

## Poiseuille resistance assignment

`src/ImageLynx/haemodynamics/poiseuille.py`

**(E1)** Legacy viscosity

$$\mu_{\text{old}} = \frac{1}{D^{1.647}}$$

**(E2)** Hagen–Poiseuille resistance

$$R = \frac{128\,\mu_{\text{old}}\,L}{\pi D^{4}}$$

---

## Coupled flow–haematocrit solver

`src/ImageLynx/haemodynamics/rheology.py`

**(E3)** Pries–Secomb reference relative viscosity at $H = 0.45$, in vivo

$$\mu_{0.45} = 6.0\,e^{-0.085D} + 3.2 - 2.44\,e^{-0.06 D^{0.645}}$$

**(E4)** Pries–Secomb reference relative viscosity at $H = 0.45$, in vitro

$$\mu_{0.45} = 220\,e^{-1.3D} + 3.2 - 2.44\,e^{-0.06 D^{0.645}}$$

**(E5)** Haematocrit shape parameter

$$C(D) = \left(0.8 + e^{-0.075D}\right)\left(-1 + \frac{1}{1 + 10^{-11}D^{12}}\right) + \frac{1}{1 + 10^{-11}D^{12}}$$

**(E6)** Haematocrit term

$$\Phi(H,D) = \frac{(1-H)^{C(D)} - 1}{(1-0.45)^{C(D)} - 1}$$

**(E7)** Cell-depleted wall layer

$$W = \left(\frac{D}{D - 1.1}\right)^{2}$$

**(E8)** Relative apparent viscosity, in vivo

$$\mu_{\text{rel}} = \Big[\,1 + (\mu_{0.45} - 1)\,\Phi\,W\,\Big]\,W$$

**(E9)** Relative apparent viscosity, in vitro

$$\mu_{\text{rel}} = 1 + (\mu_{0.45} - 1)\,\Phi$$

**(E10)** Apparent viscosity

$$\mu_{\text{app}} = \mu_{\text{rel}}\cdot\mu_{\text{plasma}}, \qquad \mu_{\text{plasma}} = 1.2$$

**(E11)** Resistance, initialisation at systemic haematocrit

$$R = R_0\,\frac{\mu_{\text{app}}(D, 0.45)}{\mu_{\text{old}}(D)}$$

This is E21 at $H = 0.45$. $R_0$ is defined there.

**(E12)** Edge conductance

$$C_{ij} = \frac{1}{R_{ij}}$$

**(E13)** Graph Laplacian

$$\mathbf{L} = \operatorname{diag}\!\left(\sum_{j} C_{ij}\right) - \mathbf{C}$$

**(E14)** Dirichlet-partitioned pressure solve

$$\mathbf{L}_{uu}\,\mathbf{p}_u = -\,\mathbf{L}_{uk}\,\mathbf{p}_k$$

**(E15)** Signed edge flow

$$Q_{ij} = \frac{p_i - p_j}{R_{ij}}$$

**(E16)** Phase separation asymmetry parameter [@pries1989; @rasmussen2018]

$$A = -13.29\,\frac{(D_1/D_2)^{2} - 1}{(D_1/D_2)^{2} + 1}\cdot\frac{1 - H_{\text{in}}}{D_F}$$

**(E17)** Phase separation steepness parameter [@pries1989; @rasmussen2018]

$$B = 1 + 6.98\,\frac{1 - H_{\text{in}}}{D_F}$$

The constants in E16–E18 are as printed in Rasmussen et al. (2018), who attribute them to
Pries, Reglin & Secomb (2003); other sources give the same form as Pries & Secomb (2005).
Which paper first printed them is unconfirmed.

$D_F$ is the diameter of the single inflowing edge. Where two or more edges merge into a node
that then splits, it is their flow-weighted mean diameter. Where no edge flows in, it falls
back to the larger of $D_1$ and $D_2$.

**(E18)** Phase separation logit relation [@pries1989; @rasmussen2018]

$$X_0 = 0.964\,\frac{1 - H_{\text{in}}}{D_F}$$

$$\operatorname{logit}(FQ_{E1}) = A + B\,\ln\!\left(\frac{FQ_1 - X_0}{1 - FQ_1 - X_0}\right), \qquad X_0 < FQ_1 < 1 - X_0$$

$$FQ_{E1} = 0 \;\text{ if }\; FQ_1 \le X_0, \qquad FQ_{E1} = 1 \;\text{ if }\; FQ_1 \ge 1 - X_0$$

The log term equals $\operatorname{logit}\!\left[(FQ_1 - X_0)/(1 - 2X_0)\right]$, the form Pries
et al. print. If $X_0 \ge 0.5$ ($D_F \lesssim 1.06$ µm at $H_{\text{in}} = 0.45$) the relation
is undefined, and both daughters take $H_{\text{in}}$.

**(E19)** Red cell flux fraction [@pries1989]

$$FQ_{E1} = \frac{1}{1 + e^{-\operatorname{logit}(FQ_{E1})}}, \qquad FQ_{E2} = 1 - FQ_{E1}$$

**(E20)** Daughter branch haematocrit [@pries1989; @pries1990]

$$H_1 = \min\!\left[\max\!\left(H_{\text{in}}\frac{FQ_{E1}}{FQ_1},\,0\right),\,0.95\right], \qquad H_2 = \min\!\left[\max\!\left(H_{\text{in}}\frac{FQ_{E2}}{FQ_2},\,0\right),\,0.95\right]$$

The clamp to $[0, 0.95]$ is not part of the Pries law. If it fires, red cell flux is no
longer conserved at the bifurcation. It has not fired for daughters of 3–30 µm.

**(E21)** Resistance, rescaled by the updated haematocrit

$$R = R_0\,\frac{\mu_{\text{app}}(D, H)}{\mu_{\text{old}}(D)}$$

$R_0$ is the resistance the edge arrives with, saved once before the solver changes it: E2,
or the integrated sphincter or pericyte resistance, all built with $\mu_{\text{old}}$ (E1).
An edge with no resistance gets $R_0 = 128\,\mu_{\text{old}}\,L / (\pi D^4)$, so E21 reduces to

$$R = \frac{128\,\mu_{\text{app}}(D, H)\,L}{\pi D^{4}}$$

For a constricted edge the rescale is approximate: $R_0$ integrates $\mu_{\text{old}}(d(x))$
along the edge, while the ratio uses the single edge diameter $D$.

The $H$ used here is under-relaxed against the previous pass,
$H \leftarrow H_{\text{old}} + \omega\,(H_{\text{new}} - H_{\text{old}})$ with $\omega = 0.5$
by default, where $H_{\text{new}}$ is from E20. Without it, the coupling can flip between two
states and never converge.

**(E22)** Wall shear stress

$$\tau_w = \frac{32\,\mu_{\text{app}}\,|Q|}{\pi D^{3}}$$

---

## Final conductance, effective resistance and flow solve

`src/ImageLynx/haemodynamics/resistance.py`

**(E23)** Edge conductance

$$C_{ij} = \frac{1}{R_{ij}}$$

**(E24)** Graph Laplacian

$$\mathbf{L} = \operatorname{diag}\!\left(\sum_{j} C_{ij}\right) - \mathbf{C}$$

**(E25)** Effective resistance between nodes $i$ and $j$

$$R_{\text{eff}}(i,j) = (\mathbf{e}_i - \mathbf{e}_j)^{\mathsf{T}}\,\mathbf{L}^{+}\,(\mathbf{e}_i - \mathbf{e}_j)$$

**(E26)** Dirichlet-partitioned pressure solve

$$\mathbf{L}_{uu}\,\mathbf{p}_u = -\,\mathbf{L}_{uk}\,\mathbf{p}_k$$

**(E27)** Edge flow

$$Q_{ij} = \frac{p_i - p_j}{R_{ij}}$$

---

## Vessel-to-grid mapping

`src/ImageLynx/haemodynamics/perfusion.py`

**(E28)** Vessel wall surface area contributed to a grid cell

$$S = 2\pi r\,\ell_{\text{cell}}$$

**(E29)** Length fraction of an edge lying in a grid cell

$$f_{\ell} = \frac{\ell_{\text{cell}}}{\ell_{\text{total}}}$$

---

## Advection–diffusion–reaction matrix assembly

`src/ImageLynx/haemodynamics/perfusion.py`

**(E30)** Total bulk flow through a grid cell

$$q_{\text{total},i} = \sum_{v \in i} Q_v\,f_{\ell,v}$$

**(E31)** P50 with Bohr shift

$$\log_{10} P_{50} = \log_{10}(26.0) - 0.4\,(\mathrm{pH} - 7.4) + 0.06\,\log_{10}\!\left(\frac{P_{\mathrm{CO_2}}}{40}\right)$$

**(E32)** Hill saturation

$$S_{\mathrm{O_2}} = \frac{P_{\mathrm{O_2}}^{\,n}}{P_{\mathrm{O_2}}^{\,n} + P_{50}^{\,n}}, \qquad n = 2.7$$

**(E33)** Blood oxygen content

$$C_{\mathrm{O_2}} = \alpha_{\mathrm{O_2}} P_{\mathrm{O_2}} + H\,c_{\mathrm{Hb,max}}\,S_{\mathrm{O_2}}, \qquad \alpha_{\mathrm{O_2}} = 1.34\times10^{-3}, \quad c_{\mathrm{Hb,max}} = \frac{0.446 \times 20.4}{0.45}$$

**(E34)** Incoming arterial oxygen flux per grid cell

$$s_{\text{in},i} = \sum_{v \in i} Q_v\,f_{\ell,v}\,C_{\mathrm{O_2}}(P_{\mathrm{O_2}}^{\text{art}}, H_v)$$

**(E35)** Diffusive face conductances, $\sigma$ in µm²/s, $\mathbf{res} = (r_z, r_y, r_x)$

$$D_z = \frac{\sigma\,r_y r_x}{r_z}, \qquad D_y = \frac{\sigma\,r_z r_x}{r_y}, \qquad D_x = \frac{\sigma\,r_z r_y}{r_x}$$

**(E36)** Seven-point diffusion operator

$$\mathbf{A}_{ii} = \sum_{j \in \mathcal{N}(i)} D_{ij} + \varepsilon, \qquad \mathbf{A}_{ij} = -D_{ij}, \qquad \varepsilon = 10^{-12}$$

---

## Multi-species perfusion solve

`src/ImageLynx/haemodynamics/perfusion.py`

**(E37)** Haldane oxygen saturation term

$$S_{\mathrm{O_2}}^{H} = \frac{P_{\mathrm{O_2}}^{2.7}}{P_{\mathrm{O_2}}^{2.7} + 26.0^{2.7}}$$

**(E38)** Base carbon dioxide carrying capacity

$$c_{\mathrm{CO_2}}^{\text{base}} = 11.02\,P_{\mathrm{CO_2}}^{0.396}$$

**(E39)** Haldane shift

$$\Delta_{\text{Haldane}} = \left(0.15 - 0.05\,S_{\mathrm{O_2}}^{H}\right) P_{\mathrm{CO_2}}$$

**(E40)** Blood carbon dioxide content

$$C_{\mathrm{CO_2}} = \alpha_{\mathrm{CO_2}} P_{\mathrm{CO_2}} + H\left(c_{\mathrm{CO_2}}^{\text{base}} + \Delta_{\text{Haldane}}\right), \qquad \alpha_{\mathrm{CO_2}} = 0.03$$

**(E41)** Henderson–Hasselbalch pH

$$\mathrm{pH} = 6.1 + \log_{10}\!\left(\frac{[\mathrm{HCO_3^-}]}{\alpha_{\mathrm{CO_2}}\,P_{\mathrm{CO_2}}}\right), \qquad [\mathrm{HCO_3^-}] = 24.0$$

**(E42)** Species-scaled diffusive face conductances

$$D_z = \frac{\sigma\,\alpha\,r_y r_x}{r_z}, \qquad D_y = \frac{\sigma\,\alpha\,r_z r_x}{r_y}, \qquad D_x = \frac{\sigma\,\alpha\,r_z r_y}{r_x}$$

**(E43)** Pseudo-washout diagonal augmentation

$$\Lambda^{\mathrm{O_2}}_i = P^{\mathrm{O_2}}_{\text{perm}}\,S_i\,\alpha_{\mathrm{O_2}}\,\gamma, \qquad \Lambda^{\mathrm{CO_2}}_i = P^{\mathrm{CO_2}}_{\text{perm}}\,S_i\,\alpha_{\mathrm{CO_2}}\,\gamma, \qquad \gamma = 1$$

**(E44)** Oxygen-dependent metabolic consumption

$$M_{\mathrm{O_2}} = M_{\max}\left(1 - e^{-k\,P_{\mathrm{O_2}}}\right)$$

**(E45)** Carbon dioxide production

$$M_{\mathrm{CO_2}} = M_{\mathrm{O_2}}\cdot RQ, \qquad RQ = 0.82$$

**(E46)** Nodal mixing of blood gas content

$$C^{\text{mix}} = \frac{\sum_{e \in \text{in}} C_e Q_e}{\sum_{e \in \text{in}} Q_e}$$

**(E47)** Inversion of content to partial pressure (root find)

$$C_{\mathrm{O_2}}(P_{\mathrm{O_2}}, H, P_{\mathrm{CO_2}}, \mathrm{pH}) - C^{\text{mix}}_{\mathrm{O_2}} = 0, \qquad C_{\mathrm{CO_2}}(P_{\mathrm{CO_2}}, H, P_{\mathrm{O_2}}) - C^{\text{mix}}_{\mathrm{CO_2}} = 0$$

**(E48)** Transmural oxygen flux into a grid cell

$$\phi_{\mathrm{O_2}} = P^{\mathrm{O_2}}_{\text{perm}}\,S\,\alpha_{\mathrm{O_2}}\left(P_{\mathrm{O_2}}^{\text{blood}} - P_{\mathrm{O_2}}^{\text{tissue}}\right)$$

**(E49)** Transmural carbon dioxide flux into a grid cell

$$\phi_{\mathrm{CO_2}} = P^{\mathrm{CO_2}}_{\text{perm}}\,S\,\alpha_{\mathrm{CO_2}}\left(P_{\mathrm{CO_2}}^{\text{blood}} - P_{\mathrm{CO_2}}^{\text{tissue}}\right)$$

**(E50)** Blood content depletion along an edge

$$C_{\mathrm{O_2}} \leftarrow C_{\mathrm{O_2}} - \frac{\phi_{\mathrm{O_2}}}{Q}, \qquad C_{\mathrm{CO_2}} \leftarrow C_{\mathrm{CO_2}} - \frac{\phi_{\mathrm{CO_2}}}{Q}$$

**(E51)** Oxygen right-hand side

$$\mathbf{b}_{\mathrm{O_2}} = \boldsymbol{\phi}_{\mathrm{O_2}} - M_{\mathrm{O_2}}V_{\text{cell}} + \Lambda^{\mathrm{O_2}}\,\mathbf{P}_{\mathrm{O_2}}$$

**(E52)** Carbon dioxide right-hand side

$$\mathbf{b}_{\mathrm{CO_2}} = \boldsymbol{\phi}_{\mathrm{CO_2}} + M_{\mathrm{CO_2}}V_{\text{cell}} + \Lambda^{\mathrm{CO_2}}\,\mathbf{P}_{\mathrm{CO_2}}$$

**(E53)** Linear systems solved per Picard iteration

$$\mathbf{A}_{\mathrm{O_2}}\,\mathbf{P}_{\mathrm{O_2}} = \mathbf{b}_{\mathrm{O_2}}, \qquad \mathbf{A}_{\mathrm{CO_2}}\,\mathbf{P}_{\mathrm{CO_2}} = \mathbf{b}_{\mathrm{CO_2}}$$

**(E54)** Picard convergence criterion

$$\frac{\lVert \mathbf{P}^{(k+1)} - \mathbf{P}^{(k)} \rVert}{\lVert \mathbf{P}^{(k+1)} \rVert + 10^{-12}} < \text{tol}$$

**(E55)** Final tissue pH

$$\mathrm{pH} = 6.1 + \log_{10}\!\left(\frac{24.0}{0.03\,P_{\mathrm{CO_2}}}\right)$$
