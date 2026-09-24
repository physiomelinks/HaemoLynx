# Equation dependency flow after vessel radius assignment

Arrows show **data dependency**: an arrow from X to Y means Y consumes a quantity X produced.
Dashed back-edges are iteration loops. Node numbering matches `post_radius_assignment_equations.md`.

Split into two charts at the 1D-to-3D handover, since all 55 equations in one chart render far too wide to read.

Legend:

- **Grey dashed outline** — available but not executed in the default configuration.
- **Amber fill** — computed, but its result is not consumed by anything downstream.

---

## Chart 1 — Network haemodynamics (E1 to E27)

```mermaid
flowchart TD

  R0(["Per-edge vessel radius r, diameter D, segment length L"])

  subgraph S1["Poiseuille resistance assignment · not taken on this path"]
    E1["E1 · mu_old = 1 / D^1.647"]
    E2["E2 · R = 128 mu_old L / pi D^4"]
    E1 --> E2
  end

  VTKR["VTK cell_data resistance"]

  subgraph S3["Coupled flow-haematocrit solver"]
    E3["E3 · mu_45 in vivo"]
    E4["E4 · mu_45 in vitro"]
    E5["E5 · C of D shape parameter"]
    E6["E6 · Phi haematocrit term"]
    E7["E7 · W cell-depleted wall layer"]
    E8["E8 · mu_rel in vivo"]
    E9["E9 · mu_rel in vitro"]
    E10["E10 · mu_app = mu_rel x mu_plasma"]
    E11["E11 · R = 128 mu_app L / pi D^4<br/>initialisation"]
    E12["E12 · C_ij = 1 / R_ij"]
    E13["E13 · Laplacian"]
    E14["E14 · Luu pu = -Luk pk"]
    E15["E15 · Q_ij = dp / R_ij"]
    E16["E16 · A bifurcation asymmetry"]
    E17["E17 · B logistic steepness"]
    E18["E18 · logit of FQe1"]
    E19["E19 · FQe1 red cell flux fraction"]
    E20["E20 · H1 and H2 daughter haematocrit"]
    E21["E21 · R = 128 mu_app L / pi D^4<br/>recomputed from the new haematocrit"]
    E22["E22 · tau_w = 32 mu_app Q / pi D^3"]

    E3 --> E8
    E4 --> E9
    E5 --> E6
    E6 --> E8
    E6 --> E9
    E7 --> E8
    E8 --> E10
    E9 --> E10
    E10 --> E11
    E11 --> E12
    E12 --> E13 --> E14 --> E15
    E15 --> E16
    E15 --> E18
    E16 --> E18
    E17 --> E18
    E18 --> E19 --> E20
    E20 --> E10
    E10 --> E21
    E10 --> E22
    E15 --> E22
    E21 -. "x15 or until tolerance 1e-4" .-> E12
  end

  subgraph S4["Final conductance, effective resistance and flow solve"]
    E23["E23 · C_ij = 1 / R_ij"]
    E24["E24 · Laplacian"]
    E25["E25 · R_eff via Laplacian pseudoinverse"]
    E26["E26 · Luu pu = -Luk pk"]
    E27["E27 · Q_ij = dp / R_ij"]
    E23 --> E24
    E24 --> E25
    E24 --> E26 --> E27
  end

  STOP["rheology stop reason, iterations,<br/>max flow change · G.graph and VTK field_data"]
  STATS["resistance-weighted betweenness<br/>and communities · reported"]
  OUT(["Q, p, tau_w per edge<br/>to tissue transport"])

  R0 --> E1
  R0 --> E3
  R0 --> E5
  R0 --> E7
  R0 --> E11
  R0 --> E16
  R0 --> E17
  R0 --> E22
  E21 --> VTKR
  E21 --> STATS
  E15 --> STOP
  STOP --> STATS
  STOP --> OUT
  E21 --> E23
  E27 --> OUT
  E22 --> OUT

  classDef unused fill:#f2f2f2,stroke:#b0b0b0,stroke-dasharray:5 4,color:#7a7a7a;
  classDef deadend fill:#fdf0dc,stroke:#d99b3d,color:#6b4a12;
  class E1,E2,E4,E9 unused;
  class E25 deadend;
```

---


## Chart 2 — Tissue transport (E28 to E55)

```mermaid
flowchart TD

  IN(["Q per edge from E27<br/>vessel radius r"])

  subgraph S5["Vessel-to-grid mapping"]
    E28["E28 · S = 2 pi r l_cell"]
    E29["E29 · f_l = l_cell / l_total"]
  end

  subgraph S6["ADR matrix assembly"]
    E30["E30 · q_total per grid cell"]
    E31["E31 · P50 with Bohr shift"]
    E32["E32 · Hill saturation S_O2"]
    E33["E33 · C_O2 blood oxygen content"]
    E34["E34 · s_in arterial oxygen flux"]
    E35["E35 · Dz Dy Dx face conductances"]
    E36["E36 · A seven-point diffusion operator"]
    E31 --> E32 --> E33 --> E34
    E35 --> E36
  end

  subgraph S7["Multi-species perfusion solve"]
    E37["E37 · S_O2 Haldane term"]
    E38["E38 · c_CO2 base capacity"]
    E39["E39 · Haldane shift"]
    E40["E40 · C_CO2 blood CO2 content"]
    E41["E41 · pH Henderson-Hasselbalch"]
    E42["E42 · Dz Dy Dx scaled by alpha"]
    E43["E43 · Lambda pseudo-washout diagonal"]
    E44["E44 · M_O2 metabolic consumption"]
    E45["E45 · M_CO2 = M_O2 x RQ"]
    E46["E46 · C_mix nodal mixing"]
    E47["E47 · invert content to partial pressure"]
    E48["E48 · phi_O2 transmural flux"]
    E49["E49 · phi_CO2 transmural flux"]
    E50["E50 · blood content depletion along edge"]
    E51["E51 · b_O2 right-hand side"]
    E52["E52 · b_CO2 right-hand side"]
    E53["E53 · A P = b conjugate gradient solve"]
    E54["E54 · Picard convergence check"]
    E55["E55 · final tissue pH"]

    E37 --> E39
    E38 --> E40
    E39 --> E40
    E42 --> E53
    E43 --> E51
    E43 --> E52
    E43 --> E53
    E44 --> E45
    E44 --> E51
    E45 --> E52
    E41 --> E47
    E46 --> E47
    E47 --> E48
    E47 --> E49
    E48 --> E50
    E49 --> E50
    E50 --> E46
    E48 --> E51
    E49 --> E52
    E51 --> E53
    E52 --> E53
    E53 --> E54
    E53 --> E55
    E54 -. "x50 or until tolerance 1e-4" .-> E41
    E54 -.-> E44
  end

  OUT2(["Tissue PO2, PCO2, pH fields"])

  IN --> E28
  IN --> E29
  IN --> E30
  IN --> E34
  IN --> E46
  E29 --> E30
  E29 --> E34
  E33 --> E46
  E33 --> E47
  E40 --> E46
  E40 --> E47
  E28 --> E43
  E28 --> E48
  E28 --> E49
  E31 --> E41
  E32 --> E37
  E53 --> OUT2
  E55 --> OUT2

  classDef deadend fill:#fdf0dc,stroke:#d99b3d,color:#6b4a12;
  class E30,E34,E36 deadend;
```

## Notes on the amber nodes

**E25** — the two-point effective resistance is reported and never used again. Since the
fix in `69dbe70` it is at least computed from the converged Pries-Secomb resistances, so the
number is now a resistance in physical units; it is simply not consumed by anything.

**E1 and E2 are no longer executed on this path.** The carotid body driver passes
`assign_resistance=False` since `aecc53d`, so the power-law viscosity and the resistance
built from it are never computed. They stay in the list, and in the chart, because the
capability remains in `set_poiseuille_resistances` and the two `resistance_network_pipeline`
drivers still use it — those have no rheology stage at all, so it is the only resistance they
have. Same status as `E4` and `E9`, the in vitro relations: available, not taken here.

`set_poiseuille_resistances` is still called on the carotid body path, for
`assigned_diameter_um` and the diameter-provenance guard. Both come from the measured
radius.

**E30, E34, E36** — `build_adr_matrix` is called unconditionally and produces `q_total`,
`s_incoming` and the diffusion operator, but `solve_multi_species_perfusion` does not receive
them and builds its own matrices via E42. They are computed and discarded in the default
configuration.

## Notes on the two loops

**E21 back to E12** — the rheology solver. Each pass recomputes viscosity from the updated
haematocrit, rebuilds the Laplacian, re-solves for pressure and re-splits red cells at every
bifurcation. Runs up to 15 times. `STOP` records why it ended — `converged`, `flow_cycle` or
`max_iterations` — from the pass-to-pass change in the `E15` flows, and travels with the
statistics and the output (`09346d7`). An asymmetric Y-split never converges: its smaller
daughter's haematocrit alternates between two values on successive passes, so the loop ends on
`max_iterations`.

**E54 back to E41 and E44** — the Picard loop. Each pass recomputes pH and metabolic rate
from the updated tissue partial pressures, re-walks the vessel tree, and re-solves both
diffusion systems. Runs up to 50 times.

## What changed in `7ea1b36` and `69dbe70`

**The resistance rescale is gone.** The in-loop update was
`R = R_original x mu_app / mu_old`, which double-applied viscosity and inflated every
resistance by 200-540x. It is now a direct Poiseuille recompute, which is why `E21` carries
the same expression as `E11`. The old `E24` no longer exists and everything after it has
shifted down by two.

**The VTK resistance array is refreshed after the solve.** `graph_to_vtk` still runs
early, because the visualiser and the flow solve both need its paths, but the post-solve pass
now rewrites `resistance` alongside `hematocrit`, `viscosity` and `wall_shear_stress_pa`. The
exported mesh no longer carries a resistance and a viscosity from two different models.

**The vessel statistics moved after the solve.** `compute_comprehensive_vessel_statistics`
weights betweenness and community detection by the edge resistance, and ran before the
solver. `STATS` now hangs off `E21`. The effect on the numbers is small — both statistics use
only relative weights, and the two models rank edges similarly — so this corrected the
provenance rather than the values.

**The two-point resistance moved after the solve.** It used to sit in its own block before
the rheology solver, computed from the `E2` power-law resistances. It is now `E25`, inside
the final block, computed from the converged ones. The separate pre-solve conductance and
Laplacian went with it, so the graph is assembled once rather than twice.
