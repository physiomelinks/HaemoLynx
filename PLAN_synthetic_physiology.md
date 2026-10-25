# Physiological synthetic networks (`PLAN_synthetic_physiology.md`)

Follows `PLAN_synthetic_network.md` (done 2026-10-25). Harvey, 2026-10-25: "The synthetic network
does not look physiological, and doesn't have defined inlets and outlets, make a new plan to make
better synthetic networks."

## What is wrong, measured (2026-10-25)

The E14.5 template run and its synthetic networks (`9e49ddfc`, both beds), rendered side by side
and measured by `outputs/synthetic_review/review.py` and `organisation.py` in the
`plan-synthetic-e08d0c` worktree (git-ignored; image `template_vs_synthetic.png`):

| | E14.5 template | synthetic (Delaunay / Voronoi) |
|---|---|---|
| large vessels' depth z, P5 / median / P95 (um) | 2.6 / 11.9 / 23.0 | 16 / 70 / 191 |
| capillary length density, 0-30 um band against the rest (mm/mm^3) | 2,340 against 250-555 | 520-840 in every band, falling to 40-70 at the top |
| capillary length within 30 degrees of z / y / x (13.4% isotropic) | 21% / 10% / 19% | 14-15% each |
| box within 20 um of a capillary | 45% | 33-36% |
| box beyond 50 um of any capillary | 4.8% | 37-40% |
| inflow through the inlets / outflow through the outlets | 46% / 79% | 74-84% / 38-43% |
| open ends (dead ends at the faces and inside) | 393 | 280-281 |
| dead ends inside the box | 305: unknown 94, island ends 94, stops short 96, artefacts 17, broken 4 | 194 blind |

What that says:

1. **The template is layered and the synthetic network is not.** E14.5's large vessels lie flat in
   a plexus on one face (z under 25 um), with the densest capillaries there; capillaries run up
   from it through the depth (z-aligned) and along x. The synthetic trees wander through the whole
   depth, and its capillaries run every way alike.
2. **The bed does not fill the tissue.** Nearly 40% of the synthetic box is more than 50 um from a
   capillary, against 5% in the template. This is decision 3 of `PLAN_synthetic_network.md` as
   built (`ff4f4dcb`): the bed spaced by the template's capillary chord, "filling only as much of
   the box as the count needs, grown from the trees". It rested on my reading that E14.5's
   capillaries were clustered, which the rendering shows to be wrong: they fill the box, and are
   short because the plexus is dense near the large vessels and the vertical capillaries are
   cross-linked, not because they gather in one part of the box.
3. **The trees are random walks.** Grown generation by generation with a space-filling score and
   bent with one `sin^2` bow each, the large vessels curl through the box; the template's are long,
   gently curving trunks in their plexus with side branches.
4. **Inlets and outlets are not what the flow uses.** The generator puts the template's one inlet
   and two outlets on the faces it found them on, but every vessel the box cuts is an open end
   (`open_end_model=estimated`): 57-62% of the synthetic outflow leaves through those, not the
   outlets. (The template run is open too: 54% of its inflow comes through open ends.) In the
   viewer the boundary-node layer, which marks them, starts hidden (`gui/results.py`
   `_from_assign_boundaries`, for every run).
5. **The arterial and venous sides do not interdigitate.** The bed joins whichever leaves are
   near; nothing makes each capillary patch fed from an arteriole and drained to a venule, so the
   arteriole-to-venule path lengths are whatever the bed gives.
6. **The template's dead ends are copied.** E14.5's 305 interior dead ends are almost all the
   segmentation and graph building's (`graph/dead_ends.py` kinds; the AIA patella's were
   fragmentation too, Harvey's marks 2026-10-20). Reproducing them as 194 blind capillaries copies
   an imaging artefact into a model network.

## Goal

A synthetic network that someone who knows the tissue would take for it, with all the flow
entering at defined inlets and leaving at defined outlets, matched to the template where the
template describes the tissue (counts, lengths, widths, branch orders, loops) and to its spatial
organisation (where the large vessels lie, how dense the capillaries are where, which way they
run), and not to its artefacts.

## Decisions for Harvey

- **D1. What "physiological" is judged by.** Proposed: (a) his marks on a grid of renders --
  template and synthetic variants, shuffled and unlabelled, each marked "real-looking" or not, the
  way `CLAUDE.md`'s marked grids calibrate a rule; and (b) numeric bands on the measures above
  (depth profile, orientation, coverage, large-vessel layer), added to the template comparison.
- **D2. Defined inlets and outlets.** (a) **closed network**: flow enters only at the inlets and
  leaves only at the outlets; vessels reaching a face are not cut open there (the bed keeps clear of
  the faces, or its face ends are closed), and `open_end_model` does not apply; (b) the template's
  face cuts kept as open ends, as now; (c) (a) with more inlets and outlets: every large vessel the
  template's box cut (E14.5's large tier leaves its faces more than once) becomes one. And the
  viewer shows the inlets and outlets of a synthetic run (the boundary-node layer visible).
- **D3. Interior dead ends.** (a) none by default (`synthetic_interior_dead_ends` off); (b) only
  the template's dead ends of the kinds that may be the tissue's own (`blind`, which only a person
  marks); (c) as now, the template's count.
- **D4. How the organisation is captured.** (a) **from the template**: measured fields --
  capillary length density and mean orientation on a coarse grid, the large vessels' occupancy --
  that the generator draws to, for any tissue; (b) tissue presets (cortex with a pial plexus and
  penetrating vessels, muscle with fibre-parallel capillaries, ...); (c) both, the preset as the
  default where the template is too small to say. Recommended: (a).
- **D5. How the trees grow.** (a) trunks with side branches confined to the template's
  large-vessel occupancy, low curvature, the side branches feeding the bed (cheap, near what is
  built); (b) constrained constructive optimisation (Karch et al. 1999): terminals added one at a
  time where they best supply the tissue at least vessel volume, Murray at every junction except
  the trunk's, giving physiological territories (slower, more work); (c) (a) now, (b) later.
- **D6. Decision 3 replaced.** The bed fills the template's capillary density field (D4) instead of
  "only as much of the box as the count needs, grown from the trees" (built `ff4f4dcb` on my wrong
  reading). Needs his OK, since he chose 3.
- **D7. Diameters.** (a) as now, the template's per class and order; (b) then adapted to the
  flow (structural adaptation, Pries, Secomb & Gaehtgens 1998), so wall shear and pressure are in
  physiological balance -- moves every width off the template's; (c) later, if wanted.

## Design (subject to D1-D7)

Phases, each leaving the suite green and the fixture templates' networks either identical or
changed only where the phase says.

1. **Measure the organisation** (`synthetic/organisation.py`, new; read from the template's graph,
   which both sources hold): a coarse grid over the box (about 40-60 um cells, the template's
   capillary spacing): capillary length density per cell, the length-weighted orientation tensor
   per cell, large-vessel length per cell; the axis of strongest variation and the profile along it;
   coverage (share of the box within 20 / 50 um of a capillary); trunk straightness per large order
   (chord over length along a run of orders); arteriole-to-venule path lengths through the bed.
   Recorded on `NetworkTemplate` (an older template loads without them), reported in the comparison
   with bands (D1b).
2. **The bed fills the tissue** (D6): bed vertices drawn with density proportional to the
   template's capillary length density (rejection-sampled jittered lattice, per-cell spacing so a
   cell's segment length is the template's), never far from where the template has capillaries;
   candidate edges scored also by alignment with the cell's orientation tensor, so the bed runs
   the way the template's capillaries run. The counts stay exact as now (degree sequence). Both
   beds (Delaunay, Voronoi) take the same density and orientation.
3. **Trees in their layer** (D5a): roots on the template's faces as now; each large order grown
   inside the cells the template's large vessels occupy, as trunks that keep their heading (bend
   limited by the template's trunk straightness) and side branches at the template's angle; the
   small tier (Art/Ven) grows from the large tier into the bed. Interdigitation: arterial and
   venous side-branch terminals placed alternately along the bed, each bed patch joined to its
   nearest feed and drain, so arteriole-to-venule paths match the template's (D2's closed network
   needs this).
4. **Defined inlets and outlets** (D2): the closed network -- the bed keeps clear of the faces,
   tree vessels the box would cut are turned back in or end in the bed -- so the only boundary
   nodes are the inlets and outlets; a setting chooses closed or open (the template's face cuts).
   The panel shows a synthetic run's inlets and outlets (the boundary-node layer visible, with its
   legend) and the run log names them.
5. **Dead ends** (D3): interior dead ends off by default; the dead-end comparison rows say what
   was left out and why.
6. **Vessel shape**: polylines with the template's curvature (curvature per micron and its spread,
   measured in phase 1) as low-frequency curves, not one `sin^2` bow per vessel; tortuosity stays
   matched.
7. **Optional, later** (D5b, D7): CCO growth; structural adaptation of the diameters.

## Tests

- `tests/test_synthetic_organisation.py` (new): the fields on hand-built networks with known
  answers -- a slab of parallel vessels (orientation tensor, density in one layer), a uniform
  lattice (flat profile, full coverage); an older template without them loads.
- `tests/test_synthetic_generate.py`: on fixture templates given a layer and a direction, the drawn
  network's depth profile, orientation and coverage are within band; large vessels inside their
  occupancy; a closed network's only boundary nodes are its inlets and outlets and a solve puts all
  inflow through the inlets (to the solver's tolerance); no interior dead ends by default;
  determinism; the counts still exact.
- `tests/test_synthetic_compare.py`: the new rows and their bands, one `TOLERANCES` table.
- `tests/test_synthetic_stage.py`: a whole closed run: every inflow through an inlet, every outflow
  through an outlet; the manifest records the new options.
- GUI (`tests/test_gui_synthetic_widget.py`, real display): the boundary-node layer shows for a
  synthetic run; the new settings sit behind the synthetic Advanced button.

## Verification

1. E14.5 as the template (the harness in `outputs/synthetic_e14p5/`): the table above again, every
   new measure within band; the inflow and outflow 100% through the inlets and outlets when closed.
2. **Harvey's grid** (D1a): about 12 renders (top and side views) -- the template, synthetic
   networks before and after this plan, both beds -- shuffled and numbered, marked by Harvey; the
   rule: the new networks are taken for the tissue as often as the template's own render is.
3. A second tissue as the template, to show the organisation is read, not built in: none of the
   saved `.haemorun` files on this machine can be one today (the AIA runs used global vessel masks,
   which a template refuses; Mason's skeletal muscle is on the often-absent E: drive), so this
   waits on a run with arteriole/venule masks or manual boundaries.

## References

- Karch, Neumann, Neumann & Schreiner (1999). A three-dimensional model for arterial tree
  representation, generated by constrained constructive optimization. Comput Biol Med 29:19-38.
  https://doi.org/10.1016/S0010-4825(98)00045-6
- Smith, Doyeux, Berg et al. (2019). Brain capillary networks across species: a few simple
  organizational requirements are sufficient to reproduce both structure and function. Front
  Physiol 10:233. https://doi.org/10.3389/fphys.2019.00233 (constrained Voronoi capillary beds)
- Linninger et al. (2013). Cerebral microcirculation and oxygen tension in the human secondary
  cortex. Ann Biomed Eng. https://doi.org/10.1007/s10439-013-0828-0
- Hartung et al. (2021). Mathematical synthesis of the cortical circulation for the whole mouse
  brain, part II: microcirculatory closure. Microcirculation, e12687 (arterial and venous trees
  joined through a synthetic capillary bed)
- Pries, Secomb & Gaehtgens (1998). Structural adaptation and stability of microvascular networks:
  theory and simulations. Am J Physiol 275:H349-H360
- Hamarneh & Jassi (2010). VascuSynth: simulating vascular trees for generating volumetric image
  data with ground truth segmentation and tree analysis. Comput Med Imaging Graph 34:605-616

## Status

**Decided by Harvey, 2026-10-25:**

- D1: **a + b** -- his marks on a shuffled grid of renders, and numeric bands on the organisation
  measures in the template comparison.
- D2: **a** -- a closed network by default (flow enters only at the inlets, leaves only at the
  outlets, no open ends); the template's face cuts kept as an option; the viewer shows a synthetic
  run's inlets and outlets.
- D3: **a** -- no interior dead ends by default.
- D4: **a** -- the organisation read from the template.
- D5: **a** -- trunks with side branches in the template's large-vessel layer.
- D6: **a** -- the bed fills the template's capillary density instead of being grown from the trees.
- D7: **a and b** -- the template's widths, and adaptation to the flow (Pries, Secomb & Gaehtgens
  1998) built now; read as an option, off by default, so the comparison's diameter rows keep
  their meaning (to confirm with Harvey).

Being built in the phase order above.

**Built:**

- Phase 1 (`af57dcd2`): `synthetic/organisation.py` measures the layout on every template (and on
  the synthetic network the comparison reads), its rows banded in `compare.TOLERANCES`.
- Phase 2: the bed's vertices are laid cell by cell in proportion to the template's capillary
  vertices (`bed.density_vertices`; the Voronoi seeds likewise, the tessellation drawn three times
  over and contracted against the spacing each cell wants), a vessel's wanted chord scales with
  the local spacing, and the fill prefers candidates along the cell's orientation tensor
  (`bed.ORIENTATION_WEIGHT`). `bed_volume_um3` and the bed grown from the trees are gone (D6). A
  template without an organisation draws exactly as before. E14.5, three seeds, no flows
  (`outputs/synthetic_e14p5/full_compare.py`): rows within band 324 -> 340 of 372 (Delaunay),
  338 (Voronoi); capillary vertices by cell correlate 0.96 with the template's (Voronoi 0.8 at
  1,000 vertices), the box beyond 50 um of a capillary 44% -> 6% (template 5%). Still out: the
  share of capillary length along z (0.14 against 0.21; a heavier orientation weight lengthened
  the capillaries out of band instead), the trees' place (phase 3).
