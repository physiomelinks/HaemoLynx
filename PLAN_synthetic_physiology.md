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
- Phase 3: each side's trees are scored by the template's tree length of that side in the cells
  they end in and the two spacings ahead along their heading, and by their room to go on before
  a face (`trees.TreeLayer`, `LAYER_WEIGHT`, `LAYER_LOOKAHEAD_SPACINGS`; read between cell
  centres), in place of their distance from the faces; a root's first vessel is aimed the same
  way. Without a layout the trees grow as before. E14.5 (seeds 1-6): the share of tree length in
  the template's tree cells 0.20 -> 0.73 (0.66-0.85; band 0.75 and up, met by 2-3 of 6), 95% within
  one cell of them, every tree in the plexus layer (z under 48 um) but for 0-10%; the rest is
  chords one 50 um cell off a corridor one cell wide (a bifurcation there cannot keep both
  daughters in it). Heavier weights, a longer lookahead, more planes per junction did not raise it.
  The trunks' turn was already within band (37 against 40 degrees). Interdigitation is not built
  separately: with each side's trees where the template's are, the feeds and drains alternate
  where the template's do.
- Phase 4: `synthetic_open_ends` (`closed`, the default, or `template`; `GenerationOptions.open_ends`).
  Closed, a tree child that would leave the box is turned back off the face at its full length
  (its direction mirrored in it), else shortened to end inside; the bed draws no face ends; a bow
  that would leave the box is turned about its chord (`polylines.BOW_TURNS`; in both modes); a
  bed vertex meant to be a junction but left with one vessel takes one more (in both modes: it
  was a dead end nobody asked for). The comparison reports the leaving-the-image row as
  "not drawn (a closed network)" and leaves the dead-end total unbanded
  (`compare.not_drawn`, read from the run's generation options). The viewer shows a synthetic
  run's boundary nodes (`gui/results.py`). E14.5 (the whole pipeline, statistics source, seed
  20240917): 100% of the inflow through the inlet and of the outflow through the outlets (the
  template's image run: 46% and 79%, the rest through 85 of its open ends). **For Harvey:** with
  the template's 91 vessels leaving the image closed off and its vessel count kept, the bed has
  about 64 more loops (295 against 231), average degree 2.60 against 2.31 and 777 branching
  points against 702, all out of band. A closed network cannot keep the vessel count, the loops
  and the dead ends at once; joining the cut ends in pairs would instead keep the loops at about
  231 + 45 with about 45 fewer vessels. Which to hold is his call.
- Phase 5: `synthetic_interior_dead_ends` off by default (`GenerationOptions.interior_dead_ends`
  likewise): a tree leaf the template ended blind joins the bed, so the boundary-node counts are
  the template's or more (reported "not held"), and the interior dead-end rows "not drawn".
  **Dead ends not drawn are joined in pairs** (`generate.dead_end_pairs_joined`): each pair of the
  template's ends leaving the image (closed) or inside the tissue (no interior dead ends) is taken
  as one vessel broken in two -- one capillary fewer, one loop more, the capillaries' total length
  kept -- and the comparison judges the vessel, loop, node and degree rows against what the pairs
  make (its tolerance column says so; the template column keeps the template's own). Kept at the
  template's vessel count instead, a bed without dead ends needs every vertex a junction: on
  E14.5 that drew 422 loops against 231 (+83%); holding the loops instead would drop the
  capillaries by 42%. The pairs (138 on E14.5) give 375 loops, expected 369. **For Harvey** (with
  Phase 4's question): the pairs are my choice of the three; say if another should hold.
- Phase 6: a vessel's curve has 1 to 4 lobes (`polylines.lobes`, `sin(pi s) sin(k pi s)`: one is
  the old bow), whichever turns nearest the template's curvature per 10 um at the same length
  (capillaries and trees each their own). That alone did nothing on E14.5: 93% of the bed's
  capillaries were straight, their chords (mean 38 um) leaving no room for the template's length
  (36 um, chord 30) -- a lattice puts neighbours a cell spacing apart, while the template's
  vertices sit 16.8 um from their nearest (half of them in a plexus sheet 12-25 um deep, thinner
  than a cell). So at the template's own size the bed's vertices are its capillary vertices each
  moved by a normal step of their median nearest-neighbour spacing (`bed.template_vertices`,
  `organisation.capillary_node_positions_rel` / `capillary_node_spacing_um`, not kept in the run's
  record); a scaled box keeps the cell field. E14.5 with the defaults: no straight capillary (was
  93%), curvature 27 degrees per 10 um against the template's 26 (was 4), the trees 30 against 29.
  The comparison now scores a layout only where the template has one (`compare.dispersion`: the
  vertices' variance over mean between cells, or along an axis, at least 2; E14.5: cells 3.6, z 226,
  y 6.6, x 0.8 -- its x profile, out at every phase, is even). E14.5, three seeds, no flows:
  322 of 348 banded rows within. Still out: average degree (3.0; every vertex a junction), loops
  by 5-7 of 369, the coverage (12% of the box beyond 50 um of a capillary against 5%: fewer,
  longer vessels), the share along z (0.11 against 0.21), the daughter angle (90 against 114,
  out before this plan), the trees' share in their cells in 2 of 3 seeds.
- **Grid D1a, marked by Harvey 2026-10-25: not met.** Twelve renders of E14.5 (from above and from
  the side, one scale), shuffled: the template, four networks drawn before this plan (Delaunay and
  Voronoi, two seeds each) and seven after (Delaunay seeds 11-14, Voronoi 11-13), at `79aa50a3`
  (`outputs/synthetic_review/grid_d1a.png` and `grid_d1a_key.csv` in the `plan-synthetic-e08d0c`
  worktree). Harvey marked only #7 T (could be the imaged tissue): #7 is the template. Every
  synthetic network, before and after, was marked S: 0 of 7 taken for the tissue against the
  template's 1 of 1. What still sets the template apart, as I read the renders (not Harvey's
  reasons, which are not recorded): its trunks run straight with short side branches at right
  angles, where the drawn ones wander; from the side its capillaries rise from the plexus as evenly
  spaced near-vertical columns, where the drawn bed is a random mesh; from above its capillaries
  are smooth curves round rounded meshes, where the Delaunay bed is straight-sided polygons. Phase 7
  (flow-adapted widths) is not built: it would not change what the renders show.
  Harvey, on the columns: "correct but only true for this particular dataset" -- so whatever
  answers them is read from the template, never built in. Measured on E14.5 (seed 1): capillaries
  whose chord lies within 30 degrees of z, 182 in the template (135 connected runs, 12 spanning
  100 um or more of depth, 5 spanning 200) against 103-112 drawn (4-7 and 0-1); a straight
  continuation through junctions does not separate them (the straightest pair at a junction:
  median 140 against 129-134 degrees). The per-layer direction tensor is matched once the bed sits
  on the template's vertices and the direction weight is raised (4-10: zz above the plexus
  0.37-0.57 against 0.36-0.64), but a tensor is an average: the template's share along z (0.21)
  needs its vessels split between near-vertical and near-flat, which matching the average does
  not ask for.
- **The columns, read from the template** (Harvey's go-ahead, 2026-10-25): the organisation records
  each cell's share of capillary length whose chord runs within 30 degrees of z, y and x
  (`chord_axis_share_by_cell`, and over the whole network `chord_axis_share`), the runs along each
  axis spanning 100 um or more (`long_runs`: capillaries joined end to end whose chords all lie in
  that axis's cone) and the mean chord of each direction class (`chord_by_class_um`). What made the
  bed follow them: E14.5's vertical capillaries are long (1.4 times the mean chord), so even the
  template's own vertices' Delaunay candidates within 60 um run only 6% along z, against 22% of its
  vessels; the fill now judges each candidate against its own direction's chord
  (`bed.class_chord_ratios`), and the per-cell direction tensor counts four times what it did
  (`ORIENTATION_WEIGHT` 4; on the old lattice bed that only lengthened the capillaries). A term
  pushing each cell's drawn share of each direction to the template's, as asked, was built and
  measured: it did nothing at weight 1-2 and made the z share worse at 4-10, so the fill does not
  use it; the shares are scored in the comparison instead. E14.5, five seeds: runs along z spanning
  100 um 10-15 (template 12; before 4-9), along x 9-14 (13), along y 2-6 (2: too few to score,
  `compare.MIN_RUNS_SCORED`); capillary chords along z 0.16-0.20 (0.22); each depth layer's
  verticality above the plexus 0.33-0.62 (0.36-0.64). 562 of 600 banded rows within (5 seeds). The
  comparison adds the chord shares (abs 0.05) and the runs per mm^2 of the face across their axis
  (rel 0.35). A new grid for Harvey with these networks.
- **Grid D1b, marked by Harvey 2026-10-25: not met.** Twelve renders, reshuffled (the template at
  #11), the other eleven drawn at `596a8296` (Delaunay seeds 21-26, Voronoi 21-25; key
  `outputs/synthetic_review/grid_d1b_key.csv`). Harvey: #11 "is the only one that looks like the
  original data, keeps its overall network shape and maintains the curvature of the original
  data". The curvature was mine to have caught: at `ORIENTATION_WEIGHT` 4 the direction bonus
  outweighed a candidate's length, the chords grew to 38-40 um and left the vessels no length to
  curve in -- 60-78% of the capillaries' 10 um turns under 5 degrees (template 7%), curvature 6-11
  against 26 degrees; only the tortuosity row (1.04-1.07 against 1.20) said so, and the curvature
  rows were reported, not banded. Fixed: the weight is 2 (E14.5, three seeds: runs along z 9-13
  against 12, chords along z 0.18-0.21 against 0.22, curvature 22-25 degrees, no capillary
  straight), and the curvature rows are banded (abs 6 degrees, not calibrated). The overall
  network shape is open. Measured on E14.5's trees (large tier with arterioles and venules, from
  each inlet and outlet): the template's have 38 leaves -- short side branches off the trunks,
  ending where the large tier hands over to capillaries -- against 11 drawn (the drawn trunk hands
  its side branches straight to the bed as one capillary each, `generate.HANDOFF_BED_EDGES`); and
  the template's trunks hold their course better, root-to-leaf chord over path 0.65-0.68 against
  0.44-0.62 drawn, though each vessel curves alike (29 against 31 degrees per 10 um).
- **The trees' shape, read from the template** (Harvey's go-ahead, 2026-10-25): the organisation
  records per branch-order label how many of its vessels end the tree (`tree_leaves_by_order`,
  E14.5: 27 leaves; the 38 above counted the leaves the two outlets share twice) and the median
  root-to-leaf chord over path (`trunk_straightness`, 0.63), the roots being the run's inlets and
  outlets (`organisation.tree_shape`). The trees end that many tips at each order -- those
  already out of the template's tree cells, then the narrowest -- and the trunk (the least
  deflected child) keeps to its tree's course from the root (`trees.TRUNK_PERSISTENCE` 2). Two
  rules tried first: ending the narrowest alone left the wider tips branching more and took the
  trees out of their corridor (share in the template's tree cells 0.68 -> 0.55); ending the tips
  heading out of the cells ended the trees' frontier and grew bushes (venous root-to-leaf reach
  77-97 um against the template's 134). Now (E14.5): 27 leaves exactly, share in the tree cells
  0.77-0.81, venous reach 115-121 um (max 226-274 against 272), arterial 230 against 244, trunk
  straightness 0.53-0.64 (0.63). Comparison rows: tree leaves (abs 3), straightness (abs 0.08),
  not calibrated.
- **Capillary lengths by tortuosity**: with the curvature rows banded, seeds 1, 4 and 5 still fell
  to 12-17 degrees per 10 um. Lengths drawn from the template's length distribution, rank-matched
  to the chords and clamped to them, left a vessel straight wherever its draw fell short of its
  chord; the bed's chords leave the template's capillaries little room (mean 35 against 41 um).
  For a template that records its layout each capillary is now its chord times a tortuosity drawn
  from the template's, the excess scaled alike on every vessel to the class total
  (`generate._capillary_lengths_by_tortuosity`; at most 3 times, `MAX_TORTUOSITY_SCALE`). E14.5,
  five seeds: every curvature row within (26 +- 6 degrees), 582 of 620 banded rows within.
  Fixture templates (no layout) draw bit for bit as before.
