# Synthetic networks

Moved verbatim out of CLAUDE.md on 2026-10-22 (commit before the move: 1a9c2ebc). These are design notes and dated measurements: what was measured, on what, and why a rule or default is what it is. They are history, not the contract, and are not kept up to date line by line; CLAUDE.md and the code are the current state.

## synthetic/

**`synthetic/`** (2026-10-18) — a network drawn to a previous run. `template.py`'s
`NetworkTemplate` is the characteristics a generator draws to and a comparison scores against:
the box and voxel size, the inlets and outlets by face, the boundary-node counts, per-class and
per-order counts, lengths, diameters and tortuosity, the daughter angle, the Murray exponent, the
intercapillary distance, loops, dead ends by kind. `template_from_statistics` reads a
`.haemorun`'s saved report through the one reader `template_from_report` (a missing section
raises `TemplateError` naming its `statistics_*` toggles, `SECTION_TOGGLES`); `template_from_graph`
measures the same sections off the graph with the statistics modules and adds per-vessel
`Samples`; `load_template(path, source)` is memoised on the file. `generate.py` draws it: roots on
the template's faces, `trees.py` grows each side one generation at a time with the order table's
counts exact (bifurcations, a trifurcation only when an odd count forces one), Murray exact at
every junction with the asymmetry fitted to the table's fall in diameter, daughter angles round
the template's; `bed.py` works the capillary bed's degree sequence out from the handshake lemma
(`degree_sequence_targets`: capillary count, loops and dead ends exact together; when they cannot
be, loops give, never dead ends), lays a jittered lattice, takes Delaunay candidates near the
template's mean chord, fills greedily, reroutes (`augment`), repairs connectivity along Delaunay
edges, merges degree-2 vertices, sets the dead ends nearest the faces on them; `polylines.py`
bends each vessel to its length with a `sin^2` envelope so it leaves its nodes along the chord
(the angles the statistics read); blind ends sit at their chain lengths. `validate_synthetic_graph`
pins the contract (int ids, `pos`/`voxels` in um, `length` the polyline's, `diameter_source=
"synthetic"`, no `branch_order`, no degree-2 node, `dead_end_kind` on every dead end, connected).
`compare.py`: exact rows (box, endpoints, tree-order counts), banded rows (counts, lengths,
diameters, tortuosity, loops, dead ends) and compare-only rows (capillary orders, the
network-wide Murray median, emergence angles, loop-length percentiles); the CSV and the report
section. On `tests/synthetic_fixtures.py`'s template: 54-55 of 55 banded rows within band across
seeds; a 10,000-vessel template draws in ~5 s. Not yet checked against a real run's template.

## Synthetic network (from the pipeline overview)

**A synthetic network instead of an image** (`network_source=synthetic`, 2026-10-18,
`PLAN_synthetic_network.md`): the Input tab's `Input` choice draws a network matched to a previous
image run's `.haemorun` (`synthetic_template_path`; `synthetic_template_source` reads its saved
statistics report, or per-vessel distributions off its graph; `synthetic_seed`, `synthetic_scale`,
`synthetic_interior_dead_ends` behind the Advanced button). `segment` reads and checks the template
(`synthetic.load_template`, remembered per file so preflight's `check_synthetic_template` and the
stage read it once; a template saved with a needed statistic off is refused naming the
`statistics_*` toggle), `skeletonise` returns an empty `SkeletonisedVolume(synthetic=True)`,
`build_network` draws the graph (`synthetic.generate_network`: trees grown to the branch-order
table under Murray's law, a capillary bed built to the capillary count, loop count and dead-end
count exactly; `{template}_synthetic_graph.pkl`), `assign_boundaries` takes the roles the generator
marked (`synthetic.boundary_nodes`; the Boundaries role tabs are greyed), the drawn widths are kept
(`diameter_source="synthetic"`, which stamping never replaces) and everything from Diameters on
runs as for an image. Every image-only Input row requires `network_source=segmented_image`, every
mask switch reads as off (`graph.vessel_masks_in_use`), `raw_image_is_set` is False, and every
output is named `{template stem}_synthetic` (`pipeline/naming.py`'s `run_stem`, the one answer).
Export writes `{stem}_template_comparison.csv` and a `synthetic_comparison` report section: each
characteristic, template against synthetic, with the band the generator is held to (`compare.py`).
A synthetic run's `.haemorun` is refused as a template.

## 2026-10-24: diameter basis, one tolerance table, validation time

`NetworkTemplate.diameter_basis` (`5b3b8fdc`): the basis most of the template's vessels' viscosity
was read on, carried onto every drawn edge, so a synthetic run's resistances are the template's
form of the Pries law whatever its own `segmentation_source` (preflight warns when they differ).
Before it, a synthetic run of a Claudin (endothelial) template under the default plasma_column
read every capillary through the in vitro form. The comparison's bands moved into one
`TOLERANCES` table, rows unchanged. `validate_synthetic_graph` was half of generation on the
fixture scaled to 9,500 vessels (11-12 s); with the same checks written without `np.allclose` per
polyline end it is 1.3 s and generation 7-10 s (`4767c3c6`). What waits on Harvey (`vessel_box`,
a Voronoi bed) is listed in `PLAN_synthetic_network.md`'s status.

## 2026-10-24: E14.5 as the template

The first real template (verification 2 of `PLAN_synthetic_network.md`, whose status has the
numbers). Generation 1.2-1.3 s. It broke the contract three ways, all fixed: stars left apart by
the connectivity repair (`89cb4b9b`), a large tier with no Art/Ven tier labelled Art/Ven
(`68d6885e`), a tree cut at its first vessel left hanging (`5dda9ac9`). What it shows about the
model, for Harvey: E14.5's large tier is a trunk with side branches, which Murray-exact
bifurcating generations shrink to sub-micron widths (the endothelial in vivo law then refuses the
run); and its capillaries are clustered, shorter than a uniform bed in the same box can draw
(16% more length), which `vessel_box` would not fix (its nodes span 99% of the box).

## 2026-10-24: Harvey's decisions, built

The large tier is a trunk with side branches at its orders' own widths (`78d7b1f0`; Murray's law
stays in the Art/Ven tier), a root's first vessel stays in the box and the tree rescale stops at
the template's longest and most tortuous vessel (`78d7b1f0`), the bed fills the template's tissue
volume or the volume its vertices take at the template's capillary chord, grown from the trees
(`ff4f4dcb`, `vessel_box_um` dropped), and `synthetic_bed_model=voronoi` draws the bed along a
Voronoi tessellation's cell edges (`9e49ddfc`). On E14.5 every large-tier order count is now the
template's, its widths are, the haemodynamics solve, and 47 of 49 exact rows are equal (the two
left are the large-vessel boundary counts); the Voronoi bed is more tortuous (1.42 against 1.20)
with closer capillaries (12 against 22 um). The table is in `PLAN_synthetic_network.md`.

## 2026-10-25: not physiological; the "clustered" reading was wrong

Harvey: the synthetic network does not look physiological and has no defined inlets and outlets.
Rendered beside its E14.5 template (`outputs/synthetic_review/`, `plan-synthetic-e08d0c` worktree)
the template is layered -- large vessels in a plexus on one face (z under 25 um), its densest
capillaries there, the rest running up through the depth and along x -- and fills its box (5% of it
beyond 50 um of a capillary). The synthetic network's trees wander through the whole depth, its
capillaries run every way alike, and 37-40% of its box is beyond 50 um of one: the bed "grown from
the trees" (`ff4f4dcb`) was built on the reading above that E14.5's capillaries are clustered, which
is wrong; they are short because the plexus is dense, not because they gather. 57-62% of the
synthetic outflow leaves through open ends, not the outlets. The measures, the decisions for
Harvey and the phases are in `PLAN_synthetic_physiology.md`.

## 2026-10-25: built to Harvey's D1-D7 (`PLAN_synthetic_physiology.md`, whose status has the numbers)

The template now records where its vessels lie and which way they run (`synthetic/organisation.py`),
scored in the comparison; the bed's vertices are the template's capillary vertices moved by their
spacing (a scaled box: its 50 um cell field), the trees grow in the template's tree cells of their
side, the network is closed with no interior dead ends by default, the dead ends not drawn joined in
pairs, and vessels curve as the template's do. On E14.5: all inflow through the inlet and outflow
through the outlets; capillary vertices by cell correlate 0.95 with the template's; capillary
curvature 27 against 26 degrees per 10 um (4 before); the trees 73% in their cells. Two things the
plan's measures showed along the way: the lattice bed drew 93% of its capillaries straight (its
chords had no room for their length), and a correlation of the template's layout means nothing where
the template has none (its x profile on E14.5), so such rows are reported, not banded. Harvey's grid
D1a (12 renders, shuffled) is in `outputs/synthetic_review/` of the `plan-synthetic-e08d0c` worktree.
Harvey marked it the same day: only the template (#7) could be the imaged tissue; all seven networks
drawn after the plan, like the four from before it, look generated. The measures moved, the look
did not: straight trunks with short side branches, capillaries rising in columns from the plexus and
curved meshes are what the template has and the drawn networks lack.

Harvey: the columns are right but particular to E14.5, so they are read from the template, not
built in. Its vertical capillaries are long -- they reach past nearer junctions -- and a bed whose
fill wanted one chord length for every direction never drew them; judged against each direction's
own chord (`bed.class_chord_ratios`) it draws E14.5's columns (10-15 runs spanning 100 um of depth
against 12), and no columns where a template has none.
