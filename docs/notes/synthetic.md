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

Grid D1b (Harvey): only the template keeps its overall network shape and curvature. The curvature I
had lost myself (a direction weight that bought chords with length); the shape was the trees: the
template's have side branches ending at each order (27 leaves on E14.5) where the drawn trunks
handed every side branch to the bed (11 leaves). Both now read from the template, with each
capillary's length its chord times a tortuosity drawn from the template's, so none is drawn
straight to fit the length budget.

Grid D1c (Harvey, the same day): three of six Delaunay networks passed with the template (#1 best,
#11 and #12 okay), no Voronoi one. What the picked share: their venous trees reach across the box
as the template's do (281-285 um against 272; the others 127-255); every Voronoi bed had lost its
capillaries' curvature (0.2-20 against 26 degrees per 10 um).

After D1c: the Voronoi bed curves again (its seeds the template's vertices, little contraction), at
the cost of a thinner plexus than the template's; and each side's trees are the one of six grown
whose leaves reach nearest the template's, so every seed's venous trees now reach across the box
(260-283 um against 272).

Grid D1d (Harvey): two of six Delaunay networks pass (#2, #9), no Voronoi one. Every network now
reaches, curves and ends its trees as the template does, and no measure separates the two from the
rest; the Voronoi beds' thinner plexus is the plain difference left between the beds.

## 2026-10-25: a batch condition as the template (`.haemobatch`, `PLAN_batch_templates.md`)

Harvey asked for a condition's average network as a template, a batch run writing one file per
condition, and chose B1a, B2a, B3a, B4b beside a and B5a. Every batch run now writes
`{condition}.haemobatch` (`batch.haemobatch` with no conditions) of the images that finished:
gzip JSON of each run's two templates (by its report and by its graph), its `.haemorun` path and
SHA-256, the runs left out with why, and the batch's `batch_settings.yaml` as written. `load_template` reads one as its runs averaged
(`synthetic/average.py`), so the panel, preflight and the stages need nothing else.

- **The box** (B1a) is each axis's mean extent, and every count the runs' mean density times its
  volume, so runs of different sizes count alike; inlets, outlets and the dead ends leaving the
  image go with the faces' area. An order a run lacks counts 0 for it, so a condition whose runs
  reach different depths draws the deepest orders rarely, not always.
- **The layout** (B2a) is pooled: each run's per-cell fields carried onto the common grid and the
  capillary vertices pooled as shares of each box. Reading a finer grid only at the coarser cells'
  centres, as first written, kept or multiplied a sparse field by chance (a test's two vessels
  counted 4.4 against 2); each cell is now shared out by its overlap, which keeps every total. This
  assumes the runs were imaged the same way up.
- **The comparison** (B4b beside a) keeps the bands round the average and adds each row's range
  over the runs, each scaled to the average's box: `Members low`, `Members high`, `Within
  members`. A correlation row has no range (each run's field is its own reference), and a value
  the drawn network lacks gets no verdict.

Checked on `tests/synthetic_fixtures.py`'s drawn runs and on E14.5 averaged with itself (counts,
orders, classes, dead ends, loops and inlets as the one run's). Not yet run on a real batch of
several images, nor marked on a grid.

## 2026-10-25: flow-adapted widths (phase 7) measured on E14.5, not built

Harvey asked for structural adaptation (Pries, Secomb & Gaehtgens 1998) read from the template.
Fitted to E14.5's solved network, the 1998 balance gives a metabolic weight of 0.69 (the paper's
0.70), a conducted weight of 0.13 and a shrinking tendency of 1.24, and explains 22% of the
template's wall shear stress. Its shear does not follow pressure over its 15-30 mmHg (r = -0.04),
so no pressure term can be read. Applied to the template's own network, the law does not keep its
widths. Under fixed inlet and outlet pressures they run away, since widening raises flow by the
fourth power and the metabolic term fades at the template's flows. Holding the inflow, the large
vessels regress to capillary width (arteriolar median 19 -> 7 um, venular 21 -> 6 um), the
shunting the 1998 paper describes; the paper's conducted weight (2.45) holds the arterioles but not
the venules. The template's widths follow its flows weakly (D ~ Q^0.11-0.17 by class, R2
0.21-0.29), and those flows are computed from the widths. So nothing was built: a law that turns
the template's own large vessels into capillaries would draw widths less like the template's.
Harvey chose to leave the widths as the template's (D7a only): phase 7 stays unbuilt.

## 2026-10-25: what still separates the template from the drawn networks

About 40 measures on E14.5 and the 44 drawn networks of grids D1a-D1d (the harness is
`outputs/synthetic_review/realism_measures.py` in the `plan-synthetic-e08d0c` worktree). Where
E14.5 sits outside every network of D1d:

- **Junction shape**, not yet measured by the comparison: E14.5's capillary junctions are wide,
  flat Ys (smallest angle between two branches, median 97 degrees, 1% under 45; the three angles
  sum to 356), the Delaunay bed's often have two branches leaving together (median 51, 41% under
  45; sum 292-298), the Voronoi bed's less so (63, 30%; 313). The same at 4, 8 and 15 um from the
  node, and with or without a dead-end stub at the junction.
- **Holes**: box beyond 50 um of a capillary 4.8% against 10-12%, out of the comparison's band in
  every network; nothing in the generator acts on it yet.
- **Columns above the plexus**: near-vertical capillary length above the plexus 0.30 against
  0.15-0.27, the columns evenly spread in E14.5 and bunched when drawn. The only measure that
  ranked Harvey's picks above the rest in both of his looks at D1d.
- **Tortuosity tail**: p90 1.50 against 1.14-1.26 with medians alike, because the generator reads
  the statistics source's mean tortuosity with a fixed spread and leaves its median unused.
- **Widths in patches**: neighbouring capillaries' diameters correlate 0.21, drawn ones about 0.

The template's fine wiggle at 2 um steps (twice the drawn networks', equal at 10 um) looks like the
skeleton's staircase, and is not one to match. Harvey's two looks at D1d agree on one network of
eleven, so his marks rank the drawn networks too loosely to fit a rule to; the template's own gaps
are the better guide.

## 2026-10-25: drawn to the junction shape, coverage, columns and tortuosity tail (grid D1e)

Harvey: read them from the template, draw to them, compare them, and produce them as statistics of
every run (`statistics/layout.py`). The template reads them from its report, or measures them off
its graph when the report predates them; the comparison bands them (not calibrated); a
`.haemobatch` averages them. What the generator does with them, on E14.5 (`outputs/synthetic_review/`,
`layout_tune.py`, `acute_source.py`, means of three seeds):

- **Junctions.** The bed's fill penalises a vessel leaving a vertex within 45 degrees of another
  (a tree leaf's tree vessel included), at both ends of the candidate, and aims each pair at the
  template's own junction angle; a template whose junctions are as acute as the fill draws
  unaided (0.38) gets no push. The greedy fill alone left one junction in eight acute whatever the
  weight (by a vertex's turn its wide neighbours are taken), so after it two edges trade ends
  where that widens a junction (`_Fill.relax_junctions`; degrees kept, never cutting the bed in
  two, which would cost a loop when the pieces are joined). Delaunay: acute 40% -> 15%, smallest
  angle median 53 -> 68 degrees (E14.5 97, 1.2%); Voronoi 28% -> 13%. Heavier weights reach 82
  degrees and 5%, but only with chords so long their sum passes the template's capillary length
  and every capillary is drawn straight, the failure grid D1b caught; so not taken.
- **Tortuosity tail.** The statistics source's median and 90th percentile now set the lognormal
  the capillaries' tortuosity is drawn from (the mean with a fixed spread before). The class's
  total length still sets the scale, so the tail's shape moved more than its value: P90 1.25
  (1.21 in D1d; E14.5 1.48).
- **Columns.** Their spread came closer without a term of its own: Clark-Evans 0.80 -> 0.86
  (0.90).
- **Holes, not drawn to.** 11% of the box beyond 50 um of a vessel, as before (0.5%). E14.5's
  dead-end chains carry 31% of its length and fill its tissue: without them its holes are 19%. Tried
  and not kept: a crowding term in the fill (holes unchanged, capillaries straightened), each
  template vertex taken at most once (holes 9%, curvature 24 -> 20 degrees per 10 um, tortuosity
  P90 1.14), the farthest vertex first (holes 6.6%, every capillary straight), the template's own
  dead ends and face cuts drawn (holes 10%), vessels' ends tapered to their chords (no change).
  Every lever that spread the bed lengthened its chords past the template's capillary length.

Grid D1e (template at #5; six Delaunay seeds 51-56, five Voronoi 51-55; key
`outputs/synthetic_review/grid_d1e_key.csv`) sent to Harvey.

Grid D1e (Harvey, 2026-10-26): "5 is best, 1, 6, 8, 9 are okay" -- the template (#5) and Delaunay
53, 55, 51 and 56: four of eleven, four of six Delaunay beds, no Voronoi (none in any grid,
D1a-D1e). On the realism battery (`outputs/synthetic_review/d1e_measures.py`, `.csv`) the Voronoi
beds are nearer E14.5 than the Delaunay ones on its holes, the evenness of its capillary density
and its mesh size; what sets them apart is the plexus: 37% of their junctions lie in the bottom
48 um against 60% of the Delaunay beds' and 61% of E14.5's. Among the Delaunay beds (four picked
against two, too few to fit a rule to) the picked lean the template's way on columns (5.1 against
4.1 per 10^4 um^2; E14.5 7.7), junction width (smallest angle 69 against 64 degrees) and holes,
and the other way on tortuosity (P90 1.27 against 1.35; E14.5 1.50).

## 2026-10-26: the Voronoi bed's corners at the template's junction depths (grid D1f)

Harvey: keep the Voronoi bed's corners in the template's plexus layer, by its junctions' share by
depth read from the template, without losing their curvature. A tessellation of seeds in a sheet
thinner than their spacing puts its corners above and below the sheet: each corner is the centre of
a sphere through four seeds, and four spanning a thin sheet put it well off it. The organisation now
records every capillary junction's depth (`capillary_junction_depths_rel`; E14.5: 644 junctions,
58.5% in the bottom 48 um), and once the corners are drawn and thinned they are moved, rank for
rank, onto those depths (`bed.depths_like_template`): each keeps its place in the image plane and
its cell edges, and the order in depth is kept. E14.5, seeds 1-3, before and after: junctions in
the bottom 48 um 36% -> 62% (template 61%), capillary curvature 20.6 -> 25.5 degrees per 10 um
(26.1), tortuosity P90 1.14 -> 1.34 (1.48); and, the cost, the layers above sparser: box beyond
50 um of a vessel 7.1% -> 10.6% (0.5%), column share of length 0.17 -> 0.14 (0.22), acute
junctions 13% -> 18%. The Delaunay bed is unchanged. Grid D1f (template #4; Voronoi 61-66,
Delaunay 61-65; key `outputs/synthetic_review/grid_d1f_key.csv`) sent to Harvey.

Grid D1f (Harvey, 2026-10-26): "4 is best, everything except 10 and 11 are okay" -- the template
(#4), all five Delaunay networks and four of six Voronoi (61, 62, 63, 66): nine of eleven, and
the first Voronoi networks taken for the tissue in any grid. The two left (Voronoi 65 and 64) do
not differ in plexus share, junction width or holes; their trees wind more (26.1 against 24.7
degrees per 10 um at 10 um steps; the template 22.9) -- four against two, a lead, not a rule
(`outputs/synthetic_review/d1f_measures.py`, `.csv`). The Delaunay bed is D1e's code: its 4 of 6
there and 5 of 5 here is the marking's spread, not a change.

## 2026-10-26: the trees curved as the template's at 2 and 10 um (grid D1g)

Harvey: draw the large vessels with the template's own tree curvature, read at both 2 and 10 um
steps. Every tree vessel was bent toward the template's mean turn per 10 um (29.3 on E14.5), while
its median vessel turns 26.8; and the drawn trees' smooth curves turned less than the template's
read every 2 um (72-75 against 88, its centrelines' fine wobble). The organisation now records each
tree vessel's turn at 10 um steps and the median vessel's at 2 um (`path_turning`,
`tree_curvature_by_vessel_deg`, `tree_fine_curvature_deg`); each drawn tree vessel takes a target
drawn from the former, and a fine wobble across it (`polylines.fine_wobble`, none within 4 um of
either node so its junctions are read as before) brings the median vessel's 2 um turn to the
template's, one amplitude for the network found by bisection. Drawn to the targets alone the bows
came out at 33 (a bow of the trees' tortuosity turns about 30 per 10 um whatever its target, so the
low targets were out of reach); two things brought them down: the wobble's added length is taken
off each tree vessel's bow, which is drawn again to the rest, and a tree vessel's bow keeps to one
plane (the bend's second, perpendicular wave off: `generate.TREE_CROSS_WAVE`). E14.5, both beds,
seeds 1-3: median vessel at 10 um 27-28 (26.8), mean turn 27 (29.3), at 2 um 85 (88); spread across
vessels p25-p75 about 21-34 (19.7-36.2). Generation 6-8 s, from 3-4. Grid D1g (template #9;
Voronoi 71-76, Delaunay 71-75; key `outputs/synthetic_review/grid_d1g_key.csv`) sent to Harvey.

Grid D1g (Harvey, 2026-10-26): "These all look good now" -- every drawn network, six Voronoi and
five Delaunay, taken for the tissue beside the template. The plan's verification 2 (drawn networks
taken for the tissue as often as the template's own render) is met. What is still open: the holes
(7-11% of the box beyond 50 um of a vessel against E14.5's 0.5%, measured but not drawn to), and
verification 3, a second tissue as the template, which waits on a run with arteriole/venule masks
or hand-picked boundaries.

## 2026-10-27: the box scaled along x and y, or all three

`synthetic_scale_axes` (`xyz`, the default and what `synthetic_scale` always did, or `xy`) picks
the axes `synthetic_scale` grows the box along; `xy` keeps the template's depth, a wider field of
the same tissue. `NetworkTemplate.scaled(factor, axes)`: counts and total lengths go as the volume
(the factor squared for `xy`), each inlet and outlet as the area of its own face (an x face grows
as y alone under `xy`; one inside the box as the surface), dead ends leaving the image as the box's
surface, each pair of faces weighted by its own growth. The organisation's cells are shares of the
extent, so under `xy` they stretch across the plane and keep their depth. The comparison now scores
a scaled network against the template scaled the same way (`compare_run` reads the record's
options); it read the template at its own size, so every box, count and endpoint row of a scaled
run was outside. At a factor like 1.5 the first tree order can be one vessel short of the scaled
count (a root grows an odd number of them, `trees.grow_side`'s note), whichever axes.

## 2026-10-27: open-end models against a known beyond (`PLAN_synthetic_surround.md` phase 1)

`scripts/surround_check.py`: a synthetic network drawn at 1.5x its template's box along x and y,
closed and solved, is the truth; a box of the template's size cut from it (`graph.crop_graph_to_box`)
is a stand-in image, fed through its widest arterial and venous open ends; the image is solved under
each `open_end_model` and every vessel's flow scored against the truth. An oracle (each open end at
the truth's pressure) gives the truth back to 1e-12. Over 10 runs on E14.5's template, the default
`estimated` is the best of the models and still far from the truth: 76% of vessels flow the true way
(82% beyond 20 um of a face), median flow error 77%, rank correlation 0.72; `closed` 68%,
`terminal_resistance` and `outlet` 60-65% with 9-28 times the true inflow through the designated
inlets. In the truth 89% of the blood entering the image comes through its open ends. At arterial
and venous open ends the `estimated` direction is right 59 times in 124, no better than a coin. 1.5x,
not more: the generator roots every tree on the outer faces, so a larger truth holds no box fed from
both sides.

Harvey (2026-10-27, D5): every arteriole and venule open end is to run on as a trunk to the
surround's outer face, whichever way its blood ran ("revisit if the harness says otherwise"). On
the harness a trunk is right at 97 of 124 such ends on E14.5 and 46 of 53 on the fixture, a count
biased toward it: the truth roots every tree outside the image. The report's last line gives the
count and says "revisit D5" at half or fewer.

## 2026-10-28: a network drawn round a hole (`PLAN_synthetic_surround.md` phase 2)

`GenerationOptions.hole` (`synthetic/hole.py`): a box the network leaves empty, the image's own box
inside a surround. Its share of the template's counts is not drawn (`NetworkTemplate.around_hole`);
roots on a face inside its footprint move along the face out of it; the trees end before it; the
bed lays its vertices outside it and a margin (0.25 bed spacings) round it, and joins none across
it; a bow is turned off it, or drawn straight; the graph records it and the validation refuses
anything inside. Without a hole the generator draws what it drew before, bit for bit (nine
networks, fixture and E14.5, digested before and after; in CI, a hole outside the box draws the
network drawn without one). Round a centred hole the template's size at 2x, the fixture's 12
networks and E14.5's at 2x xy (both beds) validate, capillaries within 3 of the target; E14.5 2x xy
takes 17 s with the hole and 23 s without.
At 2x xyz on E14.5 (about 8,000 vessels) a network round the hole takes 81-95 s over four draws
against 43 s without one; the junction trades take 39 s of it, most of them failing near the hole's
faces.

## 2026-10-28: the image's network joined to one drawn round it (`PLAN_synthetic_surround.md` phase 3)

`synthetic.surround_network(image_graph, template, options=)`: the image's graph moved, unchanged,
into the middle of a box `synthetic_scale` times its own, and a network drawn round it to its own
template (`generate_network(joins=)`, the image's box the hole) joined at each open end. A capillary
end is joined by the bed with one capillary. Every arteriole and venule end, and each designated
inlet and outlet, runs on as a trunk (D5): straight out along its heading to the outer face, at its
measured width, in pieces of its order's mean chord, each joint handing a capillary to the bed; its
end there is a new inlet or outlet, in place of one of the scaled template's. A tip short of the
hole's face is drawn on to it by a connector of the same vessel. Under `xy`, ends on the z faces stay
open. On E14.5 (91 open ends, 4 of them trunks) a 1.5x xy surround is 2,852 vessels (44 s), 1.5x xyz
4,186 (13 s), 2x xy 4,860 (63 s); its tips lie 2.9 um short of the face at the median, 8.8 at most.
Not yet in the pipeline, and not yet scored on the cut-out harness.

**Densities beyond the template's box** (Harvey's request, 2026-10-28):
`synthetic_extension_arteriole_density` and `_venule_density` draw the arterial and venous trees in
the part of a scaled box beyond the template's own at that share of the template's density, since an
image framed round its large vessels need not be typical of the tissue round it. The tree classes,
orders and boundary nodes go as 1 + s(d - 1), s the share of the drawn volume beyond the template's
box (all of it in a surround), and the inlets and outlets on each face by that face's own share
(`generate.drawn_template`, read by the generator and the comparison alike). Capillaries, loops and
dead ends keep the template's density. In a surround the image's arterioles and venules still run on
as trunks at any density; at 0 they are the only trees reaching the box. At 1, or at the template's
own size, nothing changes (nine networks digested before and after, identical).

## 2026-10-28: the surround on the cut-out harness (`PLAN_synthetic_surround.md` phase 4)

`scripts/surround_check.py` now solves each stand-in image inside a surround as well (at the
truth's 1.5x along x and y, its own seed, drawn to the template the truth was drawn to), reads the
image's pressures back and scores them like the open-end models'. Over 10 runs (5 seeds x both beds),
medians, against `estimated` (with the trunk fix below):

| | directions right | interior | median flow error | interior | P90 flow error | rank corr. | designated inflow / truth's |
|---|---|---|---|---|---|---|---|
| E14.5, `estimated` | 0.76 | 0.82 | 0.77 | 0.69 | 2.68 | 0.72 | 3.5 |
| E14.5, **surround** | **0.87** | **0.91** | **0.52** | **0.45** | 3.46 | **0.82** | 0.92 |
| fixture, `estimated` | 0.70 | 0.81 | 0.94 | 0.76 | | 0.52 | 2.1 |
| fixture, **surround** | **0.84** | **0.92** | **0.53** | **0.29** | | **0.68** | 1.1 |

The kill criterion (beat `estimated` on median flow error and on direction agreement) is met on
both: on E14.5 the flow error is lower in 9 runs of 10 and the directions better in all 10; on the
fixture 10 and 9. The tail is not better: E14.5's P90 flow error is 3.5 against 2.7. Through the
open ends the surround puts 93% of the image's inflow against the truth's 89% (`estimated` 70%).
Drawn at 2x instead of 1.5x, the fixture's numbers hold (0.50 / 0.85); at 1.25x no surround can be
drawn (a shell of about 83 capillaries cannot take ~60 joins), and a run it cannot draw counts as
lost. **A kind test**: the surround is drawn by the generator that drew the truth, to the same
template, and the truth roots every tree outside the image (D5's trunk is right at 97 of 124 tree
ends because of it). Real tissue beyond an image is not drawn by the generator, so these numbers
are a ceiling on what the surround does for a real image, not a measure of it.

Drawn instead to the template a run of the stand-in image measures for itself
(`surround_check.image_template`: its statistics, its inlets and outlets; every vessel its faces cut
a dead end, its loops cut short, its tree orders only the pieces the box holds), the surround still
meets the criterion on E14.5: directions 0.86 (0.91 interior), median flow error 0.54, share error
0.60, rank correlation 0.81, better than `estimated` in 9 and 10 runs of 10. On the fixture at 1.5x
three draws of ten fail (the shell too thin for the face-cut ends, joined in pairs on the closed
outer faces, and the image's joins: no bed, or a trunk left in a piece of its own), which by the
harness's rule fails the criterion though the seven drawn are all better; at 2x it is met.

Its first run found a bug: two E14.5 draws took 42 and 96 minutes in the bed's junction trades. An
image's template can hold an order only as the sliver its face cut (one vessel of 0.44 um), and the
trunk continuing that end was cut into pieces that long: about 400 joints, each wanting a bed
vessel, piled onto four bed vertices (46-78 vessels each) by the last-resort join, and the trades
cost the square of a vertex's degree. A trunk's pieces are now never shorter than the bed's minimum
vertex separation (half its spacing); each draw takes 8-14 s, and networks without joins are
unchanged.

## 2026-10-28: the surround in the pipeline (`PLAN_synthetic_surround.md` phase 5)

`network_source=synthetic_surround` ("The template run's network inside a synthetic one") runs a
surround as a run of its own off the template run's `.haemorun` (D1). It takes the synthetic path
(`network_is_synthetic` answers yes; `network_surrounds_image` says where it differs, and the
synthetic rows' prerequisite is now `network_source!=segmented_image`): the Input stage reads the
template and the run's own graph (`surround.load_image_run`, refusing a `.haemobatch`), the Graph
stage draws `surround_network` round it, the Boundaries stage takes the roles it marked. The image's
vessels keep their run's widths (`diameter_source="image_run"`, ahead of any reading the run left on
them; the run's own source as `image_diameter_source`) and labels as `image_branch_order`, while
branch orders run over the whole network (D7). The template comparison scores the whole network
against the template scaled to the box (the hole not left out). Outputs: `{template stem}_surrounded`,
the VTK's `network_part` and `image_branch_order`, `{stem}_image_flows.csv` (each image vessel's flow
in the template run and inside the surround, from the same node, their ratio, whether it flipped),
the manifest's `surrounds_template_run`, and `network_part` as a colour in the viewer. Preflight
(`check_synthetic_surround`) refuses a scale of 1 or less, a `.haemobatch` and a run with no open
end, and says which open ends stay open under xy and when the run has no solve to compare with. A
closed template run is surroundable: its own inlets and outlets are open ends, each a trunk. Not yet:
D6's mirrored layout (the shell takes the template's stretched over the box), and a run on E14.5.

## 2026-10-29: E14.5's own run surrounded (`PLAN_synthetic_surround.md` phase 6)

E14.5's run (`e14p5.haemorun`: 1,406 vessels, solved under `estimated`, 91 open ends, endothelial
widths) run again as `network_source=synthetic_surround` from its own settings, at 1.5x along x and
y (five seeds) and along all three (three seeds); the image's flows compared with the run solved
again under each `open_end_model` on its own conductances (`estimated` gives the run's flows back to
1e-7), over the 1,104 vessels it solved above the carrying floor. No truth: this measures how far
the surround moves the answer, and how far one draw of the surrounding network moves it from another.

| median over seeds | xy (5 seeds; 30 z-face ends left open) | xyz (3 seeds; all joined) |
|---|---|---|
| whole network, vessels; one run's time | 2,973; 17-56 s | 4,428; 26-220 s |
| image inflow through its open ends (run under `estimated`: 54%) | 34-38% | 22-27% |
| vessels the same way as `closed` / `estimated` / `terminal_resistance` k=100 / `outlet` | 83 / 77 / 74 / 71% | 83 / 79 / 78 / 76% |
| median abs log10 flow ratio to `closed` / `estimated` / `outlet` | 0.16 / 0.20 / 0.45 | 0.19 / 0.24 / 0.43 |
| one seed's surround against another's: same way, median abs log10 ratio | 88.5% (84-93), 0.13 | 85% (85-89), 0.12 |

The image's resistances are the run's own (median ratio 1.00). Inside the surround E14.5 takes less
of its blood through its open ends than `estimated` puts there, and its flows lie nearer `closed`'s
than any other model's; one draw of the surrounding network against another still turns 1 vessel
in 9 (xy) to 7 (xyz) the other way, about half the surround's difference from `estimated`. The whole
network against the template scaled to its box (xy, seed 1): 75 of 150 banded characteristics
within. (Its large tier read half the scaled template's, 30 large arterioles against 61, 84 large
venules against 133; that was a labelling bug, corrected below, not the trunks.)

With both extension densities at 2 (xy, seeds 1-3) the network has 4 inlets and 6 outlets, and the
open ends take 36-39% of the image's inflow, as at 1 (38%); at 3 (6 inlets, 9 outlets) 42-50%,
still short of `estimated`'s 54%: a richer supply beyond the faces is not what the open ends lack.
The richer the supply, the further the image's directions move from `estimated`'s (79, 75, 72% the
same way at densities 1, 2, 3) and from `closed`'s (83, 78, 79%), while one seed against another
stays at 88-90%: the density beyond the faces moves the answer about as much as the draw does.
Driver: `outputs/surround_e14p5/run_surround.py --density` (git-ignored, with each seed's run).

**Shell room in preflight** (2026-10-29). A surround's shell can be too thin to join every open end
of the image (the fixture at 1.25x and 1.5x). `generate.planned_bed` is now the generator's own bed
arithmetic (capillaries less loops less roots, the dead-end pairs joined), which
`surround.planned_surround_bed` asks for a surround without drawing it. Over the cut-out harness's
70 surrounds (fixture at 1.25, 1.5, 2x, both templates; E14.5 at 1.5x, both templates;
`outputs/surround_check/shell_room/plans.csv`): every one planned below 1 vertex failed (9 of 9), and
every failure was planned with fewer bed vertices than joins (13 of 13, against 15 such drawn).
`check_synthetic_surround` refuses the first and warns of the second
(`surround.ROOMY_VERTICES_PER_JOIN`). The connectivity CSV now says which network each vessel is
(`Network part`).

**D6: the shell's layout mirrored** (2026-10-29; Harvey's D6, the recommended answer). Round a hole
the template's layout is no longer stretched over the whole box: `Organisation.mirrored_to` puts on
the box's own grid, beside each face of the image, what lies at that face inside it (a mirror tiling,
`organisation.mirror_into`, folded again past the far side), counts and lengths at the image's own
density, orientation tensors turned over across a flipped axis, the junctions' depths folded the
same way (left alone while the box keeps the image's depth). `drawn_template` applies it to every
network drawn round a hole, and the comparison to a surround scored whole, on the grid its layout is
measured on, so a surround's per-cell layout is scored at last. On E14.5 at 1.5x along x and y (seeds
1-5 against the run before, seed 1, both measured against the mirrored layout): capillary vertices by
cell correlate 0.93-0.94 against 0.86, capillary length 0.85-0.89 against 0.81, its profile along y
0.94-0.98 against 0.59, and the tree length lies in the template's tree cells 0.78-0.87 against 0.76;
along x the template's capillaries are even (the comparison holds no layout there) and the
correlation is noise. The image's flows barely move (the same way as `estimated` at 77.5%, against
77%; seed against seed 90% against 88.5%), the open ends carry 24-40% of its inflow, and a whole run
took 25-30 s against 11 that day, which was other sessions' load, not D6 (2026-10-30, below).
Networks without a hole are unchanged (the nine digests).

**The large tier was mislabelled, not short** (2026-10-29). The connector drawn from an image
vessel's tip to its face took the vessel's class but not its large-mask tag, and the large tier is
followed along tagged vessels from the inlets (`branch_order`'s `large_tier` mode), so it stopped at
every trunk's connector: the image's own 86 large vessels lost their `Large_` labels, and only the
shell's were counted. A connector now takes its vessel's mask tags too (`surround.CONNECTOR_KEEPS`).
E14.5's surround at 1.5x along x and y has 58 large arterioles against the scaled template's 61 and
145 large venules against 133 (with D6, seeds 1-5); at extension densities 2 and 3, 87 and 221, 116
and 297. The shell's own trees were never short: they drew 30 and 84 of their 34 and 74. Branch orders
enter no resistance, so the flows above stand. At density 3 one draw (seed 2) put all of the image's
inflow through its open ends, its own inlet running out, and seed against seed fell to 68%.

**The harness again, with D6** (2026-10-29; `outputs/surround_check/p4d_*` against `p4c_*`). The kill
criterion holds where it held. E14.5 at 1.5x (10 runs): with the truth's template the median flow
error falls from 0.52 to 0.47 (directions 0.86, rank 0.83), with the image's own from 0.54 to 0.48
(directions 0.87, share error 0.60 to 0.51), each better than `estimated` (0.77, 0.76) in 9 or 10 runs
of 10 on both. The fixture with the truth's template is unchanged (the hand-set template has no
layout to mirror); with the image's own at 2x it still meets it (flow error 0.60 to 0.54, better in
all 10 runs), and at 1.5x still fails it (8 of 10 drawn, the shell too thin). An E14.5 case takes
55-77 s against about 30.

**Why E14.5's open ends carry less inside the surround** (2026-10-29). By kind of end
(`outputs/surround_e14p5/inflow_by_end.py`, `outputs/surround_check/truth_inflow_by_end.py`,
`capillary_exchange.py`). E14.5's image has 91 open ends: its designated inlet and two outlets, one
other arteriole or venule end, and 87 capillary ends. The designated inlet brings in about the same
blood in every solve (1.3-1.8e-12 m^3/s); the difference is all at the capillary ends. Under
`estimated` they are a strong net source (in 54% of the image's inflow, out 20%, net +33%); inside
the surround they let blood in and out about evenly (xy: in 0.95e-12, out 1.05e-12) or drain (xyz: in
0.61e-12, out 1.13e-12). On the harness's E14.5 crops, where the truth is known, the capillary ends
exchange a great deal both ways and nearly balance (in 57%, out 60%, net -2%, medians of 10); the
surround reproduces the exchange (in 56%, net +11%), `estimated` carries half of it (in 26%). So the
surround's capillary ends behave as the truths' do, and it is `estimated`'s net feeding on E14.5 that
stands out; with no truth for E14.5 that is consistent, not proven. The harness's 89% through the
open ends is no benchmark for E14.5: a harness crop cuts some 12 arteriole and venule ends besides
its designated ones, and its designated inlet brings in only 8% of its blood, where E14.5's image is
framed round its main arteriole, whose inlet brings in 46% of it even under `estimated`.

**The sparse solvers advised in preflight** (2026-10-29). A synthetic network or a surround is
drawn in the Graph stage, which warns when the dense solvers will not fit in the free memory only
once it is drawn: about 30 s for E14.5's surround, longer for a bigger template. Preflight now asks
the same question of the nodes the network is planned with (`generate.BedPlan.nodes`: its vessels,
the trees' and the bed's, less its loops, plus one; a surround's adds the image run's nodes to its
shell's) and recommends `haemodynamics_solver=sparse` (and `equivalent_resistance_solver=sparse`)
past half the memory free. The template's own node count is no estimate: E14.5's networks came out
with 0.65 of it (0.75-0.8 for a surround), since most of its dead ends are not drawn. The plan's
count was within 3% of the drawn one on the fixture (closed and open) and E14.5, plain and
surrounded, at 1.5x and 2x along xy and 1.5x or 2x along all three
(`outputs/surround_e14p5/node_estimate.py`): E14.5 2x xy plain 3,109 planned and 3,033 drawn,
surrounded 3,491 and 3,507. Its dense solve needs about 0.5 GB, so the advice comes only on a
machine short of memory or a bigger template.

**D6 did not slow the surround** (2026-10-30; `outputs/surround_e14p5/slow_d6/`). E14.5 at 1.5x along
x and y, seed 2, as `run_surround.py` runs seeds 2-5 (statistics off), the commit before D6
(`7c2b2a46`) and this one alternated ABBA twice: 12.8, 13.1 and 12.8 s before (16.6 s cold) against
12.5-13.7 s now, the draw 5.1-5.2 s in both. The mirroring itself takes 0.03 s; the draw alone 9.2 s
with it and 10.6 s without, and the export's comparison 8.4 s and 8.3 s. The 25-30 s against 11 was
load from other sessions on the machine.

**Citations** (2026-10-30). A surround's citations add Schmid, Tsai, Kleinfeld, Jenny & Weber (2017),
PLoS Comput Biol 13(2) e1005392, whose method it is: the realistic network "implanted into a large
artificial MVN", a hole cut in the artificial network's centre, the two joined at the deep in- and
outflow vertices, the penetrating trees' ends joined to the nearest capillary vertex, and the
pressures there taken as the realistic network's boundary conditions. Peyrounette, Davit, Quintard &
Lorthois (2018), PLoS ONE 13(1) e0189474, named beside it in the plan, is not cited: it couples
anatomical arteriole and venule trees to a homogenised continuum standing in for a synthetic
capillary bed, on a periodic domain, and does not move a measured network's boundary. Smith et al.
(2019), already cited for the generator, applies to the shell.

**D5's harder truth: trees rooted inside the image** (2026-10-30; `surround_check.py --truth-roots
z_faces`, `outputs/surround_check/d5z_*` against `p4d_*`). The harness's truths rooted every tree on
an outer face, so a tree vessel an image cuts mostly fed or drained it, which D5's trunks assume. The
new truth roots them across its z0 face (`rooted_on_z_faces`: anywhere 15% in from the x and y
edges), as a cortex's penetrating vessels enter from its surface, so an image often holds a root
and cuts its branches on their way out. Medians of 10 runs (5 seeds, both beds), flow error and
direction agreement:

| | trunk right | `estimated` | surround (truth's template) | surround (image's own) |
|---|---|---|---|---|
| fixture 1.5x, faces | 46/53 (87%) | 0.94, 0.70 | 0.53, 0.84 | 0.56, 0.80 |
| fixture 1.5x, z face | 38/67 (57%) | 0.93, 0.68 | 0.65, 0.78 (met) | 0.62, 0.77 (4 not drawn) |
| fixture 2x, z face | 40/66 (61%) | 0.92, 0.66 | 0.92, 0.75 (flow 4/10: not met) | 0.70, 0.76 (met) |
| E14.5 1.5x, faces | 97/124 (78%) | 0.77, 0.76 | 0.47, 0.86 | 0.48, 0.87 |
| E14.5 1.5x, z face | 69/125 (55%) | 0.98, 0.65 | 0.74, 0.83 (met) | 0.54, 0.84 (1 not drawn) |

The trunk is right at 55-61% of tree ends, against 78-87%: near a coin, still above the "revisit D5"
line at half. The truth is harder for every model (`estimated` 0.98 against 0.77 on E14.5), and the
surround still beats `estimated` on both medians in every row but the fixture's 2x flow error with
the truth's template, where they are level; the image's own template beats it in nearly every run
it is drawn in (fixture 1.5x all 6, 2x 9 and 10 of 10; E14.5 8 and 9 of 10). Its trees' open ends still run the truth's way more often than `estimated`'s (E14.5 0.66
and 0.70 against 0.59), since a trunk to an outer inlet need not feed the image where the image's own
root sits at the inlet pressure. So D5 holds on this truth too, as far as the harness can say; the
fixture's four image-template failures at 1.5x are its thin shell (shell-room warning), and E14.5's
one is a drawn vessel of 1.05 um on an endothelial basis, which the in vivo viscosity law refuses
(under 1.1 um): the generator draws no floor under the surface layer.

**A surround from the panel** (2026-10-30; `outputs/surround_e14p5/panel_run/`). E14.5's run picked as
the template in the real panel, 1.5x along x and y, Run pipeline: preflight passed (91 open ends, 30
left open on the z faces), the run finished in 35 s (55 s with the window shown), 2,850 vessels, and
Colour by offered `network_part`, the image's vessels inside the shell. Pinned on the fixture by
`tests/test_gui_synthetic_widget.py`'s surround run (slow, gui): the vessels layer carries both parts,
the image flows CSV, manifest and Schmid citation are written.
