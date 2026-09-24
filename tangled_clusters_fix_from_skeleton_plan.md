# Tangled skeleton clusters: diagnosis and remediation plan

**Status: not started.** Recorded for action at a later date. Nothing in the pipeline is
changed by this document.

Scope: the dense webs of centreline ("tangles") that survive skeletonisation and skeleton
post-processing, and what to do about them. Measurements below were taken on the six
reference ROIs (160^3 voxels, `PROCESSING_VOXEL_UM = (1.8639, 1.866, 1.866)`) through the
operative mask chain: hysteresis 0.90/0.95 -> `close_binary_mask(radius=1)` ->
`keep_largest_mask_components(n=1, connectivity=3)` -> `skeletonize`.

## 1. The problem, measured

Junction voxels (26-neighbour degree >= 3) clustered by connected component:

| specimen | clusters | median | p90 | max | >=10 vox | >=50 vox | % skeleton in clusters >=20 |
|---|---|---|---|---|---|---|---|
| WKY-A | 2767 | 1 | 4 | 665 | 57 | 49 | 20.6% |
| WKY-B | 2685 | 1 | 4 | 424 | 23 | 19 | 8.7% |
| WKY-C | 2392 | 1 | 4 | 501 | 50 | 36 | 16.0% |
| SHR-A | 3073 | 1 | 5 | 336 | 56 | 36 | 10.2% |
| SHR-B | 2949 | 2 | 5 | 474 | 49 | 27 | 8.5% |
| SHR-C | 2810 | 1 | 4 | 579 | 33 | 22 | 9.7% |

The median cluster is 1-2 voxels, which is what a genuine bifurcation looks like. The
problem is the tail: 19-49 clusters of >=50 voxels per specimen, absorbing 8.5%-20.6% of
the entire skeleton.

Largest tangle per specimen, with the centreline length inside it against the diagonal of
its own bounding box:

| specimen | voxels | bounding box (um) | centreline | box diagonal | ratio |
|---|---|---|---|---|---|
| WKY-A | 665 | 37 x 45 x 26 | 1241 um | 64 um | 19.4x |
| WKY-B | 424 | 30 x 30 x 34 | 791 um | 54 um | 14.7x |
| WKY-C | 501 | 37 x 41 x 19 | 935 um | 59 um | 16.0x |
| SHR-A | 336 | 30 x 22 x 21 | 627 um | 43 um | 14.7x |
| SHR-B | 474 | 30 x 32 x 21 | 884 um | 48 um | 18.4x |
| SHR-C | 579 | 43 x 32 x 24 | 1080 um | 59 um | 18.4x |

WKY-A's worst tangle folds 1.24 mm of centreline into a box occupying 0.16% of the ROI
volume while holding 1.5% of the total centreline - roughly nine times the mean centreline
density, and nineteen times more centreline than the straight-line distance across itself.

## 2. Diagnosis: artefact of sheet geometry, not of thickness

Two measurements identify the mechanism.

**The tangles are not in fat regions.** Median EDT radius inside them is 4.17-5.27 um
against 3.73-4.17 um for ordinary degree-2 path voxels - one or two EDT quantisation steps,
not a bulge. This is not thinning applied to a large lump.

**Removing the morphological closing makes it markedly worse.** WKY-A's largest cluster
grows from 665 voxels to 9593, and cluster counts rise on every specimen (WKY-A 2767 ->
3589, SHR-A 3073 -> 4017). The closing is currently suppressing the problem, not causing
it.

Together these indicate the cause is local sheet geometry. Where two capillaries run
alongside one another and the segmentation fuses them, the result is a flat ribbon - thin
in one direction, wide in the other two. Lee's algorithm (skimage 0.24.0 dispatches 3D
input to Lee, Kashyap & Chu 1994) correctly returns a medial *surface* for such a region
rather than a medial axis, and a medial surface in 3D presents as a web of interlinked
centrelines. Rough or pitted mask surfaces produce the same effect at smaller scale, which
is why smoothing them with the closing helps.

**Caveat on interpretation.** The carotid body has one of the densest capillary beds in the
body, sinusoidal and highly anastomotic around glomus cell clusters. The tangle locations
are therefore very likely real - they are where vessels pack tightest and are most likely
to fuse under segmentation. The locations are anatomy; the topology reported at them is
not. Any write-up should say the skeleton misrepresents the connectivity of genuinely dense
regions, not that the regions themselves are spurious.

## 3. Why this cannot be fixed by deletion

Clusters >=20 voxels hold 20.6% of WKY-A's skeleton - about 17 mm of its 82.4 mm total
centreline. Collapsing those to hub nodes naively removes 17 mm of vessel from a resistance
network.

Resistance scales as length / r^4. Removing that much length, concentrated in the densest
regions of the bed, will systematically underestimate total network resistance. Worse, the
loss is uneven across specimens - 20.6% for WKY-A against 8.7% for WKY-B - which introduces
a specimen-dependent bias into precisely the quantity being compared between cohorts.

Any hub replacing a tangle must therefore carry an equivalent resistance: a lumped resistor
fitted to the tangle's own length and calibre distribution, or a small reduced network
rather than a single point. This is a design decision and belongs in the methods, not in
the implementation detail.

This is also the reason to prefer interventions that prevent the tangle over interventions
that manage one after the fact.

## 4. Plan

### Step 0 - instrument first

Implement the **centreline-length to bounding-box-diagonal ratio** as a measurement, not
yet as a filter. Real bifurcations score 1-2; every tangle measured here scores 14-19. The
separation is large and the metric is scale-free, so it does not need per-specimen
recalibration.

This is the number to quote before and after every subsequent change. It is also the
contrast with `skeletonize_voxel_bundles_into_paths`, which is disabled because its density
trigger fired wherever two capillaries passed within ~17 um - normal in this bed. A ratio
criterion does not have that failure mode: two capillaries crossing cleanly give a small
cluster with a low ratio.

Consider adding a per-cluster independent-cycle count alongside it. A true bifurcation
contains zero loops; a medial-surface web contains many. "Many cycles in a small box" would
be very hard for a real junction to trigger.

### Step 1 - seeded watershed on the distance transform

Split fused vessels in the mask before thinning. Re-measure the ratio. This quantifies how
much of the tangling is segmentation fusion.

### Step 2 - TEASAR-style centreline extraction

Replace `skeletonize` with a curve-skeleton extractor (Kimimaro is the usual implementation,
built for this problem in dense EM data). Such methods trace paths through the distance
field and return curves by construction, so they cannot emit a medial surface. Re-measure.
This quantifies how much was the extractor.

Note these two steps are complementary, not alternatives. Running a TEASAR-style extractor
on a fused mask does not produce a tangle - it produces a single clean centreline running
through the fusion, silently merging two capillaries into one path. A visible tangle is at
least legible as a defect; a confident wrong curve is not. Conversely the watershed alone
leaves small webs arising from surface roughness.

### Step 3 - collapse whatever survives

Apply hub collapse only to the residue, with the resistance-preserving treatment from
section 3. If steps 1 and 2 bring the ratio near 1-2 there may be very little left to act
on, which is the desired outcome, since this is the only step that costs centreline.

### Sequencing

One change at a time, with the step 0 metric re-measured after each. Applying them together
leaves no way to attribute the improvement.

If effort is constrained, step 0 plus step 1 gives the most value for the least disruption.
Step 2 is the more principled fix but replaces a pipeline stage.

## 5. Dependencies and risks

**Both steps 1 and 2 invalidate the threshold selection chain in section 2.2 of
`cb_modelling_reference.md`.** That section selects the threshold using endpoint density
measured on a skimage skeleton of the plain-cut mask, and the fragmentation onset, the
`FRAGMENTATION_TOLERANCE = 1.5` and the frozen 0.90 are all calibrated against that
combination.

- Step 1 changes the mask, so calibre measurements and fragmentation onsets move.
- Step 2 changes the skeleton, and a TEASAR endpoint is not a thinning endpoint, so
  endpoint density changes meaning.

Either forces section 2.2 to be re-run and re-justified. Both at once means rebuilding it
on two changed inputs simultaneously.

**Retain the morphological closing.** It is currently suppressing this problem. The
intuition runs the other way, so this is worth stating explicitly wherever the change is
described.

## 6. Acceptance criteria

- Median junction-cluster size unchanged at 1-2 voxels (genuine bifurcations preserved).
- No cluster with a length-to-diagonal ratio above ~3.
- Fraction of skeleton in clusters >=20 voxels below 1% on all six specimens.
- Total centreline length not reduced by more than a few percent, and the reduction
  comparable across specimens rather than ranging 8.7%-20.6%.
- Section 2.2 re-run and its conclusions restated against whatever inputs changed.
