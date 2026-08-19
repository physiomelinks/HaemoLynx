# Viewing the H2 perfusion and glomus results in ParaView

The H1 exports show the vascular reconstruction. These add the second channel and the flow
solve: where the glomus nests sit, which capillaries penetrate them, what those capillaries
carry, and what the tissue oxygen field looks like around them. Those are the four things H2
measures, and each is far easier to judge by eye than from a table of ratios.

Regenerate with:

```bash
venv/bin/python examples/cb_h2_vtk.py --verify     # check the frames, write nothing
venv/bin/python examples/cb_h2_vtk.py              # write all six specimens
```

Everything is in physical micrometres on a common origin, so the files overlay without any
transform even though the glomus mask is at 1.866 µm and the perfusion grid at 4 µm.

`--verify` re-checks that rather than assuming it. It measures how often an edge scored as
penetrating actually has its midpoint inside the exported mask, against edges scored as
outside, and against a deliberately transposed copy of the same mask. Across the six that is
87.4–90.7% against 0.2–1.7%, with the transposed control at 13.8–42.7%. A specimen failing
that is skipped rather than written, because a misaligned overlay renders perfectly well and
is wrong.

Coverage is reported separately and does not block writing. The perfusion grid takes its
extent from the vascular bounding box, so a specimen whose vessels stop short of the region
edge gets a grid smaller than the glomus mask. **SHR-A and SHR-C are affected**: 4.35% and
7.54% of their glomus volume lies outside the solved grid and carries no PO₂. In ParaView the
perfusion volume visibly ends before the glomus surface does — SHR-A's stops at 286 µm where
the tissue runs to 298 µm. That is where the vessels stop, not a misalignment.

## The files

| File | Type | What it is |
|---|---|---|
| `<SPEC>_glomus_surface.vtp` | PolyData | The glomus tissue, contoured and smoothed — **open this first** |
| `<SPEC>_glomus_clusters.vtp` | PolyData | The same surface split into connected nests, each with an id and its volume |
| `<SPEC>_glomus_prob.vti` | ImageData | TH probability at native 1.866 µm, if you want to contour it yourself |
| `<SPEC>_perfusion.vti` | ImageData | The 4 µm solve: PO₂, tissue fraction, metabolic rate, source terms |
| `<SPEC>_vessels_h2.vtp` | PolyData lines | Centrelines carrying flow, haematocrit, viscosity, transit time and the penetrating flag |

`<SPEC>` is `WKY-A`, `WKY-B`, `WKY-C`, `SHR-A`, `SHR-B`, `SHR-C`. `export_summary.json` in the
same directory records the frame check and the counts for each.

These are the same six regions as the H1 exports, so `<SPEC>_surface.vtp` from that set loads
into the same scene and lines up.

## A first look

1. **File → Open**, select `WKY-A_glomus_surface.vtp` and `WKY-A_vessels_h2.vtp` together,
   **Apply**.
2. Set the glomus surface **Opacity** to about `0.4` and its colour to something flat.
3. Select the vessels, set **Coloring** to `penetrating`, **Line Width** to `3`.

That is the H2 stratification itself: every red segment has more than half its centreline
inside glomus tissue, and every comparison in the report is between those and the rest.

4. Add **Filters → Common → Clip**, type **Box**, and flatten one axis to roughly 15 µm. The
   region is 26–34% vessel by volume and largely opaque from outside; a slab resolves what the
   whole cube cannot.

## Useful arrays

**On `_vessels_h2.vtp` (cell data):**

| Array | Use |
|---|---|
| `penetrating` | 1 where ≥ 50% of the centreline is inside glomus tissue. The H2 stratification |
| `th_fraction` | The underlying continuous value, before the cutoff |
| `flow_um3_s` | Volumetric flow, in µm³/s. Read the second caution below before quoting it |
| `hematocrit` | Discharge haematocrit after phase separation at each bifurcation (§2.2) |
| `viscosity_cP` | Pries–Secomb apparent viscosity, in vivo law |
| `diameter_um` | Calibre, carried over from the H1 morphometry |
| `transit_time` | Arrival time from the nearest inlet along the solved flow direction (§2.4) |
| `length_um` | Segment length |

Unreachable segments carry `nan` for `transit_time` rather than infinity, which VTK cannot
hold. ParaView hides them; that is intended, and they are the disconnected fraction.

**On `_perfusion.vti` (cell data):** `PO2_mmHg`, `th_fraction`, `metabolic_rate`,
`q_total_um3_s`, `s_incoming`.

**On `_glomus_clusters.vtp` (point data):** `cluster_id`, `cluster_volume_um3`.

## Things worth looking at

**One nest at a time.** Open `_glomus_clusters.vtp`, **Threshold** on `cluster_id`. Each nest
is a separate connected component with its own id, so a single glomerulus can be isolated and
its neighbouring vessels read off. `cluster_volume_um3` on the same points gives its size
without measuring anything. Nothing in H1 or H2 reports a per-cluster quantity, so this is for
inspection rather than analysis.

**Whether penetrating vessels are actually different.** Colour the vessels by `hematocrit` and
rescale to a fixed range across specimens. §2.2 finds the two populations' medians within about
6% of each other in five of six specimens, and the view shows why: that is far less than the
variation between neighbouring segments in either population. The exception is SHR-C at 0.770,
which the §2.2 group mean rests almost entirely on.

**Where the tissue takes oxygen out.** Colour `_perfusion.vti` by `metabolic_rate`. The
glomus cells consume at twice the stromal rate, so this array is essentially the tissue
fraction rescaled, and it should coincide with the glomus surface. If it does not, the frames
have drifted and the check has missed something.

**Transit time along the tree.** Colour the vessels by `transit_time`, then **Threshold** to
the lower half. What remains is the fast path from inlet to outlet. §2.4 is the observation
that in WKY the penetrating segments sit late on this map and in SHR they sit early.

**All six at once.** Open all six `_vessels_h2.vtp` and colour by `penetrating`. The regions
are cut from different parts of each organ, so their absolute positions are not comparable,
only their contents.

## Three cautions

**The oxygen field is nearly flat, and that is a result, not a rendering problem.** Colouring
by `PO2_mmHg` gives an almost uniform volume. The diffusion length here is 20–45 µm and the
tissue-to-vessel distance is 5–8 µm, so oxygen reaches everywhere from the nearest capillary
several times over and no gradient survives. The consequence is that §2.3's glomus-specific
mechanism is inert in this tissue: raising glomus consumption to twice stromal changes almost
nothing. Do not read a flat field as a failed solve.

**Absolute perfusion is 20–100× lower than measured carotid body flow.** The ratios in the
report are reliable — the three independent unit and physics corrections moved absolute
quantities by three to five orders while every within-specimen ratio moved by at most 0.02 —
but `flow_um3_s` and everything derived from it should be read as relative, not as a
calibrated flow rate. The likely cause is the 160-voxel region cutting the supplying arteriole,
which is recorded as T1.9 and not yet resolved.

**Each file is a 0.0266 mm³ region, not a whole carotid body**, centred on each volume's own
tissue signal. And with three specimens per group, a visible difference between one WKY and
one SHR is not evidence of anything; §7 and §10 of the H2 report set out what the numbers do
and do not support.
