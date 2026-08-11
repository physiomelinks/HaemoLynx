# Viewing the H1 reconstructions in ParaView

Six specimens, one file set each. Everything is in physical micrometres with a common origin,
so the files overlay without any transform and the ParaView ruler reads true distances.

Regenerate with:

```bash
venv/bin/python examples/cb_h1_vtk.py --verify
```

`--verify` re-checks that the frames agree rather than assuming it: that centrelines sit
inside the mask to within a fraction of a voxel, and that every raw skeleton point falls in
mask foreground.

## The files

| File | Type | What it is |
|---|---|---|
| `<SPEC>_surface.vtp` | PolyData | Reconstruction, already contoured and smoothed — **open this first** |
| `<SPEC>_mask.vti` | ImageData | The binary mask itself, if you want to contour it yourself |
| `<SPEC>_vessels.vtp` | PolyData lines | The analysed centrelines, carrying the full per-edge morphometry |
| `<SPEC>_nodes.vtp` | PolyData points | Graph nodes carrying `degree` |
| `<SPEC>_skeleton.vtp` | PolyData points | Raw skeleton, before pruning and smoothing |

`<SPEC>` is `WKY-A`, `WKY-B`, `WKY-C`, `SHR-A`, `SHR-B`, `SHR-C`.

Each file carries `specimen_id`, `group`, `classifier_sha256` and the frozen threshold as
field data, so a view can always be traced back to what produced it.

## A first look

1. **File → Open**, select `WKY-A_surface.vtp` and `WKY-A_vessels.vtp` together, **Apply**.
2. Set the surface **Opacity** to about `0.3`.
3. Select the vessels, set **Coloring** to `edt_diameter_um`, **Line Width** to `2`.
4. Rescale the colour map to a fixed range of **4 to 14 µm** — not to the data range, which
   differs per specimen and would make the six incomparable.

The region is a 160-voxel cube, 297 µm on a side. At 26–34% foreground it is largely opaque
seen from outside, so:

5. Add **Filters → Common → Clip**, set the type to **Box**, and flatten one axis to roughly
   15 µm. A slab resolves individual vessels where the whole cube cannot. This is exactly
   what the figures in the report do, and for the same reason.

## Useful arrays

**On `_vessels.vtp` (cell data):**

| Array | Use |
|---|---|
| `edt_diameter_um` | Calibre — the estimator §1.2 reports |
| `tortuosity` | §1.4. Also correlates with segmentation inclusiveness, so read §9 first |
| `length_um` | Segment length |
| `radius_um` | Ready-made for **Tube** radius, though constant radius reads better |
| `diameter_provenance_code` | 0 measured_edt · 1 measured_fwhm · 2 constant · 3 synthetic |
| `edt_junction_trim_code` | 0 trimmed · 1 untrimmed_too_short · 2 no_junction · 3 not_applied |
| `centreline_smoothing_code` | 0 bspline · 1 bspline_relaxed · 2 raw_fallback · 3 raw_too_short |
| `reconnected` | 1 where terminal reconnection created the edge |

The `_code` arrays exist because ParaView cannot colour by a string. The string version is
beside each one for reading in a spreadsheet view. An unrecognised level codes to `-1` rather
than joining an existing class silently.

**On `_nodes.vtp` (point data):** `degree`, `is_branch_node`, `is_endpoint`.

To see what §1.1 counts: open `_nodes.vtp`, **Threshold** on `is_branch_node` ≥ 1, then
**Glyph → Sphere** with **Scale Array** set to `No scale array` and a radius near 2 µm.

## Things worth looking at

**Which radii are still biased.** Colour the vessels by `edt_junction_trim_code`. Code 1 is
the 32–37% of segments too short to have the junction correction applied at all; they keep a
known upward bias. Those are the segments §11.2 is about, and they are not evenly scattered.

**Where the diameters came from.** Colour by `diameter_provenance_code`. Anything other than
0 is not an EDT measurement.

**Raw against analysed.** Load `_skeleton.vtp` and `_vessels.vtp` together. The difference is
stub pruning and B-spline smoothing — the skeleton shows the voxel staircase, the centrelines
do not.

**All six at once.** Open all six `_vessels.vtp` in one session and colour by `group_code`
(0 = WKY, 1 = SHR). Note that the regions are cut from different parts of each organ, so
their absolute positions are not comparable — only their contents are.

## Two cautions

Each file is a **0.0266 mm³ region, not a whole carotid body** — roughly a fortieth of a cubic
millimetre. The region is centred on each volume's own tissue signal, which samples mid-organ
where the network is denser than at the periphery, so densities seen here overstate the organ.

And these views cannot settle a between-group question. With three specimens per group, a
visible difference between one WKY and one SHR is not evidence of anything; §7.2 and §11 of
the report set out what the numbers do and do not support.
