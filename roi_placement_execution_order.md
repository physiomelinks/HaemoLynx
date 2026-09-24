# ROI placement / selection — order of execution

Control flow rather than data dependency. Arrows show **what runs next**.
Blue hexagons are loop heads, diamonds are branches, dashed outlines are fallback paths.

Source: `src/ImageLynx/roi_placement.py`, function `place_roi`.

Note the boundary. The axial peak slice is **not** computed here — it is read from the QC
record written earlier by `preprocess_cb.py`. That upstream derivation is shown in the grey
block for completeness, but it does not run at ROI placement time.

```mermaid
flowchart TD

  subgraph U1["Upstream · preprocess_cb.py, run earlier"]
    U2{{"for each z slice of the grayscale channel"}}
    U3["take the 99th percentile of that slice's intensities"]
    U4["append to the axial brightness profile"]
    U5["smooth the profile with a moving average<br/>window = max(3, n // 20) slices"]
    U6["argmax of the smoothed profile"]
    U7["write peak_slice into the QC record"]
    U2 --> U3 --> U4
    U4 -. "next slice" .-> U2
    U2 --> U5 --> U6 --> U7
  end

  START(["Specimen, requested ROI size 160 x 160 x 160"])

  R1["read the specimen's QC record"]
  D1{"QC record contains<br/>z_profile.peak_slice?"}
  R2["centre_z = peak_slice<br/>source = z=qc_peak_slice"]
  R3["centre_z = shape[0] // 2<br/>source = z=volume_centre"]

  R4["provisional centre_y, centre_x = shape // 2"]
  D2{"ilastik input file<br/>exists on disk?"}
  R5["source = yx=volume_centre (absent)"]

  subgraph H1["read the grayscale channel"]
    R6["open the HDF5 file"]
    R7["read data[::4, ::2, ::2, 0]<br/>channel 0 only, strided to hold 16x less in RAM"]
    R6 --> R7
  end

  D3{"read succeeded?"}
  R8["source = yx=volume_centre (unreadable)"]

  subgraph C1["tissue_centroid_yx"]
    C2["project the block with max over z"]
    C3["cutoff = 99th percentile of the projection"]
    C4["mask = projection >= cutoff"]
    D4{"any pixel<br/>above the cutoff?"}
    C5["return the centre of the projection"]
    C6["ys, xs = coordinates of the masked pixels"]
    C7["weights = projection values at those pixels"]
    C8["cy, cx = weighted mean of ys and xs<br/>weighting is inert here, see note"]
    C2 --> C3 --> C4 --> D4
    D4 -- no --> C5
    D4 -- yes --> C6 --> C7 --> C8
  end

  R9["rescale for the stride<br/>centre_y = cy x 2, centre_x = cx x 2<br/>source = yx=grayscale_centroid"]

  subgraph K1["clamp_centre"]
    K2{{"for each axis z, y, x"}}
    D5{"requested size<br/>>= volume extent?"}
    K3["centre = extent // 2<br/>box cannot fit, sit it in the middle"]
    K4["clamp centre into<br/>[half, extent - (size - half)]"]
    K2 --> D5
    D5 -- yes --> K3
    D5 -- no --> K4
    K3 -. "next axis" .-> K2
    K4 -. "next axis" .-> K2
  end

  R10["centre_to_offsets<br/>fractional offset from the volume centre per axis"]
  R11["return RoiPlacement<br/>centre, size, offsets, peak_slice, source string"]
  END(["ROI bounds as three slice objects"])

  U7 -. "written to disk, read back later" .-> R1

  START --> R1 --> D1
  D1 -- yes --> R2
  D1 -- no --> R3
  R2 --> R4
  R3 --> R4
  R4 --> D2
  D2 -- no --> R5 --> K2
  D2 -- yes --> R6
  R7 --> D3
  D3 -- no --> R8 --> K2
  D3 -- yes --> C2
  C5 --> R9
  C8 --> R9
  R9 --> K2
  K2 --> R10 --> R11 --> END

  classDef upstream fill:#f2f2f2,stroke:#b0b0b0,stroke-dasharray:5 4,color:#7a7a7a;
  classDef fallback fill:#fdf0dc,stroke:#d99b3d,color:#6b4a12;
  classDef loop fill:#eef4ff,stroke:#7c9fd6,color:#24456f;
  class U2,U3,U4,U5,U6,U7 upstream;
  class R3,R5,R8,C5,K3 fallback;
  class U2,K2 loop;
```

## Reading the chart

**Three independent fallbacks are amber.** Each one substitutes the geometric volume centre
for a measured position, and each records why in the `source` string rather than failing
silently — a silent fallback would reintroduce the centring bias the function exists to
remove. On the six carotid body specimens none of them fire.

**The z and y/x centres are decided independently.** The axial position comes from a QC
record written upstream; the lateral position is measured here from the image. Either can
fall back without the other doing so, which is why `source` is a comma-joined list rather
than a single label.

**The stride is applied before the centroid and undone after it.** `data[::4, ::2, ::2, 0]`
is read, the centroid is computed in strided coordinates, then `cy` and `cx` are multiplied
by 2 to return to full resolution. The z stride of 4 does not need undoing because z is not
used by the centroid — only the max projection over it.

**`C8` is labelled inert.** The upstream preprocessing clips the top 0.02% of voxels to 1.0
and the projection takes a max over z, so 1.33–1.52% of the projection is saturated. The
99th percentile therefore lands exactly on 1.0, every surviving pixel carries the same
weight, and the weighted centroid equals the unweighted one to 0.00 px on all six volumes.
The weighting is kept for inputs that are not saturated at the cutoff.

**`clamp_centre` runs last and can move the centre.** It pulls the box inwards until it fits
wholly inside the volume, because a box hanging over the edge would be silently truncated —
making that sample smaller than the others, which is the thing a matched ROI size exists to
prevent.
