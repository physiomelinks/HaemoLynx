# LCFM image preprocessing

Raw confocal z-stacks to Ilastik-ready HDF5, for both channels of the carotid body
acquisitions. These scripts used to live next to the data and are now version controlled here,
so paths are passed on the command line rather than implied by the working directory.

## Where things are

The acquisitions and derived volumes are not in this repository: they are roughly 5 GB and are
excluded by `.gitignore`. `ImageLynx.specimens` resolves their location, and
`IMAGELYNX_CB_DATA_ROOT` overrides it. On the current machine:

    ~/Desktop/LCFM Images/CB3-WKY/raw_cb_images/   WKY acquisitions
    ~/Desktop/LCFM Images/CB3-SHR/                 SHR acquisitions
    ~/Desktop/LCFM Images/ilastik_inputs/          outputs, both channels

The kilobyte QC sidecars that record how each volume was produced *are* committed, under
`src/ImageLynx/data/preprocessing_qc/`.

## The two channels

Each acquisition is one `ZCYX` TIFF with two channels. **Channel 0 is lectin (vessels),
channel 1 is TH (glomus cells)**, verified by byte-comparison against the separately extracted
`C1-*_vessels.tif` and `C2-*_glomus_cells.tif`.

| | vessels | TH glomus cells |
|---|---|---|
| script | `preprocess_cb.py` | `preprocess_th.py` |
| channel | 0 | 1 |
| output channels | grayscale, vesselness fine, vesselness coarse | grayscale, signed soma DoG |
| output suffix | `_ilastik.h5` | `_TH_ilastik.h5` |
| protocol | `optimal-filtering-strategies-v5.md` | `th-glomus-cell-preprocessing-guide.md` |

`preprocess_th.py` imports its shared stages from `preprocess_cb.py`, so run it from this
directory or put this directory on `PYTHONPATH`.

## Running

Diagnose before processing, always. It reports the z-profile verdict, tissue occupancy and both
anchor choices without writing anything, and prints a cohort-wide anchor pair.

```bash
DATA=~/Desktop/"LCFM Images"
OUT="$DATA/ilastik_inputs"

# Vessels
python3 preprocess_cb.py --input "$DATA/CB3-WKY/raw_cb_images" --diagnose
python3 preprocess_cb.py --input "$DATA/CB3-WKY/raw_cb_images" --output-dir "$OUT"

# TH, all six specimens. Pass the cohort anchors from the --diagnose run so the
# applied gain is identical for every specimen and cannot track how much empty
# frame was acquired.
python3 preprocess_th.py \
    --input "$DATA/CB3-WKY/raw_cb_images/CB3-WKY-CB-"*-2x2x2.tif \
            "$DATA/CB3-SHR/CB3-SHR-CB-"*-2x2x2.tif --diagnose
python3 preprocess_th.py \
    --input "$DATA/CB3-WKY/raw_cb_images/CB3-WKY-CB-"*-2x2x2.tif \
            "$DATA/CB3-SHR/CB3-SHR-CB-"*-2x2x2.tif \
    --output-dir "$OUT" --anchors 708 10578
```

The six TH volumes currently in `ilastik_inputs` were produced with `--anchors 708 10578`.

Downstream, `prob_to_mask.py` turns an Ilastik probability export into a binary mask plus a
calibrated distance transform. It expects channel-last `zyxc` and a dataset named `data`, which
both preprocessors write.

## Tests

`tests/test_preprocess_th.py` in the repository root, run by the normal suite. Each test encodes
one defect found in `th_glomus_preprocessing_review.md`, so reintroducing any of them fails
there rather than silently in a cohort run.

## Note on the unbinned data

`CB3-SHR-CB-{A,B,C}-1x1x1.tif` exist at 0.933 um, but there is no WKY equivalent with a TH
channel. The unbinned stacks therefore cannot support a group comparison; the binned `2x2x2`
data is the only common ground. Every pixel-valued default in both scripts assumes the binned
resolution and would need doubling for unbinned input.
