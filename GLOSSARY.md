# Glossary

Domain terms for the carotid body (CB) study code. Use these names in code, docstrings and docs.

**Batch run**: one specimen's output folder written by `examples/cb_h1_batch.py`, either `--stage run` (`examples/outputs/cb_h1_batch/<id>/`) or `--stage sensitivity` (`examples/outputs/cb_h1_sensitivity/t<threshold>/<id>/`). It holds the edge table, `roi_placement.json` and exactly one `*_cache/` folder (`network_graph.pkl`, `skeleton.npy`, `vessel_mask.npy`).

**Edge table**: a batch run's `per_edge_morphometry.csv`. It has one row per graph edge, keyed by `(u, v, key)`. The cached graph carries no calibre, so diameters come from here.

**Placed ROI**: the box `roi_placement.place_roi` picks for a specimen, recorded in a batch run's `roi_placement.json`. Every reader of a batch run checks the record against `place_roi` before using it.
