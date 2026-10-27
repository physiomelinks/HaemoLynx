# Glossary

Domain terms for the carotid body (CB) study code and its test tooling. Use these names in code, docstrings and docs.

**Batch run**: one specimen's output folder written by `examples/cb_h1_batch.py`, either `--stage run` (`examples/outputs/cb_h1_batch/<id>/`) or `--stage sensitivity` (`examples/outputs/cb_h1_sensitivity/t<threshold>/<id>/`). It holds the edge table, `roi_placement.json` and exactly one `*_cache/` folder (`network_graph.pkl`, `skeleton.npy`, `vessel_mask.npy`).

**Edge table**: a batch run's `per_edge_morphometry.csv`. It has one row per graph edge, keyed by `(u, v, key)`. The cached graph carries no calibre, so diameters come from here.

**Placed ROI**: the box `roi_placement.place_roi` picks for a specimen, recorded in a batch run's `roi_placement.json`. Every reader of a batch run checks the record against `place_roi` before using it.

**Driver snapshot**: the files and stdout that one run of chosen `examples/cb_*.py` drivers wrote, kept in one folder so a later run can be compared with it by content. Not a batch run: it holds what the drivers make from batch runs.

**Mutation check**: one deliberate break of a source line, applied in a throwaway worktree, with the test meant to guard it. Caught when that test fails, survived when it passes (some breaks are expected to survive).

**Benchmark sub-volume**: WKY-A's Ilastik probability map cut at its placed ROI, both channels, committed as a test fixture. It stays frozen: only its own generating script changes it, and its sidecar records which map and box it came from.

**Reference run**: the chosen outputs of one pipeline run on the benchmark sub-volume, committed so a later run can be compared with it by content. A deliberate change to those outputs regenerates it, and the commit says why. Not a batch run: it is not cut from the full map at test time and holds only the chosen outputs.
