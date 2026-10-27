# CB study workflow

`python`/`python3` below means the venv interpreter (`venv/bin/python`).

## Stages in order

```bash
python3 examples/preprocessing/preprocess_cb.py --input <dir> --diagnose            # raw TIFF -> report only
python3 examples/preprocessing/preprocess_cb.py --input <dir> --output-dir ilastik_inputs
python3 examples/preprocessing/preprocess_th.py --input <dir>                        # TH (glomus) channel
# Ilastik prediction with the pooled classifier (external, headless) -> ilastik_probabilities/
python examples/carotid_image_to_model.py --record-provenance                        # stamp maps with the classifier; run right after predicting
python examples/cb_h1_batch.py --stage placement
python examples/cb_h1_batch.py --stage threshold
python examples/cb_h1_batch.py --stage run --threshold 0.95                          # ~6 min/specimen
python examples/cb_h1_batch.py --stage sensitivity                                   # frozen threshold's grid neighbours, seed held; 12 runs (~80 min) -> outputs/cb_h1_sensitivity/
python examples/carotid_image_to_model.py --specimen WKY-A --roi-voxels 160 160 160  # one specimen
python examples/carotid_image_to_model.py --list-specimens                           # also shows each specimen's readiness
python3 examples/cb_h1_th_metrics.py          # H1 §1.3, 1.5 (WKY only; --th-threshold sets the TH cuts)
python3 examples/cb_h1_th_metrics.py --all    # adds SHR; writes cb_h1_th_metrics_all.json (--out overrides)
python3 examples/cb_h1_figures.py
python3 examples/cb_h1_vtk.py                 # VTK of the H1 network (--verify checks frames only)
python3 examples/cb_h1_renders.py             # renders cb_h1_vtk.py's .vtp files
python3 examples/cb_h2_glomus_perfusion.py    # H2 §2.1, 2.2, 2.4 (--penetration)
python3 examples/cb_h2_hypoxic_fraction.py    # H2 §2.3; --pad-grid, --grid-um, --vessel-mapping, --contrast for the variant/sweep JSONs; ~40 min per run
python3 examples/cb_h2_vtk.py                 # VTK of the H2 fields (--verify checks frames only; also --specimen, --pad-grid, --vessel-mapping, --decimate); ~14 min
```

## Pipeline vs H2 drivers

The pipeline and the H2 drivers share their pressure boundaries but differ in tier and grid. Check these before comparing their outputs:

- **Pressure boundaries — shared.** Both use the face rule (`select_boundary_terminal_nodes_by_face`) on `cb_settings.BOUNDARY_AXIS` (axis 1) with 1-voxel tolerance; it raises on an empty face. The band rule (`select_boundary_terminal_nodes`) remains for the nerve pipeline (with `allow_extreme_fallback=True`) and as the comparison rule in `cb_h2_absolute_perfusion.py`. `cb_h2_error_propagation.py` uses the same face rule but solves plain Poiseuille flow, not the Pries flow, so its error floors are not the H2 flow field.
- **Tissue transport tiers** (`cb_modelling_reference.md` §6.6). Tier 1 is `solve_perfusion_steady_state` (O₂ only). Tier 2 is `solve_coupled_1d3d_perfusion` (endothelial barrier) and is unreachable at the defaults, because the dispatch checks `use_multi_species_model` before `use_endothelial_barrier_model` and both default to `True`. Tier 3 is `solve_multi_species_perfusion` (O₂, CO₂, pH). The pipeline runs Tier 3 and writes a tier-tagged `*_perfusion.vti`, which is not the H2 field: every H2 hypoxia number comes from Tier 1, called directly by `cb_h2_hypoxic_fraction.py` and `cb_h2_vtk.py` with `cb_settings.PerfusionSettings`.
- **Perfusion grid and vessel mapping** (§6.2, §6.8). The pipeline uses a 10 µm grid (`PerfusionConfig.grid_resolution_xyz`) and `map_vessels_to_grid`'s default `vessel_mapping="centreline"`. The H2 drivers use `cb_settings.GRID_UM` (3 µm) and `vessel_mapping="cross_section"`. Only the cross-section mapping converges with grid refinement.

## Benchmark reference run

`venv/bin/python -m pytest -m benchmark` runs the pipeline once on the committed WKY-A sub-volume (`tests/data/benchmark/wky_a_benchmark_subvolume.h5`) with the frozen defaults and compares it with `tests/data/benchmark/reference_run/`, part by part. A failing part prints its gaps and the kept run folder's path. The report's `environment` lines name any Python or package version that differs from the reference run's `manifest.json`.

**After a deliberate change** that moves an output (a fix, a re-run package, a closed open item):

1. Commit the code change, so the tree is clean. The update script refuses tracked changes and untracked files under `src/` or `examples/`.
2. `venv/bin/python examples/update_benchmark_reference_run.py`. It reruns the pipeline, prints how the new run differs from the current reference run part by part, then rewrites the folder whole.
3. Check that only the parts you expected moved. A mask or skeleton voxel that moves is a topology change: explain it before going on.
4. Commit the new `reference_run/` on its own, with a message that cites the reason (a re-run package letter or an open-item number) and says which parts moved.
5. Run `scripts/ci_like_run.sh -m benchmark` to check that the new reference passes under CI's Python 3.10 and latest allowed packages. Do not loosen a tolerance in `src/ImageLynx/reference_run.py` to make it pass. A gap above a tolerance needs the researcher's decision.

**Rebuilding the benchmark sub-volume** is rare: the sub-volume is frozen (`docs/adr/0001-committed-benchmark-subvolume.md`), and a new Ilastik prediction does not change it. Rebuild it only on purpose, for example after a new pooled classifier. On the machine with the real maps, with a clean tree, run `venv/bin/python examples/make_benchmark_subvolume.py --overwrite`. It recuts WKY-A's map at its placed ROI and rewrites the `.h5` and its `.placement.json` sidecar (source path and sha256, box, provenance, commit). Commit the new cut (the update script refuses tracked changes), then rewrite the reference run as above, and run `venv/bin/python -m pytest tests/test_benchmark_subvolume.py` locally (CI has no real maps). Its `realdata` test rebuilds the cut from the map on disk and checks it equals the committed file. Every committed benchmark file together must stay under 100 MB.
