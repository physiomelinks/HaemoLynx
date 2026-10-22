# Design notes and measurement history

CLAUDE.md is the contract: layout, conventions, rules, and what each module is for. This folder
holds what used to sit beside it there: why a rule or default is what it is, the marked grids it
was fitted on, the benchmarks, and the E14.5 and patella numbers. Each file was moved verbatim
out of CLAUDE.md on 2026-10-22. They are dated history and are not kept up to date line by line,
so check a claim against the code before relying on it.

Add a new measurement history here (or to the matching `PLAN_*.md`), not to CLAUDE.md.

| File | Covers |
|------|--------|
| [inputs.md](inputs.md) | One raw image per run, optional raw image |
| [vessel_masks_and_boundaries.md](vessel_masks_and_boundaries.md) | Mask loading, global masks, masks found by name, stumps, the hop fallback, mask switches, the Large/small tiers, boundary nodes after hand edits |
| [downsample_and_crop.md](downsample_and_crop.md) | Downsample on import, binarising a crop |
| [graph_building.md](graph_building.md) | `consolidate_lumen`, mask-supported bridging, mask recovery, facing ends, loops inside one lumen, short loops, skeleton re-trace, smoothing |
| [dead_ends_and_open_ends.md](dead_ends_and_open_ends.md) | The dead-end kinds and their grids, interior dead-end removal, flow states, dead ends after hand edits, open-end conditions |
| [low_ram.md](low_ram.md) | One mask reader a run, no whole-volume copies, peaks and timings, low-RAM against in-RAM |
| [thick_vessels.md](thick_vessels.md) | Thick-vessel skeletonisation, the large-stack tree, the bridges and their grids |
| [pipeline_and_post_processing.md](pipeline_and_post_processing.md) | `stages.py`, `progress.py` and the Post processing tab, provenance |
| [diameters_and_haemodynamics.md](diameters_and_haemodynamics.md) | FWHM and the diameter chain, viscosity laws and their history, solvers and memory, perturbations, open-end sensitivity |
| [statistics.md](statistics.md) | Network analyses, flux heterogeneity, red cells, exact betweenness, cell distances, tissue volume |
| [gui.md](gui.md) | `.haemorun` save timings, layout, Statistics tab, layer sets, vessel tubes |
| [batch_run.md](batch_run.md) | Batch matching, conditions, paired comparison, sections left out |
| [synthetic.md](synthetic.md) | Synthetic networks |
| [cleanup_plan.md](cleanup_plan.md) | The cleanup checklist with its finished phases |
