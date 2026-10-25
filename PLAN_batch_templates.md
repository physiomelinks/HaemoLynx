# Batch templates: a condition's average network as a synthetic template

Harvey, 2026-10-25: "allow the average characteristics from each condition (or the total batch if
there is no condition) of a batch run, to be used as a template. This would mean that a batch run
creates a .haemobatch file per condition, which can then be loaded in."

## What exists

- A batch run (`gui/batch_run.py`, `PLAN_batch_run.md`) writes one `.haemorun` per image,
  `{output root}/{stem}/{stem}.haemorun`, and `batch_summary.csv`; with conditions
  (`statistics.batch_comparison.BatchConditions`: up to two, read from the file names, independent
  or paired) it writes the batch statistics CSVs.
- A synthetic run reads one image run's `.haemorun` as its template (`synthetic.load_template`,
  by `synthetic_template_source`: its statistics report or its graph). `NetworkTemplate.scaled`
  already resizes a template's box, counts going as the volume and inlets/outlets as the faces.

## Design

1. **Writing.** At the end of `run_batch`, beside `write_statistics`: for each condition -- or,
   with none, the whole batch -- the images that finished are read as templates (both sources),
   and `{output root}/{condition}.haemobatch` (`batch.haemobatch` with no condition) is written:
   gzip JSON holding each member's stem, `.haemorun` path and SHA-256, and its two templates
   (`NetworkTemplate.to_dict`), with the condition, the batch's settings file and the version. An
   image that cannot be a template (a refusal from `synthetic.template`) is left out, named in
   the file and the log. Headless too (`python -m haemolynx.gui.batch_run`).
2. **Reading.** `load_template` accepts a `.haemobatch`: it reads the members' templates of the
   asked source and averages them (`synthetic.average_templates`, below) into one
   `NetworkTemplate` whose provenance names the members and the condition. Cached like a run.
   The panel's Template row takes `.haemorun` or `.haemobatch`; the diameter basis it sets is the
   one most members were read on; preflight names the members and warns where they disagree
   (basis, scheme).
3. **Averaging** (`synthetic/average.py`, new): each member scaled to the common box (B1), then
   counts as means (rounded), distributions pooled (samples) or averaged (moments), the order
   table label by label (a label a member lacks counts 0 for it), the dead-end, loop and Murray
   figures as means, the layout as B2, inlets and outlets as B3.
4. **Comparing.** A synthetic run from a `.haemobatch` compares against the average (B4).
5. **Tests.** `tests/test_synthetic_average.py`: averaging two hand-set templates (counts,
   pooled samples, order tables of different depth, boxes of different size); a batch of two
   fixture images writes `batch.haemobatch` and, with two conditions, one file each; a synthetic
   run from it; the panel row accepts it; a member that cannot be a template is left out.

## Decisions for Harvey

- **B1. The box.** (a) each axis the members' mean extent, every count the members' mean
  density times that box's volume; (b) the box of the member nearest the median volume, counts
  scaled to it. Recommended: (a).
- **B2. The layout** (where the vessels lie, `synthetic/organisation.py`). (a) pooled: the per-cell
  fields averaged on the common grid (cells are shares of the box), the members' capillary
  vertices pooled for the bed (as shares of each box); (b) the layout of the member most like the
  average. Recommended: (a), which needs the members imaged the same way up; (b) if they are not.
- **B3. Inlets and outlets.** (a) the mean count of each, rounded, on the faces most members use,
  at positions pooled from theirs; (b) the most typical member's. Recommended: (a).
- **B4. The comparison.** (a) as now, the template's bands round the average; (b) a characteristic
  passes when it lies within the members' own spread (their range, or mean +- 2 SD), reported
  beside the bands. Recommended: (b) beside (a): it says whether the synthetic network is
  another member of the condition.
- **B5. When the files are written.** (a) at the end of every batch run; (b) on a button after it.
  Recommended: (a).

## Status

Planned 2026-10-25; waiting on B1-B5.
