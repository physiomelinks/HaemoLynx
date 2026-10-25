# Manual diameter protocol (`diameter.8b`)

TEMPORARY: part of the release review's human kit (`PLAN_release_review.md` step R5).

`diameter.8b` passes when at least 30 vessels measured by hand agree with the method within 15%,
with the bias reported. Any of the six testers may measure; `diameter.10a` later asks for a second
annotator (George, Rebecca or Mike) or another modality.

## Choosing the vessels

1. Run the pipeline on the dataset through Diameters (E14.5 first, then a second dataset).
2. List the run's vessels with their diameters, and draw 30 or more spread across the widths:
   about a third under 5 um, a third 5-10 um, a third over 10 um, never two from the same branch.
   Draw them before measuring, by their ID, not by how clear they look. Keep the list.
3. Measure where the vessel is straight and away from a junction: the middle third of its length.

## Measuring in napari

1. Open the **raw** image (not the mask) at its voxel size, and add a **Shapes** layer named
   `manual_diameters`.
2. For each listed vessel, find its middle in the viewer and choose the slice (or the rotated plane)
   that cuts it across. Draw one **line** across the lumen, wall to wall as the stain shows it:
   for an endothelial stain the inside of the bright walls, for a plasma stain the edge of the
   bright column at half its height.
3. Draw the lines in the order of the list; one line per vessel.
4. Save the layer as CSV (*File > Save Selected Layer(s)*): `manual_diameters.csv`.

Write down, for the record: who measured, the date, the image, which stain edge was used, and
how long it took.

## Comparing

```bash
python -m release_review.human_kit compare manual_diameters.csv <run>_graph.pkl --voxel-size-zyx 1.0 0.98 0.98
```

(from `scripts/`, with `PYTHONPATH` at `src`). Each line is matched to the vessel whose centreline
passes nearest its middle; the report gives, per vessel and per method, the hand width, the
method's width, their relative difference, the share within 15% and the mean signed difference (the
bias). A line more than one hand width from any centreline is listed and left out.

Record the result in `human_checks.yaml` under `diameter.8b`, with the measurer's name, the date,
the share within 15%, the bias, and where the CSV and the report are kept.
