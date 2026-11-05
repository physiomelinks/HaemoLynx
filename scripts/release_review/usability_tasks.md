# Usability task list (`usability.6a`, `usability.8a`, `docs.8a`)

TEMPORARY: part of the release review's human kit (`PLAN_release_review.md` step R5).

**For:** a first walk-through by a developer, timed (`usability.6a`), then George, Rebecca and
Mike, each as a first-time user (`usability.8a`, `docs.8a`). Nobody shows a first-time tester the
panel beforehand: the list is only a first-time test once. `RELEASE_REVIEW.md` § Testers suggests
spending one tester now and keeping the other two until the help pages and the guide gaps are done.

**The observer** reads each task aloud, starts the clock, and does not help. If the tester is stuck
for five minutes, or asks for help, the observer may give one hint and notes it: that task then
counts as *helped*. Write down where the tester looked, what they expected, and anything they said.
The console and the code are off limits for the tester; the README, `tutorials/README.md` and the
panel's own hover text and help window are allowed.

**Data:** the Nerve crop (`tests/data/Nerve_capillaries_cropped.tif`, a segmented mask) unless the
tester brings their own segmented stack.

Record each result in `scripts/release_review/human_checks.yaml` under the check, with the tester's
name, the date and where this sheet is kept. Keep the filled sheet beside the other marked grids
(`outputs/usability_review/`, git-ignored) or on paper; the record names where.

---

Tester: ____________________  Observer: ____________________  Date: ____________

Machine and OS: ____________________  HaemoLynx version or commit: ____________________

| # | Task | Done when | Minutes | Unaided / helped / gave up | Where they got stuck |
|---|---|---|---|---|---|
| 1 | **Install** HaemoLynx with the panel, from the README. | napari opens with the HaemoLynx panel docked. | | | |
| 2 | **Open an image**: load the segmented stack and check its voxel size. | The image is shown in napari, and the tester can say its voxel size in microns. | | | |
| 3 | **Run** the pipeline from start to the end with flows solved. | The run finishes, with the network and its flows in the viewer. | | | |
| 4 | **Find a result**: the total vessel length, and the flow in the widest vessel. | The tester reads both numbers, with their units. | | | |
| 5 | **Inspect**: colour the vessels by diameter and find the narrowest one. | The colour map shows diameter, and the tester points at the narrowest vessel. | | | |
| 6 | **Edit a vessel**: delete one vessel that should not be there, and solve again. | The vessel is gone, and the flows are solved again without it. | | | |
| 7 | **Export**: save the network for ParaView and the statistics as a spreadsheet. | The tester names the files written and where they are. | | | |
| 8 | **Re-run from a stage**: change one diameter setting and run again from the Diameters stage only. | The run starts at Diameters and the earlier stages are not repeated. | | | |
| 9 | **Save and reopen the run**, then close napari and open it again. | The reopened run shows the same network and results. | | | |

Total minutes: ________  Tasks done unaided: ____ of 9

Anything the tester expected to find and could not, in their words:

&nbsp;

&nbsp;

Then hand the tester the SUS sheet (`sus_sheet.md`) before discussing anything.
