# Plan: the raw image as evidence for dead ends

*2026-10-13. Branch `claude/raw-image-input`, which makes one raw image the run's own: its path,
its channel and its stain (`raw_image_stain`: auto, plasma column or endothelial) on the Input
tab. This plan says how dead-end handling should use that image. It is decision D6 of
`PLAN_dead_ends.md` ("the raw image and the endothelial channel as evidence for broken against
blind"), now that there is one image to read.*

It builds on the "raw confirmation" session's `PLAN_raw_confirmation.md`. That plan's
`preprocessing/raw_evidence.py` (`RawEvidence`: one reader, sampled, windowed, calibrated once
per run) is the reader every rule here asks, rather than a second one. It reads both stains
(`RawEvidencePlasma`, `RawEvidenceEndothelial`: `vesselness`, `signal_along`, `wall_between`,
`hole_is_lumen`), and this plan adds the wall-stain primitives dead ends need on E14.5, whose raw
image is claudin-5.

## Status (2026-10-25)

| Phase | State | Where |
|---|---|---|
| A. Wall primitives | **Done** | `bafd3aa2`: `ring_along` (E1), `inside_walls` (E2), `walls_ahead` (E3), the ring fields on `RawCalibration`, `ring_calibration`; tests in `tests/test_raw_evidence.py` |
| Cost on E14.5 | **Measured** | Section 6 (Risks) |
| B. Measure and record, re-fit G1-G4 (R1) | **Waits** on `graph/dead_ends.py`, `graph/network_review.py` and the census, which the dead-ends and bridging sessions are finishing | Section 4 |
| C. The grid | **Built and sent** as **DR1** (`9debc4d1`): https://claude.ai/artifact/FUrmkvLXHiSobkY1zAk9Jf. Waits on Harvey's marks | Section 4 |
| D. Rules | **Waits** on DR1's marks, then a fresh grid DR2 agreeing at least 90% | Section 4 |
| E. Docs, second dataset | CLAUDE.md, the notes and `RELEASE_REVIEW.md` `graph.6b` updated for A and C; a plasma-only dataset waits on data | Section 4 |

---

## 1. What the marks and the measurements say

- **G1–G3 (Harvey, 2026-10-12, 87 marks).** 54 dead ends were "not a real vessel". No measure of
  the mask separated them: their chains lie in the mask (median in-mask fraction 0.99–1.00).
- **G4 (Harvey, 2026-10-13, 29 marks: 15 not a vessel, 12 vessel, 2 partly).** Every mark cites
  the segmentation's shape and context; none cites the endothelium.
  - **Stains:** claudin-5 contrast along the chain gets 20/27 leave-one-out (74%), against 15/27
    for marking everything one way. Dextran gets 15/27: every chain lies in mask made from it.
  - **Shape:** no measure does better than 17/27 (lumen radius, distance to another centreline,
    the mask the chain covers alone, winding, mask fill round the tip).
  - **So:** what decides "not a vessel" is a judgement of shape in context, and the stain is
    weak evidence read the way G4 read it, as a mean along the chain.
- **The two E14.5 channels across a vessel's edge** (median by signed distance from the mask):
  - Claudin-5 peaks within 1 µm of the edge (~470–510), falls to ~38 five microns inside the
    lumen, and is ~30 outside. That is a wall ring round a dark lumen, at the plasma mask's edge.
  - Dextran is brightest deep inside the vessel and falls steadily outward.
  - `read_raw_stain` tells them apart on all five seeds (dextran +0.25 to +0.33; claudin-5 −0.16
    to −0.44).
- **What G4 read, and what it did not.** It averaged the stain *along* the chain. An
  endothelial stain's information is *across* it: whether there is a ring round the chain, the
  chain's own or a neighbour's, and whether the ring goes on past the tip. Nothing has measured
  that yet. It is the cheapest evidence still unused.

## 2. The questions the raw image can help with

| Question | Dead-end kinds it bears on | Raw evidence (endothelial stain) | Raw evidence (plasma stain) |
|---|---|---|---|
| Q1. Is the chain a vessel at all? | artefact, unknown (G1 X, G4 N) | a wall ring of its own round the chain, not a neighbour's | weak: the mask was made from it |
| Q2. Does the vessel go on past the tip? | stops_short, broken, unfilled sprouts | the wall-enclosed lumen continues along the heading | dim lumen signal past the mask's end |
| Q3. Does the vessel leave the image? | leaves_image (G2) | the wall-enclosed lumen reaches the face | lumen signal reaching the face |
| Q4. Where does it go? | trace on, join to the vessel ahead (D3) | a route through the wall-enclosed lumen | a route through the lumen signal |

Q2 is where an endothelial stain is unique. Coelho-Santos et al. found unfilled sprouts only with
an endothelial label: a vessel with walls but no plasma. A dextran mask cannot see one, and the
claudin-5 channel can.

## 3. Evidence: four measures, one reader

Each measure is a number per dead end, keyed in the grids' CSVs. None changes the graph until a
fresh grid confirms it (CLAUDE.md, "Marked example grids").

- **E1. Ring support along the chain (Q1).**
  - Endothelial stain: the share of the chain's sections, every 2 µm over its last 10 µm and over
    its whole length, where a wall ring is fitted round the centreline. This reuses
    `haemodynamics/endothelial.fit_ring_section` with the run's calibrated blur and wall
    (`calibrate_from_rings`), gated by `endothelial_min_ring_contrast`.
  - Also record whether the fitted ring's centre lies on the chain or on another centreline. A
    spur inside a fat lumen sits inside its parent's ring and has none of its own.
  - Plasma stain: E1 reduces to the peer plan's `signal_along`, which G4 suggests is weak.
- **E2. Own lumen (Q1).** Whether the chain's sections fall inside the wall-enclosed region of a
  vessel that is not its own chain. `segmentation_raw_comparison.enclosed_by_walls` (this branch)
  gives the walls and the lumens they enclose. A chain that runs inside another vessel's walls
  is the "two in one lumen" artefact, seen in the raw image instead of the mask.
- **E3. Continuation past the tip (Q2, Q3).** Along the tip's heading (`DeadEnd.heading`, read
  over the whole chain), in steps of 1 µm up to 30 µm:
  - how far the wall-enclosed lumen (endothelial) or the lumen signal (plasma: brightness in the
    peer plan's units) goes on;
  - whether it meets another vessel's lumen (Q4: the join), the image face (Q3) or ends;
  - the background crossed on the way, as `bridge_mask_support` measures it in the mask.
  This is the raw analogue of `DeadEnd.mask_run_um` and `ahead_um`.
- **E4. A route (Q4).** The traced path the review's Trace on and `trace_on_dead_ends` would
  take, with a cost from the raw image instead of the mask. For an endothelial stain the cost
  is low inside `enclosed_by_walls` and high on the walls themselves. That reuses the Add-vessel
  A* (`graph/post_processing.py`'s trace through raw data), which already reads an intensity
  cost field.

All four read through `RawEvidence` windows. The stain comes from `G.graph["raw_image_stain"]`,
resolved once by the Diameters stage, or by `resolve_stain` when classification runs first.

## 4. Phases

### Phase A — The wall primitives in `RawEvidence` (no behaviour change)

Agreed with the raw-confirmation session on 2026-10-13, on the five conditions below, so that
`RawEvidence` stays one reader. Its branch `claude/raw-evidence` (48ccd51, non-slow suite green)
already reads `raw_image_path`, `raw_image_channel` and `raw_image_stain`, falling back to the
retired names.

1. **Build on what is there.** `RawEvidenceEndothelial` already has `vesselness(points, radius,
   tangents)` (the brightest point on rings across the vessel at r−1, r and r+1 µm, 8 directions
   each), `wall_between(a, b)` (a bright line across the chord) and `hole_is_lumen` (no wall
   lining the hole). Its `RawCalibration` holds the vessel, wall and background levels, their
   spread, and the blur (`sigma_vox`) they were read under.
   - `ring_along(path_um)` (E1) is `vesselness` over the path's samples, plus
     `endothelial.fit_ring_section` only where the fitted centre is needed (whose chain the ring
     belongs to).
   - `inside_walls(points_um)` (E2) and `walls_ahead(tip_um, heading)` (E3) are new methods on
     `RawEvidenceEndothelial`.
2. **Keep the cost model.** Windows only (`RawEvidence._read`, chunked to `MAX_WINDOW_VOXELS`),
   with stats counted and no whole-volume pass. `enclosed_by_walls` may run once per window, never
   once per volume.
3. **The ring calibration is optional.** The block calibration (about 1 s) stays the default.
   `calibrate_from_rings` (90–120 s on E14.5 with 8 workers) adds the ring-fitted wall level,
   thickness and blur to `RawCalibration` only when the run already pays for them (endothelial
   diameters on) or a rule asks. These become optional fields defaulting to None, and
   `from_record` must still read old graphs.
4. **Tests** go in `tests/test_raw_evidence.py`, beside its endothelial tube and walled slab
   fixtures, with one case per primitive: a ring round the chain, a ring round a neighbour,
   walls running on past the tip, and walls stopping at it.
5. **Sequence.** `claude/raw-evidence` landed on `Devel_GUI_improvements_HD` at 96b0329
   (2026-10-13) with `preprocessing/raw_evidence.py`, its `build_network` hook and
   `tests/test_raw_evidence.py`, and `claude/raw-image-input` with the settings,
   `read_raw_stain`/`resolve_stain` and `enclosed_by_walls`. Phase A is on top of both.

A `keep_dtype` option on the raw loader is not needed for now: the in-RAM path reads a channel in
its own dtype, and float32 on the memmap path is acceptable until a large stack says otherwise.

**Landed (2026-10-25, `bafd3aa2`).** All three are methods of `RawEvidenceEndothelial`; they
measure and nothing reads them yet.

- `ring_along(path_um, radius_um, *, others_um=None, fit_rings=True, min_contrast=None,
  psf_sigma_zyx=None, wall_um=None)` returns `RingSupport`:
  - `fraction` / `tip_fraction`: the share of the chain's sections, every 2 um (at 1, 3, 5 ...
    um from the tip), with a wall on both sides across it (`vesselness`), over the chain and its
    tip's 10 um;
  - `readings`: a `RingReading` per tip section (five), the ring fitted by
    `endothelial.fit_ring_section` on a window round the section, exactly what it gives on the
    whole volume (pinned). Each says `ring` or why not (`no_dark_lumen`, `low_contrast`,
    `partial_wall`, `ring_not_closed`, `off_volume`), the centre in microns, the diameter, and
    whose a seen ring is: `own`, `other` or `off_centre`;
  - **whose** is the nearest centreline to the fitted centre, the chain or one of `others_um`. The
    fit's own centring limit (1.5 um or 0.3 of the diameter) is generous on a wide ring: a strand
    2.5 um off a 12 um vessel's axis is "centred" by it, though the ring is the vessel's;
  - the blur and wall are the arguments, else the calibration's ring fields, else fitted free.
- `inside_walls(points_um, radius_um, *, others_um=None, reach_um=None)` returns `InsideWalls`,
  per point: `lumen` (enclosed by walls, off them), `wall` (on one), `shared` (the enclosed region
  it lies in also holds another centreline). Each point is judged on its own window
  (`enclosed_by_walls` once per window), so the answer never depends on the points asked with it.
- `walls_ahead(tip_um, heading, radius_um, *, others_um=None, others_radius_um=0, limit_um=30,
  step_um=1)` returns `WallsAhead`: from one lumen radius past the tip, a reading every 1 um
  (`vesselness` across the heading), `run_um` until the first that is not vessel, `stop`
  (`vessel`: into another vessel's lumen with no gap; `face`; `runs_on`; `ends`), `meets_um`,
  `face_um`, the gaps before it, and the readings themselves.
- `RawCalibration` gains `ring_psf_sigma_zyx`, `ring_wall_um` and `ring_wall_level` (None by
  default; `from_record` reads older graphs), and `ring_calibration(G, evidence, ...)` fills them
  from `calibrate_from_rings`. `RawEvidence._read(blur=False)` reads a window unblurred for the
  ring fit.
- The tests use synthetic walled tubes: a ring round the chain, a ring round a neighbour, walls
  running on past the tip (to the limit, to the face, into another vessel), walls stopping at
  it, a gap before a vessel, two lumens kept apart, and a memory map giving the in-RAM answers
  in windows under a quarter of the volume.

### Phase B — Measure on E14.5 and record (no behaviour change)

- `classify_dead_ends(..., raw=RawEvidence | None)` fills E1–E3 on every `DeadEnd`. The kinds
  stay as they are. The census (`scripts/graph_artefacts/dead_end_census.py`) and `dead_ends.csv`
  carry the measures, and the review list shows them.
- Re-read G1–G4 with them: the 116 marked dead ends' E1–E3 against Harvey's marks, fitted by
  `fit_marks.py`. This costs no marking and says whether a fifth grid is worth asking for.

**Waits (2026-10-25)** until the dead-ends and bridging sessions hand over `graph/dead_ends.py`,
`graph/network_review.py` and `scripts/graph_artefacts/dead_end_census.py` (the bridging
session's `thick_wall` kind lands in them first). What is left, in order:

1. `classify_dead_ends(..., raw=None)` fills E1-E3 on each `DeadEnd` (new fields, None without a
   raw image or under a plasma stain, whose analogues are `signal_along` along the chain and along
   the heading), with `others_um` from the classifier's own `_Centrelines`. The kinds do not change.
2. The census, `dead_ends.csv` and the review list carry them; each new CSV column gets its
   `pipeline/export_dictionary.py` entry and the page is written again.
3. R1: G1-G4's 116 marks re-fitted with E1-E3, from the keys in
   `.claude/worktrees/dead-ends/outputs/dead_end_review/` (`G1_key.csv`-`G4_key.csv`), reported
   beside `FIT.md` / `FIT_G4.md` against marking everything one way. Those keys come from the
   1,082-dead-end graph of 2026-10-12, so the measures are read on that graph and mask, not today's.

### Phase C — Grid DR1, spread by the new evidence

Named **DR1** ("dead ends, raw image"), not G5: G5 is the stops_short route grid of 2026-10-21.
Its fresh check, Phase D's, is **DR2**.

- 30 fresh interior dead ends, none shown before, spread across E1 (ring support) and E3
  (continuation), then shuffled.
- The same page as G4: segmentation, dextran, claudin-5, in two planes. It adds a third plane
  along the heading past the tip, with the E3 path drawn, because Q2 is about what lies beyond
  the tip.
- One question with G4's marks (V vessel, N not a vessel, P partly), plus a second: "Does it go
  on past the tip?" (yes / no / unsure). What decided it is recorded as before.
- Fit, and report right / wrong / all one way (CLAUDE.md). If the marks again cite shape and
  context only, say so, and keep looking for the shape measure instead of the stain.

**Built and sent (2026-10-25, `9debc4d1`, `scripts/dead_end_review/build_dr1.py`):**
https://claude.ai/artifact/FUrmkvLXHiSobkY1zAk9Jf (marks save to the page's `annotations`
collection). Page, key, candidate pool and calibration: `outputs/dead_end_review/DR1_*` in worktree
`plan-dead-ends-raw-04d14a` (git-ignored).

- **Data:** E14.5, the dead-ends session's graph of Harvey's Claudin segmentation
  (`plan-dead-ends-36f84f/outputs/de_fresh/e14_final.pkl`, 394 dead ends), the composite's dextran
  (channel 0) and claudin-5 (channel 1). The rings' blur and wall were fitted on this graph
  (`ring_calibration`, 8 workers): sigma 2.71 um along z and 1.08 um across, wall 1.13 um.
- **Fresh:** interior, with a heading, no tip within 3 um of one G1-G5, G1'-G3', the short dead
  ends or the tips-in-lumen grids showed, nor within 8 um of a facing pair M or M2 showed: 108 of
  394.
- **Spread:** thirds of E3's run, and within each the three levels of E1's fitted-ring share
  (no ring at the five tip sections; some; half or more), 4/3/3, evenly across the run. Not E1's
  vesselness share: on E14.5 it reads "walled" at three in four interior tips. The 30: runs 0-30
  um, 16 `unknown`, 10 `stops_short`, 4 `artefact`.
- **Page:** G4's two planes plus "Past the tip (its own plane)", framed from 6 um behind the tip to
  the end of the line E3 reads, drawn dashed in every channel. Each card names its tip's voxel
  position. Two questions: V / N / P / ? and Goes on / Ends here / Unsure, with "what decided it".
- **What will be fitted** once marked: Q1 against E1 (ring share, own-ring share, vesselness
  share) and E2 (lumen, shared, on-wall shares); Q2 against E3 (run, stop, gap), each reported
  right / wrong / all one way, with the IDs it gets wrong.

### Phase D — Use what DR1 confirms (each behind a setting, off until a fresh grid DR2 agrees ≥ 90%)

Nothing here is built: each item waits on DR1's marks saying a measure separates them, then on
DR2. Each would be a schema setting, off by default, behind an Advanced button.

- **Q1 ⇒ prune.** A chain with no ring of its own, or one inside another vessel's walls, joins
  the artefacts the stub prune removes (`prune.prune_vascular_stubs`, agreed with the session
  that owns it).
- **Q2 ⇒ trace on.** `trace_on_dead_ends` follows the wall-enclosed lumen (E4) where it
  continues, instead of the mask. It can then reach a vessel the dextran never filled. Joins
  across a gap (`join_dead_ends_to_vessel_sides`, decision D3) require the walls to continue
  across it.
- **Q3 ⇒ open ends.** `leaves_image` accepts E3 reaching the face as well as the mask, so the
  open-end boundary conditions (`open_end_model`) cover vessels whose plasma signal faded before
  the face.
- **Review.** The dead-end list ranks by the evidence, and Trace on from the tip follows E4.

### Phase E — Docs, release review, a second dataset

- CLAUDE.md's dead-end entry; `RELEASE_REVIEW.md` rows `graph.6b` (the fit saved) and
  `graph.8b` (the fresh grid).
- Repeat Phase B on a second dataset with a plasma raw image only (the Nerve fixture has none),
  to see what E1–E3 are worth without walls.

**State (2026-10-25):** CLAUDE.md's dead-end entry names the primitives and DR1;
`docs/notes/dead_ends_and_open_ends.md` has the E14.5 measurements; `RELEASE_REVIEW.md` `graph.6b`
names DR1 (unmarked, still failing; no score change). `graph.8b` waits on DR2. The plasma-only
repeat waits on a dataset with a plasma raw image and no endothelial one.

## 5. Decisions for Harvey

| # | Question | Recommendation |
|---|---|---|
| R1 | Read G1–G4 again with E1–E3 (Phase B) before asking for the grid? | Yes: it costs no marking, and decides whether the grid is needed. **2026-10-25:** DR1 was built and sent first, at the coordinating session's request, because Phase B's files were in other sessions' hands. Marking it can wait for R1's re-fit if Harvey prefers |
| R2 | The grid's second question ("does it go on?") | Yes: it is what an endothelial stain can answer that the mask cannot. **DR1 asks it**; Harvey may leave it blank |
| R3 | Should an unfilled sprout (walls, no plasma) become a vessel in the graph, with no flow (`blind_end`), or stay a note in the review? | Review first, since it adds vessels the segmentation never had; **open, for Harvey after DR1's marks** |
| R4 | A dataset with both a plasma and an endothelial channel: one raw channel (today), or a second for evidence? | One. The segmentation already carries the plasma; the raw channel should be the other stain |
| R5 | Which channel is E14.5's raw image? D5 (2026-10-12) said one channel only, the one the segmentation came from (dextran). On 2026-10-13 Harvey said the endothelial channel (claudin-5) "would be better" for this dataset. | **Decided: claudin-5** (Harvey, 2026-10-13, D5 overturned; relayed by the raw-confirmation session, which records it in `PLAN_raw_confirmation.md`). Dextran read along the chain had done no better than chance on G4 (15/27), and on that session's grids H and H2 (17/30 each, `outputs/doubled_run_review/FIT_raw.md`) |

## 6. Risks

- **Blur along z** (σ ≈ 2.9 µm on claudin-5) fills a capillary's ring in. Below ~3 µm lumen
  radius, E1 will read "no ring" for a real vessel, so E1 must be read with the vessel's size,
  and the grid spread across it.
- **A neighbour's wall** can sit within a capillary's ring radius. E1's ring centre test (own
  chain or not) is what keeps a spur from borrowing it.
- **Cost:** the ring fit was guessed at ~0.1 s a section in one process. **Measured on
  2026-10-25** (`outputs/dead_ends_raw/measure_e14.py` and `timing.json`, worktree
  `plan-dead-ends-raw-04d14a`): E14.5's 394 dead ends, claudin-5, one process, the rings' blur
  and wall fixed. It costs:
  - the ring fit: 0.032 s a section (1,947 sections, 63 s);
  - so 1,082 dead ends × 5 sections is about 3 min in one process, about 30-45 s on 8 workers
    (estimated, not run);
  - `ring_along`: 0.17 s a dead end, the vesselness share 6 ms of it;
  - `walls_ahead`: 7 ms a dead end;
  - `inside_walls`: 2 ms a point (5 tip points: 10 ms; whole chains every 2 um: 69 s for all 394);
  - the block calibration: 0.9 s;
  - `ring_calibration`: 86 s on 8 workers, sigma 2.71/1.08 um, wall 1.13 um, against 2.94/1.16 and
    1.19 on the 2026-10-04 graph;
  - reads: 1.3 G voxels in 36,617 windows, never the whole volume; the channel held in RAM is
    150 MB (uint16).
- **What the measures read on E14.5** (the same run, 209 interior dead ends):
  - The block calibration's wall level (197 counts) lies under the walls' peak (~470): its band
    straddles the edge of a mask that holds the wall. A reading counts as wall above ~108 counts,
    still between the lumen (~38) and the wall.
  - E1's vesselness share reads walled at three in four interior tips (25th percentile 1.0), so
    it barely spreads.
  - E1's fitted rings spread more. Of 1,036 interior tip sections, 222 fit a ring, 405 have no
    dark lumen, and 228 are low contrast. A capillary's lumen fills in under the z blur.
  - E2 puts a centreline on the wall (median wall share 1.0) and in a lumen shared with another
    centreline at none. On this stack it is unlikely to separate anything; DR1 will say.
  - E3 runs a median 6 um (IQR 2-11): `stops_short` 7, `unknown` 5, `artefact` 5. It reaches the
    face for 53 of 91 `leaves_image` ends and for 25 interior ones. 24 of those 25 tips lie within
    25 um of the top or bottom z face (`outputs/dead_ends_raw/face_check.py`).
- **G4's lesson:** the marks follow shape in context. The stain may still not separate them; if
  so, this plan's value is Q2–Q4 (continuation and routes), not Q1.
