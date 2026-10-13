# Plan: the raw image as evidence for dead ends

*2026-10-13. Branch `claude/raw-image-input`, which makes one raw image the run's own: its path,
its channel and its stain (`raw_image_stain`: auto, plasma column or endothelial) on the Input
tab. This plan says how dead-end handling should use that image. It is decision D6 of
`PLAN_dead_ends.md` ("the raw image and the endothelial channel as evidence for broken against
blind"), now that there is one image to read.*

It builds on the "raw confirmation" session's `PLAN_raw_confirmation.md`. That plan's
`preprocessing/raw_evidence.py` (`RawEvidence`: one reader, sampled, windowed, calibrated once
per run) is the reader every rule here asks, rather than a second one. Its v1 reads only a
plasma (luminal) stain, so this plan adds the wall-stain primitives dead ends need on E14.5,
whose raw image is claudin-5.

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
5. **Sequence.** `claude/raw-evidence` lands first, then Phase A on a branch off it. The
   raw-confirmation session leaves `raw_evidence.py` alone meanwhile. Landed on 2026-10-13:
   `Devel_GUI_improvements_HD` at 96b0329 carries `preprocessing/raw_evidence.py`, its
   `build_network` hook and `tests/test_raw_evidence.py`.

This branch supplies the settings, `read_raw_stain`/`resolve_stain` and `enclosed_by_walls`. A
`keep_dtype` option on the raw loader is not needed for now: the in-RAM path reads a channel in
its own dtype, and float32 on the memmap path is acceptable until a large stack says otherwise.

### Phase B — Measure on E14.5 and record (no behaviour change)

- `classify_dead_ends(..., raw=RawEvidence | None)` fills E1–E3 on every `DeadEnd`. The kinds
  stay as they are. The census (`scripts/graph_artefacts/dead_end_census.py`) and `dead_ends.csv`
  carry the measures, and the review list shows them.
- Re-read G1–G4 with them: the 116 marked dead ends' E1–E3 against Harvey's marks, fitted by
  `fit_marks.py`. This costs no marking and says whether a fifth grid is worth asking for.

### Phase C — Grid G5, spread by the new evidence

- 30 fresh interior dead ends, none shown before, spread across E1 (ring support) and E3
  (continuation), then shuffled.
- The same page as G4: segmentation, dextran, claudin-5, in two planes. It adds a third plane
  along the heading past the tip, with the E3 path drawn, because Q2 is about what lies beyond
  the tip.
- One question with G4's marks (V vessel, N not a vessel, P partly), plus a second: "Does it go
  on past the tip?" (yes / no / unsure). What decided it is recorded as before.
- Fit, and report right / wrong / all one way (CLAUDE.md). If the marks again cite shape and
  context only, say so, and keep looking for the shape measure instead of the stain.

### Phase D — Use what G5 confirms (each behind a setting, off until a fresh grid G6 agrees ≥ 90%)

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

## 5. Decisions for Harvey

| # | Question | Recommendation |
|---|---|---|
| R1 | Read G1–G4 again with E1–E3 (Phase B) before asking for G5? | Yes: it costs no marking, and decides whether G5 is needed |
| R2 | G5's second question ("does it go on?") | Yes: it is what an endothelial stain can answer that the mask cannot |
| R3 | Should an unfilled sprout (walls, no plasma) become a vessel in the graph, with no flow (`blind_end`), or stay a note in the review? | Review first, since it adds vessels the segmentation never had; decide after G5 |
| R4 | A dataset with both a plasma and an endothelial channel: one raw channel (today), or a second for evidence? | One. The segmentation already carries the plasma; the raw channel should be the other stain |
| R5 | Which channel is E14.5's raw image? D5 (2026-10-12) said one channel only, the one the segmentation came from (dextran). On 2026-10-13 Harvey said the endothelial channel (claudin-5) "would be better" for this dataset. | **Decided: claudin-5** (Harvey, 2026-10-13, D5 overturned; relayed by the raw-confirmation session, which records it in `PLAN_raw_confirmation.md`). Dextran read along the chain had done no better than chance on G4 (15/27), and on that session's grids H and H2 (17/30 each, `outputs/doubled_run_review/FIT_raw.md`) |

## 6. Risks

- **Blur along z** (σ ≈ 2.9 µm on claudin-5) fills a capillary's ring in. Below ~3 µm lumen
  radius, E1 will read "no ring" for a real vessel, so E1 must be read with the vessel's size,
  and the grid spread across it.
- **A neighbour's wall** can sit within a capillary's ring radius. E1's ring centre test (own
  chain or not) is what keeps a spur from borrowing it.
- **Cost:** the ring fit is ~0.1 s a section in one process. 1,082 dead ends × 5 sections is
  ~10 min in one process, or ~1.5 min with the 8 workers `sections.run_edge_tasks` already
  provides. Phase B measures it.
- **G4's lesson:** the marks follow shape in context. The stain may still not separate them; if
  so, this plan's value is Q2–Q4 (continuation and routes), not Q1.
