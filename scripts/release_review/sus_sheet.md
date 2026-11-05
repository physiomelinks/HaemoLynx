# System Usability Scale (`usability.8b`, `usability.10a`)

TEMPORARY: part of the release review's human kit (`PLAN_release_review.md` step R5).

The ten standard items of the System Usability Scale (Brooke, 1996), unaltered. Hand this sheet to
the tester straight after the task list (`usability_tasks.md`), before anyone discusses the panel
with them. One answer per item; if unsure, the middle (3).

Tester: ____________________  Date: ____________

| # | Statement | Strongly disagree 1 | 2 | 3 | 4 | Strongly agree 5 |
|---|---|---|---|---|---|---|
| 1 | I think that I would like to use this system frequently. | | | | | |
| 2 | I found the system unnecessarily complex. | | | | | |
| 3 | I thought the system was easy to use. | | | | | |
| 4 | I think that I would need the support of a technical person to be able to use this system. | | | | | |
| 5 | I found the various functions in this system were well integrated. | | | | | |
| 6 | I thought there was too much inconsistency in this system. | | | | | |
| 7 | I would imagine that most people would learn to use this system very quickly. | | | | | |
| 8 | I found the system very cumbersome to use. | | | | | |
| 9 | I felt very confident using the system. | | | | | |
| 10 | I needed to learn a lot of things before I could get going with this system. | | | | | |

**Scoring** (for the observer, not the tester): each odd item scores its answer minus 1, each even
item 5 minus its answer; the sum times 2.5 is the score, 0 to 100.
`release_review.human_kit.sus_score([answers])` does it. Record the score in
`human_checks.yaml` as the mark's `value`, with the sheet's whereabouts as its evidence.
`usability.8b` passes at 68 or more over George, Rebecca and Mike; `usability.10a` at 80 or more
over five testers.
