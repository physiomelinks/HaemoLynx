# Issue tracker: Local Markdown

Issues and specs for this repo live as markdown files in `.scratch/`. `.scratch/` is gitignored; nothing in it is committed.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues are one file per ticket at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`, never a single combined tickets file
- Triage state is recorded as a `Status:` line near the top of each issue file (see `triage-labels.md` for the role strings)
- Comments and conversation history append to the bottom of the file under a `## Comments` heading

## Closing a ticket

After the commit that implements a ticket, set its `Status:` line to `done (<hash>)`, using every hash if it took more than one commit. When the last ticket of a feature closes, set `Status: done` in that feature's `spec.md` too. These are the closing states for implementation tickets; the roles in `triage-labels.md` cover only the states before work starts. Also tick the ticket's acceptance boxes, following these rules, then run `venv/bin/python .scratch/test-suite-cleanup/build_progress.py` (for that feature; it warns on any `done` ticket with unticked boxes).

- Tick each box on its own, and only when a line in the ticket's `## Answer` meets it. Never tick them all with one bulk replace of `- [ ]` by `- [x]`.
- A box that waits on another ticket stays `[ ]`, with "(checked in NN)" after it.
- Every numbered finding from `/code-review` gets one line under `## Answer`, as one of: `fixed`; `declined: <reason>`; or `not a finding: <what was checked>`. None is left without an outcome.
- A follow-up found while working a ticket (something out of its scope) goes into an open ticket's acceptance criteria, or into a new ticket. A note in `## Comments` alone is not enough: it strands the follow-up in a ticket that is about to close.
- Before setting a spec to `Status: done`, check that every ticket is done; `build_progress.py` warns when a spec says `done` and one of its own tickets is not (`spec.md` owns `issues/`, `retro-spec.md` owns `retro/`).
- `ls` a path before giving it to the user, or before writing it into a ticket after a compaction. Use the full path from the repo root (`.scratch/<feature>/assets/gaps.md`), never one relative to another file.
- Write ticket text with the Edit tool or a quoted heredoc (`<<'EOF'`), never an unquoted one, which eats the backtick spans.

## Handing a ticket to an agent in a worktree

- Before dispatching, check out the integration branch in the main checkout, so new worktrees start from it (`worktree.baseRef` is `head`). The implementer still checks its base, as `implement-spec` step 4 says. A merger runs `git merge --ff-only <branch>` in the main checkout.
- When a ticket is handed to an agent, set its `Status: in progress (<branch>)`.
- A brief asks only for git commands that run inside the worktree, one plain command per call (no `git -C <main checkout>`). A worktree shares `.git/config`, so `git config core.bare` run there checks the main checkout's config too.
- The implementer writes its notes and new-ticket drafts into its own worktree's `.scratch/<feature>/` (gitignored, so they survive a `/clear` of the orchestrator's session), never into a session scratchpad.
- Whoever lands the branch copies those files into the main checkout's `.scratch/<feature>/` before `git worktree remove`, which deletes ignored files without a word. `git status --short --ignored .scratch` run in the worktree lists them.

## Checking a run for repo writes

`git status` cannot see writes into gitignored folders (`tests/outputs/`, `tests/plots/`, `examples/outputs/`). To check that a run writes nothing into the repo, `touch` a marker file outside the repo, run, then list what is newer:

```bash
touch "$SCRATCH/marker"
# ... the run ...
find . -newer "$SCRATCH/marker" -not -path './.git/*' -not -path './venv/*' -type f
```

It must print nothing. This is the method `test-suite-pruning/assets/inventory.md` used.

## Hook stamp after a full coverage pass

After `scripts/coverage_by_module.py run` passes over the whole suite on a clean tree, run `scripts/stamp-tests.sh` before committing, so the pre-commit hook does not re-run the fast tests (about 54 s). `coverage_by_module.py` does not write the stamp itself.

## Mutation evidence

A ticket that shows its tests fail against a broken version of the behaviour keeps that evidence in the repo, run with `scripts/mutation_check.py`.

- **Always save the spec.** It lives at `.scratch/<feature>/issues/NN-mutations.toml` (`NN` is the ticket number), is written before the runs, and is never a scratchpad file that is thrown away. The ticket's `## Answer` names it in the `Run:` line.
- **Keep a break that survived.** If a break passes within the test's tolerance, it stays in the spec with `expect = "survived"` and a comment saying why it is too small to be seen. The bigger break that is caught goes next to it as its own `[[mutation]]`. Never replace the surviving break with the bigger one.

## Writing a ticket from verdict rows

Applies when an area ticket is made from the rows of a verdict file (for example `test-suite-pruning/assets/verdicts/<area>.md`).

- **Name the module-level cleanups and the rows command.** Every area ticket lists the module-level cleanups for its files (the `sys.path` lines, unused imports, shared constants), and the command that prints its rows: `venv/bin/python .scratch/test-suite-pruning/assets/verdicts/rows.py AREA`. The implementer then reads only those rows, not the whole verdict file.
- **Settle every open choice in the ticket.** A row that says "rename to say that", "either ... or" or "consider" gets a decision written into the ticket before `/implement` starts, so the implementer has nothing to ask.
- **Correct a row error where you find it.** If applying a row shows it is wrong (for example it says an import is unused when the test calls it), fix the row in the verdict file in the same session, so the next area does not check it again.

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/` (creating the directory if needed).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the issue number directly.

A bare number (`/implement 13`) means the ticket whose `Status:` is not `done (<hash>)` or `resolved`; ask only if more than one match is still open.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a file with one **child** file per ticket.

- **Map**: `.scratch/<effort>/map.md` (the Notes / Decisions-so-far / Fog body).
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records the ticket type (`research`/`prototype`/`grilling`/`task`); a `Status:` line records `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when every file it lists is `resolved`.
- **Frontier**: `venv/bin/python scripts/wayfinder_progress.py .scratch/<effort> --frontier` lists the open, unblocked, unclaimed tickets; first by number wins.
- **Claim**: set `Status: claimed` and save before any work.
- **Resolve**: append the answer under an `## Answer` heading, set `Status: resolved`, then append a context pointer (gist + link) to the map's Decisions-so-far in `map.md`.
- **Session log**: at the end of each session, append a `## <mode>` entry to `.scratch/<effort>/sessions.md` (lines `Date:`, `Mode:`, `Call:`, `Ticket:`, plus `Model:`, `Effort:`, `Why:` recommending how to run the next one), then run `venv/bin/python scripts/wayfinder_progress.py .scratch/<effort>`, which rebuilds `.scratch/<effort>/progress.html`, the page the user opens to follow the map.
