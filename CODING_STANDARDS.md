# Coding standards

Rules for code review that no test can fully check. A reviewer checks every change against
each rule below, as well as against `CLAUDE.md`'s "Rules for this repo".

## 1. A value that can't be parsed raises

A blank, text, NaN or infinite cell where a number belongs raises, naming where it came from
(specimen, file, column, keys). It is never turned into NaN, 0, a default or a dropped row.
A blank is not zero and it is not NaN.

- Read edge-table numbers through `BatchRun.numeric_column`, not `float(row[...])`.
- Not allowed: `float(x or "nan")`, `except ValueError: return np.nan`, `int(x or 0)`,
  `if row.get(column)` used to skip blank cells.
- A driver may still apply its own analysis filter to valid numbers (for example, dropping
  lengths of zero or less), but it says so in a comment.

## 2. A path, file name or constant is defined once

Each one has a single home, and everything else imports it from there:
`specimens.py` for data roots and run folders, `batch_outputs.py` for batch-run file names,
`cb_settings.py` for frozen analysis values. A second copy, even an equal one, is a finding.

## 3. A review finding is fixed in the same change, or declined in writing

A finding is not "acceptable" or "for later" without a reason. Either fix it in the change
under review, or decline it with a written reason in the review. If it really belongs to
other work, add it as a row in the current `pipeline_rerun_*_notes.md` package table, and
say so in the review.
