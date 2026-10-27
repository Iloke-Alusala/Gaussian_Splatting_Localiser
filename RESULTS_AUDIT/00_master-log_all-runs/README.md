# Master run log (all runs)

`RESULTS_LOG.md` — the append-only master. Every `query_eval.py`, `run_hloc.py` and
timing run auto-appends a block here, oldest first.

- **Units: scene units** (the "cm" labels assume 1 unit = 1 m, which is NOT the real scale —
  use folder 01 for real cm).
- Structure: Part A = backfilled old runs, B = frame-mismatch bug evidence, C = chat-only
  failed runs (marked "do not cite"), D = known discrepancies, E = auto-appended new runs.

How to read a block: a date/scene header, then median translation/rotation, median time,
and recall at the four bars (5cm/5deg, 10cm/5deg, 25cm/2deg, 50cm/10deg).

Status: authoritative for *what was run and when*; for real-cm numbers use folder 01/02.
