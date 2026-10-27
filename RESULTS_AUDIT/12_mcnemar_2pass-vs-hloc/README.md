# McNemar: ours 2-pass vs HLoc (Office)

Paired per-query comparison on Office/stacey, ours 2-pass EfficientLoFTR vs HLoc (num_loc=20),
metric thresholds, n=159.

- `stacey_2pass_summary.txt` — the ours 2-pass run (aggregate) this was computed from.
- `mcnemar_and_failures.txt` — the computed result: paired success counts (b, c), McNemar exact
  p-values at 5cm/5deg and 10cm/5deg, and all 20 failures at 10cm/5deg classified into
  no-consensus / wrong-but-consistent / partial-recovery.

Headline: ours significantly beats HLoc at 10cm/5deg (87.4% vs 72.3%, p=0.0004); ahead but not
significant at the tight 5cm/5deg (79.9% vs 71.1%, p=0.08).

Status: computed 2026-10-27 (not an auto-logged run).
