# Photometric, full query sets (end-to-end)

Source: `metric_runs/item5_*` (2026-10-21). The photometric (LPIPS) refiner run end-to-end from
the real MegaLoc seed, on the **full** query sets (class1 = 104, stacey = 159). Real cm.

Result — the photometric path collapses from where MegaLoc actually lands:
- Room1/class1 (104): 5cm/5deg = 6%, 10cm/5deg = 9%, 50cm/10deg = 44%. Hit the 500-iter cap 21/104.
- Office/stacey (159): 5cm/5deg = 1%, 10cm/5deg = 1%, 50cm/10deg = 11%. Cap 4/159.

This is the "photometric is basin-limited, not a viable deployed stage" evidence: it only works
from a tight seed (folder 07), not from a real retrieval seed. ~30-60 s/query.

Status: authoritative. (This is the run that uses all 104/159 queries.)
