# Perturbation basin — photometric LPIPS (thesis F.2 column)

Source: `metric_runs/item4_*` (2026-10-21). Same setup as folder 06 but the fine stage is the
photometric (LPIPS) refiner. n=40 per offset. Filename `seed<units>-<deg>` encodes the offset.

r50 (offset at 50% recall@10cm/5deg): **Room1/class1 = 18.9 cm** (the thesis F.2 number;
reproduces exactly). The photometric basin is far tighter than 2-pass LoFTR (folder 06).

PARTIAL — this is the one incomplete column:
- Room1/class1 ran offsets 2, 5, 10, 20, 60 cm (r50 still well-bracketed between 10 and 20 cm).
- Office/stacey ran only 2 and 10 cm, so it has **no r50** (never brackets 50%).
If you need the full stacey photometric curve it has to be rerun (~2 h, photometric is slow).

Status: PARTIAL.
