# Perturbation basin — 2-pass EfficientLoFTR (thesis F.2 column)

Source: `metric_runs/item3_*` (2026-10-21). Take real queries, offset the seed pose by a known
metric amount in a random direction, run the 2-pass LoFTR+PnP fine stage, see if it recovers.
n=100 per offset. One file per offset; the filename `seed<units>-<deg>` encodes it, e.g.
`stacey_..._seed0.050439-5` = 10 cm / 5 deg (0.050439 u x 1.9826 = 0.10 m).

Offset schedule: 2cm/1, 5/2, 10/5, 20/5, 40/8, 60/12, 80/16, 100/20, 150/30 (cm/deg).
Each file reports recall at 10cm/5deg etc. and median t.

r50 (offset where recall@10cm/5deg hits 50%): **Room1/class1 ~84 cm, Office/stacey ~52 cm**.
This is the "2-pass EfficientLoFTR" column of thesis Table F.2.

Status: complete, both scenes.
