# Results log — 3DGS localiser vs HLoc

Append-only. Every run of `query_eval.py`, `hloc_baseline/run_hloc.py` and
`hloc_baseline/time_hloc_online.py` appends its summary here automatically.
Units: ours reports nerfstudio scene units ("units"); HLoc's "ns" column is the
same unit (COLMAP x dataparser scale). "cm" assumes 1 unit = 1 m.

# Part A — backfill from files on disk (copied verbatim by `cat`)
Header lines are context; the fenced block is the file's exact contents.

## [backfill] 2026-10-09 17:14 | hloc | classroom_1-f1250 sparse/0 | queries: 112 nerfstudio held-out eval-split frames (make_query_set.py) | old run_hloc
source file: 3dgs/hloc_out/classroom_1-f1250/hloc/summary.txt
```
HLoc baseline - classroom_1-f1250
queries: 112   localised: 112 (100.0%)
median t_err: 0.0051 colmap  (0.0007 ns)   median r_err: 0.063 deg
mean   t_err: 0.0113 colmap  (0.0015 ns)   mean   r_err: 0.146 deg
recall @ 0.05 ns / 2.0 deg: 100.0%
recall @ 0.1 ns / 5.0 deg: 100.0%
recall @ 0.25 ns / 10.0 deg: 100.0%
```

## [backfill] 2026-10-10 16:10 | hloc | stacey_room-f2500 sparse/0 | queries: 53 stills query_d1280, EXIF-upright (query_d1280_up), eval --upright
source file: 3dgs/hloc_out/stacey_room-f2500/hloc/summary.txt
```
HLoc baseline - stacey_room-f2500
queries: 53   localised: 53 (100.0%)
median t_err: 0.0115 colmap  (0.0024 ns)   median r_err: 0.176 deg
mean   t_err: 0.0155 colmap  (0.0032 ns)   mean   r_err: 0.204 deg
recall @ 0.05 ns / 2.0 deg: 100.0%
recall @ 0.1 ns / 5.0 deg: 100.0%
recall @ 0.25 ns / 10.0 deg: 100.0%
```

## [backfill] 2026-10-12 08:39 | ours | stacey_room-f2500 splat | queries: 53 query_d1280 | 1 pass (before --passes flag existed)
source file: Desktop/3DGS/code/query_eval_out/summary_megaloc.txt
```
megaloc on 53 queries (query_d1280)
translation: median 0.0462 units (4.62 cm)  mean 0.0830
rotation:    median 1.548 deg  mean 4.507
time:        median 285 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=51%  10cm_5deg=75%  25cm_2deg=60%  50cm_10deg=91%
gpu peak:    2.49 GB alloc  3.27 GB reserved (this process)
```

## [backfill] 2026-10-12 08:40 | hloc FULL online timing | stacey_room-f2500 | 53 upright queries | re-run inside bench_resources.sh
source file: 3dgs/hloc_out/stacey_room-f2500/hloc/timing_online.txt
```
HLoc full online timing, 53 queries, resize_max=1024, num_loc=20
  SuperPoint extract            50.3 ms/query   (2.67s / 53)
  NetVLAD extract              177.1 ms/query   (9.39s / 53)
  retrieval (NN search)         19.5 ms/query   (1.04s / 53)
  LightGlue match (20 db)      562.2 ms/query   (29.80s / 53)
  localize_sfm (PnP)            70.0 ms/query   (3.71s / 53)
  TOTAL online                 879.2 ms/query   (46.60s / 53)
  gpu peak                   1.44 GB alloc  2.03 GB reserved (this process)
```

## [backfill] 2026-10-12 08:40 | peak resources, both methods | stacey_room-f2500 | 53 queries
source file: 3dgs/hloc_out/stacey_room-f2500/resources.txt
```
=== resource benchmark Mon 12 Oct 2026 08:38:13 AM SAST ===
[ours: MegaLoc + render + LoFTR + PnP]  (nerfstudio env)
  torch gpu peak:    2.49 GB alloc  3.27 GB reserved (this process)
  system GPU peak : 3274 MiB (nvidia-smi, incl. CUDA ctx)
  RAM peak (RSS)  : 2.39 GB

[hloc: full online (extract+retrieve+match+PnP)]  (hloc env)
  torch gpu peak                   1.44 GB alloc  2.03 GB reserved (this process)
  system GPU peak : 2056 MiB (nvidia-smi, incl. CUDA ctx)
  RAM peak (RSS)  : 2.36 GB

```

## [backfill] 2026-10-12 11:29 | ours | stacey_room-f2500 splat | 53 query_d1280 | iterative ablation, 1 pass
source file: Desktop/3DGS/code/query_eval_out/summary_megaloc_p1.txt
```
megaloc on 53 queries (query_d1280), 1 fine pass(es)
translation: median 0.0462 units (4.62 cm)  mean 0.0830
rotation:    median 1.548 deg  mean 4.507
time:        median 285 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=51%  10cm_5deg=75%  25cm_2deg=60%  50cm_10deg=91%
gpu peak:    2.49 GB alloc  3.27 GB reserved (this process)
```

## [backfill] 2026-10-12 11:30 | ours | stacey_room-f2500 splat | 53 query_d1280 | iterative ablation, 2 passes
source file: Desktop/3DGS/code/query_eval_out/summary_megaloc_p2.txt
```
megaloc on 53 queries (query_d1280), 2 fine pass(es)
translation: median 0.0091 units (0.91 cm)  mean 0.0423
rotation:    median 0.355 deg  mean 2.561
time:        median 447 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=92%  10cm_5deg=94%  25cm_2deg=92%  50cm_10deg=96%
gpu peak:    2.56 GB alloc  3.27 GB reserved (this process)
```

## [backfill] 2026-10-12 11:31 | ours | stacey_room-f2500 splat | 53 query_d1280 | iterative ablation, 3 passes
source file: Desktop/3DGS/code/query_eval_out/summary_megaloc_p3.txt
```
megaloc on 53 queries (query_d1280), 3 fine pass(es)
translation: median 0.0087 units (0.87 cm)  mean 0.0374
rotation:    median 0.245 deg  mean 2.261
time:        median 608 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=94%  10cm_5deg=96%  25cm_2deg=94%  50cm_10deg=96%
gpu peak:    2.59 GB alloc  3.27 GB reserved (this process)
```

## [backfill] 2026-10-13 12:09 | ours | class1_train-f2900 splat | eval/class1 query_gt (104) | 2 passes
source file: Desktop/3DGS/code/query_eval_out/class1/summary_megaloc_p2.txt
```
megaloc on 104 queries (images), 2 fine pass(es)
translation: median 0.0101 units (1.01 cm)  mean 0.0107
rotation:    median 0.237 deg  mean 0.290
time:        median 420 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=100%  10cm_5deg=100%  25cm_2deg=99%  50cm_10deg=100%
gpu peak:    2.06 GB alloc  2.65 GB reserved (this process)
```

## [backfill] 2026-10-13 12:12 | ours | class1_train-f2900 splat | eval/class1 query_gt (104) | 1 pass
source file: Desktop/3DGS/code/query_eval_out/class1/summary_megaloc_p1.txt
```
megaloc on 104 queries (images), 1 fine pass(es)
translation: median 0.0106 units (1.06 cm)  mean 0.0148
rotation:    median 0.482 deg  mean 0.781
time:        median 311 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=98%  10cm_5deg=99%  25cm_2deg=95%  50cm_10deg=99%
gpu peak:    2.04 GB alloc  2.59 GB reserved (this process)
```

## [backfill] 2026-10-13 13:10 | hloc | class1_train-f2900 sparse/0 | eval/class1 query_gt (104) | NOTE time line = localize_sfm only, NOT full online
source file: 3dgs/hloc_out/class1/hloc/summary.txt
```
HLoc on 104 queries   localised 104/104
translation: median 0.0009 ns (0.09 cm)  mean 0.0013 ns
rotation:    median 0.085 deg  mean 0.123
time:        36.2 ms/query (amortized localize_sfm)
recall: 5cm_5deg=100%  10cm_5deg=100%  25cm_2deg=100%  50cm_10deg=100%
```

## [backfill] 2026-10-14 08:42 | ours | stacey_room_train-f4200 splat | eval/stacey query_gt (159) | 1 pass
source file: Desktop/3DGS/code/query_eval_out/stacey/summary_megaloc_p1.txt
```
megaloc on 159 queries (images), 1 fine pass(es)
translation: median 0.0244 units (2.44 cm)  mean 0.0648
rotation:    median 1.219 deg  mean 5.514
time:        median 357 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=67%  10cm_5deg=79%  25cm_2deg=67%  50cm_10deg=89%
gpu peak:    2.92 GB alloc  3.19 GB reserved (this process)
```

# Part B — class1 frame-mismatch evidence (re-run 2026-10-14 08:51, output appended verbatim)
Sim3 fit of cluster model eval/class1/colmap_with_query -> each local sparse/0, on shared frame_* camera centres.
```
classroom_1-f1250 (splat used in the FAILED runs): cluster_imgs=2826 local_imgs=1126 shared=1055  scale=0.1802 rot=176.687deg  res_rms=3.0461 res_max=7.3965 (58.76% of scene)
class1_train-f2900 (splat used in the WORKING runs): cluster_imgs=2826 local_imgs=2722 shared=2722  scale=1.0004 rot=0.026deg  res_rms=0.0011 res_max=0.0733 (0.44% of scene)
```

# Part C — runs with NO surviving file (overwritten later)
PROVENANCE: transcribed verbatim from this conversation (user-pasted terminal
output, or tool output in the session). No file on disk backs these. Kept only
as history of the failures; do not cite as primary results.

## [chat] 2026-10-10 | ours | stacey_room-f2500 | 53 query_d1280 | 1 pass | BEFORE EXIF fix (query loaded sideways)
```
megaloc on 53 queries (query_d1280)
translation: median 0.5653 units (56.53 cm)  mean 0.5756
rotation:    median 89.044 deg  mean 86.828
recall: 5cm_5deg=0%  10cm_5deg=0%  25cm_2deg=0%  50cm_10deg=0%
```

## [chat] 2026-10-10 | hloc | stacey_room-f2500 | 53 query_d1280 RAW landscape (no EXIF-upright) | superseded by the upright run in Part A
```
HLoc baseline - stacey_room-f2500
queries: 53   localised: 53 (100.0%)
median t_err: 2.5004 colmap  (0.5154 ns)   median r_err: 74.961 deg
mean   t_err: 4.5833 colmap  (0.9448 ns)   mean   r_err: 66.955 deg
recall @ 0.05 ns / 2.0 deg: 28.3%
recall @ 0.1 ns / 5.0 deg: 28.3%
recall @ 0.25 ns / 10.0 deg: 28.3%
```

## [chat, user-pasted] 2026-10-13 | ours | classroom_1-f1250 splat (WRONG reconstruction, see Part B) | old eval/class1 GT (144) | 2 passes
```
megaloc on 144 queries (images), 2 fine pass(es)
translation: median 0.3839 units (38.39 cm)  mean 0.3974
rotation:    median 27.105 deg  mean 31.021
time:        median 441 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=0%  10cm_5deg=0%  25cm_2deg=0%  50cm_10deg=0%
gpu peak:    2.10 GB alloc  2.79 GB reserved (this process)
```

## [chat, user-pasted] 2026-10-13 | ours | classroom_1-f1250 splat (WRONG reconstruction) | new eval/class1 GT (104) | 2 passes
```
megaloc on 104 queries (images), 2 fine pass(es)
translation: median 0.3732 units (37.32 cm)  mean 0.3905
rotation:    median 27.061 deg  mean 30.724
time:        median 548 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=0%  10cm_5deg=0%  25cm_2deg=0%  50cm_10deg=1%
gpu peak:    2.10 GB alloc  2.76 GB reserved (this process)
```

# Part D — known discrepancies with other docs
- EVALUATION.md quotes ours = 282 ms/query and HLoc full online = 878 ms/query
  (stacey f2500). Those came from earlier runs whose files were overwritten.
  The surviving files (Part A) say 285 ms and 879.2 ms. Treat the files as
  authoritative; the difference is run-to-run noise.

# Part E — new runs (appended automatically by the scripts from here on)
