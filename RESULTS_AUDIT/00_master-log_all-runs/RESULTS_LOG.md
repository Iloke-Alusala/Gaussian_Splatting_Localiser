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

## 2026-10-14 09:00 | ours (query_eval) | scene=stacey | passes=2
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 159 queries (images), 2 fine pass(es)
translation: median 0.0118 units (1.18 cm)  mean 0.0475
rotation:    median 0.465 deg  mean 4.597
time:        median 484 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=87%  10cm_5deg=89%  25cm_2deg=87%  50cm_10deg=90%
gpu peak:    2.93 GB alloc  3.13 GB reserved (this process)
```

## 2026-10-15 07:13 | hloc (run_hloc) | scene=stacey
db model: /home/chini/3dgs/data/stacey_room_train-f4200-processed
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
HLoc on 159 queries   localised 159/159
translation: median 0.0060 ns (0.60 cm)  mean 0.1182 ns
rotation:    median 0.320 deg  mean 19.743
time:        30.0 ms/query (amortized localize_sfm)
recall: 5cm_5deg=72%  10cm_5deg=72%  25cm_2deg=71%  50cm_10deg=73%
```

## 2026-10-18 21:24 | ours (query_eval) | scene=class1 | passes=3
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 104 queries (images), 3 fine pass(es)
translation: median 0.0099 units  mean 0.0105 units
rotation:    median 0.211 deg  mean 0.243
time:        median 559 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=100%  10cm_5deg=100%  25cm_2deg=100%  50cm_10deg=100%
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    2.06 GB alloc  2.59 GB reserved (this process)
```

## 2026-10-18 21:29 | ours (query_eval) | scene=stacey | passes=3
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 159 queries (images), 3 fine pass(es)
translation: median 0.0119 units  mean 0.0414 units
rotation:    median 0.355 deg  mean 4.243
time:        median 621 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=91%  10cm_5deg=91%  25cm_2deg=90%  50cm_10deg=92%
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    2.93 GB alloc  3.14 GB reserved (this process)
```

## 2026-10-18 21:32 | hloc online timing | scene=class1
db model: /home/chini/3dgs/data/class1_train-f2900-processed
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
HLoc full online timing, 104 queries, resize_max=1024, num_loc=20
  SuperPoint extract            38.1 ms/query   (3.96s / 104)
  NetVLAD extract              116.3 ms/query   (12.09s / 104)
  retrieval (NN search)         11.0 ms/query   (1.14s / 104)
  LightGlue match (20 db)      294.4 ms/query   (30.62s / 104)
  localize_sfm (PnP)            39.5 ms/query   (4.10s / 104)
  TOTAL online                 499.3 ms/query   (51.92s / 104)
  gpu peak                   1.23 GB alloc  1.68 GB reserved (this process)
```

## 2026-10-18 21:35 | hloc online timing | scene=stacey
db model: /home/chini/3dgs/data/stacey_room_train-f4200-processed
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
HLoc full online timing, 159 queries, resize_max=1024, num_loc=20
  SuperPoint extract            38.2 ms/query   (6.07s / 159)
  NetVLAD extract              104.1 ms/query   (16.55s / 159)
  retrieval (NN search)         11.1 ms/query   (1.77s / 159)
  LightGlue match (20 db)      411.9 ms/query   (65.50s / 159)
  localize_sfm (PnP)            28.6 ms/query   (4.55s / 159)
  TOTAL online                 594.0 ms/query   (94.44s / 159)
  gpu peak                   1.23 GB alloc  1.68 GB reserved (this process)
```

## 2026-10-18 22:27 | ours (query_eval) | scene=stacey | fine=photometric | passes=1
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 40 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 0.1610 units  mean 0.1838 units
rotation:    median 18.139 deg  mean 20.356
time:        median 29415 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=5% [1-17]  10cm_5deg=5% [1-17]  25cm_2deg=0% [-0-9]  50cm_10deg=8% [3-20]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 341  mean 301  min 5  max 500
hit cap without converging: 2/40
gpu peak:    3.25 GB alloc  3.47 GB reserved (this process)
```

## 2026-10-18 23:18 | splat resources (P3) | scenes=stacey,class1
```
splat resources (per-scene load cost; no metric scale on these maps)
[single] stacey (stacey_room-f4200): 2,083,052 gaussians  1.485 GB on disk  load 10.3 s  vram 0.90 GB alloc / 0.91 GB reserved
[single] class1 (class1_train-f2900-processed): 420,882 gaussians  0.308 GB on disk  load 1.8 s  vram 0.70 GB alloc / 0.88 GB reserved
[all rooms] 2 splats together: 2,503,934 gaussians  load 9.3 s  vram 1.40 GB alloc / 1.66 GB reserved
note: errors/counts are in nerfstudio scene units; this card is 4 GB, so the all-rooms peak is the number room-scoped loading has to stay under
```

## 2026-10-18 23:20 | ours (query_eval) | scene=class1 | fine=loftr | passes=1
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 104 queries (images), 1 fine pass(es)
translation: median 0.0106 units  mean 0.0148 units
rotation:    median 0.482 deg  mean 0.781
time:        median 348 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=98% [93-99]  10cm_5deg=99% [95-100]  25cm_2deg=95% [89-98]  50cm_10deg=99% [95-100]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    2.04 GB alloc  2.59 GB reserved (this process)
```

## 2026-10-18 23:22 | hloc online timing | scene=class1
db model: /home/chini/3dgs/data/class1_train-f2900-processed
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
HLoc full online timing, 104 queries, resize_max=1024, num_loc=20
  SuperPoint extract            50.8 ms/query   (5.28s / 104)
  NetVLAD extract              145.3 ms/query   (15.11s / 104)
  retrieval (NN search)         10.9 ms/query   (1.14s / 104)
  LightGlue match (20 db)      432.9 ms/query   (45.03s / 104)
  localize_sfm (PnP)            45.5 ms/query   (4.73s / 104)
  TOTAL online                 685.4 ms/query   (71.29s / 104)
  gpu peak                   1.23 GB alloc  1.68 GB reserved (this process)
```

## 2026-10-18 23:25 | ours (query_eval) | scene=stacey | fine=loftr | passes=1
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 159 queries (images), 1 fine pass(es)
translation: median 0.0244 units  mean 0.0648 units
rotation:    median 1.219 deg  mean 5.514
time:        median 400 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=67% [59-74]  10cm_5deg=79% [72-85]  25cm_2deg=67% [59-74]  50cm_10deg=89% [83-93]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    2.92 GB alloc  3.13 GB reserved (this process)
```

## 2026-10-18 23:28 | hloc online timing | scene=stacey
db model: /home/chini/3dgs/data/stacey_room_train-f4200-processed
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
HLoc full online timing, 159 queries, resize_max=1024, num_loc=20
  SuperPoint extract            51.5 ms/query   (8.19s / 159)
  NetVLAD extract              131.3 ms/query   (20.88s / 159)
  retrieval (NN search)         10.5 ms/query   (1.67s / 159)
  LightGlue match (20 db)      595.8 ms/query   (94.74s / 159)
  localize_sfm (PnP)            27.5 ms/query   (4.38s / 159)
  TOTAL online                 816.7 ms/query   (129.86s / 159)
  gpu peak                   1.23 GB alloc  1.68 GB reserved (this process)
```

## 2026-10-18 23:57 | ours (query_eval) | scene=stacey | fine=loftr | passes=2
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
clip on 159 queries (images), 2 fine pass(es)
translation: median 0.0143 units  mean 0.1843 units
rotation:    median 0.661 deg  mean 21.354
time:        median 498 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=67% [59-74]  10cm_5deg=70% [63-77]  25cm_2deg=67% [60-74]  50cm_10deg=71% [64-78]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    2.50 GB alloc  2.70 GB reserved (this process)
```

## 2026-10-18 23:59 | ours (query_eval) | scene=class1 | fine=loftr | passes=2
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
clip on 104 queries (images), 2 fine pass(es)
translation: median 0.0100 units  mean 0.0714 units
rotation:    median 0.235 deg  mean 9.474
time:        median 435 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=88% [80-93]  10cm_5deg=88% [80-93]  25cm_2deg=86% [78-91]  50cm_10deg=88% [80-93]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    1.67 GB alloc  2.21 GB reserved (this process)
```

## 2026-10-19 06:08 | ours (query_eval) | scene=stacey | fine=loftr | passes=1
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 159 queries (images), 1 fine pass(es)
translation: median 0.0244 units  mean 0.0648 units
rotation:    median 1.219 deg  mean 5.514
time:        median 179 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=67% [59-74]  10cm_5deg=79% [72-85]  25cm_2deg=67% [59-74]  50cm_10deg=89% [83-93]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    2.92 GB alloc  3.20 GB reserved (this process)
```

## 2026-10-19 06:09 | ours (query_eval) | scene=class1 | fine=loftr | passes=1
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 104 queries (images), 1 fine pass(es)
translation: median 0.0106 units  mean 0.0148 units
rotation:    median 0.482 deg  mean 0.781
time:        median 158 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=98% [93-99]  10cm_5deg=99% [95-100]  25cm_2deg=95% [89-98]  50cm_10deg=99% [95-100]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    2.04 GB alloc  2.58 GB reserved (this process)
```

## 2026-10-19 06:12 | ours (query_eval) | scene=stacey | fine=loftr | passes=2
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 159 queries (images), 2 fine pass(es)
translation: median 0.0118 units  mean 0.0475 units
rotation:    median 0.465 deg  mean 4.597
time:        median 317 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=87% [81-92]  10cm_5deg=89% [83-93]  25cm_2deg=87% [81-92]  50cm_10deg=90% [84-94]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    2.93 GB alloc  3.11 GB reserved (this process)
```

## 2026-10-19 06:13 | ours (query_eval) | scene=class1 | fine=loftr | passes=2
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 104 queries (images), 2 fine pass(es)
translation: median 0.0101 units  mean 0.0107 units
rotation:    median 0.237 deg  mean 0.290
time:        median 272 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=100% [96-100]  10cm_5deg=100% [96-100]  25cm_2deg=99% [95-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    2.07 GB alloc  2.58 GB reserved (this process)
```

## 2026-10-19 06:16 | ours (query_eval) | scene=stacey | fine=loftr | passes=3
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 159 queries (images), 3 fine pass(es)
translation: median 0.0119 units  mean 0.0414 units
rotation:    median 0.355 deg  mean 4.243
time:        median 455 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=91% [85-94]  10cm_5deg=91% [85-94]  25cm_2deg=90% [84-94]  50cm_10deg=92% [87-95]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    2.93 GB alloc  3.12 GB reserved (this process)
```

## 2026-10-19 06:18 | ours (query_eval) | scene=class1 | fine=loftr | passes=3
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 104 queries (images), 3 fine pass(es)
translation: median 0.0099 units  mean 0.0105 units
rotation:    median 0.211 deg  mean 0.243
time:        median 386 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=100% [96-100]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    2.07 GB alloc  2.58 GB reserved (this process)
```

## 2026-10-19 06:19 | hloc online timing | scene=class1
db model: /home/chini/3dgs/data/class1_train-f2900-processed
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
HLoc full online timing, 104 queries, resize_max=1024, num_loc=20
  SuperPoint extract            37.6 ms/query   (3.91s / 104)
  NetVLAD extract              114.9 ms/query   (11.95s / 104)
  retrieval (NN search)         10.0 ms/query   (1.04s / 104)
  LightGlue match (20 db)      293.6 ms/query   (30.53s / 104)
  localize_sfm (PnP)            37.3 ms/query   (3.87s / 104)
  TOTAL online                 493.4 ms/query   (51.31s / 104)
  gpu peak                   1.23 GB alloc  1.68 GB reserved (this process)
```

## 2026-10-19 06:21 | hloc online timing | scene=stacey
db model: /home/chini/3dgs/data/stacey_room_train-f4200-processed
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
HLoc full online timing, 159 queries, resize_max=1024, num_loc=20
  SuperPoint extract            37.9 ms/query   (6.02s / 159)
  NetVLAD extract              103.7 ms/query   (16.49s / 159)
  retrieval (NN search)         11.4 ms/query   (1.81s / 159)
  LightGlue match (20 db)      408.8 ms/query   (65.00s / 159)
  localize_sfm (PnP)            25.3 ms/query   (4.02s / 159)
  TOTAL online                 587.1 ms/query   (93.34s / 159)
  gpu peak                   1.23 GB alloc  1.68 GB reserved (this process)
```

## 2026-10-19 07:32 | ours (query_eval) | scene=class1 | fine=photometric | passes=1
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 40 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 0.0834 units  mean 0.1040 units
rotation:    median 9.750 deg  mean 11.690
time:        median 30469 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=15% [7-29]  10cm_5deg=15% [7-29]  25cm_2deg=5% [1-17]  50cm_10deg=52% [37-67]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 420  mean 401  min 153  max 500
hit cap without converging: 7/40
gpu peak:    1.72 GB alloc  1.77 GB reserved (this process)
```

## 2026-10-19 09:04 | ours (query_eval) | scene=stacey | fine=photometric | passes=1 | seed=perturb 0.05u/2deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 40 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 0.0115 units  mean 0.0189 units
rotation:    median 1.431 deg  mean 1.316
time:        median 56501 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=95% [83-99]  10cm_5deg=100% [91-100]  25cm_2deg=85% [71-93]  50cm_10deg=100% [91-100]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 487  mean 452  min 79  max 500
hit cap without converging: 10/40
gpu peak:    2.18 GB alloc  2.24 GB reserved (this process)
```

## 2026-10-19 09:05 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0.05u/2deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 40 queries (images), 2 fine pass(es)
translation: median 0.0104 units  mean 0.0101 units
rotation:    median 0.381 deg  mean 0.398
time:        median 276 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=100% [91-100]  10cm_5deg=100% [91-100]  25cm_2deg=100% [91-100]  50cm_10deg=100% [91-100]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    1.79 GB alloc  1.85 GB reserved (this process)
```

## 2026-10-19 09:29 | ours (query_eval) | scene=stacey | fine=photometric | passes=1 | seed=perturb 0.1u/5deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 40 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 0.0984 units  mean 0.0945 units
rotation:    median 5.028 deg  mean 5.003
time:        median 34708 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=0% [0-9]  10cm_5deg=32% [20-48]  25cm_2deg=0% [0-9]  50cm_10deg=100% [91-100]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 353  mean 331  min 93  max 500
hit cap without converging: 2/40
gpu peak:    2.21 GB alloc  2.30 GB reserved (this process)
```

## 2026-10-19 09:30 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0.1u/5deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 40 queries (images), 2 fine pass(es)
translation: median 0.0099 units  mean 0.0100 units
rotation:    median 0.392 deg  mean 0.474
time:        median 274 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=100% [91-100]  10cm_5deg=100% [91-100]  25cm_2deg=98% [87-100]  50cm_10deg=100% [91-100]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    1.79 GB alloc  1.85 GB reserved (this process)
```

## 2026-10-19 09:55 | ours (query_eval) | scene=class1 | fine=photometric | passes=1 | seed=perturb 0.05u/2deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 40 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 0.0143 units  mean 0.0172 units
rotation:    median 1.226 deg  mean 1.208
time:        median 34937 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=100% [91-100]  10cm_5deg=100% [91-100]  25cm_2deg=88% [74-95]  50cm_10deg=100% [91-100]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 481  mean 463  min 230  max 500
hit cap without converging: 7/40
gpu peak:    0.68 GB alloc  0.73 GB reserved (this process)
```

## 2026-10-19 09:55 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0.05u/2deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 40 queries (images), 2 fine pass(es)
translation: median 0.0087 units  mean 0.0091 units
rotation:    median 0.214 deg  mean 0.279
time:        median 221 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=100% [91-100]  10cm_5deg=100% [91-100]  25cm_2deg=100% [91-100]  50cm_10deg=100% [91-100]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    1.02 GB alloc  1.16 GB reserved (this process)
```

## 2026-10-19 10:17 | ours (query_eval) | scene=class1 | fine=photometric | passes=1 | seed=perturb 0.1u/5deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 40 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 0.0903 units  mean 0.0824 units
rotation:    median 4.730 deg  mean 4.572
time:        median 30382 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=8% [3-20]  10cm_5deg=58% [42-71]  25cm_2deg=0% [0-9]  50cm_10deg=100% [91-100]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 447  mean 417  min 95  max 500
hit cap without converging: 8/40
gpu peak:    0.71 GB alloc  0.76 GB reserved (this process)
```

## 2026-10-19 10:17 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0.1u/5deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 40 queries (images), 2 fine pass(es)
translation: median 0.0086 units  mean 0.0093 units
rotation:    median 0.233 deg  mean 0.317
time:        median 218 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=100% [91-100]  10cm_5deg=100% [91-100]  25cm_2deg=100% [91-100]  50cm_10deg=100% [91-100]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    1.03 GB alloc  1.18 GB reserved (this process)
```

## 2026-10-19 10:43 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0.2u/5deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 40 queries (images), 2 fine pass(es)
translation: median 0.0119 units  mean 0.0807 units
rotation:    median 0.602 deg  mean 4.847
time:        median 364 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=70% [55-82]  10cm_5deg=70% [55-82]  25cm_2deg=70% [55-82]  50cm_10deg=98% [87-100]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    1.83 GB alloc  1.90 GB reserved (this process)
```

## 2026-10-19 10:44 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0.28u/8deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 40 queries (images), 2 fine pass(es)
translation: median 0.2800 units  mean 0.1847 units
rotation:    median 8.000 deg  mean 8.919
time:        median 352 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=35% [22-50]  10cm_5deg=35% [22-50]  25cm_2deg=35% [22-50]  50cm_10deg=95% [83-99]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    1.82 GB alloc  1.90 GB reserved (this process)
```

## 2026-10-19 10:45 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0.2u/5deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 40 queries (images), 2 fine pass(es)
translation: median 0.0130 units  mean 0.0907 units
rotation:    median 0.667 deg  mean 7.694
time:        median 293 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=70% [55-82]  10cm_5deg=70% [55-82]  25cm_2deg=65% [50-78]  50cm_10deg=90% [77-96]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    1.04 GB alloc  1.17 GB reserved (this process)
```

## 2026-10-19 10:45 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0.28u/8deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 40 queries (images), 2 fine pass(es)
translation: median 0.2308 units  mean 0.1663 units
rotation:    median 8.000 deg  mean 11.600
time:        median 288 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=38% [24-53]  10cm_5deg=40% [26-55]  25cm_2deg=32% [20-48]  50cm_10deg=82% [68-91]  (95% wilson CI in [])
note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m
gpu peak:    1.04 GB alloc  1.23 GB reserved (this process)
```

## 2026-10-23 03:01 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=megaloc
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 2 queries (images), 2 fine pass(es)
translation: median 95.29 cm  mean 95.29 cm  (median 0.4806 u @ 1.9826 m/u)
rotation:    median 53.575 deg  mean 53.575
time:        median 4381 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=0% [0-66]  10cm_5deg=0% [0-66]  25cm_2deg=0% [0-66]  50cm_10deg=0% [0-66]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    2.98 GB alloc  3.11 GB reserved (this process)
```

## 2026-10-23 03:09 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=megaloc
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 2 queries (images), 2 fine pass(es)
translation: median 95.29 cm  mean 95.29 cm  (median 0.4806 u @ 1.9826 m/u)
rotation:    median 53.575 deg  mean 53.575
time:        median 3344 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=0% [0-66]  10cm_5deg=0% [0-66]  25cm_2deg=0% [0-66]  50cm_10deg=0% [0-66]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    2.92 GB alloc  3.14 GB reserved (this process)
```

## 2026-10-23 03:33 | ours (query_eval) | scene=stacey | fine=loftr | passes=1 | seed=megaloc
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 2 queries (images), 1 fine pass(es)
translation: median 79.36 cm  mean 79.36 cm  (median 0.4003 u @ 1.9826 m/u)
rotation:    median 23.211 deg  mean 23.211
time:        median 3001 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=0% [0-66]  10cm_5deg=0% [0-66]  25cm_2deg=0% [0-66]  50cm_10deg=0% [0-66]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    2.92 GB alloc  3.14 GB reserved (this process)
```

## 2026-10-23 03:33 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=megaloc
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 2 queries (images), 2 fine pass(es)
translation: median 95.29 cm  mean 95.29 cm  (median 0.4806 u @ 1.9826 m/u)
rotation:    median 53.575 deg  mean 53.575
time:        median 4226 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=0% [0-66]  10cm_5deg=0% [0-66]  25cm_2deg=0% [0-66]  50cm_10deg=0% [0-66]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    2.98 GB alloc  3.14 GB reserved (this process)
```

## 2026-10-23 03:34 | ours (query_eval) | scene=stacey | fine=loftr | passes=3 | seed=megaloc
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 2 queries (images), 3 fine pass(es)
translation: median 95.29 cm  mean 95.29 cm  (median 0.4806 u @ 1.9826 m/u)
rotation:    median 53.575 deg  mean 53.575
time:        median 2479 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=0% [0-66]  10cm_5deg=0% [0-66]  25cm_2deg=0% [0-66]  50cm_10deg=0% [0-66]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    2.92 GB alloc  3.14 GB reserved (this process)
```

## 2026-10-25 10:57 | ours (query_eval) | scene=stacey | fine=photometric | passes=1 | seed=perturb 0.05u/7deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 40 queries (images), photometric fine (mse, 500-iter cap)
translation: median 5.24 cm  mean 5.82 cm  (median 0.0264 u @ 1.9826 m/u)
rotation:    median 4.815 deg  mean 4.601
time:        median 21975 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=32% [20-48]  10cm_5deg=60% [45-74]  25cm_2deg=0% [0-9]  50cm_10deg=100% [91-100]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=mse  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 500  mean 491  min 175  max 500
hit cap without converging: 36/40
gpu peak:    1.61 GB alloc  2.01 GB reserved (this process)
```

## 2026-10-25 11:27 | ours (query_eval) | scene=stacey | fine=photometric | passes=1 | seed=perturb 0.05u/7deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 40 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 7.74 cm  mean 7.14 cm  (median 0.0390 u @ 1.9826 m/u)
rotation:    median 5.903 deg  mean 5.930
time:        median 37893 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=10% [4-23]  10cm_5deg=22% [12-38]  25cm_2deg=0% [0-9]  50cm_10deg=100% [91-100]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 429  mean 396  min 123  max 500
hit cap without converging: 12/40
gpu peak:    2.19 GB alloc  2.36 GB reserved (this process)
```

## 2026-10-27 13:24 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=megaloc
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 159 queries (images), 2 fine pass(es)
translation: median 2.34 cm  mean 9.42 cm  (median 0.0118 u @ 1.9826 m/u)
rotation:    median 0.465 deg  mean 4.597
time:        median 317 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=80% [73-85]  10cm_5deg=87% [81-92]  25cm_2deg=87% [81-92]  50cm_10deg=90% [84-94]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    2.93 GB alloc  3.20 GB reserved (this process)
```
