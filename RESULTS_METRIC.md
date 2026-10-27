# RESULTS_METRIC — metric-scaled reruns

Brand-new sheet for the overnight metric rerun. Everything here is in **real metres**
(scene units × m_per_unit: class1 = 3.5601, stacey = 1.9826). Thresholds are the round
metric bars 5 / 10 / 25 / 50 cm. The old `RESULTS_LOG.md` is left untouched.

Scripts append below when run with `RESULTS_LOG=$(pwd)/RESULTS_METRIC.md` in the env.

---

## 2026-10-21 01:14 | ours (query_eval) | scene=class1 | fine=loftr | passes=1 | seed=megaloc
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 104 queries (images), 1 fine pass(es)
translation: median 3.78 cm  mean 5.28 cm  (median 0.0106 u @ 3.5601 m/u)
rotation:    median 0.482 deg  mean 0.781
time:        median 157 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=62% [52-70]  10cm_5deg=89% [82-94]  25cm_2deg=95% [89-98]  50cm_10deg=99% [95-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    2.04 GB alloc  2.58 GB reserved (this process)
```

## 2026-10-21 01:15 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=megaloc
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 104 queries (images), 2 fine pass(es)
translation: median 3.61 cm  mean 3.82 cm  (median 0.0101 u @ 3.5601 m/u)
rotation:    median 0.237 deg  mean 0.290
time:        median 271 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=77% [68-84]  10cm_5deg=99% [95-100]  25cm_2deg=99% [95-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    2.07 GB alloc  2.58 GB reserved (this process)
```

## 2026-10-21 01:17 | ours (query_eval) | scene=class1 | fine=loftr | passes=3 | seed=megaloc
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 104 queries (images), 3 fine pass(es)
translation: median 3.54 cm  mean 3.75 cm  (median 0.0099 u @ 3.5601 m/u)
rotation:    median 0.211 deg  mean 0.243
time:        median 384 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=81% [72-87]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    2.07 GB alloc  2.58 GB reserved (this process)
```

## 2026-10-21 01:18 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=megaloc
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
clip on 104 queries (images), 2 fine pass(es)
translation: median 3.56 cm  mean 25.42 cm  (median 0.0100 u @ 3.5601 m/u)
rotation:    median 0.235 deg  mean 9.474
time:        median 231 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=69% [60-77]  10cm_5deg=88% [80-93]  25cm_2deg=86% [78-91]  50cm_10deg=88% [80-93]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.67 GB alloc  2.21 GB reserved (this process)
```

## 2026-10-21 01:21 | ours (query_eval) | scene=stacey | fine=loftr | passes=1 | seed=megaloc
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 159 queries (images), 1 fine pass(es)
translation: median 4.84 cm  mean 12.84 cm  (median 0.0244 u @ 1.9826 m/u)
rotation:    median 1.219 deg  mean 5.514
time:        median 178 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=52% [44-60]  10cm_5deg=67% [59-74]  25cm_2deg=65% [57-72]  50cm_10deg=89% [83-93]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    2.92 GB alloc  3.20 GB reserved (this process)
```

## 2026-10-21 01:23 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=megaloc
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 159 queries (images), 2 fine pass(es)
translation: median 2.34 cm  mean 9.42 cm  (median 0.0118 u @ 1.9826 m/u)
rotation:    median 0.465 deg  mean 4.597
time:        median 314 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=80% [73-85]  10cm_5deg=87% [81-92]  25cm_2deg=87% [81-92]  50cm_10deg=90% [84-94]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    2.93 GB alloc  3.11 GB reserved (this process)
```

## 2026-10-21 01:26 | ours (query_eval) | scene=stacey | fine=loftr | passes=3 | seed=megaloc
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 159 queries (images), 3 fine pass(es)
translation: median 2.35 cm  mean 8.22 cm  (median 0.0119 u @ 1.9826 m/u)
rotation:    median 0.355 deg  mean 4.243
time:        median 453 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=88% [82-92]  10cm_5deg=91% [85-94]  25cm_2deg=90% [84-94]  50cm_10deg=92% [87-95]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    2.93 GB alloc  3.12 GB reserved (this process)
```

## 2026-10-21 01:29 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=megaloc
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
clip on 159 queries (images), 2 fine pass(es)
translation: median 2.84 cm  mean 36.54 cm  (median 0.0143 u @ 1.9826 m/u)
rotation:    median 0.661 deg  mean 21.354
time:        median 275 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=62% [54-69]  10cm_5deg=67% [59-74]  25cm_2deg=67% [60-74]  50cm_10deg=71% [64-78]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    2.50 GB alloc  2.70 GB reserved (this process)
```

## 2026-10-21 01:30 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0u/5deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 3.24 cm  mean 3.58 cm  (median 0.0091 u @ 3.5601 m/u)
rotation:    median 0.186 deg  mean 0.207
time:        median 223 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=87% [79-92]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.06 GB alloc  1.52 GB reserved (this process)
```

## 2026-10-21 01:31 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0u/10deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 3.19 cm  mean 3.58 cm  (median 0.0090 u @ 3.5601 m/u)
rotation:    median 0.195 deg  mean 0.212
time:        median 221 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=86% [78-91]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.06 GB alloc  1.52 GB reserved (this process)
```

## 2026-10-21 01:33 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0u/15deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 3.51 cm  mean 3.59 cm  (median 0.0099 u @ 3.5601 m/u)
rotation:    median 0.182 deg  mean 0.210
time:        median 222 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=87% [79-92]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.06 GB alloc  1.52 GB reserved (this process)
```

## 2026-10-21 01:34 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0u/20deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 3.41 cm  mean 3.61 cm  (median 0.0096 u @ 3.5601 m/u)
rotation:    median 0.199 deg  mean 0.221
time:        median 219 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=87% [79-92]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.06 GB alloc  1.52 GB reserved (this process)
```

## 2026-10-21 01:35 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0u/25deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 3.42 cm  mean 3.61 cm  (median 0.0096 u @ 3.5601 m/u)
rotation:    median 0.200 deg  mean 0.225
time:        median 218 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=86% [78-91]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.06 GB alloc  1.52 GB reserved (this process)
```

## 2026-10-21 01:37 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0u/30deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 3.49 cm  mean 3.65 cm  (median 0.0098 u @ 3.5601 m/u)
rotation:    median 0.208 deg  mean 1.146
time:        median 217 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=80% [71-87]  10cm_5deg=96% [90-98]  25cm_2deg=96% [90-98]  50cm_10deg=97% [92-99]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.06 GB alloc  1.52 GB reserved (this process)
```

## 2026-10-21 01:38 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0u/40deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 2.72 cm  mean 8.94 cm  (median 0.0076 u @ 3.5601 m/u)
rotation:    median 0.266 deg  mean 14.168
time:        median 213 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=57% [47-66]  10cm_5deg=71% [61-79]  25cm_2deg=71% [61-79]  50cm_10deg=71% [61-79]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.06 GB alloc  1.52 GB reserved (this process)
```

## 2026-10-21 01:39 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0u/50deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 0.00 cm  mean 11.62 cm  (median 0.0000 u @ 3.5601 m/u)
rotation:    median 50.000 deg  mean 33.103
time:        median 208 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=29% [21-39]  10cm_5deg=36% [27-46]  25cm_2deg=35% [26-45]  50cm_10deg=36% [27-46]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.05 GB alloc  1.52 GB reserved (this process)
```

## 2026-10-21 01:41 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0u/5deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 2.05 cm  mean 2.02 cm  (median 0.0103 u @ 1.9826 m/u)
rotation:    median 0.333 deg  mean 0.370
time:        median 271 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=100% [96-100]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.79 GB alloc  2.09 GB reserved (this process)
```

## 2026-10-21 01:42 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0u/10deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 2.01 cm  mean 2.00 cm  (median 0.0101 u @ 1.9826 m/u)
rotation:    median 0.333 deg  mean 0.358
time:        median 269 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=100% [96-100]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.79 GB alloc  2.09 GB reserved (this process)
```

## 2026-10-21 01:44 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0u/15deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 2.01 cm  mean 2.02 cm  (median 0.0101 u @ 1.9826 m/u)
rotation:    median 0.324 deg  mean 0.358
time:        median 268 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=100% [96-100]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.79 GB alloc  2.09 GB reserved (this process)
```

## 2026-10-21 01:46 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0u/20deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 2.04 cm  mean 2.02 cm  (median 0.0103 u @ 1.9826 m/u)
rotation:    median 0.310 deg  mean 0.355
time:        median 267 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=100% [96-100]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.79 GB alloc  2.09 GB reserved (this process)
```

## 2026-10-21 01:47 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0u/25deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 2.01 cm  mean 2.02 cm  (median 0.0101 u @ 1.9826 m/u)
rotation:    median 0.324 deg  mean 0.359
time:        median 266 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=100% [96-100]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.79 GB alloc  2.03 GB reserved (this process)
```

## 2026-10-21 01:49 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0u/30deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 2.00 cm  mean 2.02 cm  (median 0.0101 u @ 1.9826 m/u)
rotation:    median 0.333 deg  mean 0.667
time:        median 265 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=99% [95-100]  10cm_5deg=99% [95-100]  25cm_2deg=99% [95-100]  50cm_10deg=99% [95-100]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.80 GB alloc  2.03 GB reserved (this process)
```

## 2026-10-21 01:51 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0u/40deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 1.76 cm  mean 1.42 cm  (median 0.0089 u @ 1.9826 m/u)
rotation:    median 0.407 deg  mean 12.646
time:        median 262 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=69% [59-77]  10cm_5deg=69% [59-77]  25cm_2deg=69% [59-77]  50cm_10deg=69% [59-77]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.81 GB alloc  2.03 GB reserved (this process)
```

## 2026-10-21 01:52 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0u/50deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 0.00 cm  mean 2.86 cm  (median 0.0000 u @ 1.9826 m/u)
rotation:    median 50.000 deg  mean 32.647
time:        median 256 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=35% [26-45]  10cm_5deg=35% [26-45]  25cm_2deg=35% [26-45]  50cm_10deg=36% [27-46]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.83 GB alloc  2.03 GB reserved (this process)
```

## 2026-10-21 01:54 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0.005618u/1deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 3.57 cm  mean 3.62 cm  (median 0.0100 u @ 3.5601 m/u)
rotation:    median 0.196 deg  mean 0.219
time:        median 223 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=86% [78-91]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.06 GB alloc  1.57 GB reserved (this process)
```

## 2026-10-21 01:55 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0.014045u/2deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 3.23 cm  mean 3.59 cm  (median 0.0091 u @ 3.5601 m/u)
rotation:    median 0.184 deg  mean 0.215
time:        median 222 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=84% [76-90]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.06 GB alloc  1.52 GB reserved (this process)
```

## 2026-10-21 01:56 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0.028089u/5deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 3.40 cm  mean 3.66 cm  (median 0.0095 u @ 3.5601 m/u)
rotation:    median 0.208 deg  mean 0.243
time:        median 222 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=87% [79-92]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.05 GB alloc  1.52 GB reserved (this process)
```

## 2026-10-21 01:58 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0.056178u/5deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 3.17 cm  mean 3.51 cm  (median 0.0089 u @ 3.5601 m/u)
rotation:    median 0.191 deg  mean 0.226
time:        median 221 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=87% [79-92]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.04 GB alloc  1.52 GB reserved (this process)
```

## 2026-10-21 01:59 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0.112356u/8deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 3.49 cm  mean 8.07 cm  (median 0.0098 u @ 3.5601 m/u)
rotation:    median 0.240 deg  mean 1.968
time:        median 218 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=79% [70-86]  10cm_5deg=96% [90-98]  25cm_2deg=95% [89-98]  50cm_10deg=99% [95-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.04 GB alloc  1.52 GB reserved (this process)
```

## 2026-10-21 02:00 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0.168535u/12deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 4.20 cm  mean 15.51 cm  (median 0.0118 u @ 3.5601 m/u)
rotation:    median 0.371 deg  mean 4.800
time:        median 217 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=64% [54-73]  10cm_5deg=77% [68-84]  25cm_2deg=77% [68-84]  50cm_10deg=81% [72-87]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.06 GB alloc  1.61 GB reserved (this process)
```

## 2026-10-21 02:02 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0.224713u/16deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 6.80 cm  mean 45.52 cm  (median 0.0191 u @ 3.5601 m/u)
rotation:    median 1.079 deg  mean 9.094
time:        median 216 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=45% [36-55]  10cm_5deg=54% [44-63]  25cm_2deg=54% [44-63]  50cm_10deg=60% [50-69]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.06 GB alloc  1.65 GB reserved (this process)
```

## 2026-10-21 02:03 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0.280891u/20deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 100.00 cm  mean 65.78 cm  (median 0.2809 u @ 3.5601 m/u)
rotation:    median 20.000 deg  mean 16.074
time:        median 211 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=28% [20-37]  10cm_5deg=35% [26-45]  25cm_2deg=37% [28-47]  50cm_10deg=40% [31-50]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.06 GB alloc  1.65 GB reserved (this process)
```

## 2026-10-21 02:04 | ours (query_eval) | scene=class1 | fine=loftr | passes=2 | seed=perturb 0.421336u/30deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 150.00 cm  mean 133.28 cm  (median 0.4213 u @ 3.5601 m/u)
rotation:    median 30.000 deg  mean 28.205
time:        median 201 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=8% [4-15]  10cm_5deg=9% [5-16]  25cm_2deg=10% [6-17]  50cm_10deg=11% [6-19]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.07 GB alloc  1.64 GB reserved (this process)
```

## 2026-10-21 02:06 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0.010088u/1deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 2.07 cm  mean 2.07 cm  (median 0.0105 u @ 1.9826 m/u)
rotation:    median 0.322 deg  mean 0.368
time:        median 272 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=100% [96-100]  10cm_5deg=100% [96-100]  25cm_2deg=100% [96-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.79 GB alloc  2.09 GB reserved (this process)
```

## 2026-10-21 02:07 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0.025219u/2deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 2.08 cm  mean 2.12 cm  (median 0.0105 u @ 1.9826 m/u)
rotation:    median 0.324 deg  mean 0.393
time:        median 271 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=99% [95-100]  10cm_5deg=100% [96-100]  25cm_2deg=99% [95-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.80 GB alloc  2.09 GB reserved (this process)
```

## 2026-10-21 02:09 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0.050439u/5deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 2.04 cm  mean 2.18 cm  (median 0.0103 u @ 1.9826 m/u)
rotation:    median 0.356 deg  mean 0.438
time:        median 270 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=97% [92-99]  10cm_5deg=100% [96-100]  25cm_2deg=99% [95-100]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.81 GB alloc  2.09 GB reserved (this process)
```

## 2026-10-21 02:11 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0.100878u/5deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 1.96 cm  mean 2.72 cm  (median 0.0099 u @ 1.9826 m/u)
rotation:    median 0.398 deg  mean 0.622
time:        median 269 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=93% [86-97]  10cm_5deg=97% [92-99]  25cm_2deg=97% [92-99]  50cm_10deg=100% [96-100]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.82 GB alloc  2.09 GB reserved (this process)
```

## 2026-10-21 02:12 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0.201755u/8deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 3.51 cm  mean 15.00 cm  (median 0.0177 u @ 1.9826 m/u)
rotation:    median 0.759 deg  mean 3.318
time:        median 264 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=59% [49-68]  10cm_5deg=65% [55-74]  25cm_2deg=62% [52-71]  50cm_10deg=98% [93-99]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.87 GB alloc  2.09 GB reserved (this process)
```

## 2026-10-21 02:14 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0.302633u/12deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 60.00 cm  mean 36.14 cm  (median 0.3026 u @ 1.9826 m/u)
rotation:    median 12.000 deg  mean 8.399
time:        median 257 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=36% [27-46]  10cm_5deg=40% [31-50]  25cm_2deg=39% [30-49]  50cm_10deg=44% [35-54]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.87 GB alloc  2.06 GB reserved (this process)
```

## 2026-10-21 02:15 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0.403511u/16deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 80.00 cm  mean 58.86 cm  (median 0.4035 u @ 1.9826 m/u)
rotation:    median 16.000 deg  mean 13.177
time:        median 250 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=23% [16-32]  10cm_5deg=25% [18-34]  25cm_2deg=23% [16-32]  50cm_10deg=29% [21-39]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.88 GB alloc  2.04 GB reserved (this process)
```

## 2026-10-21 02:17 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0.504388u/20deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 100.00 cm  mean 83.88 cm  (median 0.5044 u @ 1.9826 m/u)
rotation:    median 20.000 deg  mean 20.326
time:        median 246 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=12% [7-20]  10cm_5deg=13% [8-21]  25cm_2deg=12% [7-20]  50cm_10deg=15% [9-23]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.86 GB alloc  1.97 GB reserved (this process)
```

## 2026-10-21 02:18 | ours (query_eval) | scene=stacey | fine=loftr | passes=2 | seed=perturb 0.756582u/30deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 100 queries (images), 2 fine pass(es)
translation: median 150.00 cm  mean 144.33 cm  (median 0.7566 u @ 1.9826 m/u)
rotation:    median 30.000 deg  mean 28.951
time:        median 234 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=2% [1-7]  10cm_5deg=2% [1-7]  25cm_2deg=2% [1-7]  50cm_10deg=4% [2-10]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
gpu peak:    1.93 GB alloc  2.04 GB reserved (this process)
```

## 2026-10-21 04:00 | ours (query_eval) | scene=class1 | fine=photometric | passes=1 | seed=megaloc
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 104 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 33.00 cm  mean 39.61 cm  (median 0.0927 u @ 3.5601 m/u)
rotation:    median 9.196 deg  mean 10.869
time:        median 50867 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=6% [3-12]  10cm_5deg=9% [5-16]  25cm_2deg=8% [4-14]  50cm_10deg=44% [35-54]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 409  mean 397  min 120  max 500
hit cap without converging: 21/104
gpu peak:    1.75 GB alloc  1.91 GB reserved (this process)
```

## 2026-10-21 06:53 | ours (query_eval) | scene=stacey | fine=photometric | passes=1 | seed=megaloc
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 159 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 54.01 cm  mean 59.44 cm  (median 0.2724 u @ 1.9826 m/u)
rotation:    median 18.290 deg  mean 21.131
time:        median 59136 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=1% [0-3]  10cm_5deg=1% [0-4]  25cm_2deg=0% [0-2]  50cm_10deg=11% [7-17]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 348  mean 315  min 10  max 500
hit cap without converging: 4/159
gpu peak:    3.31 GB alloc  3.53 GB reserved (this process)
```

## 2026-10-21 07:09 | ours (query_eval) | scene=class1 | fine=photometric | passes=1 | seed=perturb 0.005618u/1deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 40 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 2.72 cm  mean 2.78 cm  (median 0.0076 u @ 3.5601 m/u)
rotation:    median 0.271 deg  mean 0.296
time:        median 22907 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=95% [83-99]  10cm_5deg=100% [91-100]  25cm_2deg=100% [91-100]  50cm_10deg=100% [91-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 341  mean 336  min 73  max 480
hit cap without converging: 0/40
gpu peak:    0.73 GB alloc  0.79 GB reserved (this process)
```

## 2026-10-21 07:31 | ours (query_eval) | scene=class1 | fine=photometric | passes=1 | seed=perturb 0.014045u/2deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 40 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 3.28 cm  mean 3.24 cm  (median 0.0092 u @ 3.5601 m/u)
rotation:    median 0.473 deg  mean 0.509
time:        median 31569 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=90% [77-96]  10cm_5deg=100% [91-100]  25cm_2deg=98% [87-100]  50cm_10deg=100% [91-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 452  mean 423  min 158  max 492
hit cap without converging: 0/40
gpu peak:    0.74 GB alloc  0.78 GB reserved (this process)
```

## 2026-10-21 07:57 | ours (query_eval) | scene=class1 | fine=photometric | passes=1 | seed=perturb 0.028089u/5deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 40 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 6.83 cm  mean 6.87 cm  (median 0.0192 u @ 3.5601 m/u)
rotation:    median 2.187 deg  mean 2.419
time:        median 40123 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=22% [12-38]  10cm_5deg=90% [77-96]  25cm_2deg=40% [26-55]  50cm_10deg=100% [91-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 498  mean 475  min 170  max 500
hit cap without converging: 19/40
gpu peak:    0.73 GB alloc  0.82 GB reserved (this process)
```

## 2026-10-21 08:23 | ours (query_eval) | scene=class1 | fine=photometric | passes=1 | seed=perturb 0.056178u/5deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 40 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 10.55 cm  mean 12.20 cm  (median 0.0296 u @ 3.5601 m/u)
rotation:    median 3.176 deg  mean 3.365
time:        median 36713 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=2% [0-13]  10cm_5deg=45% [31-60]  25cm_2deg=12% [5-26]  50cm_10deg=100% [91-100]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 500  mean 462  min 205  max 500
hit cap without converging: 21/40
gpu peak:    0.69 GB alloc  0.76 GB reserved (this process)
```

## 2026-10-21 10:37 | coarse | scene=class1 | retriever=megaloc | source=real
```
                10cm      25cm      50cm     100cm
  recall@1    7.7%    36.5%    64.4%    99.0%
  recall@5   11.5%    40.4%    74.0%   100.0%
  recall@10  14.4%    42.3%    74.0%   100.0%
note: recall radii are metric, converted cm->units via m_per_unit
```

## 2026-10-21 10:40 | coarse | scene=class1 | retriever=megaloc | source=rendered
```
                10cm      25cm      50cm     100cm
  recall@1    0.9%     5.8%    33.1%    69.1%
  recall@5    1.4%     9.2%    48.0%    81.9%
  recall@10   2.4%    10.2%    54.0%    87.7%
note: recall radii are metric, converted cm->units via m_per_unit
```

## 2026-10-21 10:40 | coarse | scene=class1 | retriever=clip | source=real
```
                10cm      25cm      50cm     100cm
  recall@1    5.8%    26.9%    49.0%    79.8%
  recall@5    8.7%    33.7%    63.5%    95.2%
  recall@10  10.6%    38.5%    66.3%    98.1%
note: recall radii are metric, converted cm->units via m_per_unit
```

## 2026-10-21 10:43 | coarse | scene=class1 | retriever=clip | source=rendered
```
                10cm      25cm      50cm     100cm
  recall@1    0.5%     3.1%    17.8%    50.0%
  recall@5    1.8%     5.7%    32.2%    77.6%
  recall@10   2.2%     8.0%    39.7%    86.6%
note: recall radii are metric, converted cm->units via m_per_unit
```

## 2026-10-21 10:44 | coarse | scene=stacey | retriever=megaloc | source=real
```
                10cm      25cm      50cm     100cm
  recall@1    1.3%     8.8%    44.7%    90.6%
  recall@5    2.5%    16.4%    50.9%    96.2%
  recall@10   2.5%    22.0%    53.5%    96.2%
note: recall radii are metric, converted cm->units via m_per_unit
```

## 2026-10-21 10:48 | coarse | scene=stacey | retriever=megaloc | source=rendered
```
                10cm      25cm      50cm     100cm
  recall@1    0.1%     2.1%    43.6%    74.4%
  recall@5    0.4%     4.5%    57.4%    83.3%
  recall@10   0.5%     6.3%    63.0%    88.3%
note: recall radii are metric, converted cm->units via m_per_unit
```

## 2026-10-21 10:49 | coarse | scene=stacey | retriever=clip | source=real
```
                10cm      25cm      50cm     100cm
  recall@1    0.6%    11.3%    37.1%    73.0%
  recall@5    3.1%    16.4%    45.9%    86.8%
  recall@10   4.4%    22.6%    52.8%    94.3%
note: recall radii are metric, converted cm->units via m_per_unit
```

## 2026-10-21 10:52 | coarse | scene=stacey | retriever=clip | source=rendered
```
                10cm      25cm      50cm     100cm
  recall@1    0.2%     2.3%    28.4%    61.2%
  recall@5    0.4%     4.8%    47.1%    78.2%
  recall@10   0.4%     6.4%    57.6%    86.7%
note: recall radii are metric, converted cm->units via m_per_unit
```

## 2026-10-21 10:54 | capture-radius (render-vs-render) | scene=stacey
```
capture radius render-vs-render, scene=stacey, 50 trials/offset, 2-pass, recovered = final within 5cm/5deg of base (1.9826 m/unit)
offset            recovered   recall
  2cm/1deg            50/50   100%
  5cm/2deg            50/50   100%
  10cm/5deg           48/50   96%
  20cm/5deg           48/50   96%
  40cm/8deg           41/50   82%
  60cm/12deg          32/50   64%
  80cm/16deg          16/50   32%
  100cm/20deg         11/50   22%
  150cm/30deg          5/50   10%
```

## 2026-10-22 13:58 | ours (query_eval) | scene=class1 | fine=photometric | passes=1 | seed=perturb 0.168535u/12deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 40 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 59.48 cm  mean 58.06 cm  (median 0.1671 u @ 3.5601 m/u)
rotation:    median 11.759 deg  mean 11.630
time:        median 35672 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=0% [0-9]  10cm_5deg=0% [0-9]  25cm_2deg=0% [0-9]  50cm_10deg=0% [0-9]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 390  mean 381  min 164  max 500
hit cap without converging: 4/40
gpu peak:    0.72 GB alloc  0.85 GB reserved (this process)
```

## 2026-10-22 14:00 | ours (query_eval) | scene=class1 | fine=photometric | passes=1 | seed=perturb 0.168535u/12deg
splat: /home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml
queries: /home/chini/3dgs/eval/class1/query_gt.json
```
megaloc on 40 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 59.85 cm  mean 58.14 cm  (median 0.1681 u @ 3.5601 m/u)
rotation:    median 11.697 deg  mean 11.586
time:        median 36589 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=0% [0-9]  10cm_5deg=0% [0-9]  25cm_2deg=0% [0-9]  50cm_10deg=0% [0-9]  (95% wilson CI in [])
note: scale 3.5601 m/scene-unit (door-frame measurement); recall cm are real cm
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 390  mean 388  min 114  max 500
hit cap without converging: 2/40
gpu peak:    0.81 GB alloc  0.91 GB reserved (this process)
```

## 2026-10-22 14:27 | ours (query_eval) | scene=stacey | fine=photometric | passes=1 | seed=perturb 0.010088u/1deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 40 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 1.61 cm  mean 1.63 cm  (median 0.0081 u @ 1.9826 m/u)
rotation:    median 0.312 deg  mean 0.365
time:        median 37694 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=100% [91-100]  10cm_5deg=100% [91-100]  25cm_2deg=100% [91-100]  50cm_10deg=100% [91-100]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 409  mean 382  min 166  max 482
hit cap without converging: 0/40
gpu peak:    2.19 GB alloc  2.31 GB reserved (this process)
```

## 2026-10-22 15:00 | ours (query_eval) | scene=stacey | fine=photometric | passes=1 | seed=perturb 0.050439u/5deg
splat: /home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml
queries: /home/chini/3dgs/eval/stacey/query_gt.json
```
megaloc on 40 queries (images), photometric fine (lpips, 500-iter cap)
translation: median 5.15 cm  mean 5.69 cm  (median 0.0260 u @ 1.9826 m/u)
rotation:    median 3.860 deg  mean 3.722
time:        median 45079 ms/query  (coarse+fine online; first query includes model warmup)
recall: 5cm_5deg=48% [33-63]  10cm_5deg=92% [80-97]  25cm_2deg=2% [0-13]  50cm_10deg=100% [91-100]  (95% wilson CI in [])
note: scale 1.9826 m/scene-unit (door-frame measurement); recall cm are real cm
photometric: lr_trans=0.0002 lr_rot=0.0002  loss=lpips  cap=500  schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters
iters:  median 498  mean 440  min 97  max 500
hit cap without converging: 18/40
gpu peak:    2.18 GB alloc  2.34 GB reserved (this process)
```
