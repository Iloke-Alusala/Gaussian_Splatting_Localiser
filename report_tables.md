<!-- AUTOGEN:START (make_report_tables.py - do not edit inside) -->
# Head-to-head results (generated from summary files on disk)

All translation/rotation errors are in **nerfstudio scene units** - these maps have no metric scale. Recall thresholds carry a 95% Wilson CI in [].

## class1

| method | n | med t (u) | med r (deg) | ms/query | 5cm_5deg | 10cm_5deg | 25cm_2deg | 50cm_10deg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ours 1-pass | 104 | 0.0106 | 0.482 | 158 | 98% [93-99] | 99% [95-100] | 95% [89-98] | 99% [95-100] |
| ours 2-pass | 104 | 0.0101 | 0.237 | 272 | 100% [96-100] | 100% [96-100] | 99% [95-100] | 100% [96-100] |
| ours 3-pass | 104 | 0.0099 | 0.211 | 386 | 100% [96-100] | 100% [96-100] | 100% [96-100] | 100% [96-100] |
| ours photometric | 40 | 0.0834 | 9.750 | 30469 | 15% [7-29] | 15% [7-29] | 5% [1-17] | 52% [37-67] |
| ours CLIP 2-pass | 104 | 0.0100 | 0.235 | 435 | 88% [81-93] | 88% [81-93] | 86% [78-91] | 88% [81-93] |
| HLoc | 104 | 0.0009 | 0.085 | 493 | 100% [96-100] | 100% [96-100] | 100% [96-100] | 100% [96-100] |

### resources

- gaussians: 420,882
- checkpoint on disk: 0.31 GB
- ours peak VRAM: 2.58 GB reserved
- HLoc peak VRAM: 1.68 GB reserved

## stacey

| method | n | med t (u) | med r (deg) | ms/query | 5cm_5deg | 10cm_5deg | 25cm_2deg | 50cm_10deg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ours 1-pass | 159 | 0.0244 | 1.219 | 179 | 67% [60-74] | 79% [72-85] | 67% [60-74] | 89% [84-93] |
| ours 2-pass | 159 | 0.0118 | 0.465 | 317 | 87% [81-91] | 89% [84-93] | 87% [81-91] | 90% [84-94] |
| ours 3-pass | 159 | 0.0119 | 0.355 | 455 | 91% [86-95] | 91% [86-95] | 90% [84-94] | 92% [87-95] |
| ours photometric | 40 | 0.1610 | 18.139 | 29415 | 5% [1-17] | 5% [1-17] | 0% [0-9] | 8% [3-20] |
| ours CLIP 2-pass | 159 | 0.0143 | 0.661 | 498 | 67% [60-74] | 70% [62-76] | 67% [60-74] | 71% [64-78] |
| HLoc | 159 | 0.0060 | 0.320 | 587 | 72% [64-78] | 72% [64-78] | 71% [64-78] | 73% [66-79] |

### resources

- gaussians: 2,083,052
- checkpoint on disk: 1.48 GB
- ours peak VRAM: 3.11 GB reserved
- HLoc peak VRAM: 1.68 GB reserved

## Controlled-seed / basin experiment

Photometric (iterative LPIPS) vs PnP (2-pass) from **identical** seeds per query (same RNG base). Recall @5cm_5deg, n=40 each. Shows the photometric baseline is basin-limited, not weak: it recovers from a tight seed but not from where MegaLoc lands. PnP's basin is far wider.

| seed source | seed error | photometric stacey | photometric class1 | PnP stacey | PnP class1 |
| --- | --- | --- | --- | --- | --- |
| MegaLoc | ~0.28 u (where retrieval landed) | 5% [1-17] | 15% [7-29] | 87% [81-91] | 100% [96-100] |
| perturbed | 0.05 u / 2 deg (tight) | 95% [83-99] | 100% [91-100] | 100% [91-100] | 100% [91-100] |
| perturbed | 0.10 u / 5 deg (basin edge) | 0% [0-9] | 8% [3-20] | 100% [91-100] | 100% [91-100] |
| perturbed | 0.20 u / 5 deg | - | - | 70% [55-82] | 70% [55-82] |
| perturbed | 0.28 u / 8 deg (= MegaLoc dist) | - | - | 35% [22-50] | 38% [24-53] |

Note (photometric): the ~0.10 u / 5 deg 'edge' is from the synthetic capture-radius sweep (render-vs-render). On real photos (render-vs-photo, appearance gap) the photometric basin is tighter, ~0.05 u - hence edge collapses while tight succeeds.

Note (PnP has a WIDE but FINITE basin): 100% to 0.10 u, 70% at 0.20 u, ~35% at 0.28 u (random-direction offsets). So MegaLoc's misses are NOT purely wrong-place retrieval - correct-place seeds beyond ~0.1-0.2 u also fail. BUT real MegaLoc seeds recover far better than random offsets at the same distance (full-set 2-pass = 87% stacey from a 0.28 u median seed, vs 35% here): a retrieved frame is a plausible viewpoint that keeps scene overlap, which a random perturbation loses. So these controlled numbers are a CONSERVATIVE lower bound on PnP's tolerance to real retrieved seeds, and the wrong-place-vs-too-far split needs the per-query miss analysis to settle (retrieved seed distance + LoFTR inliers on the failing queries).

<!-- AUTOGEN:END -->

---

# Metric-scaled results (door-frame scale applied)

Same runs as above, converted to **real units** with the scale recovered from a hand-measured
door frame: **class1 = 3.5601 m/unit**, **stacey = 1.9826 m/unit** (metres = scene_units x scale).

Median errors are the scene-unit values multiplied by the scale. The recall percentages are
**identical** to the scene-unit tables - only the thresholds are relabelled, because the original
recall used 0.05 / 0.10 / 0.25 / 0.50 *unit* cuts, which under the measured scale land at the real
distances shown in the column headers. (So the columns differ between scenes: 0.05 u = 9.9 cm in
stacey but 17.8 cm in class1.) Rotation is already absolute degrees and is unchanged.

To get recall at ROUND metric thresholds (5 / 10 / 25 / 50 cm) instead of these relabelled cuts,
re-run the updated `query_eval.py` - it now applies `m_per_unit` and scores against the cm
thresholds directly.

## class1 (x3.5601 m/unit)

| method | n | med t (cm) | med r (deg) | ms/query | 17.8cm_5deg | 35.6cm_5deg | 89cm_2deg | 178cm_10deg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ours 1-pass | 104 | 3.77 | 0.482 | 158 | 98% [93-99] | 99% [95-100] | 95% [89-98] | 99% [95-100] |
| ours 2-pass | 104 | 3.60 | 0.237 | 272 | 100% [96-100] | 100% [96-100] | 99% [95-100] | 100% [96-100] |
| ours 3-pass | 104 | 3.52 | 0.211 | 386 | 100% [96-100] | 100% [96-100] | 100% [96-100] | 100% [96-100] |
| ours photometric | 40 | 29.69 | 9.750 | 30469 | 15% [7-29] | 15% [7-29] | 5% [1-17] | 52% [37-67] |
| ours CLIP 2-pass | 104 | 3.56 | 0.235 | 435 | 88% [81-93] | 88% [81-93] | 86% [78-91] | 88% [81-93] |
| HLoc | 104 | 0.32 | 0.085 | 493 | 100% [96-100] | 100% [96-100] | 100% [96-100] | 100% [96-100] |

## stacey (x1.9826 m/unit)

| method | n | med t (cm) | med r (deg) | ms/query | 9.9cm_5deg | 19.8cm_5deg | 49.6cm_2deg | 99.1cm_10deg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ours 1-pass | 159 | 4.84 | 1.219 | 179 | 67% [60-74] | 79% [72-85] | 67% [60-74] | 89% [84-93] |
| ours 2-pass | 159 | 2.34 | 0.465 | 317 | 87% [81-91] | 89% [84-93] | 87% [81-91] | 90% [84-94] |
| ours 3-pass | 159 | 2.36 | 0.355 | 455 | 91% [86-95] | 91% [86-95] | 90% [84-94] | 92% [87-95] |
| ours photometric | 40 | 31.92 | 18.139 | 29415 | 5% [1-17] | 5% [1-17] | 0% [0-9] | 8% [3-20] |
| ours CLIP 2-pass | 159 | 2.84 | 0.661 | 498 | 67% [60-74] | 70% [62-76] | 67% [60-74] | 71% [64-78] |
| HLoc | 159 | 1.19 | 0.320 | 587 | 72% [64-78] | 72% [64-78] | 71% [64-78] | 73% [66-79] |

## Controlled-seed / basin experiment (metric)

Same %s as the scene-unit table; the seed magnitude is applied in scene units, so its physical size
differs per scene - both shown.

| seed source | seed (u) | seed stacey (cm) | seed class1 (cm) | photometric stacey | photometric class1 | PnP stacey | PnP class1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MegaLoc | ~0.28 | ~55.5 | ~99.7 | 5% [1-17] | 15% [7-29] | 87% [81-91] | 100% [96-100] |
| perturbed | 0.05 / 2 deg | 9.9 | 17.8 | 95% [83-99] | 100% [91-100] | 100% [91-100] | 100% [91-100] |
| perturbed | 0.10 / 5 deg | 19.8 | 35.6 | 0% [0-9] | 8% [3-20] | 100% [91-100] | 100% [91-100] |
| perturbed | 0.20 / 5 deg | 39.7 | 71.2 | - | - | 70% [55-82] | 70% [55-82] |
| perturbed | 0.28 / 8 deg | 55.5 | 99.7 | - | - | 35% [22-50] | 38% [24-53] |

