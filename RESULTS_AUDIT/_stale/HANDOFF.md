# Handoff — 3DGS localisation vs HLoc baseline

Paste this at the start of a new session. It captures the project, what's done,
where everything lives, the non-obvious gotchas, and what's open.

## Project
Final-year thesis: **camera pose localisation inside a 3D Gaussian Splatting
(3DGS) map**, benchmarked against the established **HLoc** baseline (CVPR 2019,
arXiv:1812.03506; SuperPoint + LightGlue + NetVLAD). Goal: a defensible
head-to-head (accuracy, speed, setup cost, peak resources) for the report.
**Status: the benchmark is DONE and written to `EVALUATION.md`.**

## Machine / layout
- GPU: RTX 500 Ada laptop, **4 GB VRAM** — the binding constraint throughout.
- Code: `/home/chini/Desktop/3DGS/code`.  Data+outputs: `/home/chini/3dgs`.
- Two conda envs, **do not mix** (torch pins conflict):
  - `nerfstudio` — our pipeline (splatfacto render). torch 2.4.1+cu121.
  - `hloc` — HLoc baseline. **pycolmap MUST be 3.13.0** (4.x silently breaks
    triangulation → 0 points). lightglue (git), torch 2.5.1+cu121.
- Long runs get killed on session teardown — launch with `setsid nohup ... &`
  (detached), log to a file, poll the file. `conda run` buffers stdout.

## Scene + data (stacey_room-f2500)
`/home/chini/3dgs/data/stacey_room-f2500-processed/`
- `images/` = 2500 training video frames (720x1280 **portrait**), the DB.
- `images/query_d1280/` = 53 real query stills (1280x960 raw landscape **+ EXIF
  rotate flag** → display 960x1280). `images/query_d1280_up/` = EXIF-baked
  upright copies we generated for HLoc.
- `query_gt_d1280.json` (raw-frame intrinsics+pose), `query_gt_d1280_up.json`
  (rotated intrinsics, same R/t) — GT in COLMAP world frame.
- `colmap/sparse/0` = DB model (query-free). `0_with_query*` = queries registered.
- Splat: `outputs/stacey_room-f2500/splatfacto/2026-10-08_182851/config.yml`.
- **dataparser scale = 0.2061375184571612** (COLMAP units x scale = nerf units).

## Our pipeline — `query_eval.py` (nerfstudio env)
MegaLoc retrieval (coarse) -> render RGB+depth from splat at that seed ->
EfficientLoFTR match -> PnP (fine). No viewer; loops the query set, writes
`query_eval_out/<name>_megaloc_{report.txt,summary.png,link.png}` +
`summary_megaloc.txt`. Reuses `pipeline_retriever.py` (untouched) + `my_localizer`
+ `my_pose_harness` + `coarse_harness`.
Non-obvious bits baked in:
- **EXIF orientation**: query stills load upright (`exif_transpose`), and the
  intrinsics are rotated + the GT pose is rolled 90deg to match (roll sign picked
  photometrically). Without this everything is ~90deg off.
- **COLMAP->nerf GT**: fitted empirically from the training frames
  (`fit_colmap_to_nerf`), with a self-check gate that aborts if the fit is bad.
- Per-query online timing added (GPU-synced) + torch GPU peak reported.

## HLoc baseline — `hloc_baseline/` (hloc env)
- `run_hloc.py` — 8-stage pipeline. `--query-gt <json> --query-subdir <dir>`
  triggers the external-query mode (stacey): `sparse/0` is the DB as-is (no
  deregister), query list+intrinsics from the json, feature globs widened to the
  subdir. (The no-`--query-gt` path is the held-out-eval mode for classroom.)
- `make_upright_queries.py` — writes EXIF-upright query copies + rotated-intrinsics
  json. **HLoc needs upright queries** (SuperPoint+LightGlue isn't
  rotation-invariant like COLMAP's SIFT; raw landscape → ~72% fail at ~90deg).
- `eval_hloc.py --query-gt <json> --query-subdir <dir> --scale <s> --upright` —
  scores vs COLMAP GT; translation is roll-invariant, rotation undoes the 90deg.
  (Known minor bug: `n_inliers` column reads 0 on upright runs — cosmetic, pose
  numbers are correct.)
- `time_hloc_online.py` — clean per-query online timing (warmed, GPU-synced).
- `bench_resources.sh` — peak GPU (torch + nvidia-smi) and RAM for both pipelines.
- Also clone + the vendored repo: `/home/chini/Desktop/3DGS/code/Hierarchical-Localization`.

## RESULTS (final — full detail in `EVALUATION.md`)
Same 4 GB card, stacey, 53 queries, COLMAP GT, nerf scene units.

| | Ours (MegaLoc+LoFTR) | HLoc |
|---|---|---|
| median trans / rot | 4.6 cm / 1.55 deg | **0.24 cm / 0.18 deg** |
| recall @5cm/5deg | 51% | **100%** |
| per-query online | **282 ms** | 878 ms |
| offline setup | **minutes** | hours |
| peak GPU (system) | 3.27 GB | **2.06 GB** |
| peak RAM | 2.39 GB | 2.36 GB |

**Trade-off / thesis story**: HLoc = more accurate + lighter VRAM, but slow heavy
offline SfM build and ~3x slower queries. Ours = ~3x faster per query + trivial
setup (localises straight against the splat), but less accurate + ~1.2 GB more
VRAM (splat stays resident). It is NOT a bust — we win on speed + deployment.

## Reproduce
```bash
# ours
conda run -n nerfstudio python query_eval.py
# hloc (map already built; to rebuild it's hours)
conda run -n hloc python hloc_baseline/run_hloc.py  --scene stacey_room-f2500 \
  --query-gt /home/chini/3dgs/data/stacey_room-f2500-processed/query_gt_d1280_up.json \
  --query-subdir query_d1280_up --resize-max 1024 --num-covis 20 --num-loc 20
conda run -n hloc python hloc_baseline/eval_hloc.py --scene stacey_room-f2500 \
  --query-gt .../query_gt_d1280_up.json --query-subdir query_d1280_up \
  --scale 0.2061375184571612 --upright
conda run -n hloc python hloc_baseline/time_hloc_online.py
bash hloc_baseline/bench_resources.sh
```

## Open threads / possible next steps
- **Queries saturate HLoc at 100%** (these stills overlap the dense capture). For
  a *discriminating* comparison, build a harder query set — wider baseline / the
  **50-frame video set** the user mentioned (needs registering into `sparse/0`
  via image_registrator to get GT, same shape as query_gt).
- Our weak link is **MegaLoc retrieval** (outliers IMG_2097/2101/2103 land outside
  the fine stage's capture radius). Worth analysing / improving the coarse stage.
- Other scenes exist (classroom_1/2, three_rooms); classroom_1 has an older HLoc
  held-out-eval run under `hloc_out/classroom_1-f1250/`.
- Interactive HLoc viewer was discussed and deprioritised (only needed for
  novel-view exploration, not the accuracy comparison).
- Fix the cosmetic `n_inliers=0` column in `eval_hloc.py` upright path if the
  report needs inlier counts.

## Working style (saved to memory `working-style-minimal-changes`)
Keep changes minimal / no bloat; **list intended edits before applying**; comments
in the user's voice (lowercase, casual, first-person); **verify by looking** at
images/outputs, don't hand-wave; they dislike the plan-mode approval ceremony and
multiple-choice question menus during discussion — talk in prose and act on a
clear "go".
