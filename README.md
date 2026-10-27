# 3DGS camera localisation + HLoc baseline

Localise a camera inside a **3D Gaussian Splatting** map and benchmark it against
the established **HLoc** baseline. Final-year thesis code.

**Our pipeline:** MegaLoc retrieval (coarse) -> seed pose = nearest train frame ->
render RGB+depth from the splat -> EfficientLoFTR match -> depth-lift ->
`solvePnPRansac`, iterated with `--passes N`.
**Baseline:** HLoc = SuperPoint + LightGlue + NetVLAD against a sparse SfM cloud
(`num_loc=20`, `resize_max=1024`, `pycolmap==3.13.0`).

## Scenes (names used throughout)
| name | what | queries |
|---|---|---|
| **class1** | a classroom (splat `class1_train-f2900`) | 104 |
| **stacey** (a.k.a. Office) | the office (splat `stacey_room_train-f4200`) | 159 |
| **class2** | a second classroom (`classroom_2-f1950`) | retrieval only |

Metric conversion (door-frame scale): **class1 x 3.5601 m/unit, stacey x 1.9826 m/unit**.
All "cm"/"deg" in the results are real metric unless a file says "scene units".

## Map of the repo
| path | what |
|---|---|
| `pipeline_retriever.py` | the pipeline (interactive viewer + coarse/fine stages) |
| `my_localizer.py` | fine stage: EfficientLoFTR + PnP, and the photometric (LPIPS) refiner |
| `retrievers.py` | swappable coarse retrievers (MegaLoc, CLIP) + cached indices |
| `my_pose_harness.py`, `coarse_harness.py` | pose maths and eval helpers |
| `query_eval.py` | run the pipeline over a query set, no viewer (nerfstudio env) |
| `capture_radius_metric.py` | perturbation / capture-radius basin sweeps |
| `multiroom_retrieval.py` | multi-room room-ID test (pool 3 rooms, top-1 correct room) |
| `measure_splat_resources.py` | splat size / gaussian count / VRAM |
| `mse_vs_lpips_ab.py` | photometric loss ablation (MSE vs LPIPS) |
| `render_gt_pairs.py`, `render_failure_cases.py` | figure renders |
| `make_report_tables.py` | regenerate `report_tables.md` from the logs |
| `interactive_localizer_LOFTR.py` | standalone interactive LoFTR localiser |
| `hloc_baseline/` | HLoc baseline + `sweep_num_loc.py`, timing, resource bench |
| **`RESULTS_AUDIT/`** | **the evidence archive** — one folder per experiment, each with a log copy + plain README. Start at `RESULTS_AUDIT/README.md` |
| `RESULTS_LOG.md` / `RESULTS_METRIC.md` | master append logs (scene-unit / metric) |
| `RESULTS_MULTIROOM.md`, `report_tables.md` | multi-room result, head-to-head tables |

## Quick start (nerfstudio env for ours, hloc env for baseline)
```bash
conda run -n nerfstudio python query_eval.py --scene stacey --passes 2   # ours -> query_eval_out/
conda run -n hloc      python hloc_baseline/run_hloc.py                  # baseline
conda run -n hloc      python hloc_baseline/time_hloc_online.py          # per-query timing
bash hloc_baseline/bench_resources.sh                                    # peak GPU + RAM
```

## Headline (metric, Office/stacey, n=159, single 4 GB GPU)
- **Accuracy @10cm/5deg:** ours 2-pass **87.4%** vs HLoc **72.3%** (McNemar p=0.0004).
  HLoc has a tiny *median* error but a heavy outlier tail; ours is more robust at
  this threshold. At the tight 5cm/5deg, ours 79.9% vs HLoc 71.1% (p=0.08).
- **Speed:** ours ~220 ms/query (coarse+fine) vs HLoc ~590 ms, with a near-trivial
  setup (minutes, no SfM triangulation).
- **Fine-stage recovery** (10cm/5deg seed perturbation): EfficientLoFTR collapses
  error to **2-3 cm** (full recall); photometric LPIPS only to ~5-7 cm.
- **Multi-room room-ID** (3-room, 7854-frame bank): MegaLoc **98.7%** top-1 correct
  room vs CLIP 87.7%.

Exact numbers, provenance and the honest data gaps are in `RESULTS_AUDIT/`.

## Ablations (see RESULTS_AUDIT for full tables)
- **Iterative fine stage** (`--passes N`): one extra pass is transformative, then
  plateaus at the splat-depth floor. **2 passes is the operating point.**
- **MSE vs LPIPS** photometric loss (`mse_vs_lpips_ab.py`).
- **num_loc sweep** for HLoc (`hloc_baseline/sweep_num_loc.py`) justifies num_loc=20.
- **Rotation-only capture radius** and **render-vs-render** upper bound (folders
  09/10 of the audit).
