# Localisation evaluation harnesses

Two harnesses that localise the **same query photos** against the **same scene**
and score against the **same COLMAP ground truth**, so our 3DGS pipeline and the
HLoc baseline are directly comparable.

- **Ours** — `query_eval.py` : MegaLoc retrieval -> render RGB+depth from the
  splat -> EfficientLoFTR match -> PnP. Runs in the **nerfstudio** env.
- **Baseline** — `hloc_baseline/` : HLoc (SuperPoint + LightGlue + PnP against a
  triangulated COLMAP cloud). Runs in the **hloc** env.

Default scene: `stacey_room-f2500`. Query set: the 53 registered stills in
`images/query_d1280/` (1280-long-edge), GT in `query_gt_d1280.json` (COLMAP world
frame). Hot-swap either harness onto a different query set by pointing it at a new
image folder + query_gt json.

Errors are reported in **nerfstudio scene units** on both sides so they line up
(HLoc's COLMAP-unit error is multiplied by the dataparser scale ~0.206). Whether
a scene unit is literally a metre depends on the capture; our `POSE_THRESHOLDS`
treat 1 unit == 1 m (-> cm).

---

## Ours: `query_eval.py`  (nerfstudio env)

```bash
conda run -n nerfstudio python query_eval.py
# or point it elsewhere:
conda run -n nerfstudio python query_eval.py \
    --config   /home/chini/3dgs/outputs/<scene>/splatfacto/<ts>/config.yml \
    --query-dir /home/chini/3dgs/data/<scene>-processed/images/query_d1280 \
    --query-gt  /home/chini/3dgs/data/<scene>-processed/query_gt_d1280.json \
    --out-dir  query_eval_out --retriever megaloc
```

What it does: fits the COLMAP->nerf transform off the training frames (aborts if
the fit is bad), then per query runs coarse+fine and writes, into `--out-dir`:
`<name>_megaloc_report.txt` / `_summary.png` (query | GT render | final + diffs) /
`_link.png` (LoFTR matches), plus `summary_megaloc.txt` (median cm/deg + recall).

Note: query stills are stored landscape with an EXIF rotate flag; the harness
loads them upright and rotates the intrinsics + rolls the GT pose to match, so
the query, the render and the GT all sit the same way up.

---

## Baseline: HLoc  (hloc env)

One-time env (pycolmap **must** be 3.13.0, not 4.x - see
[hloc_baseline/README.md](hloc_baseline/README.md)):

```bash
conda create -n hloc python=3.10 -y && conda activate hloc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
cd Hierarchical-Localization && pip install -e . && pip install "pycolmap==3.13.0"
```

Run + score (external query set = stacey stills; `sparse/0` is the DB as-is):

```bash
cd /home/chini/Desktop/3DGS/code
conda run -n hloc python hloc_baseline/run_hloc.py  --scene stacey_room-f2500 \
    --query-gt /home/chini/3dgs/data/stacey_room-f2500-processed/query_gt_d1280.json \
    --query-subdir query_d1280 --resize-max 1024 --num-covis 20 --num-loc 20

conda run -n hloc python hloc_baseline/eval_hloc.py --scene stacey_room-f2500 \
    --query-gt /home/chini/3dgs/data/stacey_room-f2500-processed/query_gt_d1280.json \
    --query-subdir query_d1280 --scale 0.2061375184571612
```

Outputs land in `/home/chini/3dgs/hloc_out/stacey_room-f2500/hloc/`:
`results.txt` (poses), `results.csv` (per query: `query,t_err,r_err,n_inliers,
time_ms,t_err_ns`), `summary.txt`. `t_err` is COLMAP units; `t_err_ns` is nerf
units (== ours). HLoc runs on the **raw** images (no EXIF rotation) - it never
renders the splat, so it stays self-consistent in COLMAP's frame.

The held-out-eval variant (classroom scenes, queries drawn from the training
split) still works via `make_query_set.py` + `run_hloc.py` with no `--query-gt`;
see [hloc_baseline/README.md](hloc_baseline/README.md).

---

## Results & comparison (stacey_room-f2500, 53 query stills, 4 GB laptop GPU)

Same queries, same COLMAP GT, errors in nerfstudio scene units. HLoc was run
EXIF-upright (`make_upright_queries.py`) so SuperPoint matches like-with-like;
our pipeline loads the queries upright for the same reason.

**Accuracy** (median; recall at translation/rotation thresholds)

| | median trans | median rot | 5cm/5deg | 10cm/5deg | 50cm/10deg | localised |
|---|---|---|---|---|---|---|
| Ours, 1 pass | 4.62 cm | 1.55 deg | 51% | 75% | 91% | 53/53 |
| Ours, **2 passes** (iterative) | **0.91 cm** | **0.36 deg** | **92%** | 94% | 96% | 53/53 |
| **HLoc** (SP+LightGlue) | **0.24 cm** | **0.18 deg** | **100%** | **100%** | **100%** | 53/53 |

Iterating the fine stage (`--passes 2`, iComMa-style re-render + re-PnP) closes
most of the accuracy gap for +162 ms (447 ms/query, still ~2x faster than HLoc).
It plateaus at ~0.87 cm by pass 3 — the splat-depth floor. See the ablation in
`README.md`.

**Per-query online time** (GPU-synced, warmed up; `time_hloc_online.py` /
`query_eval.py` summary)

| | time/query |
|---|---|
| **Ours** (MegaLoc + render + LoFTR + PnP) | **282 ms** |
| HLoc full online | 878 ms |

Both are warm, steady-state, GPU-synced. Ours is the median over 53 queries
(warm mean = 284 ms; the one cold first query, 3.76 s of model load, is excluded;
warm spread 264-337 ms). HLoc warms each model once then times fresh runs. Ours
is a per-query loop; HLoc's stages are batched (minor throughput edge to HLoc).

| &nbsp;&nbsp;SuperPoint extract | 50 ms |
| &nbsp;&nbsp;NetVLAD extract | 174 ms |
| &nbsp;&nbsp;retrieval (NN) | 18 ms |
| &nbsp;&nbsp;LightGlue match (20 db candidates) | 571 ms |
| &nbsp;&nbsp;localize_sfm (PnP) | 66 ms |

**Offline / setup cost**

| | setup |
|---|---|
| **Ours** | MegaLoc index: encode 2500 train imgs once, cached. **~minutes.** |
| HLoc | 2500-img features + ~50k-pair LightGlue matching + triangulation. **~hours.** |

**Peak resource usage** (same 4 GB card; `bench_resources.sh`)

| | peak GPU (torch alloc / system) | peak RAM (RSS) |
|---|---|---|
| Ours (MegaLoc+LoFTR) | 2.49 / **3.27 GB** | 2.39 GB |
| HLoc (full online) | 1.44 / **2.06 GB** | 2.36 GB |

System GPU includes the CUDA context. HLoc online is lighter on VRAM; our
pipeline must keep the full 3DGS splat resident to render, so it uses ~1.2 GB
more (≈80% vs ≈51% of the 4 GB card). RAM is comparable. (HLoc's *offline* build
is the heavier phase but is one-time; its per-op GPU peak is similar to online.)

### Takeaway for the report
- **HLoc is more accurate** (~20x translation, ~9x rotation) — expected, it's the
  mature structure-based baseline.
- **Our pipeline is ~3x faster per query** (282 vs 878 ms): it does one render +
  one dense match, where HLoc matches each query against 20 retrieved db frames
  (the 571 ms that dominates its cost).
- **Our setup is far cheaper** — no SfM triangulation, no pairwise DB matching;
  it localises directly against the splat.
- **HLoc is lighter on VRAM** (~2.1 vs ~3.3 GB): the splat must stay resident for
  us to render, whereas HLoc holds a sparse cloud + feature nets. RAM is a tie.
- Net trade-off: **HLoc** = higher accuracy + lower VRAM, at the cost of a slow,
  heavy offline build and ~3x slower queries. **Ours** = ~3x faster queries +
  trivial setup, at the cost of accuracy and ~1.2 GB more VRAM.

Caveats to state: both ran resolution-limited at 1024 px on a 4 GB GPU; HLoc's
per-query cost scales with the retrieval count `k` (k=20 here; smaller k is
faster but usually less accurate), whereas ours is fixed at one render+match.

### Reproduce the timings
```bash
conda run -n hloc python hloc_baseline/time_hloc_online.py      # -> hloc/timing_online.txt
conda run -n nerfstudio python query_eval.py                    # -> query_eval_out/summary_megaloc.txt
```

  overleaf_session2=s%3Ai3oIwfP23zLYseA6lZiKA45tCNYdHe8u.466OmccKTt8kKpevl%2B0v4cYTRV4QwxVks5B%2F51bGZNU