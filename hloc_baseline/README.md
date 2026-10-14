# HLoc baseline

Hierarchical Localization (SuperPoint + LightGlue + PnP against a sparse COLMAP
cloud) as the structure-based reference baseline for our 3DGS pose localiser.
The point of the baseline is that HLoc and our pipeline sit on **identical camera
poses and identical ground truth**, so any difference in pose error comes from
the method, not the map.

## How this differs from `HLOC_SETUP.md`

The brief assumes a `stacey_room-f2500` scene with separately-registered high-res
query stills and a `query_gt.json` produced by `register_queries.sh`. **None of
that exists on this machine.** So the query set is built a different (equivalent)
way:

- **Queries = the nerfstudio eval split.** Our splat trains with
  `eval_mode: fraction`, `train_split_fraction: 0.9`, holding out ~10% of frames
  as eval. Those same frames (112 of them for `classroom_1-f1250`) are the
  queries for *both* methods, so the comparison is apples-to-apples. They were
  **not seen during splat training**.
- **Ground truth = their COLMAP poses**, read straight from
  `data/<scene>/colmap/sparse/0/images.bin`. No `image_registrator` /
  bundle-adjustment step is needed — these frames are already registered.
- **Reference model = `sparse/0` minus the query frames** (`create_reference_sfm`),
  so HLoc triangulates a SuperPoint cloud on the database poses only and cannot
  read a query pose out of its own map. We **never** run `hloc.reconstruction`
  (from-scratch SfM); that would change the coordinate frame and void the
  comparison.

## Environment

Separate conda env `hloc` (do **not** install into `nerfstudio`):

```bash
conda create -n hloc python=3.10 -y
conda activate hloc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
cd Hierarchical-Localization && pip install -e .   # pulls pycolmap, lightglue, kornia
pip install "pycolmap==3.13.0"                     # REQUIRED pin, see below
```

Verified versions: **pycolmap 3.13.0**, torch 2.5.1+cu121, LightGlue (git), CUDA 12.1.
GPU: RTX 500 Ada Laptop, **4 GB** — the binding constraint.

> **pycolmap must be 3.13.0, not 4.x.** hloc only declares `pycolmap>=3.13.0`, so
> a fresh `pip install -e .` pulls 4.2.1. Under 4.x the triangulation database
> linkage silently breaks — `create_db_from_model` → `write_keypoints` →
> `run_triangulation` leaves every image "seeing 0/0 points", so the SfM model
> comes out with **0 points3D** and every query fails to localise. 3.13.0 is the
> version hloc's current code actually targets. Our `create_reference_sfm` uses
> `deregister_frame` / `has_pose`, which exist in 3.13.0 too.

## Resolution (resolution-limited baseline — state this in the report)

Feature extraction long edge is `--resize-max`, **default 1024**, set for the
4 GB GPU. Native frames are 720×1280, so 1024 downsizes the long edge. Drop to
`--resize-max 640` if CUDA OOMs. The value used for a run is recorded in
`hloc/timing.txt` context and should be quoted in the report: the baseline ran
resolution-limited, not at its best.

## Run order

All outputs go to `/home/chini/3dgs/hloc_out/<scene>/`; nothing under
`data/*/colmap/` is ever modified.

```bash
# 1. build the held-out query set + GT   (nerfstudio env)
conda run -n nerfstudio python make_query_set.py --scene classroom_1-f1250
#    -> query_gt.json, query_list.txt, db_list.txt, scale.json

# 2. (optional) plumbing sanity: a few db frames as queries vs the FULL model,
#    recovered poses must be ~0 error                              (hloc env)
conda run -n hloc python run_hloc.py  --scene classroom_1-f1250 --roundtrip 5
conda run -n hloc python eval_hloc.py --scene classroom_1-f1250 --roundtrip

# 3. the real baseline                                             (hloc env)
conda run -n hloc python run_hloc.py  --scene classroom_1-f1250
conda run -n hloc python eval_hloc.py --scene classroom_1-f1250
#    -> hloc/results.csv, hloc/summary.txt
```

## Output

`hloc/results.csv`, one row per query, brief's shape plus a converted column:

| column | meaning |
|---|---|
| `query` | frame name |
| `t_err` | translation error in **COLMAP world units** (‖C_est − C_gt‖, C = −Rᵀt) |
| `r_err` | rotation error in degrees (angle of R_est R_gtᵀ) |
| `n_inliers` | PnP inliers from `localize_sfm` |
| `time_ms` | **amortized** localise time per query (localize_sfm keeps no per-query timer; = wall / N, see `timing.txt`) |
| `t_err_ns` | `t_err × dataparser_scale` → **nerfstudio units**, directly comparable to our pipeline |

`scale` for `classroom_1-f1250` is ≈ 0.137 (COLMAP units × scale = nerfstudio
units). HLoc internals stay in the COLMAP world frame per the brief; only the
reported `t_err_ns` column is rescaled for comparison. Our own pipeline's
dataparser applies an additional rotation/recentre — that is *our* side's
concern and is **not** applied to HLoc output.

## Verification before trusting numbers

1. **Triangulation sanity** (end of `run_hloc` stage 5): registered-image count
   ≈ db size (~1014) and mean reprojection error in a sane range. Wildly off ⇒
   triangulation ignored our poses.
2. **Round-trip** (step 2 above): db frames localised against the full model must
   come back at ~0 error. Confirms coordinate conventions + error math.
3. Only then trust the real query set (step 3).

## Files

| file | env | role |
|---|---|---|
| `make_query_set.py` | nerfstudio | held-out query set + COLMAP GT + scale |
| `run_hloc.py` | hloc | the 8-stage HLoc pipeline → `results.txt` |
| `eval_hloc.py` | hloc | scoring → `results.csv` + `summary.txt` |
