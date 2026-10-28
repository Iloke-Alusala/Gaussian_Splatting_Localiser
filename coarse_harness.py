# RUN COMMAND:
# conda run -n nerfstudio python coarse_harness.py --scene stacey --source real --retriever megaloc --radii-cm 10 25 50 --out runs/coarse_stacey

# Tests the coarse stage on its own: recall@N (is a top-N retrieved frame within R of the query) and how far the seed it hands to the fine stage is off.
# --source rendered perturbs train poses and renders fake queries, --source real uses the registered query photos. Results go to <out>/results.csv, plots and coarse_summary.txt

# Import Libraries
import argparse
import copy
import math
import os
import sys
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nerfstudio.cameras.cameras import Cameras
import pickle, types

from my_localizer import (
    get_eval_views, load_pipeline, pose_error, so3_exp,
    render_at_pose, to_4x4, freeze_map
)
from my_pose_harness import make_perturbation, perturb_pose
# reuse the pipeline's own (RAM-safe, cached) retriever + query_eval's scene paths
# and GT loading, so the coarse harness runs on the same scenes/queries as everything
# else and hits the existing megaloc index cache instead of rebuilding it.
import pipeline_retriever as pr
from query_eval import (
    SCENES, fit_colmap_to_nerf, colmap_to_nerf_c2w,
    load_upright, rot_intrinsics_cw, query_camera, pick_roll_sign,
)

# Build up system path
sys.path.append("/home/chini/Desktop/3DGS/code/EfficientLoFTR")
from src.loftr import LoFTR, full_default_cfg, reparameter
import torch.nn.functional as F

# Location of LOFTR Weights
ELOFTR_WEIGHTS = "/home/chini/Desktop/3DGS/code/EfficientLoFTR/weights/eloftr_outdoor.ckpt"
_ELOFTR = None

# safely loads PyTorch checkpoint files saved by PyTorch Lightning without requiring PyTorch Lightning to be installed in your environment
def _pl_tolerant_load(path):
    def _dummy(name):
        return type(name, (), {"__init__": lambda self, *a, **k: None, "__setstate__": lambda self, state: None})
    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.split(".")[0] in ("pytorch_lightning", "lightning", "lightning_fabric"): return _dummy(name)
            return super().find_class(module, name)
    shim = types.SimpleNamespace(__name__="pickle", Unpickler=_Unpickler, load=pickle.load, loads=pickle.loads, Pickler=pickle.Pickler, dump=pickle.dump, dumps=pickle.dumps, HIGHEST_PROTOCOL=pickle.HIGHEST_PROTOCOL)
    return torch.load(path, map_location="cpu", pickle_module=shim, weights_only=False)

# This function acts as a lazy-loading singleton provider for the EfficientLoFTR model. It builds, initializes, and reparameterizes the network exactly once and caches it in memory.
def get_eloftr(device="cuda"):
    global _ELOFTR
    if _ELOFTR is None:
        cfg = copy.deepcopy(full_default_cfg)
        m = LoFTR(config=cfg)
        ckpt = _pl_tolerant_load(ELOFTR_WEIGHTS)
        m.load_state_dict(ckpt["state_dict"])
        m = reparameter(m)
        _ELOFTR = m.eval().to(device)
    return _ELOFTR

# This function prepares an RGB image for EfficientLoFTR by converting it to grayscale, cropping its dimensions to a multiple of 32, and calculating the scale factors to map matches back to the original image grid
def _prep_gray(img_hwc, device, max_dim=640):
    # clamp the long side to max_dim before matching: full 1080x1920 trips a
    # CUBLAS fp16 gemm limit in EfficientLoFTR on the 4GB card, and it's what
    # pipeline_retriever already does, so the fine stage matches the proven path.
    t = img_hwc.permute(2, 0, 1)[:3].unsqueeze(0).float().to(device)
    _, _, H, W = t.shape
    scale = min(1.0, max_dim / max(H, W))
    Hc = int((H * scale) // 32) * 32
    Wc = int((W * scale) // 32) * 32
    w = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
    gray = (t * w).sum(1, keepdim=True)
    gray = F.interpolate(gray, size=(Hc, Wc), mode="bilinear", align_corners=False)
    return gray, (W / Wc, H / Hc)

# This function synthesizes an RGB image and an aligned depth map from the 3D Gaussian Splatting map at an arbitrary 4x4 camera pose
@torch.no_grad()
def render_rgb_and_depth(pipeline, camera, c2w_4x4: torch.Tensor):
    c2w_tensor = c2w_4x4[:3, :].unsqueeze(0) if c2w_4x4.ndim == 2 else c2w_4x4[..., :3, :]
    cam = Cameras(
        camera_to_worlds=c2w_tensor, fx=camera.fx, fy=camera.fy, 
        cx=camera.cx, cy=camera.cy, width=camera.width, height=camera.height, camera_type=camera.camera_type
    ).to(pipeline.device) # Force camera onto the GPU where the model lives
    
    outputs = pipeline.model.get_outputs(cam)
    return outputs["rgb"], outputs["depth"]

# The fine stage. Same maths as pipeline_retriever.efficient_loftr_localise, but also returns the match/inlier counts and timing for the CSV
@torch.no_grad()
def localise_pnp(pipeline, camera, query: torch.Tensor, T_seed: torch.Tensor, device="cuda"):
    """ Fine stage refactored into a single callable returning pose and diagnostics. """
    t_start = time.perf_counter()
    T_seed = T_seed.to(device).float()
    rendered_rgb, rendered_depth = render_rgb_and_depth(pipeline, camera, T_seed)
    
    matcher = get_eloftr(device)
    q_gray, (sxq, syq) = _prep_gray(query, device)
    r_gray, (sxr, syr) = _prep_gray(rendered_rgb, device)
    
    batch = {"image0": q_gray, "image1": r_gray}
    matcher(batch)
    mkpts_query = batch["mkpts0_f"].cpu().numpy() * np.array([sxq, syq], dtype=np.float32)
    mkpts_render = batch["mkpts1_f"].cpu().numpy() * np.array([sxr, syr], dtype=np.float32)
    
    # filled in as we go, so a failed query still records how far it got
    diagnostics = {"correspondences": len(mkpts_query), "inliers": 0, "time": 0.0, "success": False}
    
    if len(mkpts_query) < 4:
        diagnostics["time"] = time.perf_counter() - t_start
        return T_seed, diagnostics

    cx, cy = camera.cx.item(), camera.cy.item()
    fx, fy = camera.fx.item(), camera.fy.item()
    depth = rendered_depth.squeeze(-1).cpu().numpy()
    H, W = depth.shape

    # Lift every render keypoint to 3D with the splat depth. Skip matches that land off the image or on empty space (depth 0)
    pts_3d, pts_2d_query = [], []
    for (u_q, v_q), (u_r, v_r) in zip(mkpts_query, mkpts_render):
        ur, vr = int(round(u_r)), int(round(v_r))
        if not (0 <= ur < W and 0 <= vr < H): continue
        d = float(depth[vr, ur])
        if d <= 0: continue
        pts_3d.append([(u_r - cx) * d / fx, (v_r - cy) * d / fy, d])
        pts_2d_query.append([u_q, v_q])

    if len(pts_3d) < 4:
        diagnostics["time"] = time.perf_counter() - t_start
        return T_seed, diagnostics

    # RANSAC + EPnP, gives the query pose relative to the render camera
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.array(pts_3d, dtype=np.float32), np.array(pts_2d_query, dtype=np.float32),
        np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32), np.zeros(4, dtype=np.float32), flags=cv2.SOLVEPNP_EPNP
    )
    
    diagnostics["inliers"] = len(inliers) if inliers is not None else 0
    if not ok:
        diagnostics["time"] = time.perf_counter() - t_start
        return T_seed, diagnostics

    R_off, _ = cv2.Rodrigues(rvec)
    T_off = np.eye(4)
    T_off[:3, :3] = R_off
    T_off[:3, 3] = tvec.squeeze()

    # nerfstudio is OpenGL (y up, -z fwd) and PnP is OpenCV (y down, +z fwd). Flip into OpenCV, apply the offset, flip back
    M = np.diag([1.0, -1.0, -1.0, 1.0])
    T_seed_cv = T_seed.cpu().numpy().astype(np.float64) @ M
    T_query_cv = T_seed_cv @ np.linalg.inv(T_off)
    T_final = torch.tensor(T_query_cv @ M, dtype=torch.float32, device=device)
    
    diagnostics["success"] = True
    diagnostics["time"] = time.perf_counter() - t_start
    return T_final, diagnostics

def build_translation_ladder(train_poses, steps=4):
    """ Translation offsets for the rendered sweep, scaled to how spread out the train cameras are (0.1x to 2x their std dev) """
    positions = torch.stack([p[:3, 3] for p in train_poses])
    std_dev = torch.std(positions, dim=0).mean().item()
    return [std_dev * f for f in np.linspace(0.1, 2.0, steps)]

def ensure_no_leakage(train_ds, eval_ds):
    """ Hard stop if any eval image is also a train image, otherwise retrieval would just find the query itself """
    train_paths = {train_ds.image_filenames[i].name for i in range(len(train_ds))}
    eval_paths = {eval_ds.image_filenames[i].name for i in range(len(eval_ds))}
    overlap = train_paths.intersection(eval_paths)
    assert len(overlap) == 0, f"FATAL: Train/Eval overlap detected! {len(overlap)} images leak."

def run_zero_offset_assertion(pipeline, camera, retriever, train_img, train_pose):
    """ Sanity check before the sweep: a train image seeded at its own pose should localise back onto itself """
    print("  [verify] Running zero-offset assertion...")
    top_k = retriever.query(train_img, top_k=1)
    assert top_k[0][0] != -1, "Retriever failed on identical image."
    T_final, diag = localise_pnp(pipeline, camera, train_img, train_pose, device=pipeline.device)
    t_err, r_err = pose_error(train_pose.double(), T_final.cpu().double())
    # 0.03 not 0.01: even a perfect seed on a real train image lands ~0.02 units
    # off here, because the splat render isn't pixel-identical to the photo (the
    # splat-depth floor). still a tight sanity bound, well under real seed errors.
    assert t_err < 0.03 and r_err < 0.2, f"Zero-offset test failed. Error: t={t_err:.4f}, r={r_err:.4f}"
    print("  [verify] Zero-offset test passed.")

def plot_harness_results(csv_path, out_dir):
    """ Plots for --source rendered: recall, retrieval gap and margin against yaw offset, plus final vs seed error """
    data = {"yaw": [], "trans": [], "recall_1": [], "final_t_err": [], "seed_t_err": [], "ret_gap": [], "margin": []}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["yaw"].append(float(row["req_yaw"]))
            data["trans"].append(float(row["req_trans"]))
            data["final_t_err"].append(float(row["final_t_err"]))
            data["seed_t_err"].append(float(row["seed_t_err"]))
            data["ret_gap"].append(float(row["retrieval_gap"]))
            data["margin"].append(float(row["margin"]))
            data["recall_1"].append(1 if float(row["seed_t_err"]) < 0.5 and float(row["seed_r_err"]) < 10.0 else 0) # seed counts as a hit inside 0.5 units / 10 deg

    yaws = sorted(list(set(data["yaw"])))
    def get_stats(x_vals, x_key, y_key):
        meds, p25, p75 = [], [], []
        for x in x_vals:
            ys = [y for i, y in enumerate(data[y_key]) if data[x_key][i] == x]
            meds.append(np.median(ys) if ys else 0)
            p25.append(np.percentile(ys, 25) if ys else 0)
            p75.append(np.percentile(ys, 75) if ys else 0)
        return meds, p25, p75

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Recall@1 vs Yaw
    med, _, _ = get_stats(yaws, "yaw", "recall_1")
    axes[0,0].plot(yaws, med, marker='o', color='b')
    axes[0,0].set_title("Recall@1 vs Yaw Offset")
    axes[0,0].set_xlabel("Yaw Offset (deg)")
    axes[0,0].set_ylabel("Recall@1")
    axes[0,0].grid(True)

    # 2. Final Error vs Seed Error
    axes[0,1].scatter(data["seed_t_err"], data["final_t_err"], alpha=0.5)
    axes[0,1].plot([0, max(data["seed_t_err"])], [0, max(data["seed_t_err"])], 'r--')
    axes[0,1].set_title("Region of Convergence: Final vs Seed Translation Error")
    axes[0,1].set_xlabel("Seed Error (Scene Units)")
    axes[0,1].set_ylabel("Final Error (Scene Units)")
    axes[0,1].grid(True)

    # 3. Retrieval Gap vs Yaw
    med, p25, p75 = get_stats(yaws, "yaw", "ret_gap")
    axes[1,0].plot(yaws, med, marker='o', color='g')
    axes[1,0].fill_between(yaws, p25, p75, color='g', alpha=0.2)
    axes[1,0].set_title("Retrieval Gap vs Yaw Offset")
    axes[1,0].set_xlabel("Yaw Offset (deg)")
    axes[1,0].set_ylabel("Gap (Scene Units)")
    axes[1,0].grid(True)

    # 4. Margin vs Yaw
    med, p25, p75 = get_stats(yaws, "yaw", "margin")
    axes[1,1].plot(yaws, med, marker='o', color='m')
    axes[1,1].fill_between(yaws, p25, p75, color='m', alpha=0.2)
    axes[1,1].set_title("Top1-Top2 Similarity Margin vs Yaw")
    axes[1,1].set_xlabel("Yaw Offset (deg)")
    axes[1,1].set_ylabel("Margin")
    axes[1,1].grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "harness_plots.png"), dpi=150)
    plt.close(fig)

def print_summary_table(csv_path):
    """ Final pose success rate at the usual t/r thresholds, in scene units """
    thresholds = [(0.05, 5), (0.10, 5), (0.25, 2), (0.50, 10)]
    success_counts = {th: 0 for th in thresholds}
    total = 0
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            t_err, r_err = float(row["final_t_err"]), float(row["final_r_err"])
            for th in thresholds:
                if t_err <= th[0] and r_err <= th[1]:
                    success_counts[th] += 1
    
    print("\n" + "="*50)
    print("SUCCESS SUMMARY (Scene Units / Degrees)")
    print("="*50)
    for th in thresholds:
        pct = (success_counts[th] / total) * 100 if total > 0 else 0
        print(f" {th[0]:.2f}u / {th[1]:.0f}deg threshold: {pct:.1f}% ({success_counts[th]}/{total})")
    print("="*50)

# recall@N reported at several 'same place' radii, since no metric scale exists
# to pin one. 0.10 u is the measured fine-stage capture radius; 0.05/0.25 bracket
# it tight/loose so the thesis shows how 'correct' depends on the threshold.
RECALL_NS = (1, 5, 10)
RECALL_RADII = (0.05, 0.10, 0.25)
RECALL_LABELS = None            # cm strings, set when --radii-cm is used


def _train_centres(poses):
    """ N x 3 camera centres of the train frames """
    return torch.stack([p[:3, 3].float() for p in poses])


def _topk_fields(top_k, centres, query_centre):
    """per-query retrieval columns. recall@N at R = 'is any of the top-N retrieved
    frames within R of the query centre' (standard VPR recall)."""
    top_indices = [i for i, _ in top_k]
    sims = [s for _, s in top_k]
    margin = sims[0] - sims[1] if len(sims) > 1 else 0.0
    qc = query_centre.float()
    retr_d = (centres[top_indices] - qc).norm(dim=1)      # retrieved frame -> query
    dist_nearest = float((centres - qc).norm(dim=1).min())  # nearest train frame
    rec = {}
    for N in RECALL_NS:
        best = float(retr_d[:N].min())
        for R in RECALL_RADII:
            rec[(N, R)] = int(best < R)
    sim5 = (sims + [""] * 5)[:5]
    return top_indices, sims, margin, dist_nearest, rec, sim5


def plot_real_results(csv_path, out_dir):
    """the breaking-point curves: seed error vs distance from the nearest train
    camera. this is the P4 core deliverable - where whole-image retrieval stops
    giving the fine stage a seed inside its capture basin."""
    d, st, sr, ft, mg = [], [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            d.append(float(row["dist_nearest_train"]))
            st.append(float(row["seed_t_err"]))
            sr.append(float(row["seed_r_err"]))
            ft.append(float(row["final_t_err"]))
            mg.append(float(row["margin"]))
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax[0, 0].scatter(d, st, alpha=0.5); ax[0, 0].set(
        title="Seed translation error vs distance to nearest train camera",
        xlabel="dist to nearest train cam (units)", ylabel="seed t_err (units)")
    ax[0, 1].scatter(d, sr, alpha=0.5, c="g"); ax[0, 1].set(
        title="Seed rotation error vs distance to nearest train camera",
        xlabel="dist to nearest train cam (units)", ylabel="seed r_err (deg)")
    ax[1, 0].scatter(st, ft, alpha=0.5, c="m")
    lim = max(st + ft) if st else 1
    ax[1, 0].plot([0, lim], [0, lim], "r--"); ax[1, 0].set(
        title="Chaining: final vs seed translation error",
        xlabel="seed t_err (units)", ylabel="final t_err (units)")
    ax[1, 1].scatter(d, mg, alpha=0.5, c="orange"); ax[1, 1].set(
        title="Top1-Top2 margin vs distance to nearest train camera",
        xlabel="dist to nearest train cam (units)", ylabel="margin")
    for a in ax.flat:
        a.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "breaking_point.png"), dpi=150)
    plt.close(fig)


def _recall_summary(csv_path):
    """mean recall@N at each radius across the run, keyed (N, R)."""
    rows = list(csv.DictReader(open(csv_path)))
    out = {}
    for N in RECALL_NS:
        for R in RECALL_RADII:
            key = f"rec{N}_{R:g}"
            v = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
            out[(N, R)] = 100.0 * sum(v) / len(v) if v else 0.0
    return out


def _fmt_recall(rec):
    """pretty block: a row per N, a column per radius."""
    labels = RECALL_LABELS or [f"R={R:g}" for R in RECALL_RADII]
    head = "            " + "  ".join(l.rjust(8) for l in labels)
    lines = [head]
    for N in RECALL_NS:
        lines.append(f"  recall@{N:<2d}" + "  ".join(
            f"{rec[(N, R)]:6.1f}%" for R in RECALL_RADII))
    return "\n".join(lines)


def main():
    global RECALL_RADII, RECALL_LABELS
    torch.set_float32_matmul_precision('high')
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=list(SCENES), default=None,
                    help="use a baked-in scene (stacey|class1); mirrors query_eval")
    ap.add_argument("--config", type=Path, default=None,
                    help="override: load this config.yml instead of a --scene")
    ap.add_argument("--retriever", choices=["clip", "megaloc"], default="megaloc")
    ap.add_argument("--source", choices=["rendered", "real"], default="rendered")
    ap.add_argument("--base-poses", type=int, default=10)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--out", type=str, default="runs/clip_rendered")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--radii-cm", type=float, nargs="+", default=None,
                    help="score recall at these metric radii (cm) instead of the "
                         "default unit radii; converts cm->units via the scene's "
                         "m_per_unit so both scenes share one physical grid")
    args = ap.parse_args()

    args.out = os.path.abspath(args.out)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=4)

    if not args.scene and not args.config:
        raise SystemExit("[abort] pass --scene stacey|class1 (or --config for a one-off)")
    sc = SCENES[args.scene] if args.scene else None
    config = args.config or sc["config"]

    # metric radii: convert cm -> scene units with the door-frame scale, so recall@N
    # lands at real distances and both scenes share one physical grid
    if args.radii_cm:
        if not sc:
            raise SystemExit("[abort] --radii-cm needs a --scene (for m_per_unit)")
        mpu = sc["m_per_unit"]
        RECALL_RADII = tuple((cm / 100.0) / mpu for cm in args.radii_cm)
        RECALL_LABELS = [f"{cm:g}cm" for cm in args.radii_cm]

    _cfg, pipeline, scale = load_pipeline(config, "cuda")
    freeze_map(pipeline)
    dev = pipeline.device

    train_ds = pipeline.datamanager.train_dataset
    train_poses = [to_4x4(train_ds.cameras[i:i+1].camera_to_worlds[0].cpu())
                   for i in range(len(train_ds))]
    centres = _train_centres(train_poses)
    print(f"[info] {len(train_poses)} train frames, recall radii {RECALL_RADII} units")

    # reuse the pipeline's cached retriever: build_index streams from a generator
    # (RAM-safe) and the on-disk megaloc index is named {name}_n{N}, so this hits
    # the same cache query_eval already built instead of re-encoding every frame.
    pr.RETRIEVER_NAME = args.retriever
    pr.ROT90_CW = False
    pr._DATASET_NAME = sc["name"] if sc else config.parents[2].name
    retriever = pr.get_retriever(pipeline, device=dev)

    # zero-offset sanity: an identical train image retrieves itself and localises
    run_zero_offset_assertion(pipeline, train_ds.cameras[0:1], retriever,
                              train_ds[0]["image"], train_poses[0])

    csv_path = os.path.join(args.out, "results.csv")
    rec_cols = [f"rec{N}_{R:g}" for N in RECALL_NS for R in RECALL_RADII]
    fields = ["source", "retriever", "base_idx", "req_yaw", "req_trans", "req_pitch",
              "retrieved_idx", "sim_score", "margin", "retrieval_gap",
              "seed_t_err", "seed_r_err", "dist_nearest_train",
              *rec_cols,
              "sim1", "sim2", "sim3", "sim4", "sim5",
              "correspondences", "inliers", "final_t_err", "final_r_err",
              "coarse_time", "fine_time"]

    def _row(source, idx, yaw, trans, ret_idx, sim_score, margin, ret_gap,
             t_seed, r_seed, dist, rec, sim5, diag, t_fin, r_fin, ct):
        row = {"source": source, "retriever": args.retriever, "base_idx": idx,
               "req_yaw": yaw, "req_trans": trans, "req_pitch": 0,
               "retrieved_idx": ret_idx, "sim_score": sim_score, "margin": margin,
               "retrieval_gap": ret_gap, "seed_t_err": t_seed, "seed_r_err": r_seed,
               "dist_nearest_train": dist, "correspondences": diag["correspondences"],
               "inliers": diag["inliers"], "final_t_err": t_fin, "final_r_err": r_fin,
               "coarse_time": ct, "fine_time": diag["time"]}
        for (N, R), v in rec.items():
            row[f"rec{N}_{R:g}"] = v
        for j in range(5):
            row[f"sim{j+1}"] = sim5[j]
        return row

    # rendered sweep: rotate the query away from a train pose by each yaw, at one fixed translation step
    yaw_sweep = [0, 5, 10, 20, 30, 45, 60, 90, 135, 180]
    trans_sweep = build_translation_ladder(train_poses)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        if args.source == "rendered":
            base_indices = np.random.choice(len(train_ds), args.base_poses, replace=False)
            total_iters = len(base_indices) * len(yaw_sweep) * args.trials
            print(f"[sweep] rendered: ~{total_iters} iterations.")

            for b_idx in base_indices:
                gt_pose = train_poses[b_idx]
                cam = train_ds.cameras[b_idx:b_idx+1]
                for yaw in yaw_sweep:
                    for t in range(args.trials):
                        delta = make_perturbation(t_mag=trans_sweep[1], r_mag_deg=yaw,
                                                  generator=torch.Generator().manual_seed(args.seed + t))
                        T_query = perturb_pose(gt_pose, delta)
                        rgb, _ = render_rgb_and_depth(pipeline, cam, T_query)

                        t0 = time.perf_counter()
                        top_k = retriever.query(rgb, top_k=10)
                        coarse_time = time.perf_counter() - t0

                        # top-1 frame's pose is the seed. retrieval gap = how much worse it is than the best train frame we could have picked
                        ret_idx = top_k[0][0]
                        T_seed = train_poses[ret_idx]
                        _ti, _s, margin, dist, rec, sim5 = _topk_fields(
                            top_k, centres, T_query[:3, 3])
                        t_seed, r_seed = pose_error(T_query, T_seed)
                        best = min(pose_error(T_query, tp)[0] for tp in train_poses)
                        T_final, diag = localise_pnp(pipeline, cam, rgb, T_seed, device=dev)
                        t_fin, r_fin = pose_error(T_query.double(), T_final.cpu().double())

                        writer.writerow(_row("rendered", int(b_idx), yaw, trans_sweep[1],
                                             ret_idx, top_k[0][1], margin, t_seed - best,
                                             t_seed, r_seed, dist, rec, sim5, diag,
                                             t_fin, r_fin, coarse_time))
                        f.flush()
                        torch.cuda.empty_cache()

        else:
            # real photos = the registered query set (same reconstruction as the
            # splat, per the single-reconstruction rule). GT loaded exactly like
            # query_eval so coarse numbers line up with the end-to-end ones.
            Rm, s, t_off, resid = fit_colmap_to_nerf(pipeline, sc["scene_dir"])
            print(f"[info] colmap->nerf fit resid {resid:.4f} units")
            if resid > 0.02:
                raise SystemExit(f"[abort] colmap->nerf fit off ({resid:.3f} units)")
            qgt = json.loads(Path(sc["query_gt"]).read_text())
            print(f"[sweep] real: {len(qgt)} registered queries from {sc['eval_dir']}")
            roll = None
            for qname, e in qgt.items():
                img_path = Path(sc["eval_dir"]) / qname
                if not img_path.exists():
                    continue
                query, k = load_upright(img_path)
                fx, fy, cx, cy = (float(x) for x in e["params"][:4])
                fx, fy, cx, cy, W, H = rot_intrinsics_cw(fx, fy, cx, cy, e["width"], e["height"], k)
                qcam = query_camera(fx, fy, cx, cy, W, H, dev)
                gt_np = colmap_to_nerf_c2w(np.asarray(e["R_world_to_cam"]),
                                           np.asarray(e["t_world_to_cam"]), Rm, s, t_off)
                if k:
                    if roll is None:
                        roll = pick_roll_sign(pipeline, qcam, query, gt_np, k)
                    gt_np = gt_np @ np.linalg.matrix_power(roll, k)
                gt_pose = torch.from_numpy(gt_np).float()

                t0 = time.perf_counter()
                top_k = retriever.query(query, top_k=10)
                coarse_time = time.perf_counter() - t0

                ret_idx = top_k[0][0]
                T_seed = train_poses[ret_idx]
                _ti, _s, margin, dist, rec, sim5 = _topk_fields(
                    top_k, centres, gt_pose[:3, 3])
                t_seed, r_seed = pose_error(gt_pose, T_seed)
                best = min(pose_error(gt_pose, tp)[0] for tp in train_poses)
                T_final, diag = localise_pnp(pipeline, qcam, query, T_seed, device=dev)
                t_fin, r_fin = pose_error(gt_pose.double(), T_final.cpu().double())

                writer.writerow(_row("real", Path(qname).stem, 0, 0,
                                     ret_idx, top_k[0][1], margin, t_seed - best,
                                     t_seed, r_seed, dist, rec, sim5, diag,
                                     t_fin, r_fin, coarse_time))
                f.flush()
                torch.cuda.empty_cache()

    # plots, the summary table and the recall block (printed, saved and appended to the results log)
    (plot_real_results if args.source == "real" else plot_harness_results)(csv_path, args.out)
    print_summary_table(csv_path)
    rec = _recall_summary(csv_path)
    tag = args.scene or str(config)
    block = _fmt_recall(rec)
    note = ("note: recall radii are metric, converted cm->units via m_per_unit"
            if RECALL_LABELS else
            "note: errors are in nerfstudio scene units; no metric scale on this map")
    print("  recall@N at same-place radii:\n" + block)
    with open(os.path.join(args.out, "coarse_summary.txt"), "w") as fsum:
        fsum.write(f"coarse stage: {args.retriever} on {args.source} queries, scene={tag}\n")
        fsum.write("recall@N (rows) at same-place radius R (cols):\n")
        fsum.write(block + "\n" + note + "\n")
    # mirror the recall block into the run log sheet ($RESULTS_LOG), so coarse sits
    # with everything else; unset = the old RESULTS_LOG.md
    log_path = os.environ.get("RESULTS_LOG", "/home/chini/Desktop/3DGS/code/RESULTS_LOG.md")
    with open(log_path, "a") as fl:
        fl.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} | coarse | scene={tag} | "
                 f"retriever={args.retriever} | source={args.source}\n```\n"
                 + block + "\n" + note + "\n```\n")
    print(f"  [done] Output saved to {args.out}")

if __name__ == "__main__":
    main()