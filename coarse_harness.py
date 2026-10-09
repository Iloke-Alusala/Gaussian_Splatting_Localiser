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
from retrievers import ClipRetriever, MegaLocRetriever

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
def _prep_gray(img_hwc, device):
    t = img_hwc.permute(2, 0, 1)[:3].unsqueeze(0).float().to(device)
    _, _, H, W = t.shape
    w = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
    gray = (t * w).sum(1, keepdim=True)
    Hc, Wc = (H // 32) * 32, (W // 32) * 32
    gray = F.interpolate(gray, size=(Hc, Wc), mode="bilinear", align_corners=False)
    return gray, (W / Wc, H / Hc)

# This function synthesizes an RGB image and an aligned depth map from the 3D Gaussian Splatting map at an arbitrary 4x4 camera pose
@torch.no_grad()
def render_rgb_and_depth(pipeline, camera, c2w_4x4: torch.Tensor):
    c2w_tensor = c2w_4x4[:3, :].unsqueeze(0) if c2w_4x4.ndim == 2 else c2w_4x4[..., :3, :]
    cam = Cameras(
        camera_to_worlds=c2w_tensor, fx=camera.fx, fy=camera.fy, 
        cx=camera.cx, cy=camera.cy, width=camera.width, height=camera.height, camera_type=camera.camera_type
    ).to(pipeline.device) # FIX: Force camera onto the GPU where the model lives
    
    outputs = pipeline.model.get_outputs(cam)
    return outputs["rgb"], outputs["depth"]

# 
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
    
    diagnostics = {"correspondences": len(mkpts_query), "inliers": 0, "time": 0.0, "success": False}
    
    if len(mkpts_query) < 4:
        diagnostics["time"] = time.perf_counter() - t_start
        return T_seed, diagnostics

    cx, cy = camera.cx.item(), camera.cy.item()
    fx, fy = camera.fx.item(), camera.fy.item()
    depth = rendered_depth.squeeze(-1).cpu().numpy()
    H, W = depth.shape

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

    M = np.diag([1.0, -1.0, -1.0, 1.0])
    T_seed_cv = T_seed.cpu().numpy().astype(np.float64) @ M
    T_query_cv = T_seed_cv @ np.linalg.inv(T_off)
    T_final = torch.tensor(T_query_cv @ M, dtype=torch.float32, device=device)
    
    diagnostics["success"] = True
    diagnostics["time"] = time.perf_counter() - t_start
    return T_final, diagnostics

def build_translation_ladder(train_poses, steps=4):
    positions = torch.stack([p[:3, 3] for p in train_poses])
    std_dev = torch.std(positions, dim=0).mean().item()
    return [std_dev * f for f in np.linspace(0.1, 2.0, steps)]

def ensure_no_leakage(train_ds, eval_ds):
    train_paths = {train_ds.image_filenames[i].name for i in range(len(train_ds))}
    eval_paths = {eval_ds.image_filenames[i].name for i in range(len(eval_ds))}
    overlap = train_paths.intersection(eval_paths)
    assert len(overlap) == 0, f"FATAL: Train/Eval overlap detected! {len(overlap)} images leak."

def run_zero_offset_assertion(pipeline, camera, retriever, train_img, train_pose):
    print("  [verify] Running zero-offset assertion...")
    top_k = retriever.query(train_img, top_k=1)
    assert top_k[0][0] != -1, "Retriever failed on identical image."
    T_final, diag = localise_pnp(pipeline, camera, train_img, train_pose, device=pipeline.device)
    t_err, r_err = pose_error(train_pose.double(), T_final.cpu().double())
    assert t_err < 0.01 and r_err < 0.1, f"Zero-offset test failed. Error: t={t_err:.4f}, r={r_err:.4f}"
    print("  [verify] Zero-offset test passed.")

def plot_harness_results(csv_path, out_dir):
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
            data["recall_1"].append(1 if float(row["seed_t_err"]) < 0.5 and float(row["seed_r_err"]) < 10.0 else 0) # Adjust thresholds as needed

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

def main():
    torch.set_float32_matmul_precision('high')
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default="/home/chini/Desktop/3DGS/outputs/agri-s12/splatfacto/2026-09-16_203014/config.yml")
    ap.add_argument("--retriever", choices=["clip", "megaloc"], default="clip")
    ap.add_argument("--source", choices=["rendered", "real"], default="rendered")
    ap.add_argument("--base-poses", type=int, default=10)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--out", type=str, default="runs/clip_rendered")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out = os.path.abspath(args.out)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=4)

    os.chdir(args.config.resolve().parents[4])
    
    _cfg, pipeline, scale = load_pipeline(args.config, "cuda")
    freeze_map(pipeline)
    
    train_ds = pipeline.datamanager.train_dataset
    eval_ds = pipeline.datamanager.eval_dataset
    ensure_no_leakage(train_ds, eval_ds)

    train_poses = [to_4x4(train_ds.cameras[i:i+1].camera_to_worlds[0].cpu()) for i in range(len(train_ds))]
    train_imgs = [train_ds[i]["image"] for i in range(len(train_ds))]
    
    retriever = ClipRetriever() if args.retriever == "clip" else MegaLocRetriever()
    dataset_name = args.config.parents[2].name
    retriever.build_index(train_imgs, train_poses, cache_dir=".index_cache", dataset_name=dataset_name)

    # Assert Zero-offset on first train image
    run_zero_offset_assertion(pipeline, train_ds.cameras[0:1], retriever, train_imgs[0], train_poses[0])

    csv_path = os.path.join(args.out, "results.csv")
    fields = ["source", "retriever", "base_idx", "req_yaw", "req_trans", "req_pitch", 
              "retrieved_idx", "sim_score", "margin", "retrieval_gap", 
              "seed_t_err", "seed_r_err", "correspondences", "inliers", 
              "final_t_err", "final_r_err", "coarse_time", "fine_time"]

    yaw_sweep = [0, 5, 10, 20, 30, 45, 60, 90, 135, 180]
    pitch_sweep = [0, 5, 10, 20, 30]
    trans_sweep = build_translation_ladder(train_poses)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        if args.source == "rendered":
            base_indices = np.random.choice(len(train_ds), args.base_poses, replace=False)
            total_iters = len(base_indices) * len(yaw_sweep) * args.trials
            print(f"[sweep] Starting rendered sweep: ~{total_iters} iterations.")
            
            for b_idx in base_indices:
                gt_pose = train_poses[b_idx]
                cam = train_ds.cameras[b_idx:b_idx+1]
                
                # Combine sweeps to avoid deeply nested loops, focusing on yaw as primary variable per instructions
                for yaw in yaw_sweep:
                    for t in range(args.trials):
                        delta = make_perturbation(t_mag=trans_sweep[1], r_mag_deg=yaw, generator=torch.Generator().manual_seed(args.seed + t))
                        T_query = perturb_pose(gt_pose, delta)
                        
                        rgb, _ = render_rgb_and_depth(pipeline, cam, T_query)
                        
                        t0 = time.perf_counter()
                        top_k = retriever.query(rgb, top_k=2)
                        coarse_time = time.perf_counter() - t0
                        
                        ret_idx, sim_score = top_k[0]
                        margin = sim_score - top_k[1][1] if len(top_k) > 1 else 0.0
                        T_seed = train_poses[ret_idx]
                        
                        t_seed, r_seed = pose_error(T_query, T_seed)
                        
                        # Calculate retrieval gap (Seed Error - Best Possible Error)
                        best_possible_err = min([pose_error(T_query, tp)[0] for tp in train_poses])
                        ret_gap = t_seed - best_possible_err
                        
                        T_final, diag = localise_pnp(pipeline, cam, rgb, T_seed, device=pipeline.device)
                        t_fin, r_fin = pose_error(T_query.double(), T_final.cpu().double())

                        writer.writerow({
                            "source": "rendered", "retriever": args.retriever, "base_idx": b_idx,
                            "req_yaw": yaw, "req_trans": trans_sweep[1], "req_pitch": 0,
                            "retrieved_idx": ret_idx, "sim_score": sim_score, "margin": margin,
                            "retrieval_gap": ret_gap, "seed_t_err": t_seed, "seed_r_err": r_seed,
                            "correspondences": diag["correspondences"], "inliers": diag["inliers"],
                            "final_t_err": t_fin, "final_r_err": r_fin,
                            "coarse_time": coarse_time, "fine_time": diag["time"]
                        })
                        f.flush()
                        torch.cuda.empty_cache()
                        
        else:
            print(f"[sweep] Starting real sweep on {len(eval_ds)} eval images.")
            for i in range(len(eval_ds)):
                gt_pose = to_4x4(eval_ds.cameras[i:i+1].camera_to_worlds[0].cpu())
                cam = eval_ds.cameras[i:i+1]
                rgb = eval_ds[i]["image"]
                
                t0 = time.perf_counter()
                top_k = retriever.query(rgb, top_k=2)
                coarse_time = time.perf_counter() - t0
                
                ret_idx, sim_score = top_k[0]
                margin = sim_score - top_k[1][1] if len(top_k) > 1 else 0.0
                T_seed = train_poses[ret_idx]
                
                t_seed, r_seed = pose_error(gt_pose, T_seed)
                best_possible_err = min([pose_error(gt_pose, tp)[0] for tp in train_poses])
                ret_gap = t_seed - best_possible_err
                
                T_final, diag = localise_pnp(pipeline, cam, rgb, T_seed, device=pipeline.device)
                t_fin, r_fin = pose_error(gt_pose.double(), T_final.cpu().double())
                
                writer.writerow({
                    "source": "real", "retriever": args.retriever, "base_idx": i,
                    "req_yaw": 0, "req_trans": 0, "req_pitch": 0,
                    "retrieved_idx": ret_idx, "sim_score": sim_score, "margin": margin,
                    "retrieval_gap": ret_gap, "seed_t_err": t_seed, "seed_r_err": r_seed,
                    "correspondences": diag["correspondences"], "inliers": diag["inliers"],
                    "final_t_err": t_fin, "final_r_err": r_fin,
                    "coarse_time": coarse_time, "fine_time": diag["time"]
                })
                f.flush()
                torch.cuda.empty_cache()

    plot_harness_results(csv_path, args.out)
    print_summary_table(csv_path)
    print(f"  [done] Output saved to {args.out}")

if __name__ == "__main__":
    main()