import argparse
import copy
import math
import os
import sys
import csv                     # NEW: writing the sweep results table
from pathlib import Path

import cv2
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras

# 3DGS Harness Imports
from my_localizer import (
    get_eval_views,
    load_pipeline,
    pose_error,
    so3_exp,
    render_at_pose,
    to_4x4,
    freeze_map
)

#perturbation helpers for the sweep. These live in my_pose_harness (the
# diagnostic). If yours are somewhere else, change this import to match.
from my_pose_harness import make_perturbation, perturb_pose

# NEW: swappable coarse-stage retrievers. Both satisfy the same interface, so
# the only thing that changes between runs is which one gets constructed.
from retrievers import ClipRetriever, MegaLocRetriever

# Adjust the path to wherever you cloned the repo
sys.path.append("/home/chini/Desktop/3DGS/code/EfficientLoFTR")

from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

import torch.nn.functional as F

ELOFTR_WEIGHTS = "/home/chini/Desktop/3DGS/code/EfficientLoFTR/weights/eloftr_outdoor.ckpt"
_ELOFTR = None

DIAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loftr_diag")
os.makedirs(DIAG_DIR, exist_ok=True)

# ============================================================================
# NEW: SWAPPABLE COARSE STAGE (CLIP or MegaLoc)
# ----------------------------------------------------------------------------
# Chosen with --retriever {clip,megaloc}. Only the coarse stage changes, so a
# CLIP run and a MegaLoc run see identical queries, identical seeds-from-poses
# and an identical fine stage. That is what makes the two comparable.
#
# The index is built once per process and cached to disk. The cache key carries
# the retriever name AND the training image count, so a MegaLoc run can never
# silently load CLIP vectors, and a retrain with a different split cannot
# silently load a stale index.
# ============================================================================
RETRIEVER_NAME = "clip"
INDEX_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".index_cache")
_RETRIEVER = None
_DATASET_NAME = "dataset"


def get_retriever(pipeline, device="cuda"):
    """Build the chosen retriever and its index once, then cache it."""
    global _RETRIEVER
    if _RETRIEVER is not None:
        return _RETRIEVER

    if RETRIEVER_NAME == "clip":
        r = ClipRetriever(device=device)
    elif RETRIEVER_NAME == "megaloc":
        r = MegaLocRetriever(device=device)
    else:
        raise ValueError(f"Unknown retriever '{RETRIEVER_NAME}'. Use clip or megaloc.")

    train_ds = pipeline.datamanager.train_dataset
    imgs = (train_ds[i]["image"] for i in range(len(train_ds)))
    poses = [to_4x4(train_ds.cameras[i:i + 1].camera_to_worlds[0].cpu())
             for i in range(len(train_ds))]

    # image count in the key so a changed split invalidates the cache
    r.build_index(imgs, poses,
                  cache_dir=INDEX_CACHE_DIR,
                  dataset_name=f"{_DATASET_NAME}_n{len(poses)}")
    _RETRIEVER = r
    return _RETRIEVER

# the perturbation grid for the capture-radius sweep.
# Each pair is (translation magnitude in SCENE UNITS, rotation magnitude in DEG).
# Your seed of 0.39 units / 8.4 deg already succeeded, so this range pushes well
# past it to find where EfficientLoFTR stops matching and the fine stage breaks.
# If the curve never peels away from y=x (i.e. it always recovers), widen the top
# end. If it breaks immediately, halve the whole range.
SWEEP_MAGNITUDES = [
    (0.01, 0.5),
    (0.05, 2.0),
    (0.10, 5.0),
    (0.20, 10.0),
]

# ============================================================================
# NEW: 90 DEGREE CLOCKWISE VIEW ROTATION
# ----------------------------------------------------------------------------
# For datasets captured rotated 90 degrees anti-clockwise. This is a VIEW-ONLY
# correction: every image written to disk or shown on screen is rotated 90 deg
# clockwise so it reads upright.
#
# It deliberately does NOT touch the matching or pose path. The splat was
# trained on the rotated images, so renders and queries are already in the same
# rotated frame and match each other correctly. Rotating the tensors fed to
# EfficientLoFTR would also require swapping fx/fy, cx/cy, width/height and
# inverse-rotating the keypoints before the depth lift, and any slip there
# silently corrupts PnP. Nothing geometric changes here.
#
# Toggled by --rot90 on the command line.
# ============================================================================
ROT90_CW = False


def _rot90_img(a: np.ndarray) -> np.ndarray:
    """Rotate an HxWx3 uint8 image 90 deg clockwise when the flag is on."""
    if not ROT90_CW:
        return a
    return cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE)


def _rot90_pts(pts, h_orig: int) -> np.ndarray:
    """
    Map (x, y) keypoints from the original frame into the 90 deg clockwise
    rotated frame, so overlays still land on the right pixels.
    A point (x, y) in an image of height H maps to (H - 1 - y, x).
    """
    p = np.asarray(pts, dtype=np.float32)
    if not ROT90_CW or len(p) == 0:
        return p
    return np.stack([(h_orig - 1) - p[:, 1], p[:, 0]], axis=1)


def _to_bgr_u8(img, rotate: bool = True):
    a = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    a = (a[..., :3] * 255).clip(0, 255).astype(np.uint8)
    a = a[..., ::-1].copy()             # RGB -> BGR
    return _rot90_img(a) if rotate else a

def _label(img, text):
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return img

def draw_loftr_matches(query_rgb, render_rgb, kpts_q, kpts_r, out_path, max_draw=200):
    """Side-by-side query|render with matched points joined. kpts in native px (x,y)."""
    # Convert WITHOUT rotation first so the original height is known, then
    # rotate both the images and the keypoints together.
    a_raw, b_raw = _to_bgr_u8(query_rgb, rotate=False), _to_bgr_u8(render_rgb, rotate=False)
    Ha, Hb = a_raw.shape[0], b_raw.shape[0]
    a, b = _rot90_img(a_raw), _rot90_img(b_raw)
    kpts_q = _rot90_pts(kpts_q, Ha)
    kpts_r = _rot90_pts(kpts_r, Hb)

    H = max(a.shape[0], b.shape[0])
    canvas = np.zeros((H, a.shape[1] + b.shape[1], 3), np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1]:] = b
    off = a.shape[1]
    n = len(kpts_q)
    sel = np.random.choice(n, min(n, max_draw), replace=False) if n > max_draw else range(n)
    for i in sel:
        p0 = (int(kpts_q[i][0]), int(kpts_q[i][1]))
        p1 = (int(kpts_r[i][0]) + off, int(kpts_r[i][1]))
        c = tuple(int(v) for v in np.random.randint(60, 255, 3))
        cv2.line(canvas, p0, p1, c, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, c, -1)
        cv2.circle(canvas, p1, 2, c, -1)
    _label(canvas, f"{n} matches   (left: query, right: render@T_init)")
    cv2.imwrite(out_path, canvas)

def _render_rgb(pipeline, camera, pose):
    rgb, _ = render_rgb_and_depth(pipeline, camera, pose.to(pipeline.device).float())
    return _to_bgr_u8(rgb)

def save_diagnostic_panel(pipeline, camera, query, gt_pose, T_init, T_final, out_path):
    """Top: query | GT render | final render.  Bottom: initial render | diff(init) | diff(final)."""
    q    = _to_bgr_u8(query)
    gt   = _render_rgb(pipeline, camera, gt_pose)
    init = _render_rgb(pipeline, camera, T_init)
    fin  = _render_rgb(pipeline, camera, T_final)
    diff_i = np.abs(q.astype(int) - init.astype(int)).astype(np.uint8)
    diff_f = np.abs(q.astype(int) - fin.astype(int)).astype(np.uint8)
    row1 = np.hstack([_label(q.copy(), "QUERY"), _label(gt.copy(), "GT render"), _label(fin.copy(), "FINAL render")])
    row2 = np.hstack([_label(init.copy(), "INIT render"), _label(diff_i.copy(), "diff QUERY-INIT"), _label(diff_f.copy(), "diff QUERY-FINAL")])
    cv2.imwrite(out_path, np.vstack([row1, row2]))

import pickle, types

def _pl_tolerant_load(path):
    """
    Load a Lightning-saved .ckpt without pytorch_lightning installed.
    Any pytorch_lightning/lightning global in the pickle is replaced with a
    throwaway dummy. We only read ['state_dict'] (plain tensors), so the
    dummies are discarded. Safe because these are your own downloaded weights.
    """
    def _dummy(name):
        return type(name, (), {
            "__init__": lambda self, *a, **k: None,
            "__setstate__": lambda self, state: None,
        })

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.split(".")[0] in ("pytorch_lightning", "lightning", "lightning_fabric"):
                return _dummy(name)
            return super().find_class(module, name)

    shim = types.SimpleNamespace(
    __name__="pickle",
    Unpickler=_Unpickler, load=pickle.load, loads=pickle.loads,
    Pickler=pickle.Pickler, dump=pickle.dump, dumps=pickle.dumps,
    HIGHEST_PROTOCOL=pickle.HIGHEST_PROTOCOL,
    )
    return torch.load(path, map_location="cpu", pickle_module=shim, weights_only=False)

def get_eloftr(device="cuda"):
    """Build EfficientLoFTR once, full precision for robustness, and cache it."""
    global _ELOFTR
    if _ELOFTR is None:
        cfg = copy.deepcopy(full_default_cfg)
        m = LoFTR(config=cfg)
        ckpt = _pl_tolerant_load(ELOFTR_WEIGHTS)
        m.load_state_dict(ckpt["state_dict"])
        m = reparameter(m)
        _ELOFTR = m.eval().to(device)
    return _ELOFTR

def _prep_gray(img_hwc, device, max_dim=640):
    """
    HWC RGB tensor in [0,1] -> (gray 1x1xH'xW' on device, (sx, sy)).
    Clamps maximum resolution to max_dim to prevent OOM on 4GB GPUs.
    """
    t = img_hwc.permute(2, 0, 1)[:3].unsqueeze(0).float().to(device)
    _, _, H, W = t.shape
    
    # Calculate safe dimensions for a 4GB VRAM limit
    scale = min(1.0, max_dim / max(H, W))
    Hc = int((H * scale) // 32) * 32
    Wc = int((W * scale) // 32) * 32
    
    w = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
    gray = (t * w).sum(1, keepdim=True)
    gray = F.interpolate(gray, size=(Hc, Wc), mode="bilinear", align_corners=False)
    
    # Return the grayscale tensor and the scaling factors to map keypoints back to native pixels
    return gray, (W / Wc, H / Hc)


def get_clip_features(image: torch.Tensor, processor, model, device):
    """ UNUSED since the coarse stage moved to retrievers.py. Left in place so
        the rest of the file is untouched. Safe to delete.
        Converts HWC tensor to a normalized 1D CLIP embedding. """
    from PIL import Image

    img_np = image.detach().cpu().numpy()
    if img_np.dtype in [np.float32, np.float64]:
        img_np = (img_np[..., :3] * 255).clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img_np)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
        feat = model.get_image_features(**inputs)

    return feat / feat.norm(dim=-1, keepdim=True)


def coarse_global_retrieval(pipeline, query: torch.Tensor, device="cuda") -> torch.Tensor:
    """ Nearest neighbour in the chosen embedding space. Returns its 4x4 pose.
        Which embedding is used is set by --retriever (clip or megaloc). """
    retriever = get_retriever(pipeline, device=device)
    names = pipeline.datamanager.train_dataset.image_filenames   # NEW

    top = retriever.query(query, top_k=5)

    print(f"  [coarse] Top 5 {retriever.name} matches:")
    for rank, (idx, sim) in enumerate(top):
        print(f"    Rank {rank+1}: train[{idx:03d}] {names[idx].name} | Cosine Sim: {sim:.4f}")

    # top1 to top2 margin. A small margin means the retriever cannot tell the
    # candidates apart, which is the failure mode to watch on a dense capture.
    if len(top) > 1:
        print(f"  [coarse] top1-top2 margin: {top[0][1] - top[1][1]:.4f}")

    best_idx = top[0][0]
    return retriever.index_poses[best_idx]


def generate_interactive_query(pipeline, camera, start_pose: torch.Tensor, step_units=0.05):
    """ Opens an interactive OpenCV window to fly around the 3DGS map. """
    current_pose = start_pose.clone().cpu()
    print("\n[info] Launching interactive viewer.")
    print("       w/a/s/d/q/e = Translate.")
    print("       Arrow Keys = Pitch / Yaw.")
    print("       , and . = Roll.")
    print("       Press SPACE to lock query.")
    if ROT90_CW:
        print("       [rot90] display rotated 90 deg CW. Controls still act in the "
              "camera frame, so on-screen motion will look rotated too.")

    rot_step = math.radians(2.0)

    while True:
        img_tensor = render_at_pose(pipeline, camera, current_pose)
        img_np = (img_tensor.numpy()[..., :3] * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_bgr = _rot90_img(img_bgr)          # NEW: view-only rotation

        cv2.imshow("Interactive Query Generator (Press SPACE to lock)", img_bgr)
        key = cv2.waitKeyEx(0)

        delta_t = torch.zeros(3)
        omega = torch.zeros(3)

        if key == 27:  # ESC key
            print("  [info] Cancelled.")
            cv2.destroyAllWindows()
            return None, None
        elif key == 32:  # SPACE key
            print("  [info] Novel query locked and saved!")
            cv2.destroyAllWindows()
            return img_tensor, current_pose

        elif key == ord('s'): delta_t[2] = step_units
        elif key == ord('w'): delta_t[2] = -step_units
        elif key == ord('a'): delta_t[0] = -step_units
        elif key == ord('d'): delta_t[0] = step_units
        elif key == ord('q'): delta_t[1] = -step_units
        elif key == ord('e'): delta_t[1] = step_units
        elif key in [65362, 82]: omega[0] = rot_step
        elif key in [65364, 84]: omega[0] = -rot_step
        elif key in [65361, 81]: omega[1] = rot_step
        elif key in [65363, 83]: omega[1] = -rot_step
        elif key == ord(','): omega[2] = -rot_step
        elif key == ord('.'): omega[2] = rot_step

        if delta_t.any() or omega.any():
            delta = torch.eye(4)
            if omega.any():
                delta[:3, :3] = so3_exp(omega)
            if delta_t.any():
                delta[:3, 3] = delta_t
            current_pose = current_pose @ delta


@torch.no_grad()
def render_rgb_and_depth(pipeline, camera, c2w_4x4: torch.Tensor):
    """ Renders both RGB and Depth maps for 2D-3D lifting. """
    c2w_tensor = c2w_4x4[:3, :].unsqueeze(0) if c2w_4x4.ndim == 2 else c2w_4x4[..., :3, :]
    cam = Cameras(
        camera_to_worlds=c2w_tensor, fx=camera.fx, fy=camera.fy,
        cx=camera.cx, cy=camera.cy, width=camera.width,
        height=camera.height, camera_type=camera.camera_type
    ).to(c2w_4x4.device)

    outputs = pipeline.model.get_outputs(cam)
    return outputs["rgb"], outputs["depth"]


@torch.no_grad()
@torch.no_grad()
def efficient_loftr_localise(pipeline, camera, query: torch.Tensor, T_init: torch.Tensor,
    device="cuda", save_viz: bool = True, viz_tag=None, viz_dir=DIAG_DIR):
    """ Single-shot fine refinement: EfficientLoFTR matches + depth lift + PnP.
        save_viz: write loftr_matches.png. Set False in the sweep so 400 trials
        don't each overwrite the same image (and to save the disk write). """
    T_init = T_init.to(device).float()
    rendered_rgb, rendered_depth = render_rgb_and_depth(pipeline, camera, T_init)

    matcher = get_eloftr(device)

    # query is already [0,1] float (a render). Do NOT divide by 255.
    q_gray, (sxq, syq) = _prep_gray(query, device)
    r_gray, (sxr, syr) = _prep_gray(rendered_rgb, device)

    batch = {"image0": q_gray, "image1": r_gray}
    matcher(batch)
    mkpts_query  = batch["mkpts0_f"].cpu().numpy()
    mkpts_render = batch["mkpts1_f"].cpu().numpy()
    # mconf = batch["mconf"].cpu().numpy()   # available if you want to weight or filter

    if len(mkpts_query) < 4:
        print("  [warn] EfficientLoFTR found <4 matches.")
        return T_init

    # scale keypoints from the resized matching grid back to native pixels
    mkpts_query  = mkpts_query  * np.array([sxq, syq], dtype=np.float32)
    mkpts_render = mkpts_render * np.array([sxr, syr], dtype=np.float32)

    # Save matches for the interactive run (fixed name) or, in the sweep, under a
    # unique tag so each sampled magnitude keeps its own image instead of overwriting.
    if save_viz or viz_tag is not None:
        os.makedirs(viz_dir, exist_ok=True)
        name = f"loftr_matches_{viz_tag}.png" if viz_tag is not None else "loftr_matches.png"
        draw_loftr_matches(query, rendered_rgb, mkpts_query, mkpts_render,
                           os.path.join(viz_dir, name))

    cx, cy = camera.cx.item(), camera.cy.item()
    fx, fy = camera.fx.item(), camera.fy.item()
    depth = rendered_depth.squeeze(-1).cpu().numpy()
    H, W = depth.shape

    pts_3d, pts_2d_query = [], []
    for (u_q, v_q), (u_r, v_r) in zip(mkpts_query, mkpts_render):
        ur, vr = int(round(u_r)), int(round(v_r))
        if not (0 <= ur < W and 0 <= vr < H):
            continue
        d = float(depth[vr, ur])
        if d <= 0:
            continue
        pts_3d.append([(u_r - cx) * d / fx, (v_r - cy) * d / fy, d])
        pts_2d_query.append([u_q, v_q])

    if len(pts_3d) < 4:
        print("  [warn] <4 matches had valid depth.")
        return T_init

    pts_3d = np.array(pts_3d, dtype=np.float32)
    pts_2d_query = np.array(pts_2d_query, dtype=np.float32)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(4, dtype=np.float32)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d_query, K, dist, flags=cv2.SOLVEPNP_EPNP
    )
    if not ok:
        print("  [warn] PnP failed.")
        return T_init

    R_off, _ = cv2.Rodrigues(rvec)
    T_off = np.eye(4)
    T_off[:3, :3] = R_off
    T_off[:3, 3] = tvec.squeeze()

    # Camera-convention fix. VERIFIED: passes zero-offset test and recovers
    # 0.39u/8.4deg -> 0.002u/0.37deg. nerfstudio poses are OpenGL (y up, -z fwd),
    # OpenCV/PnP is (y down, +z fwd), so we flip into OpenCV, solve, flip back.
    M = np.diag([1.0, -1.0, -1.0, 1.0])                       # OpenGL <-> OpenCV, self-inverse
    T_init_cv = T_init.cpu().numpy().astype(np.float64) @ M   # nerf c2w -> OpenCV c2w
    T_query_cv = T_init_cv @ np.linalg.inv(T_off)             # PnP solved in OpenCV frame
    T_final = T_query_cv @ M                                  # OpenCV c2w -> nerf c2w
    return torch.tensor(T_final, dtype=torch.float32, device=device)


def run_interactive_localization(pipeline, views, scale) -> None:
    """ Runs the full coarse-to-fine pipeline end-to-end. """
    freeze_map(pipeline)
    print("\n[info] Starting end-to-end localization test...")

    base_camera, _ = views[0]
    base_pose = to_4x4(base_camera.camera_to_worlds[0].cpu())

    query, gt_pose = generate_interactive_query(
        pipeline, base_camera, start_pose=base_pose, step_units=0.05 * scale
    )

    if query is None:
        return

    # --- Coarse Stage ---
    T_init = coarse_global_retrieval(pipeline, query, device=pipeline.device)
    t_err_coarse, r_err_coarse = pose_error(gt_pose, T_init)
    print(f"\n  [coarse] Initial guess error: {t_err_coarse:.4f} units, {r_err_coarse:.2f} deg")

    # --- Fine Stage ---
    print("  [fine] Starting Efficient LoFTR geometric refinement...")
    T_final = efficient_loftr_localise(pipeline, base_camera, query, T_init, device=pipeline.device)

    # --- Evaluation ---
    t_err_final, r_err_final = pose_error(gt_pose.double(), T_final.cpu().double())
    gt_trans = gt_pose[:3, 3].numpy()
    final_trans = T_final.cpu().numpy()[:3, 3]

    print("\n  =======================================================")
    print("  [verify] POSE MATRIX COMPARISON")
    print("  =======================================================")
    print(f"  Query Location (X,Y,Z) : [{gt_trans[0]:.4f}, {gt_trans[1]:.4f}, {gt_trans[2]:.4f}] units")
    print(f"  Final Location (X,Y,Z) : [{final_trans[0]:.4f}, {final_trans[1]:.4f}, {final_trans[2]:.4f}] units")
    print("  -------------------------------------------------------")
    print("  Query 4x4 Matrix (Ground Truth):")
    print(np.round(gt_pose.numpy(), 4))
    print("\n  Final 4x4 Matrix (Optimized):")
    print(np.round(T_final.cpu().numpy(), 4))
    print("  =======================================================\n")

    print(f"  [fine] Final localized error: {t_err_final:.4f} units, {r_err_final:.2f} deg")

    save_diagnostic_panel(pipeline, base_camera, query, gt_pose, T_init, T_final,
                          os.path.join(DIAG_DIR, "panel.png"))
    print(f"  [viz] wrote {DIAG_DIR}/panel.png and loftr_matches.png")

    if t_err_final >= t_err_coarse:
         print("  [warn] Geometric solver failed to improve the coarse guess.")


# ============================================================================
# NEW: CAPTURE-RADIUS SWEEP
# ----------------------------------------------------------------------------
# Turns the single "it worked once" result into a curve. For each eval view we
# take its known-good pose, knock it off by a controlled amount (perturb), run
# the fine stage, and record how far the final pose still is from truth.
# Reading the plot: as long as final error stays LOW while initial error grows,
# the fine stage is recovering. The initial error at which the curve snaps back
# up onto the y=x line (final == initial, i.e. no recovery) is the CAPTURE
# RADIUS, and that number is the spec your coarse room-retrieval stage must meet.
# ============================================================================

def pnp_capture_radius_sweep(pipeline, views, scale, magnitudes, n_trials=10, device="cuda",
                             sample_dir=None):
    """
    Sweep the EfficientLoFTR+PnP fine stage across perturbation magnitudes.
    If sample_dir is given, saves a panel + match image for the FIRST trial of
    each magnitude so you can eyeball the degradation without writing hundreds.
    """
    gen = torch.Generator().manual_seed(42)
    rows = []

    for v_idx, (camera, query) in enumerate(views):
        gt = to_4x4(camera.camera_to_worlds[0].cpu())

        for (t_mag, r_mag) in magnitudes:
            trials = 1 if (t_mag == 0.0 and r_mag == 0.0) else n_trials

            for k in range(trials):
                T_init = perturb_pose(gt, make_perturbation(t_mag, r_mag, generator=gen))
                t0, r0 = pose_error(gt, T_init)

                # Only the first trial of each magnitude gets pictures saved.
                tag = f"v{v_idx}_t{t_mag:.2f}_r{r_mag:.0f}" if (sample_dir and k == 0) else None

                T_final = efficient_loftr_localise(
                    pipeline, camera, query, T_init,
                    device=device, save_viz=False,
                    viz_tag=tag, viz_dir=(sample_dir or DIAG_DIR),
                )
                t1, r1 = pose_error(gt.double(), T_final.cpu().double())

                if tag is not None:
                    save_diagnostic_panel(
                        pipeline, camera, query, gt, T_init, T_final,
                        os.path.join(sample_dir, f"panel_{tag}.png"),
                    )

                rows.append(dict(
                    view=v_idx, trial=k, t_mag=t_mag, r_mag=r_mag,
                    t_init=t0, r_init=r0, t_final=t1, r_final=r1,
                    improved=bool(t1 < t0),
                ))
            print(f"  [sweep] view {v_idx} | seed t={t_mag:.2f} r={r_mag:.1f} | "
                  f"last final: {t1:.4f} units, {r1:.2f} deg")

    return rows

def plot_capture_radius(rows, out_dir):
    """ Write the CSV and a two-panel plot (translation, rotation) with the
        y=x 'no recovery' reference line. """
    import matplotlib
    matplotlib.use("Agg")                 # headless: no display needed on a cluster
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # 1. Raw table for the report / further analysis.
    csv_path = os.path.join(out_dir, "capture_radius.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  [info] wrote {csv_path}")

    # 2. Aggregate by magnitude: median final error with 25-75 percentile band.
    def agg(mag_key, val_key):
        by = {}
        for r in rows:
            by.setdefault(r[mag_key], []).append(r[val_key])
        x = sorted(by.keys())
        med = [float(np.median(by[k])) for k in x]
        lo  = [float(np.percentile(by[k], 25)) for k in x]
        hi  = [float(np.percentile(by[k], 75)) for k in x]
        return x, med, lo, hi

    xt, tmed, tlo, thi = agg("t_mag", "t_final")
    xr, rmed, rlo, rhi = agg("r_mag", "r_final")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(xt, tmed, marker="o", color="tab:blue", label="median final error")
    ax1.fill_between(xt, tlo, thi, alpha=0.2, color="tab:blue", label="25-75 pct")
    ax1.plot(xt, xt, "--", color="red", alpha=0.6, label="no recovery (y=x)")
    ax1.set_xlabel("initial translation error (scene units)")
    ax1.set_ylabel("final translation error (scene units)")
    ax1.set_title("Fine-stage capture radius (translation)")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(xr, rmed, marker="o", color="tab:green", label="median final error")
    ax2.fill_between(xr, rlo, rhi, alpha=0.2, color="tab:green", label="25-75 pct")
    ax2.plot(xr, xr, "--", color="red", alpha=0.6, label="no recovery (y=x)")
    ax2.set_xlabel("initial rotation error (deg)")
    ax2.set_ylabel("final rotation error (deg)")
    ax2.set_title("Fine-stage capture radius (rotation)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = os.path.join(out_dir, "capture_radius.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  [info] wrote {png_path}")

def run_capture_radius_sweep(pipeline, views, scale, n_trials=10) -> None:
    freeze_map(pipeline)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capture_radius_out")
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    n = len(views) * len(SWEEP_MAGNITUDES) * n_trials
    print(f"\n[info] Capture-radius sweep: {len(views)} views x "
          f"{len(SWEEP_MAGNITUDES)} magnitudes x {n_trials} trials (~{n} localisations)")

    rows = pnp_capture_radius_sweep(
        pipeline, views, scale, SWEEP_MAGNITUDES,
        n_trials=n_trials, device=pipeline.device, sample_dir=sample_dir,
    )

    plot_capture_radius(rows, out_dir)
    print(f"  [done] {len(rows)} localisations. Plot in {out_dir}, sample images in {sample_dir}")

def main():
    global ROT90_CW                      # NEW
    global RETRIEVER_NAME, _DATASET_NAME # NEW

    torch.set_float32_matmul_precision('high')

    # DEFAULT_BASE_DIR = "/home/chini/Desktop/3DGS"
    DEFAULT_BASE_DIR = "/home/chini/3dgs"
    # DEFAULT_CONFIG = f"{DEFAULT_BASE_DIR}/outputs/classroom_2-f1950/splatfacto/2026-10-07_081151/config.yml"
    # DEFAULT_CONFIG = f"{DEFAULT_BASE_DIR}/outputs/three_rooms-f3000/splatfacto/2026-10-06_180039/config.yml"
    # DEFAULT_CONFIG = f"{DEFAULT_BASE_DIR}/outputs/classroom_1-f1250/splatfacto/2026-10-07_063413/config.yml"
    # DEFAULT_CONFIG = f"{DEFAULT_BASE_DIR}/outputs/stacey_room-f2500/splatfacto/2026-10-08_182851/config.yml"
    DEFAULT_CONFIG = f"{DEFAULT_BASE_DIR}/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml"

    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, nargs="?", default=Path(DEFAULT_CONFIG))
    ap.add_argument("--device", default="cuda")
    # NEW: pick what to run.
    ap.add_argument("--mode", choices=["interactive", "sweep"], default="interactive",
                    help="interactive: fly to a query and localise once. "
                         "sweep: characterise the fine-stage capture radius.")
    ap.add_argument("--views", type=int, default=5,
                    help="number of eval views to use in sweep mode")
    ap.add_argument("--trials", type=int, default=10,
                    help="random perturbation directions per magnitude in sweep mode")
    # NEW: view-only 90 deg clockwise rotation, for datasets captured rotated
    # 90 deg anti-clockwise. Affects displayed and saved images only.
    ap.add_argument("--rot90", action="store_true",
                    help="rotate every displayed and saved image 90 deg clockwise "
                         "(view only, does not change matching or pose geometry)")
    # NEW: which coarse-stage retriever to use. Everything downstream is identical.
    ap.add_argument("--retriever", choices=["clip", "megaloc"], default="clip",
                    help="coarse stage embedding: zero-shot CLIP or MegaLoc (VPR)")
    args = ap.parse_args()

    ROT90_CW = args.rot90                # NEW
    if ROT90_CW:
        print("[info] --rot90 on: displayed and saved images rotated 90 deg clockwise. "
              "Matching and pose solving are unchanged.")

    RETRIEVER_NAME = args.retriever      # NEW
    # dataset folder name, used in the index cache key
    _DATASET_NAME = args.config.resolve().parents[2].name
    print(f"[info] coarse stage: {RETRIEVER_NAME} (dataset '{_DATASET_NAME}')")

    os.chdir(args.config.resolve().parents[4])

    print("[info] Loading pipeline...")
    _cfg, pipeline, scale = load_pipeline(args.config, args.device)

    if args.mode == "sweep":
        # No GUI here, so it runs fine over SSH on the cluster.
        views = get_eval_views(pipeline, max_views=args.views)
        run_capture_radius_sweep(pipeline, views, scale, n_trials=args.trials)
    else:
        views = get_eval_views(pipeline, max_views=1)
        run_interactive_localization(pipeline, views, scale)


if __name__ == "__main__":
    main()