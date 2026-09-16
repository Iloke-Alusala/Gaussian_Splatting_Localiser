from __future__ import annotations
# RUN COMMAND:
# cd /home/chini/Desktop/3DGS/data/desk-test-2/raw-images/
# python3 /home/chini/Desktop/3DGS/code/my_pose_harness.py outputs/processed/splatfacto/2026-08-16_104231/config.yml

# You have to be inside of the raw-images folder otherwise relative paths inside of config.yml won't work

# __future__ is a dunder, just here to stop compile of class types during compile time and allow for faster loading of program

# import required libraries
# admin libraries
import argparse
import csv
import os
from dataclasses import dataclass, field
from pathlib import Path

# science libraries
import math
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

# Step 1 is to do the pose algebra. Take a rotation around a certain axis and return a matrix to represent that rotation

def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """
    Converts a 3D axis-angle vector (omega, where magnitude = angle in radians, direction = rotation axis) into a 3x3 rotation matrix using Rodrigues' formula.

    Input : omega, a 3-vector.
    Output: the corresponding 3x3 rotation matrix.
    """
    theta = torch.linalg.norm(omega) # The magnitude which gives radians to rotate 
    # Small-angle guard: This guards us against super small rotations which are almost neglegible. We don't need to rotate at all
    if(theta < 1e-8):
        # dtype is for type safety, ensure that you return the same internal variable type, and device ensures that your returned varaible is stored on the same hardware where your input varibale was, whether that CPU or GPU
        return torch.eye(3, dtype=omega.dtype, device=omega.device)

    k = omega / theta # the normalised rotation axis vector

    # now we create the K matrix (Skew symmetric matrix), allows us to carry out cross product as matrix multiplication, more efficient
    K = torch.tensor(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=omega.dtype,
        device=omega.device,
    )
    I = torch.eye(3, dtype=omega.dtype, device=omega.device)

    # Return using Rodrigues Formula: R = I + sin(t) K + (1 - cos(t)) K^2
    return I + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)

def make_perturbation(t_mag: float, r_mag_deg: float, generator=None) -> torch.Tensor:
    """
    Make a small random perturbation, where:
    t_mag: magnitude of translational perturbation
    r_mag_deg: magnitude of rotational perturbation
    Implementation does 
    """
    # allows reproducability through multiple different runs. It uses random numbers that it generated once across all of our future runs. Allows us to test this and accurately compare perturbations
    g = generator

    # Random direction to translate in
    t_dir = torch.randn(3, generator=g)
    print(t_dir)
    t_dir = t_dir/torch.linalg.norm(t_dir)

    # Random directino to rotate around
    axis = torch.randn(3, generator=g)
    axis = axis/torch.linalg.norm(axis)

    # create the rotational matrix and convert our degrees to radians
    R = so3_exp(axis * math.radians(r_mag_deg))

    # Create the final matrix in SE(3) dimension
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t_dir * t_mag # multiply by selected magnitude
    return T

def perturb_pose(c2w_4x4: torch.tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    apply the perturbation to the pose
    Input: c2w_4x4 which is the current pose, delta which is the 4x4 matrix that gives the orientation to perturb to
    Output: the new pose in SE(3)
    """
    return c2w_4x4 @ delta

def pose_error(c2w_a: torch.Tensor, c2w_b: torch.Tensor) -> tuple[float, float]:
    """
    measure how far apart two poses are
    Input: c2w_a and c2w_b pose matrices
    Output: Translational error in meters? (scene units / dataparser_scale) and Rotation error in degrees.

    Translational is just the euclidean distance between the two points
    Rotations is the smallest rotation angle to make one orientation another (GEODESIC angle)
    """
    # get magnitude of differences
    # we use .item() at the end so we remove it from GPU storage and have a clean variable, just a float
    t_err = torch.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3]).item()

    R_rel = c2w_a[:3, :3].T @ c2w_b[:3, :3]
    cos  = (torch.trace(R_rel) - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0, 1.0) # guard against acos domain error -> we should never reach this, but sometimes rounding inaccuracies push values to 1.00000001 which then breaks acos function
    r_err = math.degrees(math.acos(cos.item()))
    return t_err, r_err

def to_4x4 (c2w_3x4: torch.Tensor) -> torch.Tensor:
    """
    nerfStudio stores poses as 3x4 and has the [0 0 0 1] at the bottom as implicit, so this just restructures them to our format
    Input: c2w3x4, which is just a 3x4 matrix
    Output: the converted 4x4 matrix
    """
    out = torch.eye(4, dtype=c2w_3x4.dtype, device=c2w_3x4.device)
    out[:3, :4] = c2w_3x4
    return out

# Section 2 - Loading the trained map

def load_pipeline(config_path: Path, device: str = "cuda"):
    """
    Load the checkpoint file from config.yml into code.
    """
    # We need the dataparser_scale to convert the normalised unit scale from inside of nerfstudio to real world dimensions. dataparser_scale is the conversion factor: Real meters = scene_units / scale.

    # eval_setup rebuilds model from config and restores checkpoint weights, and rebuilds datamanager, so we get trained splat and dataset splits in one call. (what is a dataset split)
    from nerfstudio.utils.eval_utils import eval_setup

    config, pipeline, _ckpt_path, _step = eval_setup(config_path, test_mode="test") # what does test mode = "test" do
    # what does the pipeline store? Main container for the Nerfstudio Model and its dataset
    pipeline.eval()
    pipeline.to(device)

    try:
        # checks whether we can get the dataparser value from the datamanager
        scale = float(pipeline.datamanager.train_dataparser_outputs.dataparser_scale)
    except Exception:
        # means we only have a realtive value, can't scale to real dimensions
        print("[warn] could not read dataparser_scale; reporting scene units only")
        scale = 1.0
    
    return config, pipeline, scale

def get_eval_views(pipeline, max_views: int | None = None):
    """
    Fetch held-out views (query images) together with their ground-truth poses. 
    """
    # honestly, idk if we use held out views, but let's just run it
    ds = pipeline.datamanager.eval_dataset
    n = len(ds) if max_views is None else min(max_views, len(ds))

    views = []
    for i in range(n):
        image = ds[i]["image"]
        camera = ds.cameras[i : i + 1]
        views.append((camera, image))
    return views

# Step 3 - Rendering at an arbitrary pose we choose

def render_at_pose(pipeline, camera, c2w_4x4: torch.Tensor) -> torch.Tensor:
    """
    Render the splat at a specific pose we choose.
    """

    # Reuse camera object from real view because it has it's intrinsics
    cam = camera.to(pipeline.device)
    # update the camera's pose using the c2w_4x4 matrix. camera_to_worlds just stores the camera's current pose.
    cam.camera_to_worlds = c2w_4x4[:3, :4].unsqueeze(0).to(pipeline.device).float()

    # Render the image with torch.no_grad() to detach tensor from computational graph and then load tensor on CPU and clamp rgb values to between 0 and 1
    with torch.no_grad():
        outputs  = pipeline.model.get_outputs_for_camera(cam)

    return outputs["rgb"].detach().cpu().clamp(0.0, 1.0)

# Section 4 - Scoring

@dataclass # this just creates all the boilerplate __init__ function for it
class Metrics:
    psnr: float # peak signal to noise ratio -> raw per-pixel error
    ssim: float # structural similarity -> local luminance, contrast and structure (Higher is Better)
    lpips: float | None # perceptual distance via deep features. Lower is Better

class Scorer:
    def __init__(self, device: str = "cuda", use_lpips: bool = True):
        from torchmetrics.functional import (
            peak_signal_noise_ratio,
            structural_similarity_index_measure,
        )

        self._psnr = peak_signal_noise_ratio
        self._ssim = structural_similarity_index_measure
        self._lpips = None

        if use_lpips:
            try:
                from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
                self._lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
            except Exception as e:
                print(f"[warn] LPIPS unavailable ({e}); continuing without it")

        self.device = device

    @staticmethod
    def _nchw(img: torch.Tensor) -> torch.Tensor:
        """torchmetrics wants NCHW, nerfstudio gives HWC."""
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        img = img.float()
        if img.shape[-1] == 4:          # drop alpha if the dataset carries one
            img = img[..., :3]
        return img.permute(2, 0, 1).unsqueeze(0)

    def score(self, rendered: torch.Tensor, query: torch.Tensor) -> Metrics:
        a = self._nchw(rendered).to(self.device)
        b = self._nchw(query).to(self.device)

        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: render {tuple(a.shape)} vs query {tuple(b.shape)}")
        if torch.equal(a, b):
            raise RuntimeError("render and query are identical, check the scoring inputs")

        psnr = float(self._psnr(a, b, data_range=1.0))
        ssim = float(self._ssim(a, b, data_range=1.0))

        lpips = None
        if self._lpips is not None:
            with torch.no_grad():
                lpips = float(self._lpips(a, b))

        return Metrics(psnr=psnr, ssim=ssim, lpips=lpips)
    
# These thresholds give us tolerance criteria of which we use to evaluate how accurate the localisation was. The pair is maximum translation(cm) and maximum rotation(deg)

POSE_THRESHOLDS = [
    (0.05, 5.0),   # 5 cm,  5 deg  - indoor high precision
    (0.10, 5.0),   # 10 cm, 5 deg  - indoor standard
    (0.25, 2.0),   # 25 cm, 2 deg  - the common "high precision" bar
    (0.50, 10.0),  # 50 cm, 10 deg - coarse / room-level success
]

def threshold_recall(errors_m_deg, thresholds=POSE_THRESHOLDS) -> dict[str, float]:
    """
    INPUT: an iterable with structure (translation error in metres, rotation error in degrees)
    OUTPUT: Example:
    out = {
    "5cm_5deg": 85.0,    # 85% of queries met the 5cm / 5deg threshold
    "10cm_5deg": 92.5,   # 92.5% met the 10cm / 5deg threshold
    "25cm_2deg": 60.0,   # 60% met the 25cm / 2deg threshold
    "50cm_10deg": 98.0   # 98% met the 50cm / 10deg threshold
    }
    """
    errs = list(errors_m_deg)
    out = {}
    # This returns how many hits each threshold bound had
    for t_thr, r_thr in thresholds:
        hits = sum(1 for t, r in errs if t <= t_thr and r <= r_thr)
        out[f"{int(t_thr * 100)}cm_{r_thr:g}deg"] = 100.0 * hits / max(len(errs), 1)
    return out

# Section 5: The sweep. Sweeping through 

@dataclass
class SweepConfig:
    config_path: Path
    out_dir: Path = Path("harness_out")
    n_views: int = 5 # how many held-out query views to test
    n_trials: int = 20 # random perturbation DIRECTIONS per magnitude
    # Note that t_mags is in COLMAP Scene Units
    t_mags: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    )
    r_mags_deg: list[float] = field(
        default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    )
    coupled: bool = True # perturbs translation and rotation together
    seed: int = 0
    device: str = "cuda"

def run_sweep(cfg: SweepConfig):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    _config, pipeline, scale = load_pipeline(cfg.config_path, cfg.device)
    train_files = {p.name for p in pipeline.datamanager.train_dataset._dataparser_outputs.image_filenames}
    eval_files  = [p.name for p in pipeline.datamanager.eval_dataset._dataparser_outputs.image_filenames]
    print(f"[info] train={len(train_files)} eval={len(eval_files)}")
    print(f"[info] overlap={len(train_files & set(eval_files))}")
    print(f"[info] eval images: {eval_files[:10]}")
    print(f"[info] dataparser_scale = {scale:.6f} (reporting scene units only)")

    views = get_eval_views(pipeline, max_views=cfg.n_views)
    scorer = Scorer(device=cfg.device)

    if cfg.coupled:
        pairs = list(zip(cfg.t_mags, cfg.r_mags_deg))
    else:
        pairs = [(t, 0.0) for t in cfg.t_mags] + [(0.0, r) for r in cfg.r_mags_deg]

    rows = []
    for v_idx, (camera, query) in enumerate(views):
        # Ground truth: pose where image was taken from
        gt_c2w = to_4x4(camera.camera_to_worlds[0].cpu())

        for (t_mag, r_mag) in pairs:
            trials = 1 if (t_mag == 0.0 and r_mag == 0.0) else cfg.n_trials

            for k in range(trials):
                delta = make_perturbation(t_mag, r_mag, generator=gen)
                pert_c2w = perturb_pose(gt_c2w, delta)

                rendered = render_at_pose(pipeline, camera, pert_c2w)
                m = scorer.score(rendered, query)

                t_err, r_err = pose_error(gt_c2w, pert_c2w)
                rows.append(
                    {
                        "view": v_idx,
                        "trial": k,
                        "t_mag_scene": t_mag,
                        "r_mag_deg": r_mag,
                        "t_err_scene": t_err,
                        "r_err_deg": r_err,
                        "psnr": m.psnr,
                        "ssim": m.ssim,
                        "lpips": m.lpips if m.lpips is not None else "",
                    }
                )
            print(f"[view {v_idx}] t={t_mag} r={r_mag} done")

    csv_path = cfg.out_dir / "sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[info] wrote {csv_path}")

    _plot(rows, cfg.out_dir)
    return rows

def _plot(rows, out_dir: Path):
    """
    Visualise the metrics from the Metrics class in graphs and stuff
    """
    matplotlib.use("Agg")

    def agg(metric):
        by_mag = {}
        for r in rows:
            if r[metric] == "":
                continue
            key = (r["t_mag_scene"], r["r_mag_deg"])
            by_mag.setdefault(key, []).append(float(r[metric]))
        keys = sorted(by_mag.keys())
        x = [k[0] for k in keys]
        med = [float(np.median(by_mag[k])) for k in keys]
        lo = [float(np.percentile(by_mag[k], 25)) for k in keys]
        hi = [float(np.percentile(by_mag[k], 75)) for k in keys]
        return x, med, lo, hi
    
    metrics = ["psnr", "ssim", "lpips"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))

    for ax, metric in zip(np.atleast_1d(axes), metrics):
        try:
            x, med, lo, hi = agg(metric)
        except (IndexError, ValueError):
            continue
        if not x:
            continue
        ax.plot(x, med, marker="o")
        ax.fill_between(x, lo, hi, alpha=0.2)
        ax.set_xlabel("translation perturbation (scene units)")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs pose perturbation")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = out_dir / "error_surface.png"
    fig.savefig(p, dpi=150)
    print(f"[info] wrote {p}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, help="path to nerfstudio config.yml")
    ap.add_argument("--out", type=Path, default=Path("harness_out"))
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg = SweepConfig(
        config_path=args.config,
        out_dir=args.out,
        n_views=args.views,
        n_trials=args.trials,
        device=args.device,
    )
    os.chdir(args.config.resolve().parents[4])
    run_sweep(cfg)


if __name__ == "__main__":
    main()