"""
pose_harness.py
===============
A diagnostic harness for render-and-compare localisation in a pre-trained
3D Gaussian Splatting (3DGS) map.

WHAT PROBLEM THIS SOLVES
------------------------
A trained splat is a RENDERER. It runs one way only:

        pose  --->  [ splat ]  --->  image

Localisation needs the arrow reversed: given an image, recover the pose. There
is no closed-form inverse of the renderer, so instead we SEARCH. Propose a
pose, render it, compare the render against the query image, and use the
mismatch to propose a better pose. Repeat until the render matches the query.
That is "render-and-compare", and it is a hill-descent over pose space.

That search only works if one assumption holds on YOUR scene:

    the comparison score must degrade smoothly and monotonically
    as the proposed pose moves away from the true pose.

If it does, the score is a slope you can follow downhill to the answer. If it
is flat or noisy, there is nothing to follow and no optimiser will save you.

THIS FILE DOES NOT LOCALISE ANYTHING. It tests that assumption, by taking poses
we already know are correct, deliberately knocking them off by KNOWN amounts,
and recording how the score responds. The output is a profile of the "error
surface" around the true pose.

The two numbers we want out of it:

  1. Is there any usable signal at all?
  2. The CAPTURE RADIUS: how far can a pose be knocked off before the score
     stops responding? Past that point, an optimiser is lost.

(2) is the specification for the coarse pass. If the capture radius is 15 cm,
then the coarse (room-retrieval) stage MUST deliver a starting pose within
15 cm, or the fine stage cannot finish the job. Coarse and fine are joined at
exactly this number.

TWO MODES, DO NOT CONFUSE THEM
------------------------------
  DIAGNOSTIC (this file): run once, offline. Perturbs known-good poses to
      characterise the scene. Produces a plot for the report. Never ships.

  DEPLOYMENT (built later): the actual localiser. Takes an unknown query
      image, starts from a coarse guess, and OPTIMISES the pose. Never
      perturbs anything, because it does not know the true pose.

Both call the same primitive: render at a pose, score against a query image.
That primitive is the reusable core; this file builds and validates it.

Hooks whose API moves between nerfstudio versions are marked  # >>> VERIFY.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# ============================================================================
# SECTION 1 - POSE MATHEMATICS
# ============================================================================
# A camera pose is a 4x4 matrix, "camera-to-world" (c2w):
#
#     [ R  t ]      R = 3x3 rotation    (which way the camera points)
#     [ 0  1 ]      t = 3x1 translation (where the camera is, in x,y,z)
#
# NOTE ON THE WORD "TRANSLATION": here it means POSITION / displacement in
# space. Nothing to do with language, and nothing to do with the sweep. A pose
# can be wrong in two independent ways: wrong place (translation error) or
# wrong facing (rotation error). We perturb and measure both separately.
#
# Together R and t live on the manifold SE(3) - the space of rigid-body
# motions. "Manifold" matters practically: you cannot perturb a pose by adding
# random numbers to the 12 entries of the matrix, because the result would no
# longer be a valid rotation (R must stay orthonormal with det=+1). Valid
# nudges have to be constructed as rigid transforms and COMPOSED with the pose.
# ----------------------------------------------------------------------------

# We prefer nerfstudio's own Lie-group utility where it applies, so that the
# harness uses the same machinery as nerfstudio's CameraOptimizer. The local
# fallback below means a nerfstudio version bump cannot silently break this.
try:  # >>> VERIFY: module path has moved between nerfstudio versions
    from nerfstudio.cameras.lie_groups import exp_map_SO3xR3 as _ns_exp_map

    _HAVE_NS_LIE = True
except Exception:
    _ns_exp_map = None
    _HAVE_NS_LIE = False


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """
    The EXPONENTIAL MAP for rotations (Rodrigues' formula).

    Input : omega, a 3-vector = (rotation axis) * (angle in radians).
            This is the compact, unconstrained way to write a rotation: any
            3 numbers are valid, unlike a rotation matrix whose 9 entries must
            satisfy orthonormality constraints.
    Output: the corresponding 3x3 rotation matrix.

    Intuition: omega is an angular velocity applied for one second. The
    exponential map integrates that spin and hands back where you ended up.
    This is exactly why it is called "exponential" - it is the matrix
    exponential of the skew-symmetric matrix K built from omega, and
    Rodrigues' formula is the closed form of that infinite series.

    Why we need it: it lets us say "rotate by exactly 5 degrees about a random
    axis" and get a guaranteed-valid rotation matrix out.
    """
    theta = torch.linalg.norm(omega)  # rotation angle in radians
    if theta < 1e-8:
        # Small-angle guard: the formula below divides by theta. At theta ~ 0
        # the rotation is the identity anyway.
        return torch.eye(3, dtype=omega.dtype, device=omega.device)

    k = omega / theta  # unit rotation axis

    # K is the skew-symmetric "cross-product matrix" of the axis:
    # for any vector v, K @ v == cross(k, v).
    K = torch.tensor(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=omega.dtype,
        device=omega.device,
    )
    I = torch.eye(3, dtype=omega.dtype, device=omega.device)

    # Rodrigues: R = I + sin(t) K + (1 - cos(t)) K^2
    return I + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def make_perturbation(t_mag: float, r_mag_deg: float, generator=None) -> torch.Tensor:
    """
    Build one random "nudge" - a 4x4 rigid transform with EXACTLY the requested
    translation magnitude (metres, in scene units) and rotation magnitude
    (degrees).

    "PERTURB" simply means: knock it slightly off on purpose. We take a pose we
    know is correct and displace it by a known amount, so that afterwards we can
    ask how the photometric score responded to a displacement of exactly that
    size.

    The direction is random; only the MAGNITUDE is controlled. That is
    deliberate - we sample many random directions at each magnitude so that one
    unlucky direction (e.g. straight into a textureless wall) cannot masquerade
    as a property of the scene.

    ---------------------------------------------------------------------
    WHY TRANSLATION IS SET DIRECTLY INSTEAD OF VIA THE SE(3) EXPONENTIAL
    ---------------------------------------------------------------------
    There is a standard 6-vector representation of a pose nudge: 3 numbers for
    rotation (phi), 3 for translation (rho), pushed through the SE(3)
    exponential map. But that map produces:

            R = exp(phi)          <- rotation, as expected
            t = V(phi) @ rho      <- NOT rho

    where V(phi) is a matrix depending on the rotation. The reason is that the
    SE(3) exponential describes a SCREW MOTION: rotating and translating
    simultaneously along a helical path, like a corkscrew. So rho behaves like a
    velocity along a curving path, not a final displacement. Walking at constant
    speed while turning does not leave you (speed x time) straight ahead of
    where you started, because your path curved.

    Consequence: if you set |rho| = 0.1 expecting a 10 cm nudge while also
    rotating 20 degrees, you do NOT land 10 cm away.

    That is fatal HERE specifically, because the perturbation magnitude is the
    x-axis of every plot we produce. A label reading "10 cm" that is really
    12 cm misplaces every data point and corrupts the capture radius we extract.

    So: rotation goes through the exponential map (where it is needed and
    correct), translation is set directly (so the magnitude is exactly the
    label). For the DEPLOYED optimiser later, the full SE(3) exponential IS the
    right parameterisation, because there you want a smooth unconstrained
    6-vector to take gradients in and you do not care about exact magnitudes.
    Different job, different tool.
    """
    g = generator

    # Random unit direction for the translation.
    # Sampling from a Gaussian and normalising gives a direction uniformly
    # distributed on the sphere (unlike, say, uniform-in-a-cube then normalise,
    # which over-samples the corners).
    t_dir = torch.randn(3, generator=g)
    t_dir = t_dir / torch.linalg.norm(t_dir)

    # Random unit axis for the rotation, same reasoning.
    axis = torch.randn(3, generator=g)
    axis = axis / torch.linalg.norm(axis)

    R = so3_exp(axis * math.radians(r_mag_deg))  # exact angle about that axis

    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t_dir * t_mag  # exact displacement magnitude
    return T


def perturb_pose(c2w_4x4: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Apply the nudge in the CAMERA's own frame (right-multiplication).

    ---------------------------------------------------------------
    LEFT vs RIGHT MULTIPLICATION - what the difference actually means
    ---------------------------------------------------------------
    Poses compose by matrix multiplication, and matrix multiplication does not
    commute, so the order encodes WHICH FRAME the nudge is expressed in:

        RIGHT:  T_new = T_gt @ delta     nudge is in the CAMERA's frame
        LEFT:   T_new = delta @ T_gt     nudge is in the WORLD's frame

    Everyday version: "take three steps forward" (right - depends entirely on
    which way you are currently facing) versus "move three metres north" (left -
    absolute, your facing is irrelevant). Both are valid instructions; they just
    mean different things.

    We use RIGHT, for two reasons.

    1. MEANING. Pose-estimation error is something a robot experiences in its
       own frame: "my estimate is 10 cm forward and 5 degrees off from where I
       actually am." That body-frame quantity is exactly what the coarse pass
       has to bound, so it is what we should be sweeping.

    2. DATA INTEGRITY, and this is the one that would silently corrupt results.
       A world-frame rotation rotates about the WORLD ORIGIN, which swings the
       camera through an arc. A camera sitting 5 m from the origin, rotated
       5 degrees about that origin, is dragged roughly 44 cm sideways:

            arc length = radius * angle = 5.0 * (5 * pi/180) ~ 0.44 m

       So a "pure 5 degree rotation" perturbation would have secretly injected
       44 cm of translation - and the size of that contamination depends on
       where in the scene each camera happens to sit. Different held-out views
       would receive different effective perturbations from the same nominal
       magnitude, making any average across views meaningless.

       Right-multiplication rotates the camera about ITSELF. 5 degrees is
       5 degrees of pure rotation, identically, for every view in the scene.
    """
    return c2w_4x4 @ delta


def pose_error(c2w_a: torch.Tensor, c2w_b: torch.Tensor) -> tuple[float, float]:
    """
    Measure how far apart two poses are. Returns:
        translation error (scene units - divide by dataparser_scale for metres)
        rotation error    (degrees)

    Translation error is just the Euclidean distance between the two camera
    positions.

    Rotation error is the GEODESIC angle: the single smallest rotation angle
    that would take one orientation onto the other. Derivation: compose one
    rotation with the inverse of the other to get the relative rotation R_rel;
    for any rotation matrix, trace(R) = 1 + 2 cos(angle), so invert that.

    Why geodesic rather than, say, comparing Euler angles: Euler angles are
    three numbers that are not a distance, suffer gimbal lock, and depend on
    convention. The geodesic angle is the true, convention-free distance on the
    rotation manifold, and it is what the relocalisation literature reports.

    Sanity check on this function: perturbing by magnitude X should return an
    error of X. If it does not, the perturbation is being applied wrongly.
    """
    t_err = torch.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3]).item()

    R_rel = c2w_a[:3, :3].T @ c2w_b[:3, :3]  # R_a^-1 @ R_b (transpose = inverse)
    cos = (torch.trace(R_rel) - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0, 1.0)  # guard against acos domain error
    r_err = math.degrees(math.acos(cos.item()))
    return t_err, r_err


def to_4x4(c2w_3x4: torch.Tensor) -> torch.Tensor:
    """
    nerfstudio stores poses as 3x4 (the bottom row [0,0,0,1] is implied).
    We need full 4x4 to compose transforms by matrix multiplication.
    """
    out = torch.eye(4, dtype=c2w_3x4.dtype, device=c2w_3x4.device)
    out[:3, :4] = c2w_3x4
    return out


# ============================================================================
# SECTION 2 - LOADING THE TRAINED MAP
# ============================================================================


def load_pipeline(config_path: Path, device: str = "cuda"):
    """
    Load a trained nerfstudio pipeline from its config.yml.
    This is MVP piece 1 of 3: "get the code to load the checkpoint".

    Returns (config, pipeline, dataparser_scale).

    ---------------------------------------------------------------
    THE SCALE GOTCHA - read this before trusting any number this
    harness produces
    ---------------------------------------------------------------
    nerfstudio NORMALISES scene scale when it loads a dataset, so that scenes
    of wildly different physical size all train with similar hyperparameters.
    That means distances inside the model's coordinate frame are NOT metres.
    A translation of 0.1 in model units might be 4 cm or 40 cm in the real room.

    dataparser_scale is the conversion factor. Real metres = scene_units / scale.

    If you skip this, everything still runs, the plots still look plausible, and
    every physical claim in your report is wrong. Worse, your pose-error
    thresholds (5 cm / 5 degrees etc.) become meaningless, since they are
    defined in real-world units. This is the single most likely silent failure
    in the whole harness.
    """
    from nerfstudio.utils.eval_utils import eval_setup  # >>> VERIFY

    # eval_setup rebuilds the model from config, restores the checkpoint
    # weights, and rebuilds the datamanager - so we get the trained splat AND
    # the dataset splits in one call.
    config, pipeline, _ckpt_path, _step = eval_setup(config_path, test_mode="test")
    pipeline.eval()  # disable training-time behaviour (dropout, grads, etc.)
    pipeline.to(device)

    # >>> VERIFY: attribute name varies across versions. If this raises, set
    # scale = 1.0 and report SCENE UNITS ONLY - and say so explicitly in the
    # report rather than quoting fake centimetres.
    try:
        scale = float(pipeline.datamanager.train_dataparser_outputs.dataparser_scale)
    except Exception:
        print("[warn] could not read dataparser_scale; reporting scene units only")
        scale = 1.0

    return config, pipeline, scale


def get_eval_views(pipeline, max_views: int | None = None):
    """
    Fetch held-out views together with their ground-truth poses.
    These are our stand-in "query images".

    ---------------------------------------------------------
    WHY THE EVAL SPLIT AND NOT THE TRAINING SPLIT
    ---------------------------------------------------------
    The splat was OPTIMISED to reproduce the training images. Scoring a render
    against a training image measures how well training converged - the splat
    has effectively memorised that exact view. It will look great and tell you
    nothing about localisation.

    A real query image is one the map has never been fitted to. Held-out eval
    views are the closest honest proxy available without capturing new data, so
    the ceiling PSNR we measure at zero perturbation reflects genuine novel-view
    render quality rather than memorisation.
    """
    ds = pipeline.datamanager.eval_dataset  # >>> VERIFY
    n = len(ds) if max_views is None else min(max_views, len(ds))

    views = []
    for i in range(n):
        image = ds[i]["image"]        # HWC float tensor in [0, 1]
        camera = ds.cameras[i : i + 1]  # Cameras object of length 1 (intrinsics)
        views.append((camera, image))
    return views


# ============================================================================
# SECTION 3 - RENDERING AT AN ARBITRARY POSE
# ============================================================================


def render_at_pose(pipeline, camera, c2w_4x4: torch.Tensor) -> torch.Tensor:
    """
    Render the splat from any pose we choose.
    This is MVP piece 2 of 3: "get the code to render to a specific pose".

    We reuse the Cameras object from a real view because it carries the
    INTRINSICS (focal length, principal point, image dimensions, distortion).
    We then overwrite only the EXTRINSICS (the c2w pose). Keeping intrinsics
    fixed is deliberate: we are isolating the effect of pose error, so every
    other variable must be held constant.

    Returns HWC float tensor in [0, 1] on CPU.
    """
    cam = camera.to(pipeline.device)

    # Overwrite the extrinsic. Shape must be [1, 3, 4] and float32.
    cam.camera_to_worlds = c2w_4x4[:3, :4].unsqueeze(0).to(pipeline.device).float()

    # no_grad: this is pure evaluation. The deployed optimiser will need
    # gradients here and must NOT use no_grad - that is the main structural
    # difference between the diagnostic and the localiser.
    with torch.no_grad():
        outputs = pipeline.model.get_outputs_for_camera(cam)  # >>> VERIFY

    return outputs["rgb"].detach().cpu().clamp(0.0, 1.0)


# ============================================================================
# SECTION 4 - SCORING
# ============================================================================
# This is MVP piece 3 of 3: "take an image and score it against the map".
# ----------------------------------------------------------------------------


@dataclass
class Metrics:
    psnr: float
    ssim: float
    lpips: float | None


class Scorer:
    """
    Photometric comparison between a render and a query image.

    Three metrics, because they disagree in informative ways:

      PSNR  - raw per-pixel error, in dB. HIGHER IS BETTER. Blunt: it punishes
              a small global brightness shift as harshly as a structural
              mismatch, and it is the metric 3DGS training itself optimises.

      SSIM  - structural similarity: local luminance, contrast and structure.
              HIGHER IS BETTER. Closer to "do these look like the same place".

      LPIPS - perceptual distance via deep features. LOWER IS BETTER. Most
              robust to lighting and small appearance changes, and therefore
              the most likely to still carry signal at large pose offsets where
              PSNR has gone flat.

    Watch the sign convention when reading plots: PSNR and SSIM curves should
    fall as perturbation grows, LPIPS should rise.

    Worth checking which metric holds signal furthest out - that is a genuine
    finding, and it tells you what objective the fine pass should optimise.
    """

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
                from torchmetrics.image.lpip import (
                    LearnedPerceptualImagePatchSimilarity,
                )

                # normalize=True because our images are in [0,1] and LPIPS
                # internally expects [-1,1].
                self._lpips = LearnedPerceptualImagePatchSimilarity(
                    normalize=True
                ).to(device)
            except Exception as e:
                print(f"[warn] LPIPS unavailable ({e}); continuing without it")

        self.device = device

    @staticmethod
    def _nchw(img: torch.Tensor) -> torch.Tensor:
        """torchmetrics wants NCHW; nerfstudio gives HWC."""
        return img.permute(2, 0, 1).unsqueeze(0)

    def score(self, rendered: torch.Tensor, query: torch.Tensor) -> Metrics:
        a = self._nchw(rendered).to(self.device)
        b = self._nchw(query).to(self.device)

        psnr = float(self._psnr(a, b, data_range=1.0))
        ssim = float(self._ssim(a, b, data_range=1.0))

        lpips = None
        if self._lpips is not None:
            with torch.no_grad():
                lpips = float(self._lpips(a, b))

        return Metrics(psnr=psnr, ssim=ssim, lpips=lpips)


# ----------------------------------------------------------------------------
# Localisation accuracy metrics, following the visual place recognition and
# camera relocalisation conventions (7-Scenes, Cambridge Landmarks).
#
# A pose is "correct" only if BOTH its translation and rotation errors fall
# inside a threshold pair, so results are reported as the percentage of queries
# meeting each pair. Using several pairs shows the whole accuracy profile
# instead of one cherry-picked number.
# ----------------------------------------------------------------------------

POSE_THRESHOLDS = [
    (0.05, 5.0),   # 5 cm,  5 deg  - indoor high precision
    (0.10, 5.0),   # 10 cm, 5 deg  - indoor standard
    (0.25, 2.0),   # 25 cm, 2 deg  - the common "high precision" bar
    (0.50, 10.0),  # 50 cm, 10 deg - coarse / room-level success
]


def threshold_recall(errors_m_deg, thresholds=POSE_THRESHOLDS) -> dict[str, float]:
    """
    Percentage of queries localised within each (metres, degrees) threshold.

    errors_m_deg: iterable of (translation_error_METRES, rotation_error_deg).
    Feed this REAL METRES, not scene units, or the thresholds are meaningless.

    In the diagnostic this characterises the perturbation distribution. Its real
    use is later: it is how you will report the accuracy of the finished
    localiser, and how you compare against the traditional baseline the project
    brief requires.
    """
    errs = list(errors_m_deg)
    out = {}
    for t_thr, r_thr in thresholds:
        hits = sum(1 for t, r in errs if t <= t_thr and r <= r_thr)
        out[f"{int(t_thr * 100)}cm_{r_thr:g}deg"] = 100.0 * hits / max(len(errs), 1)
    return out


def recall_at_n(*args, **kwargs):
    """
    NOT APPLICABLE YET - deliberately left as a stub.

    recall@N is a RETRIEVAL metric: given a query, does the correct entry appear
    among the top N candidates returned from a database? It needs a database of
    multiple candidates to retrieve from.

    In this project that database is the set of rooms, and recall@N is how the
    COARSE pass gets evaluated ("did room retrieval put the right room in the
    top 1?"). The perturbation sweep has a single scene and retrieves nothing,
    so reporting recall@N here would be a meaningless number.

    Implement when the CLIP-style room retrieval exists.
    """
    raise NotImplementedError(
        "recall@N belongs to the coarse retrieval pass; no room database yet"
    )


# ============================================================================
# SECTION 5 - THE SWEEP
# ============================================================================
# "Sweeping" = repeating the perturb-render-score cycle across a RANGE of
# perturbation magnitudes: 1 cm, 2 cm, 5 cm, 10 cm, 20 cm ... and many random
# directions at each. Plotting score against magnitude traces out the profile
# of the error surface around the true pose.
#
# Reading the resulting curve:
#   - steep, monotonic decline  -> strong signal, optimiser has a slope to
#                                  follow, fine pass is viable
#   - flat beyond some radius   -> that radius IS the capture radius, and the
#                                  specification your coarse pass must meet
#   - noisy / non-monotonic     -> the objective has local minima; mitigations
#                                  exist (e.g. blurring the images to smooth
#                                  the landscape, as GSLoc does)
#   - flat from the start       -> render-and-compare has no usable signal on
#                                  this scene; diagnose render quality first
# ----------------------------------------------------------------------------


@dataclass
class SweepConfig:
    config_path: Path
    out_dir: Path = Path("harness_out")

    n_views: int = 5   # how many held-out query views to test
    n_trials: int = 20 # random perturbation DIRECTIONS per magnitude
    # Why many trials: direction matters enormously. Nudging towards a
    # texture-rich bookshelf changes the image far more than nudging towards a
    # blank wall. Averaging over random directions separates the effect of
    # MAGNITUDE (what we are studying) from the effect of DIRECTION (noise here).

    # Magnitudes are in SCENE UNITS until divided by dataparser_scale.
    # Start small and widen once you see where the signal dies. If every point
    # is already flat, your range is too coarse - halve it.
    t_mags: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    )
    r_mags_deg: list[float] = field(
        default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    )

    # coupled=True perturbs translation and rotation together, which is
    # realistic (a bad pose estimate is usually wrong in both).
    # coupled=False varies one at a time, which isolates which kind of error
    # the score is more sensitive to. Run coupled first, then decoupled if the
    # result is ambiguous.
    coupled: bool = True

    seed: int = 0       # fixed seed so the sweep is reproducible for the report
    device: str = "cuda"


def run_sweep(cfg: SweepConfig):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    _config, pipeline, scale = load_pipeline(cfg.config_path, cfg.device)
    print(
        f"[info] dataparser_scale = {scale:.6f} "
        f"(real metres = scene units / scale)"
    )

    views = get_eval_views(pipeline, max_views=cfg.n_views)
    scorer = Scorer(device=cfg.device)

    if cfg.coupled:
        pairs = list(zip(cfg.t_mags, cfg.r_mags_deg))
    else:
        pairs = [(t, 0.0) for t in cfg.t_mags] + [(0.0, r) for r in cfg.r_mags_deg]

    rows = []
    for v_idx, (camera, query) in enumerate(views):
        # Ground truth: the pose this image was actually taken from.
        gt_c2w = to_4x4(camera.camera_to_worlds[0].cpu())

        for (t_mag, r_mag) in pairs:
            # The zero-perturbation case is deterministic, so one trial is
            # enough. It is also the most important single number in the sweep:
            # it is the CEILING - how well the splat reproduces a held-out view
            # from its exact true pose. Every perturbed score is measured
            # against this. If the ceiling itself is poor (say PSNR < 20), the
            # map quality is the problem, not the localisation method.
            trials = 1 if (t_mag == 0.0 and r_mag == 0.0) else cfg.n_trials

            for k in range(trials):
                delta = make_perturbation(t_mag, r_mag, generator=gen)
                pert_c2w = perturb_pose(gt_c2w, delta)

                rendered = render_at_pose(pipeline, camera, pert_c2w)
                m = scorer.score(rendered, query)

                # Measured error, recomputed from the poses rather than assumed
                # from the requested magnitude. This is a self-check: t_err
                # should match t_mag closely. A mismatch means the perturbation
                # is being applied in the wrong frame or through the wrong map.
                t_err, r_err = pose_error(gt_c2w, pert_c2w)

                rows.append(
                    {
                        "view": v_idx,
                        "trial": k,
                        "t_mag_scene": t_mag,
                        "t_mag_m": t_mag / scale if scale else float("nan"),
                        "r_mag_deg": r_mag,
                        "t_err_scene": t_err,
                        "t_err_m": t_err / scale if scale else float("nan"),
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
    Plot median score against perturbation magnitude, with an interquartile
    band showing the spread across random directions.

    Median and IQR rather than mean and standard deviation, because the spread
    across directions is usually skewed (a few directions into featureless
    regions produce outliers). The median is not dragged around by them, and the
    IQR band shows honestly how much direction matters at each magnitude.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless backend: no display on a cluster node
    import matplotlib.pyplot as plt

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
    run_sweep(cfg)


if __name__ == "__main__":
    main()
