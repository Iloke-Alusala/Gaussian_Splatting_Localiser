from __future__ import annotations

# Controlled A/B: same seed pose, MSE vs LPIPS photometric fine-refinement.
# Produces the matched side-by-side numbers for the "LPIPS vs MSE" paragraph:
# identical T_init, identical lr / iters / cosine schedule, only loss_type differs.

import argparse
from pathlib import Path

import torch

from my_pose_harness import (
    load_pipeline,
    get_eval_views,
    make_perturbation,
    perturb_pose,
    pose_error,
    to_4x4,
)
from my_localizer import localise, freeze_map, assert_map_frozen

DEFAULT_CONFIG = "/home/chini/3dgs/outputs/stacey_room-f2500/splatfacto/2026-10-08_182851/config.yml"


def run_one(pipeline, camera, query, T_init, gt_c2w, loss_type, args, lpips_fn=None):
    """Run the fine localiser once with a given loss and return (t_err, r_err, res)."""
    res = localise(
        pipeline=pipeline,
        camera=camera,
        query=query,
        T_init=T_init.to(pipeline.device).float(),
        max_iters=args.iters,
        lr_trans=args.lr,
        lr_rot=args.lr,
        loss_type=loss_type,
        debug_name=f"ab_{loss_type}",
        lpips_fn=lpips_fn,
    )
    t_err, r_err = pose_error(gt_c2w.double(), res.T_final.cpu().double())
    return t_err, r_err, res


def main():
    ap = argparse.ArgumentParser(description="MSE vs LPIPS fine-refinement A/B on one seed.")
    ap.add_argument("config", type=Path, nargs="?", default=Path(DEFAULT_CONFIG))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--view", type=int, default=0, help="eval view index to use as ground truth")
    ap.add_argument("--t-mag", type=float, default=0.05, help="translation perturbation (scene units)")
    ap.add_argument("--r-mag", type=float, default=7.0, help="rotation perturbation (degrees)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for the perturbation direction")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--trials", type=int, default=1,
                    help="number of perturbation directions (same magnitude) to average over")
    args = ap.parse_args()

    # Match the project's execution-context convention so Nerfstudio finds its checkpoints.
    import os
    os.chdir(args.config.resolve().parents[4])

    _cfg, pipeline, scale = load_pipeline(args.config, args.device)
    freeze_map(pipeline)
    assert_map_frozen(pipeline)

    views = get_eval_views(pipeline, max_views=args.view + 1)
    camera, query = views[args.view]
    gt_c2w = to_4x4(camera.camera_to_worlds[0].cpu())

    def cm(u):
        return u / scale * 100.0  # scene units -> cm

    print("=" * 64)
    print(f"[A/B] view={args.view}  seed={args.seed}  trials={args.trials}  "
          f"iters={args.iters}  lr={args.lr}")
    print(f"[A/B] scene scale={scale:.6f}  (1 scene unit = {1.0 / scale:.4f} m)")
    print(f"[A/B] seed magnitude: t={args.t_mag:.2f} u r={args.r_mag:.1f} deg")
    print("=" * 64)

    # ---- build the LPIPS model once and reuse it (4GB card OOMs otherwise) ----
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(pipeline.device)

    # One generator for the whole sweep so trials get distinct, reproducible directions.
    gen = torch.Generator().manual_seed(args.seed)
    rows = []  # (r_init, r_mse, r_lp, t_init, t_mse, t_lp)

    for k in range(args.trials):
        # shared seed per trial: build T_init ONCE, use it for both losses
        delta = make_perturbation(args.t_mag, args.r_mag, generator=gen)
        T_init = perturb_pose(gt_c2w, delta)
        t_init, r_init = pose_error(gt_c2w, T_init)

        print(f"\n--- trial {k} | seed: t={t_init:.4f} u ({cm(t_init):.1f} cm), r={r_init:.2f} deg ---")
        print("  [MSE]")
        t_mse, r_mse, res_mse = run_one(pipeline, camera, query, T_init, gt_c2w, "mse", args)
        print("  [LPIPS]")
        t_lp, r_lp, res_lp = run_one(pipeline, camera, query, T_init, gt_c2w, "lpips", args, lpips_fn=lpips_fn)

        rows.append((r_init, r_mse, r_lp, t_init, t_mse, t_lp))
        print(f"  trial {k}: seed r={r_init:.2f} | MSE r={r_mse:.2f} "
              f"[{res_mse.termination}] | LPIPS r={r_lp:.2f} [{res_lp.termination}]")

    import statistics

    def med(xs):
        return statistics.median(xs)

    r_init = med([x[0] for x in rows]); r_mse = med([x[1] for x in rows]); r_lp = med([x[2] for x in rows])
    t_init = med([x[3] for x in rows]); t_mse = med([x[4] for x in rows]); t_lp = med([x[5] for x in rows])

    print("\n" + "=" * 64)
    print(f"RESULT  (median over {args.trials} trial(s); identical seed per trial, identical lr/iters/schedule)")
    print("=" * 64)
    print(f"  seed:  t={t_init:.4f} u ({cm(t_init):.1f} cm), r={r_init:.2f} deg")
    print(f"  MSE:   t={t_mse:.4f} u ({cm(t_mse):.1f} cm), r={r_mse:.2f} deg")
    print(f"  LPIPS: t={t_lp:.4f} u ({cm(t_lp):.1f} cm), r={r_lp:.2f} deg")
    print("=" * 64)


if __name__ == "__main__":
    main()
