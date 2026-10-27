"""
capture_radius_metric.py - render-vs-render capture radius on a metric offset grid.

for each trial we take a real train pose, render the splat at it (so query and map
share appearance - the upper-bound, appearance-gap-free case), knock the pose off by
a known metric offset in a random direction, then run the loftr+pnp fine stage and ask
whether it climbs back to within 5cm/5deg of the base pose. offsets are given in cm and
converted to scene units via the scene's m_per_unit, so the grid matches the pnp basin
sweep (query_eval --seed-source perturb). runs in the nerfstudio env.

reuses the proven fine stage (coarse_harness.localise_pnp) and perturbation helpers;
nothing in those is modified.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from my_localizer import load_pipeline, freeze_map, to_4x4, pose_error
from my_pose_harness import make_perturbation, perturb_pose
from coarse_harness import render_rgb_and_depth, localise_pnp
from query_eval import SCENES

# same metric grid as the pnp basin sweep (cm translation, deg rotation)
DEFAULT_GRID = [(2, 1), (5, 2), (10, 5), (20, 5), (40, 8), (60, 12), (80, 16), (100, 20), (150, 30)]


def main():
    torch.set_float32_matmul_precision("high")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="stacey", choices=list(SCENES))
    ap.add_argument("--trials", type=int, default=50, help="random trials per offset")
    ap.add_argument("--passes", type=int, default=2, help="fine-stage passes (match the pnp basin)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sc = SCENES[args.scene]
    mpu = sc["m_per_unit"]                       # scene units -> metres (door-frame scale)
    grid = [(t / 100.0 / mpu, r, t, r) for (t, r) in DEFAULT_GRID]   # (t_units, r_deg, t_cm, r_deg)

    print(f"[info] loading pipeline ({args.scene})...")
    _cfg, pipeline, _scale = load_pipeline(sc["config"], args.device)
    freeze_map(pipeline)
    dev = pipeline.device

    train_ds = pipeline.datamanager.train_dataset
    poses = [to_4x4(train_ds.cameras[i:i + 1].camera_to_worlds[0].cpu()) for i in range(len(train_ds))]
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    rows = []
    for (t_u, r_deg, t_cm, _r) in grid:
        recovered = 0
        for _ in range(args.trials):
            b = int(rng.integers(len(poses)))
            cam = train_ds.cameras[b:b + 1]
            base = poses[b]
            # render the query from the splat at the base pose -> render-vs-render
            query, _depth = render_rgb_and_depth(pipeline, cam, base.to(dev).float())
            T = perturb_pose(base, make_perturbation(t_u, r_deg, generator=gen))
            for _p in range(args.passes):
                T, _diag = localise_pnp(pipeline, cam, query, T, device=dev)
            t_err, r_err = pose_error(base.double(), T.cpu().double())
            if t_err * mpu < 0.05 and r_err < 5.0:
                recovered += 1
            torch.cuda.empty_cache()
        recall = 100.0 * recovered / args.trials
        rows.append((t_cm, r_deg, recovered, args.trials, recall))
        print(f"  offset {t_cm:g}cm/{r_deg:g}deg: {recovered}/{args.trials} recovered ({recall:.0f}%)")

    # table: offset | recovered | recall
    lines = [f"capture radius render-vs-render, scene={args.scene}, {args.trials} trials/offset, "
             f"{args.passes}-pass, recovered = final within 5cm/5deg of base ({mpu:.4f} m/unit)",
             "offset            recovered   recall"]
    for t_cm, r_deg, rec, n, recall in rows:
        lines.append(f"  {t_cm:g}cm/{r_deg:g}deg".ljust(18) + f"{rec}/{n}".rjust(9) + f"   {recall:.0f}%")
    block = "\n".join(lines)
    print("\n" + block)

    log_path = os.environ.get("RESULTS_LOG", "/home/chini/Desktop/3DGS/code/RESULTS_LOG.md")
    with open(log_path, "a") as fl:
        fl.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} | capture-radius (render-vs-render) | "
                 f"scene={args.scene}\n```\n" + block + "\n```\n")


if __name__ == "__main__":
    main()
