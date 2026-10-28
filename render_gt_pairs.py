# RUN COMMAND:
# conda run -n nerfstudio python render_gt_pairs.py --scene stacey --query q_00093 --out <dir>

# put the real query photo next to the splat rendered from the query's ground-truth pose, for the report figures
# uses the query_eval GT loading so the pose lands exactly where the eval numbers say

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from my_localizer import load_pipeline, freeze_map
from coarse_harness import render_rgb_and_depth
from query_eval import (
    SCENES, fit_colmap_to_nerf, colmap_to_nerf_c2w,
    load_upright, rot_intrinsics_cw, query_camera, pick_roll_sign,
)


def to_np_img(x):
    """tensor or array -> HxWx3 numpy in [0,1] for plt.imsave"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.clip(x[..., :3], 0.0, 1.0)


def main():
    torch.set_float32_matmul_precision("high")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="stacey", choices=list(SCENES))
    ap.add_argument("--query", required=True, help="query stem, e.g. q_00093")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = SCENES[args.scene]

    print(f"[info] loading pipeline ({args.scene})...")
    _cfg, pipeline, _scale = load_pipeline(sc["config"], "cuda")
    freeze_map(pipeline)
    dev = pipeline.device

    # colmap -> nerf transform, same as query_eval
    Rm, s, t_off, resid = fit_colmap_to_nerf(pipeline, sc["scene_dir"])
    print(f"[info] colmap->nerf fit resid {resid:.4f} units")
    qgt = json.loads(Path(sc["query_gt"]).read_text())

    stem = args.query.replace(".jpg", "")
    qname = stem + ".jpg"
    if qname not in qgt:
        raise SystemExit(f"{qname} not in query_gt for {args.scene}")
    e = qgt[qname]

    # load the photo upright and bring its intrinsics + GT pose into the same orientation
    query, k = load_upright(Path(sc["eval_dir"]) / qname)
    fx, fy, cx, cy = (float(x) for x in e["params"][:4])
    fx, fy, cx, cy, W, H = rot_intrinsics_cw(fx, fy, cx, cy, e["width"], e["height"], k)
    qcam = query_camera(fx, fy, cx, cy, W, H, dev)

    gt_np = colmap_to_nerf_c2w(np.asarray(e["R_world_to_cam"]),
                               np.asarray(e["t_world_to_cam"]), Rm, s, t_off)
    if k:
        roll = pick_roll_sign(pipeline, qcam, query, gt_np, k)
        gt_np = gt_np @ np.linalg.matrix_power(roll, k)
    gt_pose = torch.from_numpy(gt_np).float()

    # render the splat from the GT pose with the query's own intrinsics, so the two line up pixel for pixel
    render = render_rgb_and_depth(pipeline, qcam, gt_pose.to(dev).float())[0]

    q_img = to_np_img(query)
    r_img = to_np_img(render)

    # save the two panels on their own...
    plt.imsave(out / f"{args.scene}_{stem}_query.png", q_img)
    plt.imsave(out / f"{args.scene}_{stem}_render.png", r_img)

    # ...and the pair side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(q_img); ax[0].set_title(f"{args.scene}  query  {qname}"); ax[0].axis("off")
    ax[1].imshow(r_img); ax[1].set_title("render @ same pose (from map)"); ax[1].axis("off")
    fig.tight_layout()
    pair_path = out / f"{args.scene}_{stem}_pair.png"
    fig.savefig(pair_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {pair_path}")


if __name__ == "__main__":
    main()
