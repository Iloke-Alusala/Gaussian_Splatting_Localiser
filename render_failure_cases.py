"""
render_failure_cases.py - side-by-side panels for queries where our pipeline wins
and the HLoc baseline fails (stacey_room).

For each requested query we draw one row with five panels:

  [1] query photo            the real eval image (loaded upright, as everywhere else)
  [2] ours: render@our pose  splat rendered at the pose our pipeline estimates
  [3] ours: MegaLoc pick     the train/db frame our coarse stage retrieved (the seed)
  [4] HLoc: NetVLAD pick     the top db frame HLoc's coarse (retrieval) stage picked
  [5] HLoc: render@its pose  splat rendered at the pose HLoc's PnP returned

Panels 2 and 5 are the honest comparison: both render the SAME splat, one at our
estimated pose, one at HLoc's, next to the real photo it should match. Everything
reuses the proven coarse_harness / query_eval machinery so the poses line up with
the numbers already in the result CSVs. Runs in the nerfstudio env.

  python render_failure_cases.py --scene stacey --queries q_00093 q_00067 q_00114 \
      --out <dir> --hloc-dir /home/chini/3dgs/hloc_out/stacey/hloc
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from my_localizer import load_pipeline, freeze_map, to_4x4, pose_error
from coarse_harness import render_rgb_and_depth, localise_pnp
import pipeline_retriever as pr
from query_eval import (
    SCENES, fit_colmap_to_nerf, colmap_to_nerf_c2w,
    load_upright, rot_intrinsics_cw, query_camera, pick_roll_sign, _ROLL_CW,
)


def qvec2rotmat(q):
    """quaternion (w,x,y,z) -> rotation matrix. inlined so this stays off the hloc
    env; identical convention to colmap/hloc's read_write_model.qvec2rotmat."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], float)


def parse_hloc_poses(results_txt):
    """results.txt: 'name qw qx qy qz tx ty tz' = world->cam (same frame as query_gt)."""
    poses = {}
    for ln in Path(results_txt).read_text().splitlines():
        if not ln.strip():
            continue
        p = ln.split()
        poses[p[0]] = (qvec2rotmat(np.array(p[1:5], float)), np.array(p[5:8], float))
    return poses


def parse_netvlad_top(pairs_txt):
    """first db frame listed per query = HLoc's top retrieval (coarse) pick."""
    top = {}
    for ln in Path(pairs_txt).read_text().splitlines():
        if not ln.strip():
            continue
        q, db = ln.split()[:2]
        top.setdefault(q, db)        # keep the first (highest-ranked) only
    return top


def to_np_img(x):
    """GPU/CPU tensor or array HWC [0,1] -> clipped numpy for imshow."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.clip(x[..., :3], 0.0, 1.0)


def main():
    torch.set_float32_matmul_precision("high")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="stacey", choices=list(SCENES))
    ap.add_argument("--queries", nargs="+", required=True,
                    help="query stems, e.g. q_00093 q_00067 (with or without .jpg)")
    ap.add_argument("--out", required=True, help="output directory for the panels")
    ap.add_argument("--hloc-dir", default="/home/chini/3dgs/hloc_out/stacey/hloc")
    ap.add_argument("--passes", type=int, default=1,
                    help="fine-stage passes for our pipeline (1 = matches results7.csv)")
    ap.add_argument("--retriever", choices=["clip", "megaloc"], default="megaloc")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = SCENES[args.scene]
    mpu = sc["m_per_unit"]

    print(f"[info] loading pipeline ({args.scene})...")
    _cfg, pipeline, _scale = load_pipeline(sc["config"], "cuda")
    freeze_map(pipeline)
    dev = pipeline.device

    train_ds = pipeline.datamanager.train_dataset
    train_poses = [to_4x4(train_ds.cameras[i:i+1].camera_to_worlds[0].cpu())
                   for i in range(len(train_ds))]
    names = [Path(p).name for p in train_ds._dataparser_outputs.image_filenames]
    name2idx = {n: i for i, n in enumerate(names)}
    db_img_dir = sc["scene_dir"] / "images"       # full db (colmap) frames on disk

    def db_image(name):
        """db frame by name: from the splat train set if present, else off disk.
        HLoc's db (colmap sparse/0) holds more frames than the splat's train subset,
        so its NetVLAD pick isn't always a train frame."""
        if name in name2idx:
            return train_ds[name2idx[name]]["image"]
        p = db_img_dir / name
        if p.exists():
            return load_upright(p)[0]
        return None

    pr.RETRIEVER_NAME = args.retriever
    pr.ROT90_CW = False
    pr._DATASET_NAME = sc["name"]
    retriever = pr.get_retriever(pipeline, device=dev)

    Rm, s, t_off, resid = fit_colmap_to_nerf(pipeline, sc["scene_dir"])
    print(f"[info] colmap->nerf fit resid {resid:.4f} units")
    qgt = json.loads(Path(sc["query_gt"]).read_text())

    hloc_poses = parse_hloc_poses(Path(args.hloc_dir) / "results.txt")
    hloc_top = parse_netvlad_top(Path(args.hloc_dir) / "pairs-query-netvlad20.txt")

    roll = None
    for stem in args.queries:
        stem = stem.replace(".jpg", "")
        qname = stem + ".jpg"
        if qname not in qgt:
            print(f"[skip] {qname} not in query_gt")
            continue
        e = qgt[qname]
        img_path = Path(sc["eval_dir"]) / qname
        query, k = load_upright(img_path)
        fx, fy, cx, cy = (float(x) for x in e["params"][:4])
        fx, fy, cx, cy, W, H = rot_intrinsics_cw(fx, fy, cx, cy, e["width"], e["height"], k)
        qcam = query_camera(fx, fy, cx, cy, W, H, dev)

        # GT pose in nerf frame, with the upright roll applied exactly like query_eval
        gt_np = colmap_to_nerf_c2w(np.asarray(e["R_world_to_cam"]),
                                   np.asarray(e["t_world_to_cam"]), Rm, s, t_off)
        if k:
            if roll is None:
                roll = pick_roll_sign(pipeline, qcam, query, gt_np, k)
            gt_np = gt_np @ np.linalg.matrix_power(roll, k)
        gt_pose = torch.from_numpy(gt_np).float()

        # --- our pipeline: coarse retrieve -> seed -> fine PnP ---
        top_k = retriever.query(query, top_k=10)
        ret_idx = top_k[0][0]
        T = train_poses[ret_idx]
        for _p in range(args.passes):
            T, _diag = localise_pnp(pipeline, qcam, query, T, device=dev)
        ours_t, ours_r = pose_error(gt_pose.double(), T.cpu().double())
        ours_render = render_rgb_and_depth(pipeline, qcam, T.to(dev).float())[0]
        megaloc_pick = db_image(names[int(ret_idx)])

        # --- HLoc: coarse NetVLAD pick + its PnP pose rendered on our splat ---
        hloc_render, hloc_t, hloc_r = None, None, None
        if qname in hloc_poses:
            Re, te = hloc_poses[qname]
            h_np = colmap_to_nerf_c2w(Re, te, Rm, s, t_off)
            if k:
                h_np = h_np @ np.linalg.matrix_power(roll if roll is not None else _ROLL_CW, k)
            h_pose = torch.from_numpy(h_np).float()
            hloc_t, hloc_r = pose_error(gt_pose.double(), h_pose.cpu().double())
            hloc_render = render_rgb_and_depth(pipeline, qcam, h_pose.to(dev).float())[0]

        hloc_db = hloc_top.get(qname)
        hloc_pick = db_image(hloc_db) if hloc_db else None

        # --- compose the 5-panel row ---
        panels = [
            ("query photo\n" + qname, query),
            (f"OURS: render @ est pose\n{ours_t*mpu*100:.1f} cm / {ours_r:.1f}°",
             ours_render),
            (f"OURS: MegaLoc pick (seed)\n{names[int(ret_idx)]}", megaloc_pick),
            (f"HLoc: NetVLAD pick\n{hloc_db or 'n/a'}", hloc_pick),
            (f"HLoc: render @ est pose\n"
             + (f"{hloc_t*mpu*100:.1f} cm / {hloc_r:.1f}°" if hloc_t is not None else "no pose"),
             hloc_render),
        ]
        fig, axes = plt.subplots(1, 5, figsize=(26, 6))
        for ax, (title, img) in zip(axes, panels):
            if img is not None:
                ax.imshow(to_np_img(img))
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center")
            ax.set_title(title, fontsize=12)
            ax.axis("off")
        fig.suptitle(
            f"{args.scene}  {stem}   ours {ours_t*mpu*100:.1f} cm / {ours_r:.1f}°   "
            f"vs   HLoc {hloc_t*mpu*100:.1f} cm / {hloc_r:.1f}°"
            if hloc_t is not None else f"{args.scene}  {stem}",
            fontsize=15, y=1.02)
        plt.tight_layout()
        dest = out / f"{stem}_compare.png"
        fig.savefig(dest, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[done] {stem}: ours {ours_t*mpu*100:.1f}cm/{ours_r:.1f}deg  "
              f"hloc {hloc_t*mpu*100:.1f}cm/{hloc_r:.1f}deg  -> {dest}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
