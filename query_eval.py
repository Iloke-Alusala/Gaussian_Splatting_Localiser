"""
query_eval.py - run megaloc -> loftr+pnp on a set of real query photos (the ones
registered into colmap) and log results the same way I do by hand from the
interactive viewer: one report + diagnostic panel + loftr match image per query,
plus a little summary at the end.

the query set is hot-swappable: point --query-dir and --query-gt at a different
folder/json and it just works. runs in the nerfstudio env.

nothing in pipeline_retriever.py is modified - we just borrow its coarse stage,
fine stage and the two drawing helpers.
"""
import argparse
import json
import os
import time
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary, qvec2rotmat

from my_localizer import load_pipeline, freeze_map, pose_error, to_4x4
from my_pose_harness import POSE_THRESHOLDS, threshold_recall
import pipeline_retriever as pr

# scene paths baked in; maps don't change when we swap query images.
SCENES = {
    "stacey": dict(
        name="stacey_room-f4200",
        config=Path("/home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml"),
        scene_dir=Path("/home/chini/3dgs/data/stacey_room_train-f4200-processed"),
        eval_dir=Path("/home/chini/3dgs/eval/stacey/images"),
        query_gt=Path("/home/chini/3dgs/eval/stacey/query_gt.json"),
    ),
    # "stacey": dict(
    #         name="stacey_room-f2500",
    #         config=Path("/home/chini/3dgs/outputs/stacey_room-f2500/splatfacto/2026-10-08_182851/config.yml"),
    #         scene_dir=Path("/home/chini/3dgs/data/stacey_room-f2500-processed"),
    #         eval_dir=Path("/home/chini/3dgs/eval/stacey/images"),
    #         query_gt=Path("/home/chini/3dgs/eval/stacey/query_gt.json"),
    #     ),
    "class1": dict(
        name="class1_train-f2900-processed",
        config=Path("/home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml"),
        scene_dir=Path("/home/chini/3dgs/data/class1_train-f2900-processed"),
        eval_dir=Path("/home/chini/3dgs/eval/class1/images"),
        query_gt=Path("/home/chini/3dgs/eval/class1/query_gt.json"),
    ),
}

# --- colmap world -> nerfstudio frame -----------------------------------------
# the query GT lives in colmap's frame (world->cam, opencv); the pipeline lives
# in nerfstudio's frame. re-deriving nerfstudio's orientation/scale maths by hand
# is fiddly and easy to get wrong, so instead i FIT the transform: the training
# frames exist in both worlds, so i solve for the rotation/scale/offset that
# carries their colmap poses onto their known nerf poses, then apply that to the
# queries. the fit also gives a reproduction error i use as a sanity gate.
def _c2w_opengl(R_w2c, t_w2c):
    """world->cam (opencv) -> camera-to-world in opengl cam axes."""
    w2c = np.eye(4)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1] *= -1          # opencv cam (y down, z fwd) -> opengl (y up, z back)
    c2w[:3, 2] *= -1
    return c2w


def fit_colmap_to_nerf(pipeline, data_dir):
    tj = json.loads((data_dir / "transforms.json").read_text())
    name2imid = {Path(f["file_path"]).name: f["colmap_im_id"] for f in tj["frames"]}
    id2img = {im.id: im for im in read_images_binary(
        str(data_dir / "colmap/sparse/0/images.bin")).values()}

    train = pipeline.datamanager.train_dataset
    names = [Path(p).name for p in train._dataparser_outputs.image_filenames]

    H = np.zeros((3, 3))
    Cc, Cn = [], []
    for i, nm in enumerate(names):
        if nm not in name2imid or name2imid[nm] not in id2img:
            continue
        im = id2img[name2imid[nm]]
        cgl = _c2w_opengl(qvec2rotmat(im.qvec), np.asarray(im.tvec, float))
        nerf = to_4x4(train.cameras[i:i + 1].camera_to_worlds[0].cpu()).numpy().astype(np.float64)
        H += nerf[:3, :3] @ cgl[:3, :3].T            # for the best-fit rotation
        Cc.append(cgl[:3, 3]); Cn.append(nerf[:3, 3])

    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:                         # keep it a proper rotation
        U[:, -1] *= -1; R = U @ Vt
    Cc, Cn = np.array(Cc), np.array(Cn)
    u = Cc @ R.T                                     # rotated colmap centres
    ub, cb = u.mean(0), Cn.mean(0)
    s = float(((u - ub) * (Cn - cb)).sum() / ((u - ub) ** 2).sum())
    t = cb - s * ub
    resid = float(np.abs(s * u + t - Cn).max())
    return R, s, t, resid


def colmap_to_nerf_c2w(R_w2c, t_w2c, R, s, t):
    cgl = _c2w_opengl(R_w2c, t_w2c)
    out = np.eye(4)
    out[:3, :3] = R @ cgl[:3, :3]
    out[:3, 3] = s * (R @ cgl[:3, 3]) + t
    return out


# --- orientation ---------------------------------------------------------------
# the stills are saved landscape with an EXIF rotate flag (tag 6), so the raw
# pixels sit 90 deg off from the upright training frames. colmap registered the
# raw buffer, so query_gt's intrinsics/pose are in that raw frame. we load the
# photo upright (like everything else) and rotate the intrinsics + roll the GT
# pose by the same amount so image, intrinsics and pose stay consistent.
_EXIF_ORIENT = 274
_EXIF_K = {1: 0, 3: 2, 6: 1, 8: 3}        # 90-deg CW turns to make upright
_ROLL_CW = np.array([[0., 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def load_upright(path):
    """photo in its natural orientation + how many 90-deg CW turns that took."""
    im = Image.open(path)
    k = _EXIF_K.get(im.getexif().get(_EXIF_ORIENT, 1), 0)
    arr = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), np.float32) / 255.0
    return torch.from_numpy(arr), k


def rot_intrinsics_cw(fx, fy, cx, cy, W, H, k):
    """turn raw-buffer intrinsics k times 90-deg CW to match the upright image."""
    for _ in range(k):
        fx, fy, cx, cy, W, H = fy, fx, (H - 1 - cy), cx, H, W
    return fx, fy, cx, cy, W, H


def pick_roll_sign(pipeline, cam, query, gt_np, k):
    """an image roll is a camera roll about the optical axis, but i won't guess
    the direction - render both and keep whichever matches the query better."""
    best_m, best_d = _ROLL_CW, 1e9
    for M in (_ROLL_CW, _ROLL_CW.T):
        pose = torch.from_numpy(gt_np @ np.linalg.matrix_power(M, k)).float()
        rgb, _ = pr.render_rgb_and_depth(pipeline, cam, pose.to(pipeline.device))
        d = float((rgb.detach().cpu()[..., :3] - query[..., :3]).abs().mean())
        if d < best_d:
            best_m, best_d = M, d
    return best_m


def query_camera(fx, fy, cx, cy, W, H, device):
    """a nerfstudio camera carrying the query photo's own intrinsics - we render
    the splat at these so the render and the real photo line up pixel-for-pixel."""
    return Cameras(
        camera_to_worlds=torch.eye(4)[:3, :4].unsqueeze(0),
        fx=torch.tensor([[float(fx)]]), fy=torch.tensor([[float(fy)]]),
        cx=torch.tensor([[float(cx)]]), cy=torch.tensor([[float(cy)]]),
        width=torch.tensor([[int(W)]]), height=torch.tensor([[int(H)]]),
        camera_type=CameraType.PERSPECTIVE,
    ).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="stacey", choices=list(SCENES))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--retriever", choices=["clip", "megaloc"], default="megaloc")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--passes", type=int, default=1,
                    help="iterate the fine stage N times (iComMa-style): each pass "
                         "re-renders at the refined pose, so PnP residual shrinks. "
                         "1 = unchanged single shot.")
    args = ap.parse_args()

    sc = SCENES[args.scene]
    query_dir, query_gt, scene_data = sc["eval_dir"], sc["query_gt"], sc["scene_dir"]
    out_dir = args.out_dir or Path(f"query_eval_out/{args.scene}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # point the borrowed coarse stage at the retriever we want, no GUI rotation
    pr.RETRIEVER_NAME = args.retriever
    pr.ROT90_CW = False
    pr._DATASET_NAME = sc["name"]

    print("[info] loading pipeline...")
    _cfg, pipeline, _scale = load_pipeline(sc["config"], args.device)
    freeze_map(pipeline)
    dev = pipeline.device

    # fit the colmap->nerf transform off the training frames, with a sanity gate
    Rm, s, t, resid = fit_colmap_to_nerf(pipeline, scene_data)
    print(f"[info] colmap->nerf fit, worst train reproduction error: {resid:.4f} units")
    if resid > 0.02:
        raise SystemExit(f"[abort] colmap->nerf fit is off ({resid:.3f} units) - "
                         "don't trust the query numbers")

    qgt = json.loads(query_gt.read_text())
    print(f"[info] {len(qgt)} queries from {query_dir}")

    rows = []
    roll = None                              # camera-roll sign, fixed on first turned query
    for qname, e in qgt.items():
        img_path = query_dir / qname
        if not img_path.exists():
            print(f"  [skip] {qname} not found in {query_dir}")
            continue
        stem = Path(qname).stem

        # load upright, then rotate the raw-buffer intrinsics to match
        query, k = load_upright(img_path)
        fx, fy, cx, cy = (float(x) for x in e["params"][:4])
        fx, fy, cx, cy, W, H = rot_intrinsics_cw(fx, fy, cx, cy, e["width"], e["height"], k)
        qcam = query_camera(fx, fy, cx, cy, W, H, dev)

        gt_np = colmap_to_nerf_c2w(np.asarray(e["R_world_to_cam"]),
                                   np.asarray(e["t_world_to_cam"]), Rm, s, t)
        if k:                                # roll the GT pose to match the upright image
            if roll is None:
                roll = pick_roll_sign(pipeline, qcam, query, gt_np, k)
            gt_np = gt_np @ np.linalg.matrix_power(roll, k)
        gt = torch.from_numpy(gt_np).float()

        # run coarse + fine, capturing the console so the report reads exactly
        # like the ones I save from the interactive viewer
        buf = StringIO()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t = time.perf_counter()                 # time the online localisation (coarse+fine)
        with redirect_stdout(buf):
            T_init = pr.coarse_global_retrieval(pipeline, query, device=dev)
            t0, r0 = pose_error(gt, T_init)
            print(f"\n  [coarse] Initial guess error: {t0:.4f} units, {r0:.2f} deg")
            print("  [fine] Starting Efficient LoFTR geometric refinement...")
            # iterate the fine stage: re-render at the refined pose each pass so
            # the render lines up better with the query and PnP residual shrinks.
            # only the last pass saves the match image.
            T_final = T_init
            for _p in range(args.passes):
                T_final = pr.efficient_loftr_localise(
                    pipeline, qcam, query, T_final, device=dev, save_viz=False,
                    viz_tag=stem if _p == args.passes - 1 else None,
                    viz_dir=out_dir)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt_ms = 1000.0 * (time.perf_counter() - _t)
            t1, r1 = pose_error(gt.double(), T_final.cpu().double())
            gtt, fit = gt[:3, 3].numpy(), T_final.cpu().numpy()[:3, 3]
            print("\n  =======================================================")
            print("  [verify] POSE MATRIX COMPARISON")
            print("  =======================================================")
            print(f"  Query Location (X,Y,Z) : [{gtt[0]:.4f}, {gtt[1]:.4f}, {gtt[2]:.4f}] units")
            print(f"  Final Location (X,Y,Z) : [{fit[0]:.4f}, {fit[1]:.4f}, {fit[2]:.4f}] units")
            print("  -------------------------------------------------------")
            print("  Query 4x4 Matrix (Ground Truth):")
            print(np.round(gt.numpy(), 4))
            print("\n  Final 4x4 Matrix (Optimized):")
            print(np.round(T_final.cpu().numpy(), 4))
            print("  =======================================================")
            print(f"  [fine] Final localized error: {t1:.4f} units, {r1:.2f} deg")

        (out_dir / f"{stem}_{args.retriever}_report.txt").write_text(buf.getvalue())
        pr.save_diagnostic_panel(pipeline, qcam, query, gt, T_init, T_final,
                                 str(out_dir / f"{stem}_{args.retriever}_summary.png"))
        # efficient_loftr_localise wrote loftr_matches_<stem>.png; rename to my scheme
        raw_link = out_dir / f"loftr_matches_{stem}.png"
        if raw_link.exists():
            os.replace(raw_link, out_dir / f"{stem}_{args.retriever}_link.png")

        rows.append((qname, t1, r1, dt_ms))
        print(f"  [{stem}] {t1:.4f} units, {r1:.2f} deg, {dt_ms:.0f} ms")

    # little summary so I can see at a glance how the whole set did
    if rows:
        t = np.array([r[1] for r in rows])
        d = np.array([r[2] for r in rows])
        tm = np.array([r[3] for r in rows])
        rec = threshold_recall([(r[1], r[2]) for r in rows], POSE_THRESHOLDS)
        lines = [
            f"{args.retriever} on {len(rows)} queries ({query_dir.name}), "
            f"{args.passes} fine pass(es)",
            f"translation: median {np.median(t):.4f} units ({np.median(t)*100:.2f} cm)  "
            f"mean {t.mean():.4f}",
            f"rotation:    median {np.median(d):.3f} deg  mean {d.mean():.3f}",
            f"time:        median {np.median(tm):.0f} ms/query  (coarse+fine online; "
            f"first query includes model warmup)",
            "recall: " + "  ".join(f"{k}={v:.0f}%" for k, v in rec.items()),
        ]
        if torch.cuda.is_available():
            lines.append(f"gpu peak:    {torch.cuda.max_memory_allocated()/1e9:.2f} GB alloc  "
                         f"{torch.cuda.max_memory_reserved()/1e9:.2f} GB reserved (this process)")
        (out_dir / f"summary_{args.retriever}_p{args.passes}.txt").write_text(
            "\n".join(lines) + "\n")
        print("\n" + "\n".join(lines))
        # append every run to the permanent log, so results never live only in chat
        with open("/home/chini/Desktop/3DGS/code/RESULTS_LOG.md", "a") as f:
            f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} | ours (query_eval) | "
                    f"scene={args.scene} | passes={args.passes}\n"
                    f"splat: {sc['config']}\nqueries: {sc['query_gt']}\n```\n"
                    + "\n".join(lines) + "\n```\n")
    print(f"\n[done] per-query report/summary/link + summary_{args.retriever}.txt in {out_dir}")


if __name__ == "__main__":
    main()
