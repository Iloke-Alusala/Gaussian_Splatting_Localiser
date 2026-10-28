# RUN COMMAND:
# conda run -n nerfstudio python query_eval.py --scene stacey --passes 2 [--fine photometric --photo-loss lpips] [--seed-source perturb]

# run megaloc -> loftr+pnp (or the photometric refiner) on the real query photos registered into colmap. no viewer, same reports i save by hand
# one report + diagnostic panel + loftr match image per query, then a summary that also gets appended to $RESULTS_LOG. swap the query set via SCENES below

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

from my_localizer import load_pipeline, freeze_map, pose_error, to_4x4, localise
from my_pose_harness import (POSE_THRESHOLDS, threshold_recall, wilson_ci,
                             make_perturbation, perturb_pose)
import pipeline_retriever as pr

# scene paths baked in; maps don't change when we swap query images.
SCENES = {
    "stacey": dict(
        name="stacey_room-f4200",
        config=Path("/home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml"),
        scene_dir=Path("/home/chini/3dgs/data/stacey_room_train-f4200-processed"),
        eval_dir=Path("/home/chini/3dgs/eval/stacey/images"),
        query_gt=Path("/home/chini/3dgs/eval/stacey/query_gt.json"),
        # metric scale: door frame measured in cloudcompare -> metres per scene unit
        # (mean of accepted trials). scene units are per-reconstruction, so each scene
        # carries its own factor - see the scale-recovery sheet.
        m_per_unit=1.9826,
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
        # metric scale (see stacey above): metres per scene unit from the door frame
        m_per_unit=3.5601,
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
    """fit rotation R, scale s, offset t that carry the train frames' colmap poses onto their nerf poses.
    resid is the worst train camera centre error after the fit (scene units), used as the sanity gate."""
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
    """apply the fitted transform to one colmap world->cam pose, giving a nerfstudio c2w."""
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
                         "1 = unchanged single shot. (loftr fine stage only)")
    # fine stage: the loftr+pnp path (default) or the 3dgs photometric refiner.
    # photometric is the baseline the thesis measures pnp against, so it has to run
    # from the SAME coarse seeds on the SAME query set and report the same summary.
    ap.add_argument("--fine", choices=["loftr", "photometric"], default="loftr",
                    help="fine stage. photometric = my_localizer.localise (Adam on "
                         "se(3), descends the render-vs-query loss). it iterates to "
                         "convergence internally, so it ignores --passes.")
    ap.add_argument("--photo-iters", type=int, default=250,
                    help="photometric iteration cap")
    ap.add_argument("--photo-lr-trans", type=float, default=2e-3,
                    help="Adam lr for the translation group (rho)")
    ap.add_argument("--photo-lr-rot", type=float, default=2e-3,
                    help="Adam lr for the rotation group (phi)")
    ap.add_argument("--photo-loss", choices=["mse", "lpips"], default="mse",
                    help="photometric objective")
    ap.add_argument("--photo-downscale", type=int, default=4,
                    help="render + optimise the photometric stage at 1/N resolution "
                         "(iNeRF-style coarse optimisation): N^2 fewer pixels per "
                         "iteration, and the smoother loss widens the capture basin. "
                         "pose error is still measured at full scale. 1 = full res.")
    ap.add_argument("--limit", type=int, default=0,
                    help="only run the first N queries (0 = all). handy for the slow "
                         "photometric baseline, where a subset + CI is enough.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these query stems, e.g. q_00052 q_00112")
    ap.add_argument("--no-link", action="store_true",
                    help="skip the per-query LoFTR match image. it is drawn after the "
                         "timer either way, so this only saves wall-clock, not ms/query.")
    # seed source: the real MegaLoc coarse seed (default), or a controlled seed made
    # by perturbing GT by a known magnitude - lets us ask whether the fine stage
    # works when seeded INSIDE its basin, vs wherever MegaLoc happened to land.
    ap.add_argument("--seed-source", choices=["megaloc", "perturb"], default="megaloc",
                    help="perturb = seed from GT by --seed-trans/--seed-rot instead of MegaLoc")
    ap.add_argument("--seed-trans", type=float, default=0.05,
                    help="perturb seed: translation magnitude (scene units)")
    ap.add_argument("--seed-rot", type=float, default=2.0,
                    help="perturb seed: rotation magnitude (degrees)")
    ap.add_argument("--seed-base", type=int, default=0,
                    help="base for the per-query perturbation RNG; fix it so the "
                         "photometric and loftr runs get IDENTICAL seeds (like-for-like)")
    args = ap.parse_args()

    sc = SCENES[args.scene]
    query_dir, query_gt, scene_data = sc["eval_dir"], sc["query_gt"], sc["scene_dir"]
    m_per_unit = sc["m_per_unit"]        # scene units -> real metres (door-frame scale)
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
    photo_stats = []                         # (iters, termination) per query, photometric only
    roll = None                              # camera-roll sign, fixed on first turned query
    # build the LPIPS model once and reuse it - rebuilding per query OOMs the card
    photo_lpips = None
    if args.fine == "photometric" and args.photo_loss == "lpips":
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        photo_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(dev)
    items = list(qgt.items())
    if args.limit:
        items = items[:args.limit]
    if args.only:
        want = set(args.only)
        items = [(q, e) for (q, e) in items if Path(q).stem in want]
    for _qi, (qname, e) in enumerate(items):
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
            if args.seed_source == "perturb":
                # controlled seed: perturb GT by a fixed magnitude, random direction.
                # generator keyed by query index + --seed-base so a photometric run
                # and a loftr run at the same base get the exact same seeds.
                g = torch.Generator().manual_seed(args.seed_base + _qi)
                delta = make_perturbation(args.seed_trans, args.seed_rot, generator=g)
                T_init = perturb_pose(gt, delta).float()
            else:
                T_init = pr.coarse_global_retrieval(pipeline, query, device=dev)
            t0, r0 = pose_error(gt, T_init)
            print(f"\n  [coarse] Initial guess error: {t0:.4f} units, {r0:.2f} deg")
            T_final = T_init
            T_pre_last = T_init
            if args.fine == "loftr":
                print("  [fine] Starting Efficient LoFTR geometric refinement...")
                # iterate the fine stage: re-render at the refined pose each pass so
                # the render lines up better with the query and PnP residual shrinks.
                # NO viz in here - drawing + writing the match png costs ~105 ms a
                # query, and timing that inflates every ms/query i report. the link
                # image gets regenerated after the timer stops instead.
                for _p in range(args.passes):
                    T_pre_last = T_final
                    T_final = pr.efficient_loftr_localise(
                        pipeline, qcam, query, T_final, device=dev, save_viz=False,
                        viz_tag=None, viz_dir=out_dir)
            else:
                # photometric baseline: descend the render-vs-query loss from the
                # same coarse seed. iterates internally; record iters + why it
                # stopped so we can characterise the convergence basin.
                print("  [fine] Starting 3DGS photometric refinement...")
                # optimise at 1/ds res: fewer pixels per render = much faster, and
                # a coarser image is a smoother objective = wider capture basin.
                ds = args.photo_downscale
                if ds > 1:
                    pw, ph = max(int(W) // ds, 1), max(int(H) // ds, 1)
                    pcam = query_camera(fx / ds, fy / ds, cx / ds, cy / ds, pw, ph, dev)
                    q_opt = torch.nn.functional.interpolate(
                        query[..., :3].permute(2, 0, 1).unsqueeze(0),
                        size=(ph, pw), mode="bilinear", align_corners=False
                    ).squeeze(0).permute(1, 2, 0).contiguous()
                else:
                    pcam, q_opt = qcam, query
                res = localise(
                    pipeline, pcam, q_opt, T_init.to(dev).float(),
                    max_iters=args.photo_iters,
                    lr_trans=args.photo_lr_trans, lr_rot=args.photo_lr_rot,
                    loss_type=args.photo_loss, debug_name=stem, lpips_fn=photo_lpips)
                T_final = res.T_final
                photo_stats.append((res.iterations, res.termination))
                del res                       # drop the pose/loss history before the next query
                torch.cuda.empty_cache()
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

        # append each run instead of clobbering, so successive refinement passes
        # (e.g. the p1/p2/p3 sweep) stack in one file rather than wiping each other.
        run_hdr = (f"\n{'='*60}\n"
                   f"# {time.strftime('%Y-%m-%d %H:%M')} | fine={args.fine} | "
                   f"passes={args.passes} | seed={args.seed_source}\n"
                   f"{'='*60}\n")
        with open(out_dir / f"{stem}_{args.retriever}_report.txt", "a") as _rf:
            _rf.write(run_hdr + buf.getvalue())
        # redo the last fine pass purely to draw the match image, now that the
        # clock has stopped. costs a render+match in wall-clock but keeps the
        # reported ms/query honest. --no-link skips it entirely.
        if args.fine == "loftr" and not args.no_link:
            with redirect_stdout(StringIO()):
                pr.efficient_loftr_localise(pipeline, qcam, query, T_pre_last,
                                            device=dev, save_viz=False,
                                            viz_tag=stem, viz_dir=out_dir)
        pr.save_diagnostic_panel(pipeline, qcam, query, gt, T_init, T_final,
                                 str(out_dir / f"{stem}_{args.retriever}_summary.png"))
        # efficient_loftr_localise wrote loftr_matches_<stem>.png; rename to my scheme
        raw_link = out_dir / f"loftr_matches_{stem}.png"
        if raw_link.exists():
            os.replace(raw_link, out_dir / f"{stem}_{args.retriever}_link.png")
        # localise() drops its query|init|final / diff panel in room_output_LPIP;
        # pull it into this scene's out_dir so the photometric runs stay separated
        photo_panel = Path("/home/chini/Desktop/3DGS/code/room_output_LPIP") / f"{stem}.png"
        if photo_panel.exists():
            os.replace(photo_panel, out_dir / f"{stem}_{args.retriever}_photo.png")

        rows.append((qname, t1, r1, dt_ms))
        print(f"  [{stem}] {t1*m_per_unit*100:.1f} cm, {r1:.2f} deg, {dt_ms:.0f} ms  ({t1:.4f} u)")

    # little summary so I can see at a glance how the whole set did
    if rows:
        t = np.array([r[1] for r in rows])
        t_m = t * m_per_unit          # scene units -> real metres via the door-frame scale
        d = np.array([r[2] for r in rows])
        tm = np.array([r[3] for r in rows])
        # thresholds are in metres, so feed the metric error: the cm labels are now real cm
        rec = threshold_recall([(tv, r[2]) for tv, r in zip(t_m, rows)], POSE_THRESHOLDS)
        n = len(rows)
        # each recall is a proportion over a small sample, so quote a 95% wilson
        # interval next to it: name=87% [80-92]
        rec_parts = []
        for name, v in rec.items():
            lo, hi = wilson_ci(round(v * n / 100), n)
            rec_parts.append(f"{name}={v:.0f}% [{lo:.0f}-{hi:.0f}]")
        fine_desc = (f"{args.passes} fine pass(es)" if args.fine == "loftr"
                     else f"photometric fine ({args.photo_loss}, {args.photo_iters}-iter cap)")
        lines = [
            f"{args.retriever} on {n} queries ({query_dir.name}), {fine_desc}",
            f"translation: median {np.median(t_m)*100:.2f} cm  mean {t_m.mean()*100:.2f} cm  "
            f"(median {np.median(t):.4f} u @ {m_per_unit:.4f} m/u)",
            f"rotation:    median {np.median(d):.3f} deg  mean {d.mean():.3f}",
            f"time:        median {np.median(tm):.0f} ms/query  (coarse+fine online; "
            f"first query includes model warmup)",
            "recall: " + "  ".join(rec_parts) + "  (95% wilson CI in [])",
            # metric scale recovered from a hand-measured door frame, so the recall
            # cm are now real centimetres (not the old 1-unit=1-m placeholder)
            f"note: scale {m_per_unit:.4f} m/scene-unit (door-frame measurement); recall cm are real cm",
        ]
        # photometric: report the settings actually used + the convergence basin
        # evidence (iteration distribution, how many never converged inside the cap)
        if args.fine == "photometric" and photo_stats:
            its = np.array([s[0] for s in photo_stats])
            cap_hits = sum(1 for _, term in photo_stats if term == "max_iters")
            lines += [
                f"photometric: lr_trans={args.photo_lr_trans:g} lr_rot={args.photo_lr_rot:g}  "
                f"loss={args.photo_loss}  cap={args.photo_iters}  "
                f"schedule=cosine(eta_min=1e-6)  stops: loss<1e-9 / |step|<1e-6 / max_iters",
                f"iters:  median {int(np.median(its))}  mean {its.mean():.0f}  "
                f"min {its.min()}  max {its.max()}",
                f"hit cap without converging: {cap_hits}/{len(photo_stats)}",
            ]
        if torch.cuda.is_available():
            lines.append(f"gpu peak:    {torch.cuda.max_memory_allocated()/1e9:.2f} GB alloc  "
                         f"{torch.cuda.max_memory_reserved()/1e9:.2f} GB reserved (this process)")
        tag = f"p{args.passes}" if args.fine == "loftr" else "photo"
        # keep controlled-seed runs in their own files so they never clobber the
        # MegaLoc-seeded rows: e.g. summary_megaloc_photo_seed0.05-2.txt
        if args.seed_source == "perturb":
            tag += f"_seed{args.seed_trans:g}-{args.seed_rot:g}"
        (out_dir / f"summary_{args.retriever}_{tag}.txt").write_text(
            "\n".join(lines) + "\n")
        print("\n" + "\n".join(lines))
        seed_desc = (f"perturb {args.seed_trans:g}u/{args.seed_rot:g}deg"
                     if args.seed_source == "perturb" else "megaloc")
        # append every run to the permanent log, so results never live only in chat.
        # $RESULTS_LOG lets these metric reruns go to a fresh sheet without touching
        # the old RESULTS_LOG.md; unset = old behaviour.
        log_path = os.environ.get("RESULTS_LOG", "/home/chini/Desktop/3DGS/code/RESULTS_LOG.md")
        with open(log_path, "a") as f:
            f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} | ours (query_eval) | "
                    f"scene={args.scene} | fine={args.fine} | passes={args.passes} | "
                    f"seed={seed_desc}\n"
                    f"splat: {sc['config']}\nqueries: {sc['query_gt']}\n```\n"
                    + "\n".join(lines) + "\n```\n")
    print(f"\n[done] per-query report/summary/link + summary_{args.retriever}.txt in {out_dir}")


if __name__ == "__main__":
    main()
