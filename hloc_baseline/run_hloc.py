# RUN COMMAND:
# conda run -n hloc python hloc_baseline/run_hloc.py --scene stacey

# HLoc baseline: SuperPoint + LightGlue + NetVLAD, localised with PnP against a SuperPoint cloud triangulated on our COLMAP poses (db = sparse/0, already query-free)
# the map only depends on the train frames, so it is built once and reused when the query set changes. pycolmap must be 3.13.0 (4.x breaks triangulation -> 0 points)

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from hloc import (extract_features, match_features, pairs_from_covisibility,
                  pairs_from_retrieval, triangulation, localize_sfm)
from hloc.utils.read_write_model import read_images_binary, qvec2rotmat

# scene paths baked in. maps don't change when query images do.
SCENES = {
    "stacey": dict(
        scene_dir=Path("/home/chini/3dgs/data/stacey_room_train-f4200-processed"),
        config=Path("/home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/2026-10-11_105953/config.yml"),
        eval_dir=Path("/home/chini/3dgs/eval/stacey/images"),
        query_gt=Path("/home/chini/3dgs/eval/stacey/query_gt.json"),
    ),
    # "stacey": dict(
    #     scene_dir=Path("/home/chini/3dgs/data/stacey_room-f2500-processed"),
    #     config=Path("/home/chini/3dgs/outputs/stacey_room-f2500/splatfacto/2026-10-08_182851/config.yml"),
    #     eval_dir=Path("/home/chini/3dgs/eval/stacey/images"),
    #     query_gt=Path("/home/chini/3dgs/eval/stacey/query_gt.json"),
    # ),
    "class1": dict(
        scene_dir=Path("/home/chini/3dgs/data/class1_train-f2900-processed"),
        config=Path("/home/chini/3dgs/outputs/class1_train-f2900/splatfacto/2026-10-11_104755/config.yml"),
        eval_dir=Path("/home/chini/3dgs/eval/class1/images"),
        query_gt=Path("/home/chini/3dgs/eval/class1/query_gt.json"),
    ),
}

# standard pose-recall thresholds, (translation metres/nerf-units, rotation deg)
THRESHOLDS = [(0.05, 5.0), (0.10, 5.0), (0.25, 2.0), (0.50, 10.0)]


def scale_of(sc):
    """colmap -> nerfstudio scale from dataparser_transforms.json, so hloc errors land in the same units as ours"""
    dp = sc["config"].parent / "dataparser_transforms.json"
    return float(json.loads(dp.read_text())["scale"])


def write_query_intrinsics(qgt, out):
    """Aachen-style '<name> MODEL w h params...' per query, straight from query_gt."""
    lines = [" ".join(map(str, [n, e["camera_model"], e["width"], e["height"]] + list(e["params"])))
             for n, e in qgt.items()]
    out.write_text("\n".join(lines) + "\n")


def centre(R, t):
    """camera centre in world from a world->cam R, t"""
    return -R.T @ np.asarray(t, float)


def score(results, qgt, scale, logs_path, time_ms, out_dir):
    """Compare estimated poses to COLMAP GT; write results.csv + summary.txt."""
    preds = {}
    for ln in results.read_text().splitlines():
        if not ln.strip():
            continue
        p = ln.split()
        preds[p[0]] = (qvec2rotmat(np.array(p[1:5], float)), np.array(p[5:8], float))
    inliers = {}
    if logs_path.exists():
        import pickle
        for n, lg in pickle.load(open(logs_path, "rb")).get("loc", {}).items():
            inliers[n] = int((lg.get("PnP_ret") or {}).get("num_inliers", 0))

    rows, t_errs, r_errs, n_loc = [], [], [], 0
    for name, e in qgt.items():
        Rg = np.asarray(e["R_world_to_cam"], float)
        tg = np.asarray(e["t_world_to_cam"], float)
        if name in preds:
            Re, te = preds[name]
            t_err = float(np.linalg.norm(centre(Re, te) - centre(Rg, tg)))
            cos = np.clip((np.trace(Rg.T @ Re) - 1) / 2, -1, 1)
            r_err = float(np.rad2deg(np.arccos(cos)))
            n_loc += 1
        else:
            t_err, r_err = float("inf"), 180.0
        rows.append((name, t_err, r_err, inliers.get(name, 0)))
        t_errs.append(t_err); r_errs.append(r_err)

    import csv
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "t_err", "r_err", "n_inliers", "time_ms", "t_err_ns"])
        for n, te, re, ni in rows:
            w.writerow([n, round(te, 5), round(re, 4), ni, time_ms, round(te * scale, 5)])

    # wilson 95% CI for a proportion, as percents. copied (not imported from
    # my_pose_harness) on purpose: that module pulls nerfstudio deps and this
    # runs in the hloc env, which must stay off them.
    def wilson_ci(k, nn, z=1.96):
        if nn <= 0:
            return 0.0, 0.0
        p = k / nn
        den = 1.0 + z * z / nn
        c = (p + z * z / (2 * nn)) / den
        h = z * math.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / den
        return max(0.0, 100.0 * (c - h)), min(100.0, 100.0 * (c + h))

    t, r = np.array(t_errs), np.array(r_errs)
    fin = np.isfinite(t)
    rec = {f"{int(tt*100)}cm_{rr:g}deg":
           100 * np.mean((t * scale < tt) & (r < rr)) for tt, rr in THRESHOLDS}
    lines = [
        f"HLoc on {len(rows)} queries   localised {n_loc}/{len(rows)}",
        f"translation: median {np.median(t[fin])*scale:.4f} units  "
        f"mean {t[fin].mean()*scale:.4f} units",
        f"rotation:    median {np.median(r[fin]):.3f} deg  mean {r[fin].mean():.3f}",
        f"time:        {time_ms:.1f} ms/query (amortized localize_sfm)",
        "recall: " + "  ".join(
            f"{k}={v:.0f}% [{wilson_ci(round(v*len(rows)/100), len(rows))[0]:.0f}-"
            f"{wilson_ci(round(v*len(rows)/100), len(rows))[1]:.0f}]"
            for k, v in rec.items()) + "  (95% wilson CI in [])",
        # same unit as query_eval's "units" (colmap x dataparser scale). no metric
        # scale on these maps, so the cm in the recall names is just an assumption
        "note: errors are in nerfstudio scene units; the recall labels' cm assume 1 unit = 1 m",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def archive_run(out, sc, args):
    """snapshot this run's query-specific logs so a later run on a different query
    set can't clobber them. map files (feats/sfm/netvlad) stay put and get reused."""
    import shutil
    stamp = time.strftime("%Y-%m-%d_%H%M")
    arc = out / "runs" / stamp
    arc.mkdir(parents=True, exist_ok=True)
    keep = ["summary.txt", "results.csv", "results.txt", "results.txt_logs.pkl",
            "query_intrinsics.txt", f"pairs-query-netvlad{args.num_loc}.txt",
            "loc_matches.h5"]
    for f in keep:
        p = out / f
        if p.exists():
            shutil.copy2(p, arc / f)
    # keep the exact script that made these numbers right next to them
    shutil.copy2(__file__, arc / "run_hloc.py")
    (arc / "meta.txt").write_text(
        f"scene={args.scene}\ndb_model={sc['scene_dir']}\nquery_gt={sc['query_gt']}\n"
        f"resize_max={args.resize_max} num_covis={args.num_covis} num_loc={args.num_loc}\n"
        f"time={stamp}\n")
    print(f"[archive] run logs saved -> {arc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="stacey", choices=list(SCENES))
    ap.add_argument("--resize-max", type=int, default=1024)
    ap.add_argument("--num-covis", type=int, default=20)
    ap.add_argument("--num-loc", type=int, default=20)
    args = ap.parse_args()

    sc = SCENES[args.scene]
    images_dir = sc["scene_dir"] / "images"           # db frames
    ref_model = sc["scene_dir"] / "colmap/sparse/0"    # db poses, query-free
    out = Path(f"/home/chini/3dgs/hloc_out/{args.scene}/hloc")
    out.mkdir(parents=True, exist_ok=True)

    qgt = json.loads(sc["query_gt"].read_text())
    query_names = list(qgt)                             # basenames, e.g. q_00001.jpg
    db_names = sorted(im.name for im in read_images_binary(str(ref_model / "images.bin")).values())
    print(f"[info] {args.scene}: {len(query_names)} queries, {len(db_names)} db frames")

    q_intr = out / "query_intrinsics.txt"
    write_query_intrinsics(qgt, q_intr)

    feat_conf = dict(extract_features.confs["superpoint_max"])
    feat_conf["preprocessing"] = dict(feat_conf["preprocessing"], resize_max=args.resize_max)
    nv_conf = extract_features.confs["netvlad"]
    mt_conf = match_features.confs["superpoint+lightglue"]
    feats = out / (feat_conf["output"] + ".h5")

    # local features: db (cached) + queries straight from the eval dir (no symlink)
    extract_features.main(feat_conf, images_dir, feature_path=feats, image_list=db_names, as_half=True)
    extract_features.main(feat_conf, sc["eval_dir"], feature_path=feats, image_list=query_names, as_half=True)

    # map: build once from the db, reuse ever after (queries don't change it)
    reference_sfm = out / "sfm_superpoint+lightglue"
    if (reference_sfm / "points3D.bin").exists():
        print("[map] reusing existing triangulated map")
    else:
        print("[map] building db map (one-time): covis + LightGlue + triangulation")
        sfm_pairs = out / f"pairs-db-covis{args.num_covis}.txt"
        pairs_from_covisibility.main(ref_model, sfm_pairs, num_matched=args.num_covis)
        sfm_matches = match_features.main(mt_conf, sfm_pairs, feats, matches=out / "db_matches.h5")
        triangulation.main(reference_sfm, ref_model, images_dir, sfm_pairs, feats, sfm_matches)

    # retrieval: query -> db, then match those pairs
    global_desc = out / (nv_conf["output"] + ".h5")
    extract_features.main(nv_conf, images_dir, feature_path=global_desc, image_list=db_names)
    extract_features.main(nv_conf, sc["eval_dir"], feature_path=global_desc, image_list=query_names)
    loc_pairs = out / f"pairs-query-netvlad{args.num_loc}.txt"
    pairs_from_retrieval.main(global_desc, loc_pairs, args.num_loc,
                              query_list=query_names, db_list=db_names)
    loc_matches = match_features.main(mt_conf, loc_pairs, feats, matches=out / "loc_matches.h5")

    # localise (timed) + score
    results = out / "results.txt"
    t0 = time.perf_counter()
    localize_sfm.main(reference_sfm, q_intr, loc_pairs, feats, loc_matches, results,
                      covisibility_clustering=False)
    time_ms = 1000.0 * (time.perf_counter() - t0) / max(1, len(query_names))
    print(f"[done] localize {time_ms:.1f} ms/query -> {results}")
    score(results, qgt, scale_of(sc), Path(f"{results}_logs.pkl"), time_ms, out)
    # append every run to the permanent log, so results never live only in chat
    with open("/home/chini/Desktop/3DGS/code/RESULTS_LOG.md", "a") as f:
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} | hloc (run_hloc) | scene={args.scene}\n"
                f"db model: {sc['scene_dir']}\nqueries: {sc['query_gt']}\n```\n"
                + (out / "summary.txt").read_text() + "```\n")
    # permanent per-run snapshot so re-runs on other query sets don't overwrite it
    archive_run(out, sc, args)


if __name__ == "__main__":
    main()
