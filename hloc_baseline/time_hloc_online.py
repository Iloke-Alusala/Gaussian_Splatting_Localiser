"""
Clean per-query ONLINE timing for HLoc, so we compare against our pipeline on
real numbers instead of log timestamps. The offline map (db features, db
matches, triangulation) is already built and reused; here we time only the work
HLoc does per query: SuperPoint + NetVLAD extraction, retrieval, LightGlue
matching against the retrieved db frames, and localize_sfm (PnP).

Each stage warms the model up once (untimed) then runs fresh with overwrite=True,
GPU-synced around the timed call. Runs in the hloc env.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from hloc import (extract_features, match_features, pairs_from_retrieval,
                  localize_sfm)
from hloc.utils.read_write_model import read_images_binary

from run_hloc import SCENES      # same paths as the run that built the map


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    """run once to warm the model, then time a fresh run."""
    fn()                                     # warmup (loads model / fills nothing we keep)
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="stacey", choices=list(SCENES))
    ap.add_argument("--resize-max", type=int, default=1024)
    ap.add_argument("--num-loc", type=int, default=20)
    args = ap.parse_args()

    sc = SCENES[args.scene]
    HLOC = Path(f"/home/chini/3dgs/hloc_out/{args.scene}/hloc")   # run_hloc's output dir
    tmp = HLOC / "_timing"
    tmp.mkdir(exist_ok=True)

    qgt = json.loads(sc["query_gt"].read_text())
    query_names = list(qgt)                              # basenames, as in run_hloc
    db_names = sorted(im.name for im in
                      read_images_binary(str(sc["scene_dir"] / "colmap/sparse/0/images.bin")).values())
    N = len(query_names)

    sp_conf = dict(extract_features.confs["superpoint_max"])
    sp_conf["preprocessing"] = dict(sp_conf["preprocessing"], resize_max=args.resize_max)
    nv_conf = extract_features.confs["netvlad"]
    mt_conf = match_features.confs["superpoint+lightglue"]

    main_feats = HLOC / "feats-superpoint-n4096-rmax1600.h5"      # db+query, cached
    global_h5 = HLOC / "global-feats-netvlad.h5"                  # db+query, cached
    ref_sfm = HLOC / "sfm_superpoint+lightglue"
    q_intr = HLOC / "query_intrinsics.txt"
    loc_pairs = HLOC / f"pairs-query-netvlad{args.num_loc}.txt"
    warm_pairs = tmp / "warm_pairs.txt"
    warm_pairs.write_text("\n".join(loc_pairs.read_text().splitlines()[:2]) + "\n")

    # 1. SuperPoint on the queries
    t_sp = timed(lambda: extract_features.main(
        sp_conf, sc["eval_dir"], feature_path=tmp / "sp.h5",
        image_list=query_names, overwrite=True))

    # 2. NetVLAD on the queries
    t_nv = timed(lambda: extract_features.main(
        nv_conf, sc["eval_dir"], feature_path=tmp / "nv.h5",
        image_list=query_names, overwrite=True))

    # 3. retrieval (NN search over cached global descriptors)
    t_re = timed(lambda: pairs_from_retrieval.main(
        global_h5, tmp / "pairs.txt", args.num_loc,
        query_list=query_names, db_list=db_names))

    # 4. LightGlue matching, query vs its retrieved db frames
    def _warm_then_match():
        match_features.main(mt_conf, warm_pairs, main_feats,
                            matches=tmp / "warm_m.h5", overwrite=True)
    _warm_then_match()                       # load LightGlue
    sync(); t0 = time.perf_counter()
    match_features.main(mt_conf, loc_pairs, main_feats,
                        matches=tmp / "loc_m.h5", overwrite=True)
    sync(); t_mt = time.perf_counter() - t0

    # 5. localize_sfm (covis clustering + PnP)
    sync(); t0 = time.perf_counter()
    localize_sfm.main(ref_sfm, q_intr, loc_pairs, main_feats, tmp / "loc_m.h5",
                      tmp / "results.txt", covisibility_clustering=False)
    sync(); t_lo = time.perf_counter() - t0

    total = t_sp + t_nv + t_re + t_mt + t_lo
    rows = [("SuperPoint extract", t_sp), ("NetVLAD extract", t_nv),
            ("retrieval (NN search)", t_re),
            (f"LightGlue match ({args.num_loc} db)", t_mt),
            ("localize_sfm (PnP)", t_lo), ("TOTAL online", total)]
    out = [f"HLoc full online timing, {N} queries, resize_max={args.resize_max}, "
           f"num_loc={args.num_loc}"]
    for name, sec in rows:
        out.append(f"  {name:<26} {1000*sec/N:7.1f} ms/query   ({sec:.2f}s / {N})")
    if torch.cuda.is_available():
        out.append(f"  {'gpu peak':<26} {torch.cuda.max_memory_allocated()/1e9:.2f} GB alloc  "
                   f"{torch.cuda.max_memory_reserved()/1e9:.2f} GB reserved (this process)")
    text = "\n".join(out)
    (HLOC / "timing_online.txt").write_text(text + "\n")
    # append every run to the permanent log, so results never live only in chat
    with open("/home/chini/Desktop/3DGS/code/RESULTS_LOG.md", "a") as f:
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} | hloc online timing | scene={args.scene}\n"
                f"db model: {sc['scene_dir']}\nqueries: {sc['query_gt']}\n```\n{text}\n```\n")
    print(text)


if __name__ == "__main__":
    main()
