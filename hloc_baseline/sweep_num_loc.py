# RUN COMMAND:
# conda run -n hloc python hloc_baseline/sweep_num_loc.py --scene stacey --num-loc 1 2 5 10 20

# HLoc accuracy vs latency against num_loc (how many retrieved db frames get LightGlue-matched per query), so the speed comparison is a curve and not one point
# reuses run_hloc's map + features, and only redoes retrieval -> LightGlue -> PnP per num_loc. scored with run_hloc.score(), latency adds the fixed extraction cost from timing_online.txt

import argparse
import json
import re
import time
from pathlib import Path

import torch
from hloc import match_features, pairs_from_retrieval, localize_sfm
from hloc.utils.read_write_model import read_images_binary

from run_hloc import SCENES, score, scale_of


def sync():
    """wait for the GPU to finish, so the timers measure the actual work"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def fixed_cost_ms(hloc_dir):
    """per-query cost that does NOT depend on num_loc: SuperPoint + NetVLAD
    extraction + NN retrieval search, read straight from the headline timing run."""
    txt = (hloc_dir / "timing_online.txt").read_text()
    def grab(label):
        m = re.search(rf"{re.escape(label)}\s+([\d.]+) ms/query", txt)
        return float(m.group(1)) if m else 0.0
    sp = grab("SuperPoint extract")
    nv = grab("NetVLAD extract")
    ret = grab("retrieval (NN search)")
    return sp + nv + ret, (sp, nv, ret)


def recall_from_summary(summary_path):
    """pull the 'recall: 5cm_5deg=72% ...' line HLoc's score() wrote."""
    for ln in Path(summary_path).read_text().splitlines():
        if ln.startswith("recall:"):
            return dict(re.findall(r"(\d+cm_\d+deg)=(\d+)%", ln))
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="stacey", choices=list(SCENES))
    ap.add_argument("--num-loc", type=int, nargs="+", default=[1, 5, 10, 20])
    args = ap.parse_args()

    sc = SCENES[args.scene]
    out = Path(f"/home/chini/3dgs/hloc_out/{args.scene}/hloc")   # run_hloc's dir
    sweep = out / "_sweep"
    sweep.mkdir(exist_ok=True)

    qgt = json.loads(sc["query_gt"].read_text())
    query_names = list(qgt)
    db_names = sorted(im.name for im in read_images_binary(
        str(sc["scene_dir"] / "colmap/sparse/0/images.bin")).values())
    N = len(query_names)

    feats = out / "feats-superpoint-n4096-rmax1600.h5"
    global_desc = out / "global-feats-netvlad.h5"
    reference_sfm = out / "sfm_superpoint+lightglue"
    q_intr = out / "query_intrinsics.txt"
    mt_conf = match_features.confs["superpoint+lightglue"]
    for p in (feats, global_desc, q_intr):
        assert p.exists(), f"missing cached artifact {p} (run run_hloc.py first)"

    fixed, (sp, nv, ret) = fixed_cost_ms(out)
    print(f"[info] fixed per-query cost {fixed:.1f} ms (SuperPoint {sp:.1f} + "
          f"NetVLAD {nv:.1f} + retrieval {ret:.1f}); map + features reused\n")

    scale = scale_of(sc)
    rows = []
    for nl in args.num_loc:
        d = sweep / f"nl{nl}"
        d.mkdir(exist_ok=True)
        loc_pairs = d / f"pairs-netvlad{nl}.txt"
        loc_matches = d / "loc_matches.h5"
        results = d / "results.txt"

        # retrieval top-nl (timed)
        sync(); t0 = time.perf_counter()
        pairs_from_retrieval.main(global_desc, loc_pairs, nl,
                                  query_list=query_names, db_list=db_names)
        sync(); t_ret = time.perf_counter() - t0

        # LightGlue match those nl pairs/query. warm once, then timed fresh run.
        match_features.main(mt_conf, loc_pairs, feats, matches=loc_matches, overwrite=True)
        sync(); t0 = time.perf_counter()
        match_features.main(mt_conf, loc_pairs, feats, matches=loc_matches, overwrite=True)
        sync(); t_match = time.perf_counter() - t0

        # PnP localisation (timed)
        sync(); t0 = time.perf_counter()
        localize_sfm.main(reference_sfm, q_intr, loc_pairs, feats, loc_matches,
                          results, covisibility_clustering=False)
        sync(); t_loc = time.perf_counter() - t0

        match_ms = 1000.0 * t_match / N
        loc_ms = 1000.0 * t_loc / N
        total_ms = fixed + match_ms + loc_ms
        score(results, qgt, scale, Path(f"{results}_logs.pkl"), total_ms, d)
        rec = recall_from_summary(d / "summary.txt")
        rows.append((nl, rec, match_ms, loc_ms, total_ms))
        print(f"\n[num_loc={nl:2d}] match {match_ms:6.1f}  pnp {loc_ms:5.1f}  "
              f"TOTAL {total_ms:6.1f} ms/query   recall {rec}\n" + "-"*70)

    # final table
    print("\n" + "=" * 78)
    print(f"HLoc num_loc sweep  scene={args.scene}  ({N} queries, map+features reused)")
    print("=" * 78)
    hdr = ["num_loc", "match_ms", "pnp_ms", "TOTAL_ms", "5cm_5deg", "10cm_5deg",
           "25cm_2deg", "50cm_10deg"]
    print("  ".join(h.rjust(9) for h in hdr))
    for nl, rec, mm, lm, tm in rows:
        print("  ".join([f"{nl:9d}", f"{mm:9.1f}", f"{lm:9.1f}", f"{tm:9.1f}",
                         f"{rec.get('5cm_5deg','?')+'%':>9}", f"{rec.get('10cm_5deg','?')+'%':>9}",
                         f"{rec.get('25cm_2deg','?')+'%':>9}", f"{rec.get('50cm_10deg','?')+'%':>9}"]))
    print("=" * 78)
    print(f"note: TOTAL = measured match+PnP here + fixed {fixed:.0f} ms/query "
          f"(SuperPoint+NetVLAD+retrieval, constant in num_loc)")


if __name__ == "__main__":
    main()
