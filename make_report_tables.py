"""
make_report_tables.py - build the thesis comparison tables straight from the
summary files the harnesses already write, so the numbers in the report can't
drift from the numbers on disk. reads only; writes report_tables.md.

sources per scene:
  ours : query_eval_out/<scene>/summary_megaloc_p{1,2,3}.txt, summary_megaloc_photo.txt,
         summary_clip_p2.txt
  hloc : /home/chini/3dgs/hloc_out/<scene>/hloc/summary.txt + timing_online.txt
  map  : the splat checkpoint size, + gaussian counts from the P3 block in RESULTS_LOG
recall gets a 95% wilson CI recomputed from (pct, n) so even older logged runs
(which pre-date the inline CI) get an interval. no metric scale: everything is units.
"""
import math
import re
from pathlib import Path

CODE = Path("/home/chini/Desktop/3DGS/code")
HLOC = Path("/home/chini/3dgs/hloc_out")
RESULTS_LOG = CODE / "RESULTS_LOG.md"

# scene -> (ours summary dir, hloc dir, splat checkpoint) - inline, like query_eval
SCENES = {
    "class1": dict(
        ours=CODE / "query_eval_out/class1",
        hloc=HLOC / "class1/hloc",
        ckpt=Path("/home/chini/3dgs/outputs/class1_train-f2900/splatfacto/"
                  "2026-10-11_104755/nerfstudio_models/step-000029999.ckpt"),
    ),
    "stacey": dict(
        ours=CODE / "query_eval_out/stacey",
        hloc=HLOC / "stacey/hloc",
        ckpt=Path("/home/chini/3dgs/outputs/stacey_room_train-f4200/splatfacto/"
                  "2026-10-11_105953/nerfstudio_models/step-000029999.ckpt"),
    ),
}


def wilson_ci(k, n, z=1.96):
    """95% wilson score interval as percents; closed-form, no deps."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, 100.0 * (c - h)), min(100.0, 100.0 * (c + h))


def _f(pattern, text, cast=float, default=None):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else default


def parse_summary(path):
    """pull the common fields out of one ours/hloc summary.txt."""
    if not path.exists():
        return None
    txt = path.read_text()
    n = _f(r"on (\d+) queries", txt, int)
    recalls = {}
    rec_line = re.search(r"recall:(.*)", txt)
    if rec_line and n:
        for name, pct in re.findall(r"(\d+cm_\d+deg)=(\d+)%", rec_line.group(1)):
            pct = int(pct)
            lo, hi = wilson_ci(round(pct * n / 100), n)
            recalls[name] = (pct, lo, hi)
    return dict(
        n=n,
        t_med=_f(r"translation: median ([\d.]+)", txt),
        r_med=_f(r"rotation:\s+median ([\d.]+)", txt),
        ms=_f(r"time:\s+median (\d+) ms", txt) or _f(r"time:\s+([\d.]+) ms", txt),
        gpu_resv=_f(r"gpu peak:.*?([\d.]+) GB reserved", txt),
        recalls=recalls,
    )


def parse_hloc_timing(path):
    if not path.exists():
        return {}
    txt = path.read_text()
    return dict(
        total_ms=_f(r"TOTAL online\s+([\d.]+) ms", txt),
        gpu_resv=_f(r"gpu peak\s+[\d.]+ GB alloc\s+([\d.]+) GB reserved", txt),
    )


def parse_p3_counts():
    """gaussian counts from the most recent P3 block in RESULTS_LOG, if present."""
    if not RESULTS_LOG.exists():
        return {}
    txt = RESULTS_LOG.read_text()
    out = {}
    for name, gauss in re.findall(r"\[single\] (\w+) \([^)]*\): ([\d,]+) gaussians", txt):
        out[name] = int(gauss.replace(",", ""))
    return out


def _recall_cell(rec, key):
    if key not in rec:
        return "-"
    pct, lo, hi = rec[key]
    return f"{pct}% [{lo:.0f}-{hi:.0f}]"


def scene_section(scene, cfg, p3):
    lines = [f"## {scene}", ""]
    ours = {tag: parse_summary(cfg["ours"] / f"summary_{tag}.txt")
            for tag in ("megaloc_p1", "megaloc_p2", "megaloc_p3",
                        "megaloc_photo", "clip_p2")}
    hloc = parse_summary(cfg["hloc"] / "summary.txt")
    hloc_t = parse_hloc_timing(cfg["hloc"] / "timing_online.txt")

    keys = ["5cm_5deg", "10cm_5deg", "25cm_2deg", "50cm_10deg"]
    hdr = ["method", "n", "med t (u)", "med r (deg)", "ms/query"] + keys
    rows = [hdr, ["---"] * len(hdr)]

    label = {"megaloc_p1": "ours 1-pass", "megaloc_p2": "ours 2-pass",
             "megaloc_p3": "ours 3-pass", "megaloc_photo": "ours photometric",
             "clip_p2": "ours CLIP 2-pass"}
    for tag in ("megaloc_p1", "megaloc_p2", "megaloc_p3", "megaloc_photo", "clip_p2"):
        s = ours[tag]
        if not s:
            continue
        rows.append([label[tag], str(s["n"]), f"{s['t_med']:.4f}",
                     f"{s['r_med']:.3f}", f"{s['ms']:.0f}" if s["ms"] else "-"]
                    + [_recall_cell(s["recalls"], k) for k in keys])
    if hloc:
        ms = f"{hloc_t['total_ms']:.0f}" if hloc_t.get("total_ms") else f"{hloc['ms']:.0f}*"
        rows.append(["HLoc", str(hloc["n"]), f"{hloc['t_med']:.4f}",
                     f"{hloc['r_med']:.3f}", ms]
                    + [_recall_cell(hloc["recalls"], k) for k in keys])
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    if hloc and not hloc_t.get("total_ms"):
        lines.append("")
        lines.append("\\* HLoc ms is localize_sfm only (full-online timing not found).")

    # resources
    lines += ["", "### resources", ""]
    size_gb = cfg["ckpt"].stat().st_size / 1e9 if cfg["ckpt"].exists() else None
    ng = p3.get(scene)
    res = []
    if ng:
        res.append(f"- gaussians: {ng:,}")
    if size_gb:
        res.append(f"- checkpoint on disk: {size_gb:.2f} GB")
    ours_vram = next((ours[t]["gpu_resv"] for t in ("megaloc_p2", "megaloc_p1")
                      if ours[t] and ours[t]["gpu_resv"]), None)
    if ours_vram:
        res.append(f"- ours peak VRAM: {ours_vram:.2f} GB reserved")
    if hloc_t.get("gpu_resv"):
        res.append(f"- HLoc peak VRAM: {hloc_t['gpu_resv']:.2f} GB reserved")
    lines += res or ["- (run measure_splat_resources.py for map/VRAM numbers)"]
    lines.append("")
    return lines


def basin_section():
    """controlled-seed experiment: photometric vs PnP-2pass from IDENTICAL seeds,
    to separate 'weak method' from 'seed out of reach'. @5cm_5deg, n=40 subsets."""
    # (label, seed error, photo summary tag, pnp summary tag). ordered by seed
    # magnitude so the PnP-vs-seed-error trend is readable. the 0.20/0.28 rows are
    # PnP-only (no photometric run) - photometric already failed well below these.
    conds = [("MegaLoc", "~0.28 u (where retrieval landed)", "photo", "p2"),
             ("perturbed", "0.05 u / 2 deg (tight)", "photo_seed0.05-2", "p2_seed0.05-2"),
             ("perturbed", "0.10 u / 5 deg (basin edge)", "photo_seed0.1-5", "p2_seed0.1-5"),
             ("perturbed", "0.20 u / 5 deg", "photo_seed0.2-5", "p2_seed0.2-5"),
             ("perturbed", "0.28 u / 8 deg (= MegaLoc dist)", "photo_seed0.28-8", "p2_seed0.28-8")]

    def cell(scene, tag):
        s = parse_summary(SCENES[scene]["ours"] / f"summary_megaloc_{tag}.txt")
        return _recall_cell(s["recalls"], "5cm_5deg") if s else "-"

    lines = ["## Controlled-seed / basin experiment", "",
             "Photometric (iterative LPIPS) vs PnP (2-pass) from **identical** seeds "
             "per query (same RNG base). Recall @5cm_5deg, n=40 each. Shows the "
             "photometric baseline is basin-limited, not weak: it recovers from a "
             "tight seed but not from where MegaLoc lands. PnP's basin is far wider.", ""]
    hdr = ["seed source", "seed error", "photometric stacey", "photometric class1",
           "PnP stacey", "PnP class1"]
    rows = [hdr, ["---"] * len(hdr)]
    for label, err, pt, qt in conds:
        rows.append([label, err, cell("stacey", pt), cell("class1", pt),
                     cell("stacey", qt), cell("class1", qt)])
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines += ["",
              "Note (photometric): the ~0.10 u / 5 deg 'edge' is from the synthetic "
              "capture-radius sweep (render-vs-render). On real photos (render-vs-photo, "
              "appearance gap) the photometric basin is tighter, ~0.05 u - hence edge "
              "collapses while tight succeeds.",
              "",
              "Note (PnP has a WIDE but FINITE basin): 100% to 0.10 u, 70% at 0.20 u, "
              "~35% at 0.28 u (random-direction offsets). So MegaLoc's misses are NOT "
              "purely wrong-place retrieval - correct-place seeds beyond ~0.1-0.2 u also "
              "fail. BUT real MegaLoc seeds recover far better than random offsets at the "
              "same distance (full-set 2-pass = 87% stacey from a 0.28 u median seed, vs "
              "35% here): a retrieved frame is a plausible viewpoint that keeps scene "
              "overlap, which a random perturbation loses. So these controlled numbers are "
              "a CONSERVATIVE lower bound on PnP's tolerance to real retrieved seeds, and "
              "the wrong-place-vs-too-far split needs the per-query miss analysis to "
              "settle (retrieved seed distance + LoFTR inliers on the failing queries).", ""]
    return lines


def main():
    p3 = parse_p3_counts()
    out = ["# Head-to-head results (generated from summary files on disk)", "",
           "All translation/rotation errors are in **nerfstudio scene units** - these "
           "maps have no metric scale. Recall thresholds carry a 95% Wilson CI in [].", ""]
    for scene, cfg in SCENES.items():
        out += scene_section(scene, cfg, p3)
    out += basin_section()

    # only this script owns the text BETWEEN these markers; anything outside them
    # (hand-added tables, the metric-scaled section) is left untouched on a rerun.
    START = "<!-- AUTOGEN:START (make_report_tables.py - do not edit inside) -->"
    END = "<!-- AUTOGEN:END -->"
    block = START + "\n" + "\n".join(out) + "\n" + END

    dst = CODE / "report_tables.md"
    if not dst.exists():
        dst.write_text(block + "\n")
    else:
        old = dst.read_text()
        if START in old and END in old:
            pre = old.split(START, 1)[0]
            post = old.split(END, 1)[1]
            dst.write_text(pre + block + post)
        else:
            # no markers = someone's hand-edited file. don't clobber it.
            raise SystemExit(
                f"[abort] {dst} exists without the AUTOGEN markers - refusing to "
                "overwrite hand-edited content. wrap the generated block in\n"
                f"  {START}\n  ...\n  {END}\nthen rerun.")
    print("\n".join(out))
    print(f"\n[done] wrote {dst}")


if __name__ == "__main__":
    main()
