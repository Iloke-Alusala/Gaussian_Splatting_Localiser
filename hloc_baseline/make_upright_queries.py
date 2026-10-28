# RUN COMMAND:
# python hloc_baseline/make_upright_queries.py --scene-dir <scene>-processed --subdir query_d1280 --query-gt query_gt_d1280.json

# the stills are stored landscape with an EXIF rotate flag, which cv2/colmap ignore. SuperPoint+LightGlue isn't rotation-invariant, so write upright copies to images/<subdir>_up/ plus a query_gt with rotated intrinsics
# GT R/t stays in the raw frame: the camera centre doesn't change with roll, so translation still scores directly. PIL + numpy only, runs in any env

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_EXIF_ORIENT = 274
_EXIF_K = {1: 0, 3: 2, 6: 1, 8: 3}        # 90-deg CW turns to make upright


def rot_intrinsics_cw(fx, fy, cx, cy, W, H, k):
    """turn raw-buffer intrinsics k times 90-deg CW to match the upright image (same as query_eval)"""
    for _ in range(k):
        fx, fy, cx, cy, W, H = fy, fx, (H - 1 - cy), cx, H, W
    return fx, fy, cx, cy, W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", type=Path,
                    default=Path("/home/chini/3dgs/data/stacey_room-f2500-processed"))
    ap.add_argument("--subdir", default="query_d1280")
    ap.add_argument("--query-gt", default="query_gt_d1280.json")
    args = ap.parse_args()

    src = args.scene_dir / "images" / args.subdir
    dst = args.scene_dir / "images" / f"{args.subdir}_up"
    dst.mkdir(parents=True, exist_ok=True)
    qgt = json.loads((args.scene_dir / args.query_gt).read_text())

    out = {}
    for name, e in qgt.items():
        im = Image.open(src / name)
        k = _EXIF_K.get(im.getexif().get(_EXIF_ORIENT, 1), 0)
        # bake the EXIF rotation into the pixels, then rotate the intrinsics the same number of turns
        ImageOps.exif_transpose(im).convert("RGB").save(dst / name, quality=95)

        fx, fy, cx, cy = (float(x) for x in e["params"][:4])
        fx, fy, cx, cy, W, H = rot_intrinsics_cw(fx, fy, cx, cy,
                                                 e["width"], e["height"], k)
        out[name] = {**e, "width": W, "height": H,
                     "params": [fx, fy, cx, cy] + list(e["params"][4:])}

    out_json = args.scene_dir / f"{Path(args.query_gt).stem}_up.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[done] {len(out)} upright images -> {dst}")
    print(f"       rotated intrinsics -> {out_json}")


if __name__ == "__main__":
    main()
