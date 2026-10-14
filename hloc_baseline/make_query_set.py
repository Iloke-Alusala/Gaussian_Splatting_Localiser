#!/usr/bin/env python
"""
Build the held-out query set + pseudo-ground-truth for the HLoc baseline.

Run in the *nerfstudio* conda env.

Why this exists
---------------
HLOC_SETUP.md assumes separately-registered high-res query stills and a
``query_gt.json`` produced by ``register_queries.sh``. Neither exists on this
machine for any scene. Instead we reuse the frames nerfstudio already holds out
as its *eval* split (``eval_mode: fraction``, ``train_split_fraction: 0.9``) -
the exact same frames our own pipeline queries via
``my_pose_harness.get_eval_views`` - and take their COLMAP poses (already in
``images.bin``) as pseudo ground truth. The HLoc reference model is later built
as ``sparse/0`` *minus* these frames so HLoc cannot read the answer out of its
own map.

Outputs (written under <out_dir>, default /home/chini/3dgs/hloc_out/<scene>/):
  query_gt.json   per-eval-frame GT in COLMAP world frame (brief's shape)
  query_list.txt  eval frame names (the queries)
  db_list.txt     the remaining registered frames (the HLoc database)
  scale.json      dataparser scale (COLMAP units -> nerfstudio units)

Nothing under data/*/colmap/ is modified; this script only reads it.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

# nerfstudio's own COLMAP reader (confirmed available in the nerfstudio env)
from nerfstudio.data.utils.colmap_parsing_utils import (
    read_cameras_binary,
    read_images_binary,
    qvec2rotmat,
)

SCENE = "classroom_1-f1250"
DATA_ROOT = Path("/home/chini/3dgs/data")
DEFAULT_CONFIG = Path(
    "/home/chini/3dgs/outputs/classroom_1-f1250/splatfacto/2026-10-07_063413/config.yml"
)


def eval_frame_names(config_path: Path, data_dir: Path) -> list[str]:
    """Exact eval-split frame basenames, straight from the nerfstudio dataparser.

    We only use the dataparser to learn *which* frames are held out; the poses
    come from raw COLMAP below (we never use the dataparser's transformed poses).
    """
    import yaml

    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    dp_config = config.pipeline.datamanager.dataparser
    dp_config.data = data_dir  # point at the processed scene dir (has transforms.json)

    dp = dp_config.setup()
    eval_out = dp.get_dataparser_outputs(split="val")  # "val"/"test" == i_eval
    return [Path(p).name for p in eval_out.image_filenames]


def dataparser_scale(config_path: Path, data_dir: Path) -> float:
    """COLMAP->nerfstudio scale, read from the saved dataparser_transforms.json."""
    tfm = config_path.parent / "dataparser_transforms.json"
    if tfm.exists():
        return float(json.loads(tfm.read_text())["scale"])
    print(f"[warn] {tfm} missing; scale defaults to 1.0")
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    data_dir = DATA_ROOT / f"{args.scene}-processed" if "-processed" not in args.scene \
        else DATA_ROOT / args.scene
    out_dir = args.out_dir or Path(f"/home/chini/3dgs/hloc_out/{args.scene}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sparse = data_dir / "colmap/sparse/0"
    images = read_images_binary(str(sparse / "images.bin"))
    cameras = read_cameras_binary(str(sparse / "cameras.bin"))
    name2img = {im.name: im for im in images.values()}
    print(f"[info] COLMAP model: {len(images)} registered images, "
          f"{len(cameras)} camera(s)")

    # Which frames are the queries (nerfstudio eval split)?
    eval_names = eval_frame_names(args.config, data_dir)
    # keep only eval frames that are actually registered in COLMAP
    query_names = sorted(n for n in eval_names if n in name2img)
    missing = sorted(set(eval_names) - set(query_names))
    if missing:
        print(f"[warn] {len(missing)} eval frames not in COLMAP model "
              f"(dropped), e.g. {missing[:3]}")
    db_names = sorted(n for n in name2img if n not in set(query_names))
    print(f"[info] queries: {len(query_names)}   database: {len(db_names)}")

    # Build pseudo-GT in COLMAP world frame for each query frame.
    gt = {}
    for name in query_names:
        im = name2img[name]
        R = qvec2rotmat(im.qvec)                 # world -> cam
        t = np.asarray(im.tvec, dtype=float)     # world -> cam
        cam = cameras[im.camera_id]
        centre = (-R.T @ t)                       # camera centre in world
        gt[name] = {
            "R_world_to_cam": R.tolist(),
            "t_world_to_cam": t.tolist(),
            "camera_centre": centre.tolist(),
            "camera_model": cam.model,
            "width": int(cam.width),
            "height": int(cam.height),
            "params": [float(x) for x in cam.params],
            "colmap_im_id": int(im.id),
        }

    (out_dir / "query_gt.json").write_text(json.dumps(gt, indent=2))
    (out_dir / "query_list.txt").write_text("\n".join(query_names) + "\n")
    (out_dir / "db_list.txt").write_text("\n".join(db_names) + "\n")
    scale = dataparser_scale(args.config, data_dir)
    (out_dir / "scale.json").write_text(json.dumps({"scale": scale}, indent=2))

    print(f"[done] wrote query_gt.json ({len(gt)} queries), query_list.txt, "
          f"db_list.txt, scale.json (scale={scale:.6g}) -> {out_dir}")


if __name__ == "__main__":
    main()
