"""
measure_splat_resources.py - the P3 numbers: per-scene Gaussian count, on-disk
checkpoint size, and the VRAM + wall-clock cost of loading a splat. this is the
evidence for the "room-scoped loading keeps it viable on a low-VRAM robot" claim.

two measurements that matter:
  - single room: load one scene's splat, peak VRAM + load time (what the robot
    actually pays when it loads just the room the coarse stage picked).
  - all rooms:   load every scene's splat at once, peak VRAM + total load time
    (the naive "load everything" the room-scoping is supposed to avoid).

no metric scale on these maps, so sizes are bytes/counts, not metres. runs in
the nerfstudio env. appends to RESULTS_LOG.md like the other harnesses.
"""
import time
from pathlib import Path

import torch

from my_localizer import load_pipeline, freeze_map
from query_eval import SCENES          # reuse the same baked-in scene paths


def _ckpt_path(config: Path) -> Path:
    """the trained checkpoint sits next to the config, under nerfstudio_models."""
    return config.parent / "nerfstudio_models" / "step-000029999.ckpt"


def _gaussian_count(pipeline) -> int:
    return int(pipeline.model.gauss_params["means"].shape[0])


def _gb(nbytes) -> float:
    return nbytes / 1e9


def load_one(name, device):
    """load a single scene's splat, timing it and measuring its own VRAM peak."""
    sc = SCENES[name]
    ckpt = _ckpt_path(sc["config"])
    size_b = ckpt.stat().st_size if ckpt.exists() else 0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _cfg, pipeline, _scale = load_pipeline(sc["config"], device)
    freeze_map(pipeline)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    n = _gaussian_count(pipeline)
    return pipeline, dict(
        name=sc["name"], n_gauss=n, ckpt_gb=_gb(size_b), load_s=dt,
        vram_alloc_gb=_gb(torch.cuda.max_memory_allocated()),
        vram_resv_gb=_gb(torch.cuda.max_memory_reserved()),
    )


def main():
    device = "cuda"
    names = list(SCENES)

    lines = ["splat resources (per-scene load cost; no metric scale on these maps)"]

    # single-room: load each on its own so the VRAM peak is that room alone.
    singles = []
    for nm in names:
        pipe, r = load_one(nm, device)
        singles.append(r)
        lines.append(
            f"[single] {nm} ({r['name']}): {r['n_gauss']:,} gaussians  "
            f"{r['ckpt_gb']:.3f} GB on disk  load {r['load_s']:.1f} s  "
            f"vram {r['vram_alloc_gb']:.2f} GB alloc / {r['vram_resv_gb']:.2f} GB reserved")
        del pipe
        torch.cuda.empty_cache()

    # all-rooms: load every splat together, one combined VRAM peak + total time.
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    held = []
    for nm in names:
        _cfg, pipeline, _scale = load_pipeline(SCENES[nm]["config"], device)
        freeze_map(pipeline)
        held.append(pipeline)
    torch.cuda.synchronize()
    dt_all = time.perf_counter() - t0
    total_gauss = sum(_gaussian_count(p) for p in held)
    lines.append(
        f"[all rooms] {len(names)} splats together: {total_gauss:,} gaussians  "
        f"load {dt_all:.1f} s  "
        f"vram {_gb(torch.cuda.max_memory_allocated()):.2f} GB alloc / "
        f"{_gb(torch.cuda.max_memory_reserved()):.2f} GB reserved")
    lines.append(
        "note: errors/counts are in nerfstudio scene units; this card is 4 GB, so "
        "the all-rooms peak is the number room-scoped loading has to stay under")

    out = "\n".join(lines)
    print("\n" + out)
    with open("/home/chini/Desktop/3DGS/code/RESULTS_LOG.md", "a") as f:
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} | splat resources (P3) | "
                f"scenes={','.join(names)}\n```\n" + out + "\n```\n")


if __name__ == "__main__":
    main()
