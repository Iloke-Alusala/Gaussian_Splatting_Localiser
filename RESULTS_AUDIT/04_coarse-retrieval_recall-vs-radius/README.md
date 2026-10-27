# Coarse retrieval: recall vs same-place radius

Source: `runs/coarse_*`. One file per (scene, retriever, real/rendered). This is the **coarse
stage alone** — how often the top-N retrieved train frame is within radius R of the query's GT.

Each file: recall@1/5/10 (rows) at radius R = 0.05 / 0.1 / 0.25 scene-units (columns).

Quick read:
- **MegaLoc beats CLIP** at the tight radius on both scenes — why MegaLoc is the coarse retriever.
- Retrieval almost never lands inside the fine stage's tight basin on its own; it gets the right
  *place*, and the fine stage does the rest.

Note: radii here are scene units (these runs predate the metric relabel). Full per-query data is in
the matching `results.csv` back in `runs/` (not copied — large).

Status: authoritative.
