# 3DGS camera localisation + HLoc baseline

Localise a camera inside a **3D Gaussian Splatting** map (MegaLoc retrieval ->
render RGB+depth from the splat -> EfficientLoFTR match -> PnP) and benchmark it
against the established **HLoc** baseline (SuperPoint + LightGlue + PnP vs a
sparse SfM cloud). Final-year project.

## Map of the repo
| path | what |
|---|---|
| `query_eval.py` | our pipeline as a no-viewer harness over a query set (nerfstudio env) |
| `pipeline_retriever.py` | the pipeline itself (interactive viewer + coarse/fine stages) |
| `retrievers.py` | swappable coarse retrievers (CLIP, MegaLoc) |
| `my_localizer.py`, `my_pose_harness.py`, `coarse_harness.py` | pose maths, eval helpers, fine-stage callable |
| `hloc_baseline/` | HLoc baseline: `run_hloc.py`, `eval_hloc.py`, `make_upright_queries.py`, `time_hloc_online.py`, `bench_resources.sh` |
| **`EVALUATION.md`** | **the results** — accuracy, timing, peak resources, head-to-head |
| `HANDOFF.md` | full context for picking the project back up in a new session |

## Quick start (scene: stacey_room-f2500, hloc env needs pycolmap==3.13.0)
```bash
conda run -n nerfstudio python query_eval.py                 # our pipeline -> query_eval_out/
conda run -n hloc python hloc_baseline/time_hloc_online.py   # hloc per-query timing
bash hloc_baseline/bench_resources.sh                        # peak GPU + RAM, both
```
See `EVALUATION.md` for the full comparison and `HANDOFF.md` for everything else.

## Headline (stacey, 53 queries, 4 GB GPU)
HLoc is more accurate (0.24 cm / 0.18 deg vs our 4.6 cm / 1.55 deg) and lighter on
VRAM; **ours is ~3x faster per query** (282 vs 878 ms) with a near-trivial setup
(minutes vs hours, no SfM triangulation). Trade-off: accuracy vs speed + deploy
simplicity. The *fundamental* difference: HLoc localises against a purpose-built
sparse SfM cloud; we render a dense Gaussian map and use its depth.

---

## Ablation studies

### A1. Iterative fine stage (closing the accuracy gap)
**Hypothesis.** Our single-shot fine stage renders at the MegaLoc seed, matches,
and solves PnP once. When the seed is off, the render doesn't align with the
query, so the matches and the lifted depth are noisy. **Iterating** the fine stage
(re-render at the refined pose -> re-match -> re-PnP, iComMa / iNeRF style) should
tighten the PnP residual and rescue marginal queries, at a cost of ~+200 ms per
extra pass. It should plateau at the splat's depth accuracy (depth is
photometric, not metric) — so expect improvement, not HLoc-level precision.
`--passes N` iterates; `1` is the unchanged single shot.

```bash
for p in 1 2 3; do conda run -n nerfstudio python query_eval.py --passes $p; done
# -> query_eval_out/summary_megaloc_p{1,2,3}.txt
```

| passes | median trans | median rot | recall @5cm/5deg | time/query | notes |
|---|---|---|---|---|---|
| 1 (baseline) | 4.62 cm | 1.55 deg | 51% | 285 ms | single shot |
| **2** | **0.91 cm** | **0.36 deg** | **92%** | 447 ms | big jump — **sweet spot** |
| 3 | 0.87 cm | 0.25 deg | 94% | 608 ms | plateau (splat-depth floor) |

(HLoc reference: 0.24 cm / 0.18 deg / 100% / 878 ms.)

**Result.** A single extra pass is transformative: **4.62 -> 0.91 cm and 51 -> 92%
recall** for +162 ms, and we're **still ~2x faster than HLoc** (447 vs 878 ms).
Pass 3 barely moves translation (0.91 -> 0.87 cm) but keeps tightening rotation
(0.36 -> 0.25 deg) — translation plateaus at ~0.87 cm, the splat-depth floor we
predicted (Gaussian depth is photometric, not metric, so render-and-refine can't
reach HLoc's 0.24 cm). **2 passes is the recommended operating point**: nearly all
the accuracy, still well under HLoc's latency. Natural follow-up: **adaptive
passes** (iterate only while the pose update exceeds a threshold) so easy queries
stay at 1 pass and only hard ones pay the extra render+match.

### Ideas parked for later
- Harder / wider-baseline query set (current stills saturate HLoc at 100%; a
  registered 50-frame video pass would be more discriminating).
- Improve the coarse stage (MegaLoc) — the few failures land outside the fine
  stage's capture radius.
