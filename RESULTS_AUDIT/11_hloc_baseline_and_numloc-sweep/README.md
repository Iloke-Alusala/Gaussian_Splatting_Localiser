# HLoc baseline + num_loc sweep

The structure-based baseline: SuperPoint + LightGlue + NetVLAD, resize_max 1024, num_loc 20.

Per scene:
- `<scene>_summary.txt` — recall + median errors (scene units).
- `<scene>_results.csv` — per-query: t_err, r_err, inliers, time, and `t_err_ns` (scaled). Use
  `t_err_ns` for metric recall, not the raw `t_err`.
- `<scene>_timing_online.txt` — full online timing breakdown (SuperPoint / NetVLAD / retrieval /
  LightGlue match / localize_sfm / total).

`stacey_numloc_nl*.txt` — num_loc sweep (1,2,5,10,20) on Office. nl20 is the default/reported run
(~593 ms total). Use these to justify num_loc = 20.

Quick read: HLoc localises 100% and has a tiny *median* error, but on Office its recall@5cm is only
~72% — a heavy outlier tail (high mean). That's the key nuance vs ours.

Status: authoritative. Source: /home/chini/3dgs/hloc_out/.
