# Results audit

Central, read-only **copies** of every results log in the project, one folder per experiment.
Each folder has the raw log(s) plus a `README.md` that says what it is and how to read it.
Nothing here is a source of truth for the code — these are snapshots copied on 2026-10-27.

## Units — read this once
- **Scene units** = nerfstudio map units (no physical scale). Older logs are in these.
- **Metric** = real cm/deg, using the door-frame scale **class1 (Room1) = 3.5601 m/unit,
  stacey (Office) = 1.9826 m/unit**. The 2026-10-21 reruns and anything labelled "metric" use these.
- Rooms: **Room1 = class1** (a classroom), **Office = stacey**, **class2** = the second classroom.

## Folders
| folder | what | status |
|---|---|---|
| 00_master-log_all-runs | every run ever, append-only (scene units) | authoritative history |
| 01_master-log_metric | the 2026-10-21 metric reruns (real cm) | authoritative (metric) |
| 02_headtohead_tables | clean ours-vs-HLoc tables, unit + metric | authoritative summary |
| 03_accuracy-timing_1-2-3pass_metric | ours 1/2/3-pass + CLIP, full sets, metric | authoritative |
| 04_coarse-retrieval_recall-vs-radius | MegaLoc vs CLIP retrieval, recall vs radius | authoritative |
| 05_multiroom_room-id | 3-room correct-room test (ATP2) | authoritative |
| 06_perturbation_2pass-loftr_basin | 2-pass basin vs seed offset (thesis F.2 col) | complete |
| 07_perturbation_photometric_basin | photometric basin vs offset (F.2 col) | PARTIAL (see its README) |
| 08_photometric_fullset | photometric end-to-end on full sets | authoritative |
| 09_capture-radius_render-vs-render | render-vs-render geometric basin | stacey only |
| 10_rotation-only_sweep | recovery vs pure rotation offset | complete |
| 11_hloc_baseline_and_numloc-sweep | HLoc baseline + num_loc 1..20 sweep | authoritative |
| 12_mcnemar_2pass-vs-hloc | paired stats + failure modes, Office | computed 2026-10-27 |
| _stale | superseded docs — DO NOT CITE | stale |

## Not copied
Per-query PNGs (16 G + 13 G of images), and the handoff/context docs
(`megaHandoff.md`, `report_experiments_handoff.md`) which are notes, not result logs.
