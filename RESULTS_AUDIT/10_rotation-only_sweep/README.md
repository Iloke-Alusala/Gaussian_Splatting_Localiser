# Rotation-only capture radius

Source: `metric_runs/item7_*` (2026-10-21). Seed is offset in **rotation only** (translation
error 0), by 5 .. 50 deg, then 2-pass LoFTR+PnP tries to recover. limit 100 queries.
Filename `seed0-<deg>` = 0 translation, that many degrees.

Isolates how much *pure* rotation error the fine stage tolerates, separate from translation
(folders 06/07 move both at once). Each file: recall at the bars + median t/r.

Status: complete, both scenes.
