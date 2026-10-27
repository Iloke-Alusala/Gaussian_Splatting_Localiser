# Capture radius — render vs render

The fine stage's *geometric* basin with the appearance gap removed: the "query" is itself a
render of the map from a known pose, offset, then recovered with 2-pass LoFTR+PnP. Pure upper bound.

- `stacey_render_vs_render.log` — the metric grid (2026-10-21), 50 trials/offset, recovered =
  final within 5cm/5deg of base. Offsets 2..150 cm. r50 ~ 69 cm. (thesis F.2 render-vs-render column)
- `OLD_unit-grid_capture_radius.csv` — the earlier version on a scene-unit offset grid (0.01/0.05/
  0.10/0.20 u), 50 trials each, full per-trial rows. Superseded by the metric log but kept for the
  per-trial detail (e.g. median t at 2 cm = 0.41 cm).

Only **Office/stacey** was run render-vs-render; Room1/class1 was not.

Status: stacey only.
