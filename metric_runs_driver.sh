#!/usr/bin/env bash
# RUN COMMAND:
# nohup bash metric_runs_driver.sh &      (progress in metric_runs/progress.log, touches metric_runs/DONE at the end)

# overnight metric rerun. stable (unmodified query_eval compute) first, modified
# code (coarse_harness --radii-cm, capture_radius_metric) last. each run is
# isolated under a timeout so one hang can't sink the night. logs -> RESULTS_METRIC.md
CODE=/home/chini/Desktop/3DGS/code
cd "$CODE" || exit 1
source /home/chini/miniconda3/etc/profile.d/conda.sh
conda activate nerfstudio
export RESULTS_LOG="$CODE/RESULTS_METRIC.md"

OUT="$CODE/metric_runs"; LOGS="$OUT/logs"; mkdir -p "$LOGS"
DRV="$OUT/progress.log"
trap 'echo "### $(date +%F_%H:%M) DRIVER EXIT" >>"$DRV"; touch "$OUT/DONE"' EXIT
echo "### $(date +%F_%H:%M) DRIVER START" >"$DRV"

# run one job under a timeout, log it to $LOGS/NAME.log and put its recall line in the progress log
run(){ # run NAME TIMEOUT -- cmd...
  local name="$1" tmo="$2"; shift 3
  echo "### $(date +%H:%M) START $name" >>"$DRV"
  timeout "$tmo" "$@" >"$LOGS/$name.log" 2>&1
  echo "### $(date +%H:%M) END   $name rc=$? | $(grep -m1 'recall:' "$LOGS/$name.log" 2>/dev/null | cut -c1-90)" >>"$DRV"
}
st(){ awk "BEGIN{printf \"%.6f\", $1/100/$2}"; }   # cm, m_per_unit -> scene units
declare -A MPU=( [class1]=3.5601 [stacey]=1.9826 )

# ========================= STABLE BLOCK =========================
# item 1 - accuracy / timing, both scenes (1/2/3-pass + CLIP 2-pass)
for sc in class1 stacey; do
  for p in 1 2 3; do
    run "item1_${sc}_p${p}" 2400 -- python query_eval.py --scene "$sc" --passes "$p" --no-link --out-dir "$OUT/item1_${sc}"
  done
  run "item1_${sc}_clip_p2" 2400 -- python query_eval.py --scene "$sc" --retriever clip --passes 2 --no-link --out-dir "$OUT/item1_${sc}"
done

# item 7 - rotation-only sweep, 2-pass, n=100 (translation fixed at 0)
for sc in class1 stacey; do
  for r in 5 10 15 20 25 30 40 50; do
    run "item7_${sc}_r${r}" 2400 -- python query_eval.py --scene "$sc" --passes 2 \
      --seed-source perturb --seed-trans 0 --seed-rot "$r" --seed-base 0 --limit 100 \
      --no-link --out-dir "$OUT/item7_${sc}"
  done
done

# item 3 - PnP basin, 2-pass, n=100, offsets in cm (converted per scene)
for sc in class1 stacey; do
  for pair in 2:1 5:2 10:5 20:5 40:8 60:12 80:16 100:20 150:30; do
    tcm=${pair%:*}; rd=${pair#*:}; su=$(st "$tcm" "${MPU[$sc]}")
    run "item3_${sc}_t${tcm}_r${rd}" 2400 -- python query_eval.py --scene "$sc" --passes 2 \
      --seed-source perturb --seed-trans "$su" --seed-rot "$rd" --seed-base 0 --limit 100 \
      --no-link --out-dir "$OUT/item3_${sc}"
  done
done

# item 5 - photometric fine stage (slow, 5 h timeout per scene) on the FULL query sets, MegaLoc seed
for sc in class1 stacey; do
  run "item5_${sc}" 18000 -- python query_eval.py --scene "$sc" --fine photometric \
    --photo-loss lpips --photo-iters 500 --photo-lr-trans 2e-4 --photo-lr-rot 2e-4 \
    --no-link --out-dir "$OUT/item5_${sc}"
done

# item 4 - photometric basin, tight end, n=40, same seeds as item 3 (seed-base 0)
for sc in class1 stacey; do
  for pair in 2:1 5:2 10:5 20:5; do
    tcm=${pair%:*}; rd=${pair#*:}; su=$(st "$tcm" "${MPU[$sc]}")
    run "item4_${sc}_t${tcm}_r${rd}" 5400 -- python query_eval.py --scene "$sc" --fine photometric \
      --photo-loss lpips --photo-iters 500 --photo-lr-trans 2e-4 --photo-lr-rot 2e-4 \
      --seed-source perturb --seed-trans "$su" --seed-rot "$rd" --seed-base 0 --limit 40 \
      --no-link --out-dir "$OUT/item4_${sc}"
  done
done

# ======================= MODIFIED BLOCK =======================
# item 2 - coarse retrieval recall at metric radii (new --radii-cm logic)
for sc in class1 stacey; do
  for ret in megaloc clip; do
    for src in real rendered; do
      run "item2_${sc}_${ret}_${src}" 5400 -- python coarse_harness.py --scene "$sc" \
        --retriever "$ret" --source "$src" --radii-cm 10 25 50 100 \
        --out "$OUT/item2_${sc}_${ret}_${src}"
    done
  done
done

# item 6 - capture radius render-vs-render, stacey (new script)
run "item6_stacey" 3600 -- python capture_radius_metric.py --scene stacey --trials 50 --passes 2

echo "### $(date +%F_%H:%M) ALL DONE" >>"$DRV"
