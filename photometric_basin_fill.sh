#!/usr/bin/env bash
# RUN COMMAND:
# nohup bash photometric_basin_fill.sh &      (progress in metric_runs/progress_photofill.log)

# Fills the missing (`--`) photometric-basin cells of tab:basin. Extends item 4
# of metric_runs_driver.sh to the wider offsets, keeping every photometric setting
# identical to the tight-end runs so the column stays comparable. Isolated under a
# timeout each; logs -> RESULTS_METRIC.md. Run in the nerfstudio env.
CODE=/home/chini/Desktop/3DGS/code
cd "$CODE" || exit 1
source /home/chini/miniconda3/etc/profile.d/conda.sh
conda activate nerfstudio
export RESULTS_LOG="$CODE/RESULTS_METRIC.md"

OUT="$CODE/metric_runs"; LOGS="$OUT/logs"; mkdir -p "$LOGS"
DRV="$OUT/progress_photofill.log"
echo "### $(date +%F_%H:%M) PHOTO-FILL START" >"$DRV"

# same helper as metric_runs_driver.sh: one job under a timeout, recall line into the progress log
run(){ # run NAME TIMEOUT -- cmd...
  local name="$1" tmo="$2"; shift 3
  echo "### $(date +%H:%M) START $name" >>"$DRV"
  timeout "$tmo" "$@" >"$LOGS/$name.log" 2>&1
  echo "### $(date +%H:%M) END   $name rc=$? | $(grep -m1 'recall:' "$LOGS/$name.log" 2>/dev/null | cut -c1-90)" >>"$DRV"
}
st(){ awk "BEGIN{printf \"%.6f\", $1/100/$2}"; }   # cm, m_per_unit -> scene units
declare -A MPU=( [class1]=3.5601 [stacey]=1.9826 )

# exactly the `--` cells, per scene (tcm:rdeg)
declare -A MISSING=(
  [class1]="60:12"
  [stacey]="2:1 10:5"
)

for sc in class1 stacey; do
  for pair in ${MISSING[$sc]}; do
    tcm=${pair%:*}; rd=${pair#*:}; su=$(st "$tcm" "${MPU[$sc]}")
    run "item4_${sc}_t${tcm}_r${rd}" 5400 -- python query_eval.py --scene "$sc" --fine photometric \
      --photo-loss lpips --photo-iters 500 --photo-lr-trans 2e-4 --photo-lr-rot 2e-4 \
      --seed-source perturb --seed-trans "$su" --seed-rot "$rd" --seed-base 0 --limit 40 \
      --no-link --out-dir "$OUT/item4_${sc}"
  done
done

echo "### $(date +%F_%H:%M) PHOTO-FILL DONE" >>"$DRV"
