#!/bin/bash
# Peak GPU + RAM for both localisation pipelines, on the same 53-query set.
# Each script reports its own torch GPU peak; we also poll nvidia-smi (system
# GPU, incl. CUDA context) and the process RSS (RAM) while it runs.
OUT=/home/chini/3dgs/hloc_out/stacey_room-f2500/resources.txt
cd /home/chini/Desktop/3DGS/code

poll() {   # $1=cmd pattern  $2=gpu samples  $3=rss samples
  : > "$2"; : > "$3"
  while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$2"
    for pid in $(pgrep -f "$1"); do ps -o rss= -p "$pid" 2>/dev/null; done \
      | sort -n | tail -1 >> "$3"
    sleep 0.3
  done
}
peak_mib() { sort -n "$1" | tail -1; }
peak_gb()  { awk '{if($1>m)m=$1} END{printf "%.2f", m/1024/1024}' "$1"; }  # KB->GB

echo "=== resource benchmark $(date) ===" > "$OUT"

echo "[ours: MegaLoc + render + LoFTR + PnP]  (nerfstudio env)" >> "$OUT"
poll "query_eval.py" /tmp/g_ours.txt /tmp/r_ours.txt & PP=$!
conda run -n nerfstudio python query_eval.py > /tmp/ours.log 2>&1
kill $PP 2>/dev/null; wait $PP 2>/dev/null
grep "gpu peak" /tmp/ours.log | sed 's/^/  torch /' >> "$OUT"
echo "  system GPU peak : $(peak_mib /tmp/g_ours.txt) MiB (nvidia-smi, incl. CUDA ctx)" >> "$OUT"
echo "  RAM peak (RSS)  : $(peak_gb /tmp/r_ours.txt) GB" >> "$OUT"
echo "" >> "$OUT"

echo "[hloc: full online (extract+retrieve+match+PnP)]  (hloc env)" >> "$OUT"
poll "time_hloc_online.py" /tmp/g_hloc.txt /tmp/r_hloc.txt & PP=$!
conda run -n hloc python hloc_baseline/time_hloc_online.py > /tmp/hloc.log 2>&1
kill $PP 2>/dev/null; wait $PP 2>/dev/null
grep -i "gpu peak" /tmp/hloc.log | sed 's/^ */  torch /' >> "$OUT"
echo "  system GPU peak : $(peak_mib /tmp/g_hloc.txt) MiB (nvidia-smi, incl. CUDA ctx)" >> "$OUT"
echo "  RAM peak (RSS)  : $(peak_gb /tmp/r_hloc.txt) GB" >> "$OUT"

echo "" >> "$OUT"
echo "[done] wrote $OUT"
cat "$OUT"
