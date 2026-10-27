# Accuracy & timing, 1/2/3-pass (metric)

Source: `metric_runs/item1_*`. Full query sets, MegaLoc seed, real cm.
Files are `<scene>_summary_<retriever>_p<N>.txt` (N = number of fine passes); `clip_p2` is the
CLIP-retriever variant.

Each file: median t (cm) and r (deg), ms/query, recall at 5/10/25/50 cm bars, GPU peak.

Quick read:
- **Room1/class1**: ~100% by 2 passes (easy set, queries sit close to train views).
- **Office/stacey**: discriminating set — 1-pass ~67%, 2-pass ~80%, 3-pass ~91% at 5cm/5deg.

Status: authoritative (metric).
