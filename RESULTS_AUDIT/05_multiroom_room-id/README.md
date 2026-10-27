# Multi-room retrieval (correct-room / ATP2)

`RESULTS_MULTIROOM.md` — pool all three rooms' train frames into one bank (7854 frames, each
tagged with its room), then for each query check whether the top-1 retrieved frame is from the
right room. 600 queries = 200 each from Room1/class1, class2, Office/stacey.

Result: **MegaLoc 98.7%** correct-room, **CLIP 87.7%**. CLIP collapses to 67.5% on class2 (confuses
the two classrooms); MegaLoc separates them. Median retrieval time: MegaLoc 40 ms, CLIP 6.6 ms.

class2 queries were extracted from a separate test video and verified **not** present in the bank
(0 pixel-identical frames). Built by `multiroom_retrieval.py`.

Status: authoritative.
