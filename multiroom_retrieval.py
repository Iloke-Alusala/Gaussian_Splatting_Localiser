#!/usr/bin/env python3
# RUN COMMAND:
# conda run -n nerfstudio python multiroom_retrieval.py

# multi-room coarse retrieval: pool every room's cached train index into one bank, tag each row with its room, and check top-1 lands in the right room
# no splat / no pose here, room-id only needs the embeddings. writes RESULTS_MULTIROOM.md
import time, glob, pickle
import torch
from torchvision.io import read_image
from retrievers import MegaLocRetriever, ClipRetriever

DEVICE = "cuda"
CACHE = "/home/chini/Desktop/3DGS/code/.index_cache"
EVAL = "/home/chini/3dgs/eval"

# room -> (cached megaloc pkl, cached clip pkl, query image dir)
ROOMS = {
    "class1": (f"{CACHE}/class1_train-f2900-processed_n2450_megaloc_index.pkl",
               f"{CACHE}/class1_train-f2900-processed_n2450_clip_index.pkl",
               f"{EVAL}/class1/images"),
    "class2": (f"{CACHE}/classroom_2-f1950_n1624_megaloc_index.pkl",
               f"{CACHE}/classroom_2-f1950_n1624_clip_index.pkl",
               f"{EVAL}/class2/images"),
    "stacey": (f"{CACHE}/stacey_room_train-f4200_n3780_megaloc_index.pkl",
               f"{CACHE}/stacey_room-f4200_n3780_clip_index.pkl",
               f"{EVAL}/stacey/images"),
}

def load_img(path):
    # uint8 CHW -> float[0,1] HWC, which is what both retrievers expect
    return (read_image(path)[:3].float() / 255.0).permute(1, 2, 0).to(DEVICE)

def build_bank(pkl_key):
    """ stack every room's cached features into one bank. pkl_key 0 = megaloc, 1 = clip. labels[i] is the room of row i """
    feats, labels = [], []
    for room, paths in ROOMS.items():
        with open(paths[pkl_key], "rb") as f:
            d = pickle.load(f)
        fe = d["features"].to(DEVICE)
        feats.append(fe)
        labels += [room] * fe.shape[0]
    return torch.cat(feats, 0), labels

def run(retriever, pkl_key, tag):
    """ swap the pooled bank into the retriever, query every room's images, and count top-1 correct-room per room + overall """
    bank, labels = build_bank(pkl_key)
    retriever.index_features = bank
    retriever.index_poses = list(range(len(labels)))  # query() only needs its length
    print(f"\n[{tag}] pooled bank = {len(labels)} frames "
          f"({' + '.join(f'{r}:{labels.count(r)}' for r in ROOMS)})")

    per = {r: [0, 0] for r in ROOMS}  # room -> [correct, total]
    times = []
    for room, paths in ROOMS.items():
        for qi, qp in enumerate(sorted(glob.glob(paths[2] + "/*"))):
            img = load_img(qp)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            top = retriever.query(img, top_k=1)[0][0]
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
            per[room][1] += 1
            per[room][0] += (labels[top] == room)
    times = sorted(times[1:])  # drop first (warmup)
    med = times[len(times) // 2]
    tot_c = sum(v[0] for v in per.values()); tot_n = sum(v[1] for v in per.values())
    lines = [f"  {r:7s} top-1 correct-room: {per[r][0]:3d}/{per[r][1]:3d} = "
             f"{100*per[r][0]/per[r][1]:.1f}%" for r in ROOMS]
    lines.append(f"  OVERALL              : {tot_c}/{tot_n} = {100*tot_c/tot_n:.1f}%")
    lines.append(f"  median retrieval time: {med:.1f} ms/query")
    print("\n".join(lines))
    return [f"## {tag}  (bank {len(labels)} frames)"] + lines

if __name__ == "__main__":
    out = ["# Multi-room retrieval - top-1 correct-room rate (ATP #2)",
           "query set: 200 per room (class1, class2, stacey) = 600; class2 from classroom2_test.MOV"]
    print("loading MegaLoc...");  out += run(MegaLocRetriever(DEVICE), 0, "MegaLoc")
    print("loading CLIP...");     out += run(ClipRetriever(DEVICE), 1, "CLIP")
    with open("/home/chini/Desktop/3DGS/code/RESULTS_MULTIROOM.md", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\nwrote RESULTS_MULTIROOM.md")
