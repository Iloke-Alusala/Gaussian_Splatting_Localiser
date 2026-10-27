# Multi-room retrieval - top-1 correct-room rate (ATP #2)
query set: 200 per room (class1, class2, stacey) = 600; class2 from classroom2_test.MOV
## MegaLoc  (bank 7854 frames)
  class1  top-1 correct-room: 200/200 = 100.0%
  class2  top-1 correct-room: 192/200 = 96.0%
  stacey  top-1 correct-room: 200/200 = 100.0%
  OVERALL              : 592/600 = 98.7%
  median retrieval time: 40.2 ms/query
## CLIP  (bank 7854 frames)
  class1  top-1 correct-room: 197/200 = 98.5%
  class2  top-1 correct-room: 135/200 = 67.5%
  stacey  top-1 correct-room: 194/200 = 97.0%
  OVERALL              : 526/600 = 87.7%
  median retrieval time: 6.6 ms/query
