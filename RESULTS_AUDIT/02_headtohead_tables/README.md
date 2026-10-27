# Head-to-head tables

`report_tables.md` — auto-generated clean comparison tables (from the summary files on disk).

Contains, per scene: ours 1/2/3-pass, ours CLIP 2-pass, ours photometric, and HLoc, with
median t, median r, ms/query and recall at the four bars. Has both a **scene-unit** section and
a **metric** section, plus the controlled-seed / basin comparison.

Headline to remember: on **Office (stacey)** ours 2/3-pass beats HLoc on *recall* (robust) while
HLoc keeps a lower *median* on the queries it solves. On **Room1 (class1)** both saturate near 100%.

Status: authoritative summary. Regenerate with `make_report_tables.py`.
