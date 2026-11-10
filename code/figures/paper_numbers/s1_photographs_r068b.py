#!/usr/bin/env python3
"""The photographed sections of each brain's blockface volume (paper: Supplementary Table "Blockface photographs of each brain",
column Photographed).

You need
  each brain's <ID>.info in the products (products_blockface `data`), and settings/photograph_counts.json: per brain, the number of
  frames of 00_slices and the placeholder frames among the volume's, counted once from the frames.
What it does
  The volume holds the native frames 0 .. z_out - 1 of 00_slices (code/blockface/reconstruct.py: the caudal cutoff is a
  z-truncation). A section never photographed is a black placeholder frame (black, under a 2D mask covering more than 90 % of the
  frame). photographed_sections = z_out minus the placeholders among the volume's frames.
  interpolated_sections is the .info's interp_native list (never photographed or rejected, filled from the neighbors), so the two
  columns overlap by the rejected photographs. table_r057_photographed is the earlier column, every frame of 00_slices.
You get
  $LINC_NUMBERS/s1_photographs_r068b.json, per brain (LINC ID): name, photographed_sections, interpolated_sections,
  volume_sections, placeholder_frames_in_volume, frames, frames_past_cutoff, table_r057_photographed. Seconds (CPU).
    python code/figures/paper_numbers/s1_photographs_r068b.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from shared import paths, provenance  # noqa: E402

NOTE = ("photographed_sections = sections of the volume (native frames 0 .. volume_sections - 1) whose frame is a photograph, not a "
        "black placeholder; replaces s1_perbrain_r057.photographed_sections (all frames)")


def main():
    lab = json.loads((REPO / "settings" / "specimen_labels.json").read_text())["labels"]
    data = paths.products("products_blockface")["data"]
    brains = sorted((s for s in lab if (data / f"sub-{s}" / "blockface" / f"{s}.info").exists()), key=lambda s: (lab[s][:2], int(lab[s][2:])))
    assert len(brains) == 21, brains
    counts = json.loads((REPO / "settings" / "photograph_counts.json").read_text())
    per, ins = {}, []
    for S in brains:
        bd = data / f"sub-{S}" / "blockface"
        info = dict(l.strip().split("=", 1) for l in open(bd / f"{S}.info") if "=" in l); ins.append(bd / f"{S}.info")
        nv = int(info["z_out"]); interp = [int(x) for x in info.get("interp_native", "").split(",") if x]
        frames, black = counts[S]["frames"], counts[S]["black"]
        per[S] = dict(name=lab[S], photographed_sections=nv - len(black), interpolated_sections=len(interp), volume_sections=nv,
                      placeholder_frames_in_volume=len(black), frames=frames, frames_past_cutoff=frames - nv,
                      table_r057_photographed=frames)
        print(S, per[S], flush=True)
    provenance.write_numbers("s1_photographs_r068b", dict(per_brain=per, note=NOTE), inputs=ins, md5=False,
                             cmd="python code/figures/paper_numbers/s1_photographs_r068b.py")


if __name__ == "__main__":
    main()
