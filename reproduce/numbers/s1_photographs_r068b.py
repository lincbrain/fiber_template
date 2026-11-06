#!/usr/bin/env python3
"""The photographed sections of each brain's blockface volume (paper: Supplementary Table "Blockface photographs of each brain",
column Photographed).

You need
  each brain's reconstruction in the products (products_blockface `data`: blockface/00_slices, the 2D masks
  01_brainmask/01_final and <ID>.info).
What it does
  The volume holds the native frames 0 .. z_out - 1 of 00_slices (linc/blockface/reconstruct.py: the caudal cutoff is a
  z-truncation). A section never photographed is a black placeholder frame (its 2D mask covers about the whole frame; only frames
  whose mask covers > 90 % of it are opened). photographed_sections = z_out minus the placeholders among the volume's frames.
  interpolated_sections is the .info's interp_native list (never photographed or rejected, filled from the neighbors), so the two
  columns overlap by the rejected photographs. table_r057_photographed is the earlier column, every frame of 00_slices.
You get
  $LINC_NUMBERS/s1_photographs_r068b.json, per brain (LINC ID): name, photographed_sections, interpolated_sections,
  volume_sections, placeholder_frames_in_volume, frames, frames_past_cutoff, table_r057_photographed. About 12 min (CPU).
    python reproduce/numbers/s1_photographs_r068b.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from linc import paths, provenance  # noqa: E402

NOTE = ("photographed_sections = sections of the volume (native frames 0 .. volume_sections - 1) whose frame is a photograph, not a "
        "black placeholder; replaces s1_perbrain_r057.photographed_sections (all frames)")


def main():
    lab = json.loads((REPO / "docs" / "specimen_labels.json").read_text())["labels"]
    data = paths.products("products_blockface")["data"]
    brains = sorted((s for s in lab if (data / f"sub-{s}" / "blockface" / f"{s}.info").exists()), key=lambda s: (lab[s][:2], int(lab[s][2:])))
    assert len(brains) == 21, brains
    per, ins = {}, []
    for S in brains:
        bd = data / f"sub-{S}" / "blockface"
        info = dict(l.strip().split("=", 1) for l in open(bd / f"{S}.info") if "=" in l); ins.append(bd / f"{S}.info")
        nv = int(info["z_out"]); interp = [int(x) for x in info.get("interp_native", "").split(",") if x]
        frames = sorted((bd / "00_slices").glob("SLICE_*.png"))
        black = []
        for f in frames:
            k = int(f.name[6:10])
            m = bd / "01_brainmask" / "01_final" / f"SLICE_{k:04d}.png"
            if k < nv and m.exists() and (np.asarray(Image.open(m)) > 0).mean() > 0.9 and np.asarray(Image.open(f)).max() == 0:
                black.append(k)
        per[S] = dict(name=lab[S], photographed_sections=nv - len(black), interpolated_sections=len(interp), volume_sections=nv,
                      placeholder_frames_in_volume=len(black), frames=len(frames), frames_past_cutoff=len(frames) - nv,
                      table_r057_photographed=len(frames))
        print(S, per[S], flush=True)
    provenance.write_numbers("s1_photographs_r068b", dict(per_brain=per, note=NOTE), inputs=ins, md5=False,
                             cmd="python reproduce/numbers/s1_photographs_r068b.py")


if __name__ == "__main__":
    main()
