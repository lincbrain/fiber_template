#!/usr/bin/env python3
"""Mappings between the MRI template and three published macaque templates: NMT v2 (symmetric), INIA19 and CIVM-R (paper: Methods and
Results, "Relation to published templates"). Each mapping is a symmetric deformable registration without landmarks (the template's T2 channel
against the atlas image; the b0 channel for CIVM-R). Per atlas, from its score record (products: atlas_v2/<atlas>/score.json):
brain-mask Dice of each mask carried across (both directions), the outline gate status (both directions), the fraction of folded
voxels of the four maps, the largest symmetric round-trip error (mm), the Jacobian range of both maps inside the brain, and, for
CIVM-R, the pointwise distance (mm, inside the CIVM-R brain) to an independent direct registration of the two b0 images.
Numbers only: no ONPRC18 or CIVM-R image data is read or written here.
-> $LINC_NUMBERS/atlas_final_v2_summary.json
    python code/figures/paper_numbers/atlas_final_v2_summary.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402

ATLASES = ["NMT_v2.0_sym", "INIA19", "CIVM-R"]                     # folder names under atlas_v2/, in record order
NOTE = "landmark-free (author 6 Oct); the rebuilt template (4 Oct)"  # the record's own note, verbatim (part of its data)


def main():
    A = paths.products("products_mri")["atlas_v2"]
    rows = {n: json.loads((A / n / "score.json").read_text()) for n in ATLASES if (A / n / "score.json").exists()}
    out = {n: dict(mask_dice=r["mask_dice"], outline={k: v["status"] for k, v in r["outline"].items()}, folds_fraction=r["folds_fraction"],
                   roundtrip_max_mm=max(v["max"] for v in r["roundtrip_mm"].values()) if r["roundtrip_mm"] else None,
                   detJ=r["detJ"],
                   **({"vs_item11_anatomix_direct_mm": r["vs_item11_anatomix_direct_mm"]} if "vs_item11_anatomix_direct_mm" in r else {}))
           for n, r in rows.items()}
    provenance.write_numbers("atlas_final_v2_summary", dict(atlases=out, note=NOTE), inputs=[A / n / "score.json" for n in rows],
                             md5=False, cmd="python code/figures/paper_numbers/atlas_final_v2_summary.py")


if __name__ == "__main__":
    main()
