#!/usr/bin/env python3
"""The blockface segmentation networks (Supplementary S1.1): their training data, the human-in-the-loop rounds of the two 3D
networks, and their performance. The paper describes these networks in Supplementary S1.1; the masks of record they made are
data (settings/brains/<ID>.yaml masks_of_record), and the release segments with the final models (settings/models/README.txt).

    python code/figures/paper_numbers/s1_segmentation_r057.py

You need
  the networks' training-set records (settings/internal/products_blockface.yaml nnunet_training_sets: each training set's dataset.json and,
  for the 3D networks, imagesTr.txt, its training images' names; no weights) and the model-card record bf_models_card.json
  (products: numbers_of_record).
What it does
  It recomputes:
  2D brain mask (Dataset500_SLICE, used for every photograph): numTraining from its dataset.json (trained 26 Dec 2024, before this project, on 149 photographs whose source brains are not recorded); performance from
     bf_models_card `mask2d` (the Dice coefficient between the network's photograph masks and the hand-corrected masks of 14
     brains; not cross-validated).
  3D brain mask, round 5 (Dataset506_BFmask5, the network of record): its training cases (the training set's imagesTr.txt: one case per
     brain + one validation slab), its validation Dice (bf_models_card `dataset506_bfmask3d`).
  3D marker tubes, rounds 2 and 3 (Dataset507_holes2, Dataset508_holes3 = the network of record): training cases, validation Dice
     (bf_models_card `dataset508_holes`).
  The brain-mask rounds 1-4 and the tubes' round 1 cannot be recomputed (their training folders were deleted after round 5): they
  are listed with their source (the QC report made for each round at the time, and the project log), not as measurements.
You get
  $LINC_NUMBERS/s1_segmentation_r057.json.

Ported from the analysis project's paper/r057_l1/s1_segmentation_r057.py (inputs re-rooted; values unchanged).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
LABELS = REPO / "settings" / "specimen_labels.json"
# the record's documented rounds, verbatim
DOCUMENTED = {
    "brain_mask": [
        dict(round=1, dataset="Dataset501_BFmask", brains=1, date="2026-08-10", added=["MR243"],
             source="round 1 QC report 'Blockface brain-mask model - QC (iteration 1)' (10 Aug 2026; not kept as a project file)"),
        dict(round=2, dataset="Dataset502_BFmask2", brains=2, date="2026-08-11", added=["MR250"],
             source="round 2 QC report 'iteration 2 (2 samples, ResEnc)' (11 Aug 2026; not kept as a project file): 'trained on 2 hand-edited samples'"),
        dict(round=3, dataset="Dataset503_BFmask3", brains=4, date="2026-08-12", added=["MR244", "MR248"],
             source="round 3 QC report 'iteration 3 (4 samples)' (12 Aug 2026; not kept as a project file): '4 curated samples'"),
        dict(round=4, dataset="Dataset504_BFmask4", brains=8, date="2026-08-13", added=["MF278", "MR249", "MR251", "MR252"],
             source="round 4 QC report 'iteration 4' (13 Aug 2026; not kept as a project file): description and heading '8 curated samples' (title stale); project log 13 Aug"),
    ],
    "tubes": [
        dict(round=1, dataset="holes1", brains=1, date="2026-08-14", added=["MF278"], source="project log 13-14 Aug (MF278 painted by hand)"),
    ],
}
LOOP = ("each round's network segmented every brain; the predictions of the brains added next were corrected by hand and joined the "
        "training set; the final masks of every brain were checked and corrected by hand where needed (masks_of_record)")


def cases(raw, ds):
    """brains in a training set: imagesTr case ids (channel suffix stripped); 'val'/'v' cases = validation slabs of a brain."""
    ids = sorted({re.sub(r"_\d{4}\.nii\.gz$", "", n) for n in (raw / ds / "imagesTr.txt").read_text().split()})
    brains = sorted({re.sub(r"^(BFF?|H)", "", i).rstrip("v") for i in ids if i not in ("BFval",)})
    return ids, brains


def mapname(n, labels):
    S = ("MF" if n.startswith("F") or n in ("275", "278", "282") else "MR") + n.lstrip("F")
    return labels.get(S, S)


def main():
    prod = paths.products("products_blockface")
    raw, card_p = prod["nnunet_training_sets"], prod["numbers_of_record"] / "bf_models_card.json"
    d2_p = raw / "Dataset500_SLICE" / "dataset.json"
    card = json.loads(card_p.read_text())["data"]; labels = json.loads(LABELS.read_text())["labels"]
    d2 = json.loads(d2_p.read_text())
    r5_ids, r5 = cases(raw, "Dataset506_BFmask5"); h2_ids, h2 = cases(raw, "Dataset507_holes2"); h3_ids, h3 = cases(raw, "Dataset508_holes3")
    out = dict(
        mask2d=dict(dataset="Dataset500_SLICE", task="brain vs background in each blockface photograph", input="RGB photograph",
                    training_photographs=d2["numTraining"], trained="2024-12-26 (before this project; source brains not recorded)",
                    deployed="fold 0 alone, unchanged",
                    dice_vs_curated=card["mask2d"]["summary"]["incumbent"], n_brains_scored=card["mask2d"]["n_subjects"],
                    cross_validated=False, note=card["caveats"][0]),
        mask3d=dict(task="brain vs background in the aligned color stack", input="RGB stack, 0.32 mm in-plane, 0.40 mm between sections",
                    rounds=DOCUMENTED["brain_mask"] + [dict(round=5, dataset="Dataset506_BFmask5", brains=len(r5), date="2026-08-14",
                                                             cases=r5_ids, brain_names=[mapname(b, labels) for b in r5],
                                                             source="models/nnUNet_raw/Dataset506_BFmask5/imagesTr (recomputed)")],
                    n_rounds=5, validation=card["dataset506_bfmask3d"]),
        tubes3d=dict(task="the drilled marker tubes in the masked color brain", input="masked RGB brain, 0.32 x 0.32 x 0.40 mm",
                     rounds=DOCUMENTED["tubes"] + [dict(round=2, dataset="Dataset507_holes2", brains=len(h2), date="2026-08-14", cases=h2_ids,
                                                        brain_names=[mapname(b, labels) for b in h2], source="imagesTr (recomputed)"),
                                                   dict(round=3, dataset="Dataset508_holes3", brains=len(h3), date="2026-08-19", cases=h3_ids,
                                                        brain_names=[mapname(b, labels) for b in h3], source="imagesTr (recomputed)")],
                     n_rounds=3, validation=card["dataset508_holes"]),
        loop=LOOP)
    out["mask3d"]["first_brains"] = out["mask3d"]["rounds"][0]["brains"]; out["mask3d"]["last_brains"] = len(r5)
    out["tubes3d"]["first_brains"] = 1; out["tubes3d"]["last_brains"] = len(h3)
    print("2D", out["mask2d"]["training_photographs"], out["mask2d"]["dice_vs_curated"])
    print("3D mask rounds", [r["brains"] for r in out["mask3d"]["rounds"]], r5_ids)
    print("tubes rounds", [r["brains"] for r in out["tubes3d"]["rounds"]], h2_ids, len(h3))
    provenance.write_numbers("s1_segmentation_r057", out, inputs=[card_p, LABELS, d2_p] + [raw / ds / f for f in ("dataset.json", "imagesTr.txt") for ds in
                             ("Dataset506_BFmask5", "Dataset507_holes2", "Dataset508_holes3")],
                             md5=False, cmd="python code/figures/paper_numbers/s1_segmentation_r057.py")


if __name__ == "__main__":
    main()
