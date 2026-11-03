#!/usr/bin/env python3
"""The blockface cohort for Table 1 and Methods: one row per photographed brain, every field read from its source of record, never
typed.

    python reproduce/numbers/bf_cohort.py [--raw $LINC_RAW] [--participants $LINC_PARTICIPANTS]

You need
  the products in configs/products_blockface.yaml (`data`, `mri_src`, `registration_qc_store`, `folds`, `histo_arm_final`), the raw
  photographs and the participants table.
What it does
  It reads, per brain:
  configs/subjects/<ID>.yaml      vx, vz, rotate180, jumps, missing, jump_mode, the raw photographs' folder and pattern
  <ID>.info                       native sections, output sections, interpolated sections, exposure steps, tube voxels, L1 shape,
                                  orientation correlation (written by the reconstruction)
  curated/                        the hand-corrected mask and tubes (the _edited files)
  <ID>_holes.nii.gz               the number of marker tubes (3D connected components >= 3 voxels on the L4 grid)
  the raw photographs             the photograph count, zero-byte JPEGs (data lost at acquisition)
  mri/<ID>_mri.info, the b0       the MRI used for the registration to the brain's own MRI (b0), its voxel size, whether a T2 exists
  registration_qc.json            the registration QC store's production rows (blockface to b0; the blockface-based alignment,
                                  tag to_bf_syn_t3_tstep)
  folds_v2.json                   each brain's role in the template builds of the time
  06_histo2bf/tirl                the histology components registered (this tree, and the final histology registration of record)
  participants.tsv                the species of record (LINC dataset 000003)
  The native section step caps the resolution any isotropic product of the brain can claim (stated per row).
You get
  $LINC_NUMBERS/bf_cohort.json (the rows' text fields are the record's, kept verbatim).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import yaml
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from linc import paths, provenance  # noqa: E402

R3TAG = "to_bf_syn_t3_tstep"                                            # the route-A direct rows of the time (26 Sep)
SUBJ = ["MF191", "MF196", "MF204", "MF275", "MF278", "MF282", "MR243", "MR244", "MR247", "MR248", "MR249", "MR250",
        "MR251", "MR252", "MR253", "MR254", "MR256", "MR257", "MR261", "MR263", "MR273"]
INLINE = ("MF191", "MF204", "MR261")                                    # brains whose acquisition inputs were not in the spreadsheet
NOTES = ["Raw photographs with zero bytes are original data loss (the same files are empty upstream); "
         "they are treated as missing frames.",
         "OME-Zarr axis caveat: in <S>_blockface.ome.zarr the physical slice (stack) axis is the one labelled 'y' "
         "(array index 2); the axis labelled 'z' carries the in-plane 2*vx scale.",
         "isotropic_cap_mm = max(native z spacing, L1 in-plane): no isotropic product of that subject carries "
         "information finer than this along its coarsest axis.",
         "species_of_record: from LINC dandiset 000003 participants.tsv only (the MF/MR prefix is NOT used as evidence); "
         "brain_volume_ml: measured from the deliverable mask (photographed blockface, i.e. caudally truncated)."]   # verbatim


def subject(S):
    return yaml.safe_load((REPO / "configs" / "subjects" / f"{S}.yaml").read_text())


def info(path):
    d = {}
    for line in open(path):
        if "=" in line:
            k, v = line.strip().split("=", 1); d[k] = v
    return d


def ints(v):
    return [int(x) for x in v.split(",") if x.strip()] if v else []


def species_of_record(path):
    out = {}
    if path and os.path.exists(path):
        for line in open(path):
            f = line.split()
            if len(f) >= 3 and f[0].startswith("sub-"):
                out[f[0][4:]] = f"{f[1]} {f[2]}".strip()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.environ.get("LINC_RAW"), help="root of the raw photographs (configs/subjects/<ID>.yaml `raw`)")
    ap.add_argument("--participants", default=os.environ.get("LINC_PARTICIPANTS"), help="LINC dandiset 000003 participants.tsv")
    a = ap.parse_args()
    assert a.raw, "set --raw or LINC_RAW"
    prod = paths.products("products_blockface")
    data, folds_p, qp = prod["data"], prod["folds"], prod["registration_qc_store"]
    rec = species_of_record(a.participants)
    folds = json.load(open(folds_p))
    fd = folds["folds"] if isinstance(folds["folds"], dict) else json.loads(folds["folds"])
    role = {}
    for k, f in fd.items():
        for s in f["template_no_mri_members"]:
            role[s] = "T1 member, no MRI (scaffold completion, synthetic weight 0; R3-placed)"
        for s in f["heldout"]:
            role[s] = f"T1 member with MRI; CV {k} held-out" + (" (Wednesday fold)" if k == folds["wednesday_fold"] else "")
    role[folds["external"].split()[0]] = "external (never in a template; R1 + both routes onto the shipped template)"
    for s in folds["robustness_route_A_only"]:
        role[s] = "not a member; robustness, route A only (R3-placed)"
    r3, qc = {}, {}
    if os.path.exists(qp):
        for r in json.load(open(qp)):
            if r["family"] == "bf2b0" and r.get("tag", "") == "":              # production row only (not study arms)
                qc[r["subject"]] = r
            elif r["family"] == "bf2bftemplate_direct" and r.get("tag", "") == R3TAG:
                r3[r["subject"]] = r
    rows = []
    for S in SUBJ:
        c = subject(S)
        bd = data / f"sub-{S}" / "blockface"; md = data / f"sub-{S}" / "mri"
        inf = info(bd / f"{S}.info")
        raw = sorted(glob.glob(os.path.join(a.raw, c["raw"], c["raw_glob"])))
        zero = sum(1 for f in raw if os.path.getsize(f) == 0)
        hp = bd / f"{S}_holes.nii.gz"
        ntubes = None
        if hp.exists():
            h = np.asanyarray(nib.load(str(hp)).dataobj) > 0
            lab, n = ndi.label(h)
            ntubes = int(sum(1 for s in ndi.sum(h, lab, range(1, n + 1)) if s >= 3)) if n else 0
        cur = bd / "curated"
        mri = md / f"{S}_mri.info"
        mri_mod = None; mri_vox = None; t2 = False
        if mri.exists():
            mi = dict(line.strip().split("=", 1) for line in open(mri) if "=" in line)
            mri_mod = mi.get("modality")
            b0 = md / f"{S}_mri_b0_native.nii.gz"
            if b0.exists():
                mri_vox = round(float(nib.load(str(b0)).header.get_zooms()[0]), 3)
            t2 = (prod["mri_src"] / S / "msme_echo_denoised_debiased_brain.nii.gz").exists()
        mri_vol = None
        mk = md / f"{S}_mri_mask_native.nii.gz"
        if mk.exists():
            mi_ = nib.load(str(mk))
            mri_vol = round(float((np.asanyarray(mi_.dataobj) > 0).sum() * np.prod(mi_.header.get_zooms()[:3])) / 1000.0, 2)
        interp = ints(inf.get("interp_native", ""))
        vz = float(c["vz"])
        fm = nib.load(str(bd / f"{S}_mask.nii.gz"))
        row = dict(
            subject=S, species_of_record=rec.get(S, "not recorded"),
            brain_volume_ml=round(float((np.asanyarray(fm.dataobj) > 0).sum() * np.prod(fm.header.get_zooms()[:3])) / 1000.0, 2),
            metadata_source="inline config.py" if S in INLINE else "macaque_datasets.xlsx",
            vx_um=float(c["vx"]), L1_inplane_um=2 * float(c["vx"]), vz_um=vz, rotate180=bool(c["rotate180"]),
            raw_photos=len(raw), zero_byte_raw=zero, native_slices=int(inf.get("native_nz", 0) or 0),
            output_slices=int(inf.get("z_out", 0) or 0), jumps=len(c["jumps"]), jump_mode=c.get("jump_mode", "affine"),
            missing_intervals=len(c["missing"]), interpolated_sections=len(interp),
            exposure_steps=len(ints(inf.get("intensity_steps", ""))), hole_voxels=int(inf.get("hole_voxels", 0) or 0),
            hole_tubes_L4=ntubes, mask_hand_curated=(cur / f"{S}_mask_edited.nii.gz").exists(),
            holes_hand_curated=(cur / f"{S}_holes_edited.nii.gz").exists(),
            L1_shape=inf.get("L1_shape"), orient_corr=float(inf["orient_corr"]) if "orient_corr" in inf else None,
            mri=mri_mod, mri_voxel_mm=mri_vox,
            mri_tier=(f"{mri_vox:g} mm b0" + (" + T2" if t2 else "")) if mri_mod and mri_vox else None,
            mri_brain_volume_ml=mri_vol,
            t2_available=t2, bf2b0_gate=(qc[S]["gate_pass"] if S in qc else None),
            bf2b0_qc_note=(qc[S].get("note") if S in qc else None),
            bf2b0_drift_p95_mm=(qc[S]["jacobian"].get("drift_p95_mm") if S in qc else None),
            r3_gate=(r3[S]["gate_pass"] if S in r3 else None),
            role=role.get(S, "UNASSIGNED"),
            histo_pieces_registered_here=len(glob.glob(str(bd / "06_histo2bf" / "tirl" / "*"))) or 0,
            histo_pieces_final_arm=len(glob.glob(str(prod["histo_arm_final"] / f"sub-{S}" / "blockface" / "06_histo2bf" / "tirl" / "*/"))),
            histo_pieces_final_arm_complete=len(glob.glob(str(prod["histo_arm_final"] / f"sub-{S}" / "blockface" / "06_histo2bf" / "tirl" / "*" /
                                                              "fixed4_nonlinear.timg"))),
            isotropic_cap_mm=round(max(vz, 2 * float(c["vx"])) / 1000.0, 3),
        )
        rows.append(row)
        print(f"{S}: vx {row['vx_um']} vz {vz:.0f} slices {row['output_slices']} interp {len(interp)} tubes {ntubes} mri {mri_mod} {mri_vox}",
              flush=True)
    provenance.write_numbers("bf_cohort", dict(rows=rows, notes=NOTES), inputs=[folds_p, qp], cmd="python reproduce/numbers/bf_cohort.py",
                             md5=False)


if __name__ == "__main__":
    main()
