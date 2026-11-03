#!/usr/bin/env python3
"""What can be verified about the MRI of the ten photographed M. mulatta brains with MRI (MM1-MM10 = MR243, MR252, MR256, MR249, MR250,
MR251, MR253, MR254, MR257, MR261) (paper: Methods, "Photographed brains and histology", and Table 1).
  curated     products: curated_mri/<ID>/dwi.nii.gz (the diffusion series whose volume 0 is the b0 image used): number of volumes and
              voxel size from the header.
  raw Bruker  where a raw ParaVision session survives (<raw Bruker root>/<ID>_*/<scan>/{method,acqp}): the diffusion scans with
              Method=daiDtiEpi, with PVM_SpatResol (mm), PVM_EchoTime, PVM_RepetitionTime (ms), PVM_DwNDiffExp (number of diffusion
              experiments) from method and BF1 (MHz; 200.3 MHz = 4.7 T proton frequency) from acqp, as written in the files.
  unpacked    exports named *daiDtiEpi*.bvals in <unpacked root>/<ID>/unpack, listed as evidence of the Bruker method (no parameters).
Summary: brains whose curated series has 515 volumes at 0.7 mm isotropic, brains with a raw session read, brains with an unpacked export.
The raw data live outside the products; both roots are given on the command line (the record lists the method files it read by path).
-> $LINC_NUMBERS/mm_mri_check_r054.json
    python reproduce/numbers/mm_mri_check_r054.py --raw-bruker-root <dir> --unpacked-root <dir>"""
import argparse
import re
import sys
from glob import glob
from pathlib import Path

import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

MM = ["MR243", "MR252", "MR256", "MR249", "MR250", "MR251", "MR253", "MR254", "MR257", "MR261"]   # LINC IDs of MM1-MM10
DSI_VOLUMES, DSI_VOXEL_MM = 515, [0.7, 0.7, 0.7]                    # the 515-point DSI protocol at 0.7 mm (Methods)
DTI_METHODS = ("Method=daiDtiEpi", "Method=<Bruker:daiDtiEpi>")     # ParaVision method names of the diffusion EPI scans


def method_text(raw_root, unpacked_root):
    """The record's own method text, verbatim (it is part of the record's data); the two data roots are the command-line ones."""
    return [
        "MR256, MR249, MR250, MR251, MR253, MR254, MR257, MR261).",
        "  curated     data/_lcn_src/data/curated/<S>/dwi.nii.gz (the series whose volume 0 is the b0 image used): number of volumes and voxel",
        "              size from the header.",
        f"  raw Bruker  where a raw ParaVision session survives ({raw_root}/<session>/<scan>/{{method,acqp}}): the",
        "              diffusion scans with Method=daiDtiEpi: PVM_SpatResol (mm), PVM_EchoTime, PVM_RepetitionTime (ms), PVM_DwNDiffExp",
        "              (number of diffusion experiments), and BF1 (MHz; 200.3 MHz = 4.7 T proton frequency) from acqp. Unpacked exports named",
        f"              *.daiDtiEpi.* ({unpacked_root}/<S>/unpack) are listed as Bruker-method evidence without parameters.",
        "-> paper/numbers/mm_mri_check_r054.json",
        "    mri/env/mmorf/bin/python paper/scripts/mm_mri_check_r054.py"]


def par(txt, key):
    """Value of a ParaVision parameter ##$<key>=..., including multi-line array values, as written (string), or None."""
    m = re.search(r"##\$" + key + r"=(\( [^)]*\)\s*)?([^\n#$]*(?:\n[^#$][^\n]*)*)", txt)
    return m.group(2).strip() if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-bruker-root", required=True, type=Path, help="folder holding the raw ParaVision sessions <ID>_*/")
    ap.add_argument("--unpacked-root", required=True, type=Path, help="folder holding <ID>/unpack/ with the unpacked Bruker exports")
    a = ap.parse_args()
    G, U = a.raw_bruker_root, a.unpacked_root
    CUR = paths.products("products_mri")["curated_mri"]
    out = dict(method=method_text(G, U), brains={}); ins = []
    for S in MM:
        r = {}
        dw = CUR / S / "dwi.nii.gz"
        if dw.exists():
            try:
                h = nib.load(str(dw)).header
            except nib.filebasedimages.ImageFileError:               # some files are uncompressed despite the .nii.gz name
                with open(dw, "rb") as fh:
                    h = nib.Nifti1Header.from_fileobj(fh)
            ins.append(dw)
            r["curated_n_volumes"] = int(h.get_data_shape()[3]) if len(h.get_data_shape()) > 3 else 1
            r["curated_voxel_mm"] = [round(float(z), 3) for z in h.get_zooms()[:3]]
        scans = []
        for m in sorted(glob(str(G / f"{S}_*" / "*" / "method"))):
            t = open(m, errors="ignore").read()
            if not any(x in t for x in DTI_METHODS):
                continue
            acqp = Path(m).with_name("acqp"); at = open(acqp, errors="ignore").read() if acqp.exists() else ""
            scans.append(dict(method=m, spat_resol_mm=par(t, "PVM_SpatResol"), echo_time_ms=par(t, "PVM_EchoTime"),
                              rep_time_ms=par(t, "PVM_RepetitionTime"), n_diff_exp=par(t, "PVM_DwNDiffExp"), bf1_mhz=par(at, "BF1")))
            ins.append(Path(m))
        r["raw_bruker_daiDtiEpi_scans"] = scans
        r["unpacked_daiDtiEpi_exports"] = sorted(Path(p).name for p in glob(str(U / S / "unpack" / "*daiDtiEpi*.bvals")))
        r["raw_verified"] = bool(scans)
        out["brains"][S] = r
    B = out["brains"]
    out["summary"] = dict(
        curated_515_volumes_0p7mm=[S for S, r in B.items() if r.get("curated_n_volumes") == DSI_VOLUMES and r.get("curated_voxel_mm") == DSI_VOXEL_MM],
        raw_session_read=[S for S, r in B.items() if r["raw_verified"]],
        unpacked_bruker_export=[S for S, r in B.items() if r["unpacked_daiDtiEpi_exports"]])
    provenance.write_numbers("mm_mri_check_r054", out, inputs=ins, md5=False,
                             cmd=" ".join(["python reproduce/numbers/mm_mri_check_r054.py"] + sys.argv[1:]))


if __name__ == "__main__":
    main()
