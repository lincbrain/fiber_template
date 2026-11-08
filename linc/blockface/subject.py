"""One brain's blockface inputs and folder layout, read from configs/subjects/<ID>.yaml and configs/blockface.yaml.

You need
  configs/subjects/<ID>.yaml, the hand inputs of a reconstruction:
  raw, raw_glob, raw_regex, idx_range   the photographs ($LINC_RAW/<raw>/<files>, numbered by the regex' group 1, in this range)
  rotate180                             photographs taken upside down
  vx, vz                                the pixel size and section step (um) written into the products' headers
  reconstruction_spacing_um             the spacing the pixels were reconstructed at (equal to vx, vz for a new brain; for the
                                        paper's brains, the spacing before the calibration against each brain's MRI)
  jumps                                 camera jumps (before, after), listed by hand
  missing                               missing or damaged sections: (before, after) brackets of sections to interpolate;
                                        (-1, k) drops the leading frames 0..k-1 without brain; (n, -1) is the caudal cutoff
                                        (keep 0..n)
  stack_pad                             the stacks' margin (px per side); 0 = the default (zarrio.STACK_PAD)
  stack_origin_frames                   optional: the frame count the stacks' world origin and <ID>.info's native_nz count, when the
                                        stack of record held frames that are not this brain's photographs (MR261); unset = the
                                        stack's own count
  curation_f                            native sections per curated section of the hand-edit grid (fixed per brain)
  masks_of_record                       which masks were corrected by hand (they win over the models' outputs)
What it does
  Subject(<ID>) reads both files and names every folder and file of the brain's blockface steps.
You get
  the layout under $LINC_DATA/sub-<ID>/blockface/:
  00_slices/                 the photographs, renamed and turned upright (step 1)
  01_brainmask/00_predict/   the 2D network's masks (x50 PNGs); 01_final/ the accepted masks (hand corrections survive reruns)
  _work/                     working stores: the raw and mask stacks, the camera-jump registrations, the 3D network's inputs
  02_blockface*.ome.zarr     the aligned, cropped stack and its mask (stage 3; also read by the histology steps)
  nii/                       stage 3's NIfTI exports (the volume with its background kept, read by the 3D brain-mask network)
  curated/                   the hand-edit grid (L4 in-plane, 0.40 mm sections): <ID>_bg (the network's input), <ID>_mask and
                             <ID>_holes (the networks' outputs), <ID>_mask_edited and <ID>_holes_edited (the hand
                             corrections, which win)
  <ID>_blockface.ome.zarr    the reconstruction (L1 .. L4), <ID>_mask and <ID>_holes (L4), <ID>.info, steps/ (before the tubes are
                             filled)

Ported from the analysis project's code/config.py (the fields and paths the final pipeline uses) and code/bf_geometry.py
(curation_f).
"""
import os
from pathlib import Path

import yaml

from linc import paths

REPO = Path(__file__).resolve().parents[2]


def params():
    """configs/blockface.yaml as a dict."""
    return yaml.safe_load((REPO / "configs" / "blockface.yaml").read_text())


def subject_yaml(subject):
    """A brain's configuration file: $LINC_SUBJECTS/<ID>.yaml when that folder holds one (a user's own brains, kept outside the
    repository; the self-test's toy brain), else configs/subjects/<ID>.yaml."""
    d = os.environ.get("LINC_SUBJECTS")
    if d and (Path(d) / f"{subject}.yaml").exists():
        return Path(d) / f"{subject}.yaml"
    return REPO / "configs" / "subjects" / f"{subject}.yaml"


class Subject:
    """The blockface configuration of one brain. `vx`/`vz` are the RECONSTRUCTION spacing (what every processing step uses);
    `vx_header`/`vz_header` the calibrated spacing the products carry after linc/blockface/spacing.py."""

    def __init__(self, subject, data_root=None, raw_root=None):
        p = subject_yaml(subject)
        if not p.exists():
            raise SystemExit(f"no configs/subjects/{subject}.yaml (or $LINC_SUBJECTS/{subject}.yaml): write one for a new brain "
                             "(configs/subjects/TEMPLATE.yaml; the paper's brains are examples)")
        y = yaml.safe_load(p.read_text())
        self.yaml = y
        self.subject = y["subject"]
        rr = raw_root or os.environ.get("LINC_RAW")
        self.raw_dir = str(Path(rr) / y["raw"]) if rr else None
        self.raw_glob, self.raw_regex = y["raw_glob"], y["raw_regex"]
        self.idx_range = tuple(y["idx_range"]) if y.get("idx_range") else None
        self.reduce = y.get("reduce", "mean")
        self.rotate180 = bool(y.get("rotate180", False))
        rs = y.get("reconstruction_spacing_um") or {}
        self.vx_header, self.vz_header = float(y["vx"]), float(y["vz"])
        self.vx, self.vz = float(rs.get("vx", y["vx"])), float(rs.get("vz", y["vz"]))
        self.jumps = tuple(tuple(int(v) for v in j) for j in (y.get("jumps") or []))
        self.missing = tuple(tuple(int(v) for v in m) for m in (y.get("missing") or []))
        self.stack_pad = int(y.get("stack_pad") or 0)
        self.stack_origin_frames = int(y["stack_origin_frames"]) if y.get("stack_origin_frames") else None
        self.jump_mode = y.get("jump_mode", "affine")
        self.pinned_curation_f = y.get("curation_f")
        self.masks_of_record = y.get("masks_of_record") or {}
        self._data_root = Path(data_root) if data_root else None

    # ---------------------------------------------------------------- layout
    @property
    def root(self) -> str:                      # $LINC_DATA/sub-<S>/blockface
        base = (self._data_root / f"sub-{self.subject}") if self._data_root else paths.subject_dir(self.subject)
        return os.path.join(str(base), "blockface")

    @property
    def work(self) -> str:                      # transient working data (regenerable; safe to delete)
        return os.path.join(self.root, "_work")

    @property
    def seg3d_dir(self) -> str:                 # scratch for the 3D networks (RGB channels in/ + prediction out/)
        return os.path.join(self.work, "seg3d")

    @property
    def curated_dir(self) -> str:               # the hand-edit grid
        return os.path.join(self.root, "curated")

    @property
    def curated_bg(self) -> str:                # RGB background-intact volume the brain-mask model segments
        return os.path.join(self.curated_dir, f"{self.subject}_bg.nii.gz")

    @property
    def mask_model(self) -> str:                # the 3D brain-mask model's output
        return os.path.join(self.curated_dir, f"{self.subject}_mask.nii.gz")

    @property
    def mask_edited(self) -> str:               # hand-corrected brain mask (wins over the model output)
        return os.path.join(self.curated_dir, f"{self.subject}_mask_edited.nii.gz")

    @property
    def curated_mask(self) -> str:              # the mask the reconstruction uses: edited if present, else the model's
        return self.mask_edited if os.path.exists(self.mask_edited) else self.mask_model

    @property
    def holes_model(self) -> str:               # the 3D marker-tube model's output
        return os.path.join(self.curated_dir, f"{self.subject}_holes.nii.gz")

    @property
    def holes_edited(self) -> str:              # hand-corrected marker tubes (win over the model output)
        return os.path.join(self.curated_dir, f"{self.subject}_holes_edited.nii.gz")

    @property
    def curated_holes(self) -> str:             # the tubes the reconstruction uses: edited if present, else the model's
        return self.holes_edited if os.path.exists(self.holes_edited) else self.holes_model

    def final(self, name: str) -> str:          # <S>_<name>.nii.gz
        return os.path.join(self.root, f"{self.subject}_{name}.nii.gz")

    def final_zarr(self, name: str) -> str:     # <S>_<name>.ome.zarr
        return os.path.join(self.root, f"{self.subject}_{name}.ome.zarr")

    @property
    def steps_dir(self) -> str:
        return os.path.join(self.root, "steps")

    @property
    def color_store(self) -> str:               # steps/<S>_03_color.ome.zarr: masked + corrected, before the tube inpainting
        return os.path.join(self.steps_dir, f"{self.subject}_03_color.ome.zarr")

    @property
    def info_file(self) -> str:                 # <S>.info: the reconstruction's grid contract and provenance
        return os.path.join(self.root, f"{self.subject}.info")

    @property
    def slices(self) -> str:                    # 00: the renamed photographs (the 2D network's input)
        return os.path.join(self.root, "00_slices")

    def brainmask(self, kind: str = "00_predict") -> str:   # 01: 2D brain masks (x50 PNGs): 00_predict | 01_final
        return os.path.join(self.root, "01_brainmask", kind)

    @property
    def blockface(self) -> str:                 # 02: the aligned, cropped, masked stack
        return os.path.join(self.root, "02_blockface.ome.zarr")

    @property
    def blockface_mask(self) -> str:            # 02: its mask
        return os.path.join(self.root, "02_blockface_mask.ome.zarr")

    @property
    def blockface_bg(self) -> str:              # 02: the background-intact twin (transient)
        return os.path.join(self.root, "02_blockface_bg.ome.zarr")

    @property
    def nii_dir(self) -> str:
        return os.path.join(self.root, "nii")

    def snap(self, name) -> str:                # nii/<name>.nii.gz
        return os.path.join(self.nii_dir, name + ".nii.gz")

    @property
    def raw_stack(self) -> str:                 # the raw stack (jump-corrected in place)
        return os.path.join(self.work, "blockface_raw.ome.zarr")

    @property
    def mask_stack(self) -> str:                # the 2D masks, stacked on the same grid
        return os.path.join(self.work, "brainmask.ome.zarr")

    @property
    def mask_edit_nii(self) -> str:             # full-resolution mask stack exported for hand editing
        return self.snap("01_brainmask_L0")

    @property
    def mask_edit_zarr(self) -> str:
        return os.path.join(self.work, "brainmask_edit.ome.zarr")

    @property
    def jumps_dir(self) -> str:                 # the camera-jump frames (TIRL input)
        return os.path.join(self.work, "camera_jumps")

    @property
    def tirl_jumps_dir(self) -> str:            # the camera-jump registrations
        return os.path.join(self.work, "tirl_jumps")

    # ---------------------------------------------------------------- geometry
    def curation_f(self, path=None, check=True):
        """Native sections per curated slice: the brain's pin; else the curated grid's own header (`path`, else the hand mask,
        the model mask or the bg, whichever exists); else (a new brain) round(curation_z_target_um / vz). With check=True the pin
        and the curated header must agree (the header carries the calibrated section step, vz_header). Ported from
        code/bf_geometry.curation_f."""
        import nibabel as nib
        import numpy as np
        target_um = float(params()["reconstruction"]["curation_z_target_um"])
        pin = self.pinned_curation_f
        cands = [path] if path else [self.mask_edited, self.mask_model, self.curated_bg]
        p = next((c for c in cands if c and os.path.exists(c)), None)
        if p is None:
            return int(pin) if pin else max(1, int(round(target_um / float(self.vz))))
        step_um = float(np.linalg.norm(nib.load(p).affine[:3, 1])) * 1000.0
        # a curated header carries either the reconstruction step (a file written during a rebuild) or the calibrated one (after
        # spacing.py, e.g. the hand edits of record): take the reading that is closer to an integer number of sections
        ratios = [step_um / self.vz, step_um / self.vz_header]
        f_hdr = max(1, int(round(min(ratios, key=lambda r: abs(r - round(r))))))
        if check and pin and int(pin) != f_hdr:
            raise RuntimeError(f"{self.subject}: curated grid {os.path.basename(p)} implies f = {f_hdr} (step {step_um:.1f} um) but "
                               f"the pin says f = {pin}: the section step and the curated headers disagree. Refused.")
        return int(pin) if pin else f_hdr
