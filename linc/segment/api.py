"""Prediction over a folder: predict_folder(model, IN, OUT). A case in IN is <case>_<cccc><ext>: one file per channel, cccc = 0000, 0001, ...,
ext = .nii.gz / .nii / .png / .tif. Its mask is written as OUT/<case><ext> (uint8: 0 background, 1.. the classes) in the
input's grid.

    model_dir(task)     $LINC_MODELS/segment/<task>/ (model.pt, spec.json, card.md; models/segmentation.md), or a model folder
    predict_folder(model, in_dir, out_dir, native=False, postprocess="spec", masks=None, device="cuda", mirror=None)

native: the images are photographs at native size; they are first reduced by the model's native_downsample (block means; bf_mask2d:
4 x 4), the network's working level, and the mask comes back at native size.
postprocess: "spec" (the model's own, spec.json inference.postprocess), "none", "largest_component" or "largest_cc_fill".
masks: the folder of brain masks <case><ext> on the images' grids, needed by tubes_left_hemisphere (bf_holes) with postprocess "spec".
"""
import os
import re
import time
from pathlib import Path

import nibabel as nib

TASKS = ("mri_brainmask", "bf_mask2d", "bf_mask3d", "bf_holes")
_CASE = re.compile(r"^(?P<case>.+)_(?P<ch>\d{4})(?P<ext>\.nii\.gz|\.nii|\.png|\.tiff?)$")


def model_dir(model):
    """a model folder as given, or a task name -> $LINC_MODELS/segment/<task>"""
    p = Path(model)
    if p.is_dir():
        return p
    if str(model) not in TASKS:
        raise ValueError(f"{model}: neither a model folder nor one of {TASKS}")
    root = os.environ.get("LINC_MODELS")
    if not root:
        raise RuntimeError("set LINC_MODELS (the folder holding segment/<task>/; models/segmentation.md)")
    return Path(root) / "segment" / str(model)


def cases_in(in_dir):
    """{case: (ext, [channel files in order])} for the <case>_<cccc><ext> files of a folder"""
    groups = {}
    for f in sorted(Path(in_dir).iterdir()):
        m = _CASE.match(f.name)
        if m:
            groups.setdefault((m["case"], m["ext"]), {})[int(m["ch"])] = f
    out = {}
    for (case, ext), ch in sorted(groups.items()):
        if sorted(ch) != list(range(len(ch))):
            raise ValueError(f"{in_dir}: case {case} has channels {sorted(ch)}")
        if case in out:
            raise ValueError(f"{in_dir}: case {case} has files with two extensions")
        out[case] = (ext, [ch[i] for i in range(len(ch))])
    return out


def _n_channels(files, dim):
    """the channels of one case, from the file headers: a NIfTI with one more axis than the model's dim holds channels last;
    a PNG / TIFF holds 1 (grey) or 3 (RGB, alpha dropped), as io.read_image reads them"""
    n = 0
    for f in files:
        if str(f).endswith((".nii", ".nii.gz")):
            sh = [s for s in nib.load(str(f)).shape]
            n += sh[dim] if len(sh) > dim and not (dim == 2 and len(sh) == 3 and sh[2] == 1) else 1
        else:
            from PIL import Image
            with Image.open(f) as im:
                n += 1 if len(im.getbands()) == 1 else 3
    return n


def predict_folder(model, in_dir, out_dir, native=False, postprocess="spec", masks=None, device="cuda", mirror=None):
    """predict every case of in_dir into out_dir; returns {case: seconds}"""
    from . import io
    from .predict import load, predict_case
    net, spec = load(model_dir(model), device)
    cases = cases_in(in_dir)
    if not cases:
        raise RuntimeError(f"{in_dir}: no <case>_<cccc> image files")
    want = spec["net"]["in_channels"]
    pp = None if postprocess == "spec" else postprocess
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    took = {}
    for case, (ext, files) in cases.items():
        got = _n_channels(files, spec["net"]["dim"])
        if got != want:
            raise ValueError(f"{case}: {got} channels in {len(files)} file(s); the model takes {want}")
        cm = Path(masks) / f"{case}{ext}" if masks else None
        t0 = time.time()
        mask, meta, _ = predict_case(net, spec, [str(f) for f in files], device, mirror, pp, native,
                                     str(cm) if cm else None)
        io.write_mask(mask, meta, out_dir / f"{case}{ext}")
        took[case] = round(time.time() - t0, 2)
        print(f"  {case}: {took[case]:.1f} s", flush=True)
    return took
