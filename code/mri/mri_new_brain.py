"""A new brain's MRI, prepared for `linc-register mri --place-b0` (code/mri/linc-mri prep): from one b0 or T2-weighted NIfTI to the files
the paper's brains have under $LINC_DATA/sub-<ID>/mri/ (settings/mri_construction.yaml `prep.layout`), by the same steps as the paper's
preprocessing (Methods, "MRI preprocessing"; mri_prep.stage_dwi / stage_t2):

    b0   stage -> [denoise] -> brain mask -> b0 x mask -> N4 (inside the mask) -> x mask, 1-99 percentile -> (0, 4095]
         dwi/b0.nii.gz, [dwi/b0_denoised.nii.gz], masks/dwi_brain_mask.nii.gz, dwi/b0_brain.nii.gz, dwi/b0_brain_n4.nii.gz (+ _bias)
    t2   stage -> [denoise] -> brain mask -> N4 of the head (inside the mask) -> x mask, normalized
         t2/echo.nii.gz, [t2/echo_denoised.nii.gz], masks/t2_brain_mask.nii.gz, t2/echo_n4.nii.gz (+ _bias), t2/echo_n4_brain.nii.gz

stage      one 3-D volume of the input (volume `volume` of a 4-D series), its header kept; nothing is resampled here
denoise    optional (off by default; the paper's images were denoised upstream): ANTs DenoiseImage (Rician), skipped with a message
           when ANTs is not found
mask       the given mask (same grid as the image, values {0, 1}), or the MRI brain-mask network (linc-segment `mri_brainmask`,
           code/blockface/registration/mri_mask.py: $LINC_MODELS/segment/mri_brainmask, run with $LINC_SEGMENT_PYTHON; the network resamples
           to 0.5 mm itself and writes the mask on the image's grid). The mask is required: there is no skipping it.
N4         ANTs N4BiasFieldCorrection with the recipe's parameters (`prep.n4`: fitting grid 0.5 mm, B-spline [60,3], convergence
           [80x60x40,1e-7]); skipped with a message when ANTs is not found (the uncorrected image is normalized instead)
Every step and its tool (path, version, or "not found: skipped") is written to <output>.provenance.json beside the main output.
Then: `linc-register mri --subject <ID> --place-b0 <main output> <mask> --new-brain` (the b0 placed on the template's 0.5 mm frame).
Validated 3 Nov 2026 on MF283 (MF1), ANTs 2.5.4: with the brain masks of record, the b0 path reproduces dwi/b0_brain.nii.gz,
dwi/b0_brain_n4.nii.gz and its bias field, and the T2 path (echo 5 of the MSME, --volume 4) t2/echo.nii.gz, t2/echo_n4.nii.gz, its bias
and t2/echo_n4_brain.nii.gz, voxel for voxel (0 voxels differ, same headers; N4 is deterministic here). Without a mask, the network's
mask has Dice 0.980 with the mask of record (2.9 s on an RTX 6000 Ada), the whole b0 path takes 22 s wall, and `linc-register`'s
placement of its outputs on the template's 0.5 mm frame gives a placed mask with Dice 0.94 against the frame's silhouette.
--denoise adds about 10 CPU-minutes (DenoiseImage, 0.25 mm b0). Without ANTs, both optional steps are skipped with a message.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from . import mri_prep as P
from template.tools import recipe

MODALITY = {                        # per modality: the staged image, the mask, the N4 image (+ bias), the brain image, the main output
    "b0": dict(folder="dwi", staged="b0", mask="dwi_mask", n4="b0_n4_brain", bias="b0_n4_bias", brain="dwi/b0_brain.nii.gz", out="b0_n4_brain"),
    "t2": dict(folder="t2", staged="t2_echo", mask="t2_mask", n4="t2_head", bias=None, brain="t2_brain", out="t2_brain"),
}                                   # values: keys of prep.layout, or (b0_brain, as mri_prep.stage_dwi names it) a path under the MRI folder


def _ants_tool(name):
    """Path and version of an ANTs program, or (None, reason) when it is not there (the step is then skipped)."""
    try:
        from template.tools import ants
        p = ants(name)
    except RuntimeError as e:
        return None, str(e)
    out = subprocess.run([str(Path(p).with_name("antsRegistration")), "--version"], capture_output=True, text=True).stdout
    return p, out.split("Version:")[1].split()[0] if "Version:" in out else "unknown"


def stage(image, dst, volume=0):
    """One 3-D volume of `image` under its own header -> dst (float32 kept as is; integer images kept as is)."""
    im = nib.load(str(image))
    d = np.asanyarray(im.dataobj)
    series = d.ndim == 4
    if series:
        d = d[..., volume]
    elif d.ndim != 3:
        raise ValueError(f"{image}: {d.ndim}-D; give a 3-D image or a 4-D series (--volume picks one volume)")
    hdr = im.header.copy(); hdr.set_data_shape(d.shape)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(d, im.affine, hdr), str(dst))
    return dict(shape=[int(s) for s in d.shape], voxel_mm=[round(float(z), 4) for z in im.header.get_zooms()[:3]], volume=int(volume) if series else None)


def check_mask(mask, image):
    m, i = nib.load(str(mask)), nib.load(str(image))
    if m.shape[:3] != i.shape[:3] or not np.allclose(m.affine, i.affine, atol=1e-4):
        raise ValueError(f"{mask}: not on the image's grid ({m.shape[:3]} vs {i.shape[:3]})")
    v = np.unique(np.asanyarray(m.dataobj))
    if not set(v.tolist()) <= {0, 1}:
        raise ValueError(f"{mask}: values {v[:6]}..., expected 0 and 1 only")


def prep(subject, image, modality="b0", mask=None, volume=0, denoise=False, n4=True, mri_root=None, device="cuda", log=print):
    """Run the steps above; returns the record written to the provenance file (steps, tools, outputs)."""
    from shared import paths, provenance
    from blockface.registration import mri_mask
    L = recipe()["prep"]["layout"]; M = MODALITY[modality]
    root = Path(mri_root) if mri_root else paths.subject_dir(subject) / "mri"
    f = lambda key: root / L.get(key, key)                                             # noqa: E731
    t0 = time.time(); steps = []; inputs = [Path(image)]; outputs = []

    st = stage(image, f(M["staged"]), volume); outputs.append(f(M["staged"]))
    steps.append(dict(step="stage", input=str(Path(image).resolve()), output=str(f(M["staged"])), **st)); log(f"  stage: {f(M['staged'])} {st['shape']} {st['voxel_mm']} mm")
    cur = f(M["staged"])

    if denoise:
        tool, ver = _ants_tool("DenoiseImage")
        if tool:
            dn = root / M["folder"] / f"{Path(L[M['staged']]).name[:-7]}_denoised.nii.gz"
            P.run([tool, "-d", 3, "-v", 0, "-n", "Rician", "-i", str(cur), "-o", str(dn)])
            steps.append(dict(step="denoise", tool=tool, version=ver, noise_model="Rician", output=str(dn))); outputs.append(dn); cur = dn
            log(f"  denoise: DenoiseImage {ver} (Rician) -> {dn}")
        else:
            steps.append(dict(step="denoise", skipped=True, reason=ver)); log(f"  denoise: DenoiseImage (ANTs) not found, skipped ({ver})")
    else:
        steps.append(dict(step="denoise", skipped=True, reason="not requested (--denoise)"))

    if mask:
        check_mask(mask, cur); f(M["mask"]).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(mask, f(M["mask"])); inputs.append(Path(mask))
        steps.append(dict(step="mask", source="given", input=str(Path(mask).resolve()), output=str(f(M["mask"])))); log(f"  mask: given -> {f(M['mask'])}")
    else:
        mri_mask.segment_b0(str(cur), str(f(M["mask"])), device=device)            # the network's mask + its own provenance .json
        steps.append(dict(step="mask", source="linc-segment mri_brainmask", output=str(f(M["mask"])), provenance=str(f(M["mask"]))[:-7] + ".json"))
        log(f"  mask: linc-segment mri_brainmask -> {f(M['mask'])}")
    outputs.append(f(M["mask"]))

    tool, ver = _ants_tool("N4BiasFieldCorrection") if n4 else (None, "not requested (--no-n4)")
    if modality == "b0":
        P.mask_image(cur, f(M["mask"]), f(M["brain"])); outputs.append(f(M["brain"]))
        steps.append(dict(step="mask_image", input=str(cur), output=str(f(M["brain"]))))
        src = f(M["brain"])
    else:
        src = cur
    if tool:
        bias = str(f(M["bias"])) if M["bias"] else None
        P.n4(src, f(M["mask"]), f(M["n4"]), bias=bias)
        p = recipe()["prep"]["n4"]
        steps.append(dict(step="n4", tool=tool, version=ver, parameters=dict(fit_grid_mm=p["fit_grid_mm"], bspline=p["bspline"], convergence=p["convergence"]),
                          input=str(src), output=str(f(M["n4"])), bias=bias or str(f(M["n4"]))[:-7] + "_bias.nii.gz"))
        outputs += [f(M["n4"]), Path(steps[-1]["bias"])]; log(f"  N4: N4BiasFieldCorrection {ver} -> {f(M['n4'])}")
    else:
        shutil.copyfile(src, f(M["n4"])); outputs.append(f(M["n4"]))
        steps.append(dict(step="n4", skipped=True, reason=ver, output=str(f(M["n4"])) + " (a copy of its input)"))
        log(f"  N4: skipped ({ver}); {f(M['n4'])} is the uncorrected image")
    P.mask_image(f(M["n4"]), f(M["mask"]), f(M["out"]), normalize=True)
    if f(M["out"]) not in outputs:
        outputs.append(f(M["out"]))
    c = recipe()["canonical"]
    steps.append(dict(step="mask_normalize", input=str(f(M["n4"])), output=str(f(M["out"])), rule=f"x mask; 1-99 percentile inside the mask -> ({c['norm_eps']}, {c['norm_max']}]"))
    log(f"  masked + normalized: {f(M['out'])}")

    nxt = f"code/blockface/registration/linc-register mri --subject {subject} --place-b0 {f(M['out'])} {f(M['mask'])} --new-brain"
    rec = dict(subject=subject, modality=modality, steps=steps, next=nxt, wall_s=round(time.time() - t0, 1))
    provenance.write_provenance(f(M["out"]), inputs=inputs, outputs=outputs, cmd=f"linc-mri prep --subject {subject} --image {image} --modality {modality}" +
                                (f" --mask {mask}" if mask else "") + (" --denoise" if denoise else "") + ("" if n4 else " --no-n4"), wall_s=rec["wall_s"], extra=rec)
    log(f"  provenance: {provenance.provenance_path(f(M['out']))}\n  next: {nxt}")
    return rec
