"""Feature-based deformable registration (moving -> fixed), the method of every pairwise deformable registration in the paper
(Methods, "Image registration and segmentation"): features from a pretrained network (anatomix) and self-similarity context descriptors
(MIND-SSC, which encode how each voxel's neighborhood resembles its surroundings), registered with FireANTs.

You need
  the fixed and moving images and their brain masks (every input brain-masked: inputs.py checks it and stops otherwise); the
  anatomix environment ($ANATOMIX_PYTHON, settings/env/anatomix.yml: neel-dey/anatomix@4c86aee with neel-dey/FireANTs@21ae072, run through
  anatomix_runner.py) and a GPU.
What it does
  register(fixed, moving, fmask, mmask, outdir, **kw):
  * puts the fixed image (and its mask) on an isotropic `resolution` grid when it is not on one; the moving image is passed as it
    is (the program resamples it once, by the final transform); both are clipped at 0 before normalization;
  * registers all 32 anatomix channels and 12 MIND-SSC channels, the loss evaluated `channel_chunk` channels at a time and
    including the background by default (`loss="cc"`, local cross-correlation); two deformable stages (the second computes the
    moving image's features again);
  * `gate_features_by_mask=False` leaves the feature channels unmultiplied by the masks (the masks still score the overlap);
  * fused CUDA kernels off and a fixed seed, for reproducibility.
You get
  in `outdir`: `mri_in_psoct_workres.nii.gz` (the moving image on the fixed working grid), `warp_mri_to_psoct.nii.gz` (the ITK/ANTs
  LPS displacement field on that grid, fixed -> moving; the file names are historical and the code downstream reads them),
  `metrics.json`, and the program's own files (`reg-*`).

Ported from the analysis project's code/featurereg.py (its `upstream` engine, unchanged).
"""
import csv
import json
import os
import subprocess
import time
from pathlib import Path

from blockface.registration.inputs import check_masked

UPSTREAM_VERSION = "neel-dey/anatomix@4c86aee + neel-dey/FireANTs@21ae072"
RUNNER = Path(__file__).resolve().parent / "anatomix_runner.py"


def _python():
    p = os.environ.get("ANATOMIX_PYTHON")
    if not p:
        raise RuntimeError("set ANATOMIX_PYTHON to the python of the anatomix environment (" + UPSTREAM_VERSION + ")")
    return p


def _child_cuda_visible(dev):
    """CUDA_VISIBLE_DEVICES for the CLI child that should use this process's device index `dev`: mapped through this process's own
    CUDA_VISIBLE_DEVICES (the index is relative to the devices this process sees)."""
    vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    i = int(dev)
    return vis[i] if i < len(vis) else dev


# GPU memory of the deformable stage, measured: peak GiB = 0.10 + 1.286 x million voxels of the fixed working grid (33 runs of the
# registrations to each brain's own b0 and of the cross-modal alignments, 2 Oct 2026; the 3 Nov dry run: MR252 18.0 Mvox, 23.2 GiB;
# MF191 12.4 Mvox, 16.1 GiB, both as predicted).
GPU_GIB_BASE, GPU_GIB_PER_MVOX = 0.10, 1.286


def gpu_check(n_vox, env, base=GPU_GIB_BASE, per_mvox=GPU_GIB_PER_MVOX, label="the deformable stage"):
    """Stop at once, with one line, when the GPU the child process will see (env's CUDA_VISIBLE_DEVICES, its device 0) has less free
    memory than the stage needs (base + per_mvox x million grid voxels, GiB). Read with the engine's own torch."""
    need = base + per_mvox * n_vox / 1e6
    code = "import torch; f, t = torch.cuda.mem_get_info(0); print(f / 2**30, t / 2**30)"
    out = subprocess.run([_python(), "-c", code], env=env, capture_output=True, text=True)
    if out.returncode != 0:
        return                                                          # no CUDA here: the engine itself will say so
    free, total = (float(v) for v in out.stdout.split())
    card = f"GPU {env.get('CUDA_VISIBLE_DEVICES', '0')}"
    print(f"  [featurereg] {label}: needs about {need:.1f} GiB ({n_vox / 1e6:.1f} million voxels); {card} has {free:.1f} of "
          f"{total:.1f} GiB free", flush=True)
    if free < need:
        raise SystemExit(f"{label} needs about {need:.1f} GiB free on the GPU; {card} has {free:.1f} GiB free. Run on a freer card "
                         "(e.g. --device cuda:1, or CUDA_VISIBLE_DEVICES=1) or wait until this one frees up; nothing was registered.")


def _x(s):
    """'8,4,2,1' -> '8x4x2x1' (the CLI's per-level separator)."""
    return "x".join(t.strip() for t in str(s).split(",") if t.strip())


def _grid_equal(a, b):
    import numpy as np
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-4)


def register(fixed, moving, fmask, mmask, outdir, *, affine=None, resolution=0.15,
             backbone="anatomix-dev-vit", scales="8,4,2,1", iterations="200,200,200,200", cc="17,9,5,3",
             two_stage=True, stage2_scales="2,1", stage2_cc="5,3", stage2_iterations="200,200",
             smooth_grad_sigma=1.5, smooth_warp_sigma=0.5, fixed_clip_percentile=99.0,
             moving_clip_percentile=None, mindssc=True, device="cuda:0", lr=1.0, loss="cc",
             channel_chunk=8, fused_ops="off", fill_fixed_mask_holes=True, gate_features_by_mask=True,
             seed=12345, verbose=True, allow_unmasked_reason=None, gpu_per_mvox=GPU_GIB_PER_MVOX):
    import numpy as np
    import nibabel as nib
    import nibabel.processing as nip
    from scipy import ndimage as ndi

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    work = outdir / "work"; work.mkdir(exist_ok=True)
    t0 = time.time()
    # --- fixed image + mask on the `resolution` iso working grid
    fi = nib.load(str(fixed))
    zooms = np.abs(np.asarray(fi.header.get_zooms()[:3], float))
    if resolution and not np.allclose(zooms, resolution, atol=1e-4):
        fi = nip.resample_to_output(fi, [resolution] * 3, order=1)
    F = np.asanyarray(fi.dataobj).astype(np.float32)
    fm_img = nib.load(str(fmask))
    FM = (np.asanyarray(fm_img.dataobj) > 0) if _grid_equal(fm_img, fi) else \
        (np.asanyarray(nip.resample_from_to(fm_img, (fi.shape[:3], fi.affine), order=0).dataobj) > 0)
    if fill_fixed_mask_holes:
        FM = ndi.binary_fill_holes(FM)
    F = np.maximum(F, 0.0)                                              # clipped at 0 before min-max
    fp, fmp = str(work / "fixed.nii.gz"), str(work / "fixed_mask.nii.gz")
    nib.save(nib.Nifti1Image(F, fi.affine), fp)
    nib.save(nib.Nifti1Image(FM.astype(np.uint8), fi.affine), fmp)
    # --- moving: the ORIGINAL image (a clipped copy if it has negative values); its mask on its own grid
    mi = nib.load(str(moving))
    if float(np.asanyarray(mi.dataobj).min()) < 0:
        _m = np.maximum(np.asanyarray(mi.dataobj).astype(np.float32), 0.0)
        moving = work / ("clip0_" + Path(str(moving)).name)
        nib.save(nib.Nifti1Image(_m, mi.affine, mi.header), str(moving)); mi = nib.load(str(moving))
    mm_img = nib.load(str(mmask))
    MM = (np.asanyarray(mm_img.dataobj) > 0) if _grid_equal(mm_img, mi) else \
        (np.asanyarray(nip.resample_from_to(mm_img, (mi.shape[:3], mi.affine), order=0).dataobj) > 0)
    mstem = Path(str(moving)).name.replace(".nii.gz", "").replace(".nii", "")
    mmp = str(work / f"{mstem}_mask.nii.gz")
    nib.save(nib.Nifti1Image(MM.astype(np.uint8), mi.affine), mmp)
    check_masked(fp, fmp, label=f"featurereg fixed {Path(str(fixed)).name}", allow_reason=allow_unmasked_reason)
    check_masked(str(moving), mmp, label=f"featurereg moving {Path(str(moving)).name}", allow_reason=allow_unmasked_reason)
    # --- clipping: the fixed image at its in-mask percentile; the moving image unclipped unless asked
    fclip = float(np.percentile(F[FM], fixed_clip_percentile)) if FM.any() and fixed_clip_percentile else None
    mclip = None
    if moving_clip_percentile:
        M = np.asanyarray(mi.dataobj).astype(np.float32)
        mclip = float(np.percentile(M[MM], moving_clip_percentile)) if MM.any() else None
    # --- stages
    kinds = ["deformable"] + (["deformable"] if two_stage else [])
    st = lambda v_def1, v_def2: ",".join([str(v_def1)] + ([str(v_def2)] if two_stage else []))  # noqa: E731
    argv = ["--fixed", fp, "--moving", str(moving), "--fixed-mask", fmp, "--moving-mask", mmp,
            "--fixed-seg", fmp, "--moving-seg", mmp,                     # -> mask Dice after, and the moved mask
            "--transform", ",".join(kinds),
            "--loss", ",".join([loss] * len(kinds)),
            "--step-size", st(lr, lr),
            "--shrink-factors", st(_x(scales), _x(stage2_scales)),
            "--iterations", st(_x(iterations), _x(stage2_iterations)),
            "--cc-kernel-widths", st(_x(cc) if "cc" in loss else "na", _x(stage2_cc) if "cc" in loss else "na"),
            "--smooth-grad-sigma", st(smooth_grad_sigma, smooth_grad_sigma),
            "--smooth-warp-sigma", st(smooth_warp_sigma, smooth_warp_sigma),
            "--features", "anatomix+mindssc" if mindssc else "anatomix", "--backbone", backbone,
            "--loss-channel-chunk", "none" if channel_chunk is None else str(channel_chunk),
            "--fused-ops", fused_ops, "--seed", str(seed),
            "--output-dir", str(outdir), "--exp-name", "reg", "--collapse-output-transforms", "1",
            "--device", "cuda:0" if str(device).startswith("cuda") else str(device)]
    if fclip is not None:
        argv += ["--fixed-maxclip", repr(fclip)]
    if mclip is not None:
        argv += ["--moving-maxclip", repr(mclip)]
    if affine is not None:
        argv += ["--initial-transform", str(affine)]
    if not verbose:
        argv += ["--no-verbose"]
    env = dict(os.environ)
    env["FIREANTS_LOCK"] = env.get("FIREANTS_LOCK", f"/tmp/fireants_image_lock_{os.getuid()}")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    dev = str(device).split(":")[-1] if str(device).startswith("cuda") else ""
    if dev.isdigit():
        env["CUDA_VISIBLE_DEVICES"] = _child_cuda_visible(dev)          # the CLI sees a single visible GPU
    env.pop("LINC_UNGATED_FEATURES", None)
    if not gate_features_by_mask:
        env["LINC_UNGATED_FEATURES"] = "1"
    if str(device).startswith("cuda"):
        gpu_check(int(np.prod(F.shape[:3])), env, per_mvox=gpu_per_mvox)   # gpu_per_mvox: a caller's own measured rate
    cmd = [_python(), str(RUNNER)] + argv
    print("  [featurereg] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    wall = time.time() - t0
    # --- outputs under the historical names too (callers read them), plus metrics.json
    moved = outdir / f"reg-moved-{mstem}.nii.gz"
    warp = outdir / f"reg-warp-{mstem}.nii.gz"
    mseg = outdir / f"reg-moved-seg-{mstem}.nii.gz"
    for src, dst in ((moved, outdir / "mri_in_psoct_workres.nii.gz"), (warp, outdir / "warp_mri_to_psoct.nii.gz")):
        if not src.exists():
            raise FileNotFoundError(f"engine output missing: {src}")
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.link(src, dst)
    row = {}
    mc = outdir / "reg-metrics.csv"
    if mc.exists():
        rows = list(csv.DictReader(open(mc)))
        row = rows[-1] if rows else {}
    num = lambda k: (float(row[k]) if row.get(k) not in (None, "", "nan", "None") else None)  # noqa: E731
    mb = None
    if affine is None:                                                   # overlap before, at the physical identity
        r = nip.resample_from_to(nib.Nifti1Image(MM.astype(np.uint8), mi.affine), (fi.shape[:3], fi.affine), order=0)
        R = np.asanyarray(r.dataobj) > 0
        mb = float(2 * (R & FM).sum() / max(R.sum() + FM.sum(), 1))
    ma = num("dice")
    if ma is None and mseg.exists():
        S = np.asanyarray(nib.load(str(mseg)).dataobj) > 0
        ma = float(2 * (S & FM).sum() / max(S.sum() + FM.sum(), 1))
    nf = num("num_folds")
    metrics = dict(engine="upstream", engine_version=UPSTREAM_VERSION, mask_dice_before=mb, mask_dice_after=ma,
                   num_folds=nf, folds_percent=(100.0 * nf / F.size if nf is not None else None),
                   runtime_seconds=round(wall, 1), register_wall_seconds=round(wall, 1), psoct_clip_max=fclip,
                   stages=kinds, loss=loss, lr=lr, channels="anatomix(32)+mindssc(12)" if mindssc else "anatomix(32)",
                   gate_features_by_mask=gate_features_by_mask, argv=argv,
                   grid=dict(shape=list(F.shape), voxel_mm=float(abs(fi.affine[0, 0]))))
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    metrics["moved"] = str(outdir / "mri_in_psoct_workres.nii.gz")
    metrics["warp"] = str(outdir / "warp_mri_to_psoct.nii.gz")
    return metrics
