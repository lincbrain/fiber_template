"""The MRI chain from the scanner's files to the inputs of the MRI build, run as the paper's data were.

You need
  the brain's Bruker ParaVision session folders: the diffusion scans and the multi-echo spin-echo (MSME) scan of the first
  session, the microstructure diffusion scans (dt2) of the second;
  Julia 1.11.4 (JULIA) with the environment env/julia installed in JULIA_DEPOT_PATH (Fibers.jl reads the Bruker files,
  Microstructure.jl fits SANDI), MRtrix3 3.0.2 (MRTRIX), FSL 6.0.7.4 (FSLDIR), ANTs 2.5.4 (ANTSPATH), and for dt2
  TORTOISE V4.1.0 (TORTOISE; it runs on a GPU).

What it does
  dwi    the diffusion session, step by step:
           convert   reads the scans in the order given and writes one NIfTI series with its b-values and b-vectors
                     (bruker_to_nifti.jl, Fibers.jl);
           reorient  labels the image axes with the brain's own directions, left-anterior-superior (the brain lay on its ventral
                     surface, pointing out of the scanner). Only the header changes: the voxels, their size and the image center
                     stay the same;
           mask      a brain mask for the bias correction: MRtrix dwi2mask on the b0 and the next three volumes, joined with FSL
                     bet2 on the b0 (fractional threshold 0.1, gradient 0.01);
           denoise   removes thermal noise by MP-PCA (MRtrix dwidenoise, a 7 x 7 x 7 voxel window);
           eddy      corrects eddy-current distortion and motion by registering every volume to the first one (FSL
                     eddy_correct). It keeps the per-volume matrices (.ecclog), which the tensor step uses to rotate the
                     b-vectors;
           bias      corrects the intensity bias (MRtrix dwibiascorrect with ANTs N4 on the b0 inside the mask, B-spline
                     distance 100 mm, order 3) and divides every volume by the same field.
  msme   the MSME scan: convert, denoise (a 5 x 5 x 5 voxel window), reorient. Here the denoising comes before the new axis
         labels, as it did for the paper's data: MRtrix reads the axes in label order, and its result depends on that order.
  dt2    the microstructure session: convert; the acquisition parameters (echo and repetition time, phase-encoding axis and
         percent sampling for TORTOISE; each volume's echo time and gradient timing for SANDI); reorient; mask (dwi2mask on the
         whole series, joined with bet2); gradcheck (MRtrix dwigradcheck: the axis order and signs of the b-vectors that fit
         the image best); motion and eddy-current correction with TORTOISE DIFFPREP; mask and bias as above, on TORTOISE's
         output; gradcheck again; SANDI with Microstructure.jl (sandi.jl).

You get, in --out, under the file names of the paper's data:
  dwi    dwi_orig.nii.gz with dwi_orig.bvals and dwi_orig.bvecs, dwi_realori.nii.gz, brain_mask.nii.gz, dwi_denoised.nii.gz,
         dwi_eddycorrect.nii.gz with dwi_eddycorrect.ecclog, dwi_biascorr.nii.gz with bias_est.nii.gz;
  msme   image_orig.nii.gz, msme_denoised.nii.gz, msme_denoised_las.nii.gz (the 32 echoes; the MRI build takes the fifth);
  dt2    dwi_orig, dwi_realori.JSON, dwi.techo / .tdelta / .tsmalldel, dwi_realori (+ the checked .bvals / .bvecs),
         brain_mask, and in tortoise/: dwi_las.nii (TORTOISE's output and working files), las_brain_mask, dwi_biascorr
         (+ the checked .bvals / .bvecs), bias_est, microstructure/SANDIdot/est.fracs.mean.nii.gz and the other SANDI and
         SANDIdot maps (the build reads SANDIdot's fractions and the b0 of dwi_biascorr);
  each with <name>.provenance.json (the command, the program and its version, the inputs). A step whose provenance file exists
  is skipped, so a stopped run resumes where it stopped.

    python -m linc.construction.mri_raw dwi --session <Bruker session folder> --scans 15-36 --out <folder>
    python -m linc.construction.mri_raw msme --session <Bruker session folder> --scan 39 --out <folder>
    python -m linc.construction.mri_raw dt2 --session <second Bruker session folder> --scans 14-21 --out <folder>

Paper: Methods, "MRI preprocessing". The paper's data went through the same steps with the same programs, except FSL 6.0.6
for the eddy-current correction. On MF1 (MF283) the outputs are byte-identical to the paper's files up to the denoised series and
TORTOISE's output, and so is the bias-corrected series made from the paper's eddy-corrected one. TORTOISE's output is also
byte-identical for MF2 and MF3. Three things differ slightly:
  eddy      with FSL 6.0.7.4, 170 of the 171 volumes are byte-identical; one moves by 0.005 mm (median relative difference 2e-4).
  TORTOISE  the paper's run for MF4 shared the volumes between two GPUs; on one GPU, 93 of its 184 volumes register slightly
            differently (median relative difference 2e-3).
  SANDI     in the paper's microstructure processing of all four MRI brains (MF1-MF4), the brain mask of TORTOISE's corrected
            series combined that run's dwi2mask with a bet2 mask left over from an earlier run (bet2 stopped on a stale file).
            This chain makes the mask as intended, so its SANDI fractions agree with the paper's at r = 0.97 to 0.99 for every
            brain and fraction (median difference 0.06 at most); given the paper's own inputs, sandi.jl reproduces them exactly.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from .. import provenance
from .tools import REPO, ants, check_tortoise, fsl, fsl_env, julia, mrtrix, recipe, run, tortoise as tortoise_exe

CONVERT = Path(__file__).with_name("bruker_to_nifti.jl")
SANDI = Path(__file__).with_name("sandi.jl")
JULIA_ENV = REPO / "env" / "julia"
ORIENTATION = "LAS"                       # the brain's directions along the three image axes (all the paper's sessions)
AXES = {"R": (0, 1), "L": (0, -1), "A": (1, 1), "P": (1, -1), "S": (2, 1), "I": (2, -1)}
DWI_WINDOW, MSME_WINDOW = 7, 5            # MP-PCA window (voxels per side)
BET = ["-f", "0.1", "-w", "0.01"]
N4_BSPLINE = "[100,3]"
TORTOISE_OPTIONS = ["--interp_POW", 2, "--gibbs", 0, "--is_human_brain", 0, "--drift", "off", "--epi", "off", "--dti_bval", 4000,
                    "--hardi_bval", 8000, "--b0_id", 1, "--do_QC", 1, "--output_orientation", ORIENTATION]


def _done(out, tag=None):
    """True when `out` exists and its provenance file (or that of `tag`, for steps whose outputs share a stem) does."""
    if provenance.provenance_path(tag or out).exists() and Path(out).exists():
        print(f"  done already: {out}")
        return True
    return False


def _record(out, inputs, cmd, t0, **extra):
    provenance.write_provenance(out, inputs=inputs, cmd=" ".join(map(str, cmd)), wall_s=round(time.time() - t0, 1),
                                extra=extra, md5=False)


def convert(session, scans, out):
    """The scans (folder names inside the session) as one NIfTI series; Fibers writes <out>.bvals/.bvecs for diffusion scans."""
    if _done(out):
        return
    cmd = [julia(), f"--project={JULIA_ENV}", CONVERT, session, out, *scans]
    t0 = time.time(); run(cmd)
    _record(out, [Path(session) / s for s in scans], cmd, t0, step="convert", program=f"Julia {recipe()['tools']['julia']}, env/julia")


def reorient(src, out):
    """New axis labels (ORIENTATION) about the image center (voxel dim/2, FreeSurfer's convention); the voxels are unchanged."""
    if _done(out):
        return
    t0 = time.time()
    im = nib.load(src)
    f32 = np.float32                                  # single precision, as FreeSurfer computes it: the same header bytes
    half = (np.array(im.shape[:3]) / 2).astype(f32)
    M = im.affine.astype(f32)
    center = M[:3, :3] @ half + M[:3, 3]
    A = np.zeros((4, 4), f32); A[3, 3] = 1
    for i, c in enumerate(ORIENTATION):
        axis, sign = AXES[c]
        A[axis, i] = sign * f32(im.header.get_zooms()[i])
    A[:3, 3] = center - A[:3, :3] @ half
    hdr = im.header.copy(); hdr.set_qform(A, 1); hdr.set_sform(A, 1)
    if hdr.get_xyzt_units()[1] == "msec":            # the time step in seconds, as FreeSurfer writes it
        pixdim = hdr["pixdim"].copy(); pixdim[4] /= 1000; hdr["pixdim"] = pixdim; hdr.set_xyzt_units("mm", "sec")
    nib.save(nib.Nifti1Image(np.asanyarray(im.dataobj), A, hdr), str(out))
    _record(out, [src], ["reorient", src, out, ORIENTATION], t0, step="reorient", program=f"nibabel {nib.__version__}",
            orientation=ORIENTATION)


def mask(src, bvals, bvecs, out, volumes=None):
    """MRtrix dwi2mask (on the first `volumes` volumes, else on the whole series) joined with FSL bet2 on volume 0."""
    if _done(out):
        return
    env, t0 = fsl_env(), time.time()
    with tempfile.TemporaryDirectory(dir=Path(out).parent) as tmp:
        t = Path(tmp)
        series = src
        if volumes:
            series, bvals_in, bvecs_in = t / "first.nii.gz", t / "first.bvals", t / "first.bvecs"
            run([fsl("fslroi"), src, series, 0, volumes], env=env)
            np.savetxt(bvals_in, np.loadtxt(bvals).ravel()[:volumes], fmt="%g")
            g = np.loadtxt(bvecs); g = g if g.shape[-1] == 3 else g.T
            np.savetxt(bvecs_in, g[:volumes], fmt="%g")
        else:
            bvals_in, bvecs_in = bvals, bvecs
        cmd = [mrtrix("dwi2mask"), series, "-fslgrad", bvecs_in, bvals_in, t / "dwi2mask.nii.gz", "-quiet", "-force"]
        run(cmd, env=env)
        run([fsl("fslroi"), series, t / "b0.nii.gz", 0, 1], env=env)
        run([fsl("bet2"), t / "b0.nii.gz", t / "bet", "-m", *BET], env=env)
        run([fsl("fslmaths"), t / "dwi2mask.nii.gz", "-add", t / "bet_mask.nii.gz", "-bin", out], env=env)
    _record(out, [src, bvals, bvecs], cmd + ["+ bet2", *BET], t0, step="mask", program=f"MRtrix3 dwi2mask, FSL {recipe()['tools']['fsl']} bet2")


def denoise(src, out, window):
    if _done(out):
        return
    cmd = [mrtrix("dwidenoise"), "-extent", window, src, out, "-force"]
    t0 = time.time(); run(cmd)
    _record(out, [src], cmd, t0, step="denoise", program="MRtrix3 dwidenoise", window=window)


def eddy(src, out_stem):
    """FSL eddy_correct with volume 0 as the reference: <out_stem>.nii.gz and <out_stem>.ecclog."""
    out = f"{out_stem}.nii.gz"
    if _done(out):
        return
    cmd = [fsl("eddy_correct"), src, out_stem, 0]
    t0 = time.time(); run(cmd, env=fsl_env())
    _record(out, [src], cmd, t0, step="eddy", program=f"FSL {recipe()['tools']['fsl']} eddy_correct", reference_volume=0)


def bias(src, bvals, bvecs, brain_mask, out, field):
    if _done(out):
        return
    env = fsl_env()                                   # MRtrix's Python scripts run with FSL's python
    env["PATH"] = str(Path(ants("N4BiasFieldCorrection")).parent) + ":" + env["PATH"]
    cmd = [mrtrix("dwibiascorrect"), "ants", "-bias", field, "-fslgrad", bvecs, bvals, "-ants.b", N4_BSPLINE, "-mask", brain_mask,
           "-nthreads", 4, "-scratch", Path(out).parent, "-force", src, out]
    t0 = time.time(); run(cmd, env=env)
    _record(out, [src, bvals, bvecs, brain_mask], cmd, t0, step="bias", program="MRtrix3 dwibiascorrect, ANTs N4",
            bspline=N4_BSPLINE)


def _write_table(path, m):
    """A gradient table as text, each number in its shortest exact form, one row per line."""
    Path(path).write_text("".join(" ".join(repr(float(x)) for x in r) + "\n" for r in np.atleast_2d(m)))


def _round_bvals(b):
    """b-values rounded to the nearest 50 s/mm^2, values up to 150 set to 0 (as in convert)."""
    b = np.round(np.asarray(b, float) / 50) * 50
    b[b <= 150] = 0
    return b


def gradcheck(src, brain_mask, bvals, bvecs, out_stem):
    """MRtrix dwigradcheck: the axis order and signs of the b-vectors that fit the image best, written as <out_stem>.bvecs (one
    row per volume) and <out_stem>.bvals (rounded as in convert)."""
    out = Path(f"{out_stem}.bvecs")
    if _done(out, tag=f"{out_stem}_gradcheck"):
        return
    env, t0 = fsl_env(), time.time()
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        g = np.loadtxt(bvecs); g = g.T if g.shape[-1] == 3 else g       # dwigradcheck reads the FSL layout: three rows
        _write_table(f"{tmp}/in.bvecs", g); _write_table(f"{tmp}/in.bvals", np.loadtxt(bvals).ravel())
        cmd = [mrtrix("dwigradcheck"), src, "-mask", brain_mask, "-fslgrad", f"{tmp}/in.bvecs", f"{tmp}/in.bvals",
               "-export_grad_fsl", f"{tmp}/g.bvecs", f"{tmp}/g.bvals", "-scratch", tmp, "-force"]
        run(cmd, env=env)
        g, b = np.loadtxt(f"{tmp}/g.bvecs"), np.loadtxt(f"{tmp}/g.bvals").ravel()
    g = g.T if g.shape[0] == 3 and g.shape[1] != 3 else g
    _write_table(out, g); _write_table(f"{out_stem}.bvals", _round_bvals(b)[:, None])
    _record(f"{out_stem}_gradcheck", [src, brain_mask, bvals, bvecs], cmd, t0, step="gradcheck", program="MRtrix3 dwigradcheck",
            outputs=[str(out), f"{out_stem}.bvals"])


def acquisition(session, scans, out_dir):
    """dwi_realori.JSON for TORTOISE (echo and repetition time in s, the phase-encoding axis, the percent sampling: the first
    scan's visu_pars; TORTOISE wants it beside its input, under the same name) and dwi.techo / dwi.tdelta / dwi.tsmalldel for
    SANDI (per volume: the echo time and the diffusion gradients' separation and duration in ms, from each scan's method file)."""
    o = Path(out_dir)
    if _done(o / "dwi_realori.JSON", tag=o / "acquisition"):
        return
    t0 = time.time()

    def values(lines, key):                       # the value of ##$key=..., or the lines after ##$key=( n ) up to the next key
        i = next(k for k, l in enumerate(lines) if l.startswith(f"##${key}="))
        head = lines[i].split("=", 1)[1].strip()
        if not head.startswith("("):
            return head.split()
        words = []
        for l in lines[i + 1:]:
            if l.startswith(("##", "$$")):
                break
            words += l.split()
        return words

    visu = (Path(session) / scans[0] / "pdata" / "1" / "visu_pars").read_text().splitlines()
    enc = values(visu, "VisuAcqGradEncoding")
    meta = dict(EchoTime=float(values(visu, "VisuAcqEchoTime")[0]) / 1000,
                RepetitionTime=float(values(visu, "VisuAcqRepetitionTime")[0]) / 1000,
                PhaseEncodingDirection="ijk"[enc.index("phase_enc")],
                PercentSampling=float(values(visu, "VisuMrPercentSampling")[0]))
    (o / "dwi_realori.JSON").write_text(json.dumps(meta, indent=2) + "\n")
    rows = {"techo": [], "tdelta": [], "tsmalldel": []}
    for scan in scans:
        method = (Path(session) / scan / "method").read_text().splitlines()
        n = len(values(method, "PVM_DwEffBval"))
        for key, name in (("techo", "EchoTime"), ("tdelta", "PVM_DwGradSep"), ("tsmalldel", "PVM_DwGradDur")):
            rows[key] += values(method, name) * n
    for key, words in rows.items():
        (o / f"dwi.{key}").write_text("".join(w + "\n" for w in words))
    _record(o / "acquisition", [Path(session) / s for s in scans], ["acquisition", session, *scans], t0, step="acquisition",
            program="mri_raw (visu_pars, method)", outputs=[str(o / f) for f in ("dwi_realori.JSON", "dwi.techo", "dwi.tdelta", "dwi.tsmalldel")])


def tortoise(src, acq_json, brain_mask, out_dir):
    """TORTOISE DIFFPREP: motion and eddy-current correction of every volume, with the output axes in ORIENTATION;
    <out_dir>/dwi_las.nii with its .bmtxt, .bvals, .bvecs (TORTOISE's working files stay in <out_dir>)."""
    out = Path(out_dir) / "dwi_las.nii"
    if _done(out):
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.parent / "tortoise.log"
    cmd = [tortoise_exe(), "-u", src, "--up_json", acq_json, "--b0_mask_img", brain_mask, "-t", out.parent, "-o", out,
           *TORTOISE_OPTIONS]
    t0 = time.time(); run(cmd, log=log)
    version = check_tortoise(log.read_text())
    _record(out, [src, acq_json, brain_mask], cmd, t0, step="tortoise", program=f"TORTOISE {version}")


def sandi(data, brain_mask, timing_stem, out_dir):
    """sandi.jl: the spherical means beside <data>, then SANDI and SANDIdot under <out_dir>."""
    out = Path(out_dir)
    if _done(out / "SANDIdot" / "est.fracs.mean.nii.gz", tag=out):
        return
    cmd = [julia(), f"--project={JULIA_ENV}", SANDI, data, brain_mask, timing_stem, out]
    t0 = time.time(); run(cmd)
    _record(out, [data, brain_mask, f"{timing_stem}.techo"], cmd, t0, step="sandi",
            program=f"Julia {recipe()['tools']['julia']}, env/julia (Microstructure.jl)")


def dwi(session, scans, out):
    o = Path(out); o.mkdir(parents=True, exist_ok=True)
    bvals, bvecs = o / "dwi_orig.bvals", o / "dwi_orig.bvecs"
    convert(session, scans, o / "dwi_orig.nii.gz")
    reorient(o / "dwi_orig.nii.gz", o / "dwi_realori.nii.gz")
    mask(o / "dwi_realori.nii.gz", bvals, bvecs, o / "brain_mask.nii.gz", volumes=4)
    denoise(o / "dwi_realori.nii.gz", o / "dwi_denoised.nii.gz", DWI_WINDOW)
    eddy(o / "dwi_denoised.nii.gz", o / "dwi_eddycorrect")
    bias(o / "dwi_eddycorrect.nii.gz", bvals, bvecs, o / "brain_mask.nii.gz", o / "dwi_biascorr.nii.gz", o / "bias_est.nii.gz")


def msme(session, scan, out):
    o = Path(out); o.mkdir(parents=True, exist_ok=True)
    convert(session, [scan], o / "image_orig.nii.gz")
    denoise(o / "image_orig.nii.gz", o / "msme_denoised.nii.gz", MSME_WINDOW)
    reorient(o / "msme_denoised.nii.gz", o / "msme_denoised_las.nii.gz")


def dt2(session, scans, out):
    o = Path(out); o.mkdir(parents=True, exist_ok=True); t = o / "tortoise"
    convert(session, scans, o / "dwi_orig.nii.gz")
    acquisition(session, scans, o)
    reorient(o / "dwi_orig.nii.gz", o / "dwi_realori.nii.gz")
    mask(o / "dwi_realori.nii.gz", o / "dwi_orig.bvals", o / "dwi_orig.bvecs", o / "brain_mask.nii.gz")
    gradcheck(o / "dwi_realori.nii.gz", o / "brain_mask.nii.gz", o / "dwi_orig.bvals", o / "dwi_orig.bvecs", o / "dwi_realori")
    tortoise(o / "dwi_realori.nii.gz", o / "dwi_realori.JSON", o / "brain_mask.nii.gz", t)
    mask(t / "dwi_las.nii", t / "dwi_las.bvals", t / "dwi_las.bvecs", t / "las_brain_mask.nii.gz")
    bias(t / "dwi_las.nii", t / "dwi_las.bvals", t / "dwi_las.bvecs", t / "las_brain_mask.nii.gz", t / "dwi_biascorr.nii.gz",
         t / "bias_est.nii.gz")
    gradcheck(t / "dwi_biascorr.nii.gz", t / "las_brain_mask.nii.gz", t / "dwi_las.bvals", t / "dwi_las.bvecs", t / "dwi_biascorr")
    sandi(t / "dwi_biascorr.nii.gz", t / "las_brain_mask.nii.gz", o / "dwi", t / "microstructure")


def _scans(words):
    """'15-36' or '15 16 17' -> ['15', ..., '36']."""
    out = []
    for w in words:
        a, _, b = w.partition("-")
        out += [str(i) for i in range(int(a), int(b or a) + 1)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dwi", help="the diffusion session")
    p.add_argument("--session", required=True); p.add_argument("--scans", nargs="+", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("msme", help="the multi-echo spin-echo scan")
    p.add_argument("--session", required=True); p.add_argument("--scan", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("dt2", help="the microstructure diffusion session")
    p.add_argument("--session", required=True); p.add_argument("--scans", nargs="+", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.umask(0o002)
    if a.cmd == "msme":
        msme(a.session, a.scan, a.out)
    else:
        (dwi if a.cmd == "dwi" else dt2)(a.session, _scans(a.scans), a.out)


if __name__ == "__main__":
    sys.exit(main())
