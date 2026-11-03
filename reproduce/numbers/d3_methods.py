#!/usr/bin/env python3
"""Methods numbers of the completion network and the completed volumes of record (paper: Methods, caudal completion; the paper
reads groups.no_mri.n, models.mri.training.n_brains, finish.clean_face_rule.window_mm, finish.crossfade_mm,
evaluation.brains.n_scored).

Recomputed from the records: the two networks' configurations and training logs (d3_full_A_s0 for the 13 brains with MRI,
d3p_full_A_s0 for the 8 without), the parameter count of the network built from the configuration, the export summaries of the
21 completed volumes, and per brain the predicted and brain volumes counted from <ID>_synth_mask and the completion brain (same
0.25 mm grid), checked against the export summary; the rostral rim (predicted voxels rostral of the cut, i.e. brain voxels outside
the photographed outline) and its distance to photographed tissue; the clean cut face of each brain; the withheld-slab evaluation's
geometry (slab sizes, scored volumes) and the interpolation floor sigma^2 of the photographs.
-> $LINC_NUMBERS/d3_methods.json
    python reproduce/numbers/d3_methods.py
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402
from linc.completion import CFG, deploy, network  # noqa: E402

RUNS = {"mri": "d3_full_A_s0", "no_mri": "d3p_full_A_s0"}
LSO = ("d3_lso_A_s0", "d3p_lso_A_s0")
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()  # noqa: E731
# a released copy of a network (models/completion_checksum_map.tsv: the weights of record, no machine paths) counts as its record
_MAP = Path(__file__).resolve().parents[2] / "models" / "completion_checksum_map.tsv"
OF_RECORD = {r["md5_public"]: r["md5_of_record"] for r in csv.DictReader((x for x in open(_MAP) if not x.startswith("#")), delimiter="\t")}

# the GPU model of the card both released networks were trained on (models/completion.md); the training logs record the machine
# only, so the script checks that every epoch of a run ran on one machine and writes the card's model, never the machine's name
GPU_OF_RECORD = "NVIDIA RTX 6000 Ada Generation"


def train_gpu(log):
    assert len({r.get("host") for r in log}) == 1, "a run trained on more than one machine: give each one's GPU model"   # a scrubbed log has none
    return GPU_OF_RECORD


# the record's own texts, verbatim (they are part of the record's data)
UPSAMPLING = ("x2 transposed conv (kernel 2, stride 2) + two 3x3x3 conv blocks (instance norm, LeakyReLU 0.1) at the output grid, "
              "1x1x1 conv, sigmoid")
INPUT_MRI = "own b0 (0.5 mm), normalized by its own 0.5/99.5 percentiles over the completion brain, masked by it"
INPUT_NO_MRI = ("the S2 pseudo-b0 (the template b0 pulled into the brain), trained on the 13 MRI brains' PROD pseudo-b0s; "
                "deployed on the 8 with lo = 0, x = clip(b0 / hi) x brain (decision of 4 Oct, option 2: the training scaling)")
JOIN = ("a PLANE per brain, the clean cut face z_f (clean_face.py, protocol v2.1 section 2); every voxel caudal of it is "
        "synthesized, including the photographed but incomplete tissue between the most caudal photographed plane and z_f")
WHY_12 = ("the 12 brains of the earlier synthesis study's evaluation; MF275 joined the training set later and is never "
          "held out, so every slab and fold stays paired brain by brain with the existing evaluation")
SLAB_RULE = ("per brain, the d* planes immediately rostral of the clean cut face z_f, where d* is the slab of complete "
             "photographed tissue whose photographed volume equals the brain volume missing caudal of z_f (clean_face.py)")
SLAB_TRAINING = "the leave-slab-out models never see any brain's slab: their training targets are the photographed tissue rostral of z_f + d*"
SIMULATED_CUT = ("z_f + d* (the slab's rostral edge); deployment as in the products: one isotonic map on the first 8 mm "
                 "of photographed tissue rostral of the simulated cut, 2 mm cosine cross-fade")
SCORED = ("the photographed voxels (photo fraction >= 0.999) in the slab's first d* - 1 planes; the plane touching the "
          "retained tissue is not scored (as in the earlier study)")
# the implementation that computed the evaluation of record (eval_v3/d3_lso_A_s0.json, d3_domain_shift.json): a fact about those
# records, so it is not read from the running environment (env/completion.yml pins the same version)
SSIM_IMPLEMENTATION = "scikit-image 0.25.2 structural_similarity (2-D)"
SSIM_SETTINGS = ("library defaults: 7 x 7 uniform window, K1 = 0.01, K2 = 0.03; data range 1 (photographs and predictions on [0, 1] by the "
                 "brain's 1st/99th percentiles of retained photographed tissue)")
SSIM_AGGREGATION = ("per coronal slice over the bounding box of the scored voxels; the local SSIM map is averaged over the scored voxels "
                    "only; the slice values are averaged over slices with >= 50 scored voxels and a box of >= 7 voxels per side")
INTERVAL = ("paired bootstrap over brains: resample the brains with replacement, take the mean paired difference; 95 % = the "
            "2.5th and 97.5th percentiles")
S2_DEFINITION = ("per brain, the mean squared grey difference between each photographed voxel of the first scored plane "
                 "past the simulated cut and its photographed neighbour 0.25 mm away along the anterior-posterior axis; "
                 "averaged over the 12 brains; units = the products' grey")
S2_WHY_SAME = "it compares photographs with photographs, on the same slabs; no model enters it"
CITE = ["d3_s1_comparison", "d3_domain_shift", "d3p_vs_d3_pseudo", "d3p_norm_sensitivity",
        "d42_d3u_vs_d3", "d42_d4v_vs_v21", "d42_d3_vs_d4v", "d42_d3u_vs_d4v", "d42_d3u_vs_v21",
        "d3_A_sigma2_pred_flat", "d3p_A_sigma2_pred_flat"]
NOTE_RIM = ("per_brain.<S>.rim: synthetic voxels rostral of the join plane (seam_depth >= 0), i.e. brain voxels outside "
            "the photographed outline (deploy real_mask): volume, share of the brain's synthetic tissue, fraction within "
            "1 mm (3-D) of photographed tissue, distance percentiles")
NOTE_SYNTH_FRAC = ("synthesized share of the completion brain = synth_mask / brain on the 0.25 mm product grid "
                   "(synth_mask = brain minus photographed tissue rostral of the clean cut face)")


def model_record(rd, inputs):
    cfg = json.load(open(rd / "config.json")); inputs.append(rd / "config.json")
    log = [json.loads(line) for line in open(rd / "train_log.jsonl")]; inputs.append(rd / "train_log.jsonl")
    n_par = int(sum(p.numel() for p in network.build(cfg).parameters()))
    ck = md5(rd / "last.pt")
    return dict(run=rd.name, checkpoint_md5=OF_RECORD.get(ck, ck), parameters=n_par,
                architecture=dict(backbone="MONAI BasicUNet at the input grid", features=cfg["features"], head_channels=cfg["head"],
                                  upsampling=UPSAMPLING, activation="LeakyReLU 0.1", norm="instance"),
                grids=dict(frame=cfg["frame"], input_mm=cfg["in_res_mm"], output_mm=cfg["out_res_mm"],
                           output_grid="the input box's centre-aligned 2x refinement",
                           patch_input_vox=cfg["patch_coarse"], patch_output_vox=2 * cfg["patch_coarse"],
                           patch_mm=cfg["patch_coarse"] * cfg["in_res_mm"]),
                training=dict(n_brains=len(cfg["train_subs"]), brains=cfg["train_subs"], extra_train=cfg["extra_train"],
                              epochs=cfg["epochs"], epochs_logged=len(log), patches_per_epoch=cfg["length"], batch=cfg["batch"],
                              optimizer="Adam", lr=cfg["lr"], warmup_epochs=cfg["warmup"], schedule="cosine decay after warm-up",
                              ema_decay=cfg["ema_decay"], grad_clip=cfg["clip"],
                              loss=dict(l1=cfg["w_l1"], gradient_l1=cfg["w_grad"], one_minus_ms_ssim=cfg["w_ssim"]),
                              augmentation=cfg["aug_params"], photographed_only=cfg["photographed_only"],
                              wall_min=round(sum(r["s"] for r in log) / 60.0, 1), gpu=train_gpu(log),
                              root=Path(cfg["root"]).name))


def main():
    P = paths.products("products_completion")
    OUT, CACHE, IN = P["completed"], P["completed_masks"], P["completion_inputs"]
    inputs = []
    models = {k: model_record(P["completion_runs"] / r, inputs) for k, r in RUNS.items()}
    models["mri"]["input"] = INPUT_MRI
    models["no_mri"]["input"] = INPUT_NO_MRI
    rows = {}
    for tag in ("mri", "no_mri"):
        sp = OUT / f"summary_{RUNS[tag]}.json"
        summ = json.load(open(sp)); inputs.append(sp)
        assert summ["checkpoint_md5"] == models[tag]["checkpoint_md5"], f"{sp}: checkpoint differs from {models[tag]['run']}"
        for S, m in sorted(summ["members"].items()):
            sm = nib.load(OUT / S / f"{S}_synth_mask.nii.gz"); bm = nib.load(CACHE / S / "deploy/brain.nii.gz")
            assert np.allclose(sm.affine, bm.affine, atol=1e-5) and sm.shape == bm.shape, f"{S}: synth mask and brain on different grids"
            vml = abs(np.linalg.det(sm.affine[:3, :3])) / 1000.0
            syn = int((np.asanyarray(sm.dataobj) > 0).sum()); brain = int((np.asanyarray(bm.dataobj) > 0).sum())
            frac = syn / brain
            assert abs(frac - m["synth_frac"]) < 1e-4, f"{S}: counted synthetic share {frac} vs summary {m['synth_frac']}"
            # the rostral rim: predicted voxels ROSTRAL of the join plane (seam_depth >= 0) = brain voxels outside the photographed
            # outline (real_mask); their distance to the nearest photographed voxel (3-D, mm)
            smk = np.asanyarray(sm.dataobj) > 0
            sdp = np.asanyarray(nib.load(OUT / S / f"{S}_seam_depth.nii.gz").dataobj)
            rmk = np.asanyarray(nib.load(CACHE / S / "deploy/real_mask.nii.gz").dataobj) > 0
            rim = smk & (sdp >= 0)
            vox_mm = float(abs(np.linalg.det(sm.affine[:3, :3])) ** (1 / 3))
            dist = ndi.distance_transform_edt(~rmk) * vox_mm if rim.any() else None
            dr = dist[rim] if rim.any() else np.zeros(0)
            rimrec = dict(rostral_synth_ml=round(int(rim.sum()) * vml, 3), rostral_share_of_synth=round(float(rim.sum() / max(syn, 1)), 4),
                          rostral_within_1mm_of_photographed=round(float((dr <= 1.0).mean()) if dr.size else 0.0, 4),
                          rostral_dist_mm_p50=round(float(np.median(dr)), 3) if dr.size else None,
                          rostral_dist_mm_p95=round(float(np.percentile(dr, 95)), 3) if dr.size else None)
            inputs += [OUT / S / f"{S}_seam_depth.nii.gz", CACHE / S / "deploy/real_mask.nii.gz"]
            rows[S] = dict(group=tag, species="fascicularis" if S.startswith("MF") else "rhesus", model=models[tag]["run"],
                           synth_frac=round(float(m["synth_frac"]), 5), synth_ml=round(syn * vml, 3), brain_ml=round(brain * vml, 3),
                           rim=rimrec,
                           checks=dict(provenance_le_photo_fraction=m["provenance_le_photo_fraction"],
                                       seam_depth_nonneg_exactly_rostral=m["seam_depth_nonneg_exactly_rostral"]))
            inputs += [OUT / S / f"{S}_synth_mask.nii.gz", CACHE / S / "deploy/brain.nii.gz"]
    # the join plane z_f per brain, from the clean-face record of the batch that prepared the brain (the same file its seam.json used)
    canon = {**json.load(open(IN / "canon_d3/canon_root.json"))["brains"], **json.load(open(IN / "canon_d3n_prod/canon_root.json"))["brains"]}
    rule = None
    for S in rows:
        b = canon[S]["batch"]; cfp = IN / f"cache_{b}/clean_face.json"; cf = json.load(open(cfp)); inputs.append(cfp)
        e = cf["brains"][S]
        croot = "canon_d3n_prod" if b == "d3n_prod" else "canon_d3"
        assert json.load(open(IN / croot / S / "seam.json"))["seam"]["z_f"] == e["z_f"], f"{S}: seam z_f differs from {cfp}"
        rule = rule or cf["rule"]; assert cf["rule"] == rule, f"{cfp}: a different clean-face rule"
        rows[S].update(z_f_mm_from_photographed_end=e["z_f_mm_from_photographed_end"], photographed_discarded_ml=e["photographed_discarded_ml"],
                       true_gap_mm=e["true_gap_mm"], missing_ml=e["missing_ml"])
    finish = dict(join=JOIN, clean_face_rule=rule,
                  intensity="one monotone (isotonic, PAVA) map per brain from the network output to the photographs' grey, fitted on "
                            f"the first {deploy.RECAL_MM:g} mm of fully photographed tissue (photo fraction >= 0.999) rostral of z_f; no "
                            "per-column scaling, no clipped factor",
                  recal_band_mm=deploy.RECAL_MM, recal_photo_fraction=CFG["fully_photographed"],
                  crossfade=f"cosine ramp over {deploy.FADE_MM:g} mm rostral of z_f on retained photographed tissue: w = 0.5 - 0.5 cos(pi * "
                            f"clip(d / {deploy.FADE_MM:g} mm, 0, 1)); completed = w * photograph + (1 - w) * synthesis",
                  crossfade_mm=deploy.FADE_MM, colour="constant chroma per brain (deploy.py's rule)",
                  per_brain_keys=["z_f_mm_from_photographed_end", "photographed_discarded_ml", "true_gap_mm", "missing_ml"])
    # the withheld-slab evaluation (leave-slab-out networks; deploy.regions protocol "lso", metrics.region_metrics)
    evp = P["completion_eval"] / "d3_lso_A_s0.json"; ev_own = json.load(open(evp)); inputs.append(evp)
    lso_cfg = {r: json.load(open(P["completion_runs"] / r / "config.json")) for r in LSO}
    inputs += [P["completion_runs"] / r / "config.json" for r in lso_cfg]
    members = list(CFG["training"]["members"])
    assert sorted(ev_own["subjects"]) == sorted(members) and len(members) == 12
    slabs = {}
    for S in members:
        e = ev_own["subjects"][S]
        sj = json.load(open(IN / "canon_d3" / S / "seam.json"))["seam"]
        assert sj["z_f"] == e["z_f"] and sj["dstar_planes"] == e["dstar_planes"], f"{S}: slab geometry differs between seam and evaluation"
        slabs[S] = dict(z_f=e["z_f"], withheld_planes=e["dstar_planes"], withheld_mm=e["dstar_planes"] * deploy.ISO,
                        scored_planes=e["dstar_planes"] - 1, scored_vox=e["score_vox"], scored_ml=round(e["score_vox"] * deploy.ISO ** 3 / 1000.0, 3))
    wm = np.array([slabs[S]["withheld_mm"] for S in members]); sml = np.array([slabs[S]["scored_ml"] for S in members])
    s2p = P["numbers_of_record"] / "d3_A_sigma2_pred.json"; s2 = json.load(open(s2p))["data"]; inputs.append(s2p)
    s2pp = P["numbers_of_record"] / "d3p_A_sigma2_pred.json"; s2p_ = json.load(open(s2pp))["data"]; inputs.append(s2pp)
    evaluation = dict(
        brains=dict(n_scored=len(members), scored=members, never_held_out=lso_cfg["d3_lso_A_s0"]["extra_train"], why_12=WHY_12),
        slab=dict(rule=SLAB_RULE, training=SLAB_TRAINING, simulated_cut=SIMULATED_CUT, scored=SCORED,
                  per_brain=slabs, withheld_mm_median=float(np.median(wm)), withheld_mm_range=[float(wm.min()), float(wm.max())],
                  scored_ml_median=float(np.median(sml)), scored_ml_range=[float(sml.min()), float(sml.max())]),
        models=dict(own_b0="d3_lso_A_s0", pseudo_b0="d3p_lso_A_s0",
                    protocol={r: c["protocol"] for r, c in lso_cfg.items()}, n_training_brains={r: len(c["train_subs"]) for r, c in lso_cfg.items()}),
        ssim=dict(citation="wangImageQualityAssessment2004", implementation=SSIM_IMPLEMENTATION,
                  settings=SSIM_SETTINGS, aggregation=SSIM_AGGREGATION),
        interval=dict(method=INTERVAL, n_boot=CFG["evaluation"]["n_boot"], seed="fixed per script (see each numbers file's script)",
                      unit="brain (12 per comparison)"),
        sigma2_interp=dict(value=s2["s2_interp"], value_pseudo_file=s2p_["s2_interp"], n_brains=s2["s2_interp_detail"]["n_members"],
                           per_brain=s2["s2_interp_detail"]["per_member"], definition=S2_DEFINITION, why_same_in_both_files=S2_WHY_SAME))
    assert evaluation["sigma2_interp"]["value"] == evaluation["sigma2_interp"]["value_pseudo_file"]
    groups = {}
    for g in ("mri", "no_mri", "all"):
        ss = [S for S in rows if g == "all" or rows[S]["group"] == g]
        f = np.array([rows[S]["synth_frac"] for S in ss]); v = np.array([rows[S]["synth_ml"] for S in ss])
        zf = np.array([rows[S]["z_f_mm_from_photographed_end"] for S in ss]); dml = np.array([rows[S]["photographed_discarded_ml"] for S in ss])
        rr = lambda k: [rows[S]["rim"][k] for S in ss]  # noqa: E731
        groups[g] = dict(n=len(ss), synth_frac_median=float(np.median(f)), synth_frac_range=[float(f.min()), float(f.max())],
                         synth_ml_median=float(np.median(v)), synth_ml_range=[float(v.min()), float(v.max())],
                         z_f_mm_from_photographed_end_median=float(np.median(zf)), z_f_mm_from_photographed_end_range=[float(zf.min()), float(zf.max())],
                         photographed_discarded_ml_median=float(np.median(dml)), photographed_discarded_ml_range=[float(dml.min()), float(dml.max())],
                         rostral_synth_ml_median=float(np.median(rr("rostral_synth_ml"))),
                         rostral_synth_ml_range=[float(min(rr("rostral_synth_ml"))), float(max(rr("rostral_synth_ml")))],
                         rostral_share_of_synth_median=float(np.median(rr("rostral_share_of_synth"))),
                         rostral_share_of_synth_range=[float(min(rr("rostral_share_of_synth"))), float(max(rr("rostral_share_of_synth")))],
                         rostral_within_1mm_median=float(np.median(rr("rostral_within_1mm_of_photographed"))))
    assert groups["mri"]["n"] == 13 and groups["no_mri"]["n"] == 8
    out = dict(what="d3 synthesis network and the completions of record (Methods numbers)", models=models, finish=finish, evaluation=evaluation,
               per_brain=rows, groups=groups, cite_for_performance=CITE, notes=dict(rim=NOTE_RIM, synth_frac=NOTE_SYNTH_FRAC))
    provenance.write_numbers("d3_methods", out, inputs=inputs, md5=False, cmd="python reproduce/numbers/d3_methods.py")
    for k in ("mri", "no_mri"):
        m = models[k]; t = m["training"]
        print(f"{k}: {m['run']} params {m['parameters']} | {t['n_brains']} brains, {t['epochs']} epochs | in {m['grids']['input_mm']} mm "
              f"-> out {m['grids']['output_mm']} mm")
    for g, v in groups.items():
        print(f"{g}: n {v['n']} predicted share median {v['synth_frac_median']:.3f} [{v['synth_frac_range'][0]:.3f}, {v['synth_frac_range'][1]:.3f}]")


if __name__ == "__main__":
    main()
