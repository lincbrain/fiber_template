"""Task specs, subject-grouped folds, a per-case cache and the patch sampler.

A task spec (JSON) lists the cases and the folds:
  {"task": "...", "dim": 2|3, "classes": {"0": "background", "1": "brain"}, "channels": 1,
   "cases": [{"id": "...", "subject": "...", "image": "<path>" | ["<path ch0>", ...], "label": "<path>"}, ...],
   "folds": {"0": ["<subject>", ...], ...},          # the subjects HELD OUT by each fold
   "patch": [...], "spacing": [..] or "spacing_by_subject": {subject: [..]} (2D images without a header: mm metrics), "target_spacing": [..] (resample to it for training and
   prediction; the mask goes back to the input grid), "notes": "..."}
"folds" is written once, before any training, by `python -m linc.segment.folds --task <spec>` (make_folds: seed 2026, grouped by subject).
or, instead of "cases", "nnunet_raw": "<Dataset folder>" with "subject_regex" (its first group = the subject; default: the case id).
A path may start with a root placeholder ($LINC_PRODUCTS/..., $LINC_DATA/..., ...; linc.paths), as the task files of record do.
"train" (optional) holds the task's own training arguments where they differ from the trainer's defaults (batch, workers, ...).
Training on fold k uses every case whose subject is not in folds[k]; fold "all" uses every case.

The cache (<cache>/<case>.npz-like pair of .npy files) holds each image z-scored per channel as float16 (C, *S), its label as uint8,
and up to 20,000 foreground voxel indices for oversampling; the arrays are opened memory-mapped, so a large set stays on disk.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from . import io


def remap(p, pairs):
    """a case path on this host: each (src, dst) prefix pair applied (a task's images copied to the host's local disk)"""
    for s, d in pairs or ():
        if str(p).startswith(s):
            return d + str(p)[len(s):]
    return p


def _root(p):
    """a task path: a root placeholder ($LINC_PRODUCTS/...) expanded on this machine (linc.paths); any other path as written"""
    if not str(p).startswith("$"):
        return p
    from ..paths import rerooted
    return rerooted(p)


def load_task(path, remap_pairs=None):
    spec = json.load(open(path))
    for c in spec.get("cases", []):
        c["image"] = [_root(q) for q in c["image"]] if isinstance(c["image"], list) else _root(c["image"])
        for k in ("label", "mask"):
            if k in c:
                c[k] = _root(c[k])
    if "cases" not in spec:
        root = Path(_root(spec["nnunet_raw"])); ds = json.load(open(root / "dataset.json"))
        ext = ds["file_ending"]; nch = len(ds["channel_names"])
        rx = re.compile(spec.get("subject_regex", r"(.*)"))
        cases = []
        for lab in sorted((root / "labelsTr").glob(f"*{ext}")):
            cid = lab.name[: -len(ext)]
            m = rx.search(cid)
            assert m, f"{cid}: subject_regex {rx.pattern} does not match"
            files = [root / "imagesTr" / f"{cid}_{c:04d}{ext}" for c in range(nch)]
            if ext in (".png", ".tif", ".tiff", ".jpg") and not files[-1].exists():
                files = files[:1]                                       # nnU-Net's natural images: one RGB file holds the 3 channels
            cases.append(dict(id=cid, subject=m.group(1), label=str(lab), image=[str(f) for f in files]))
        spec["cases"] = cases
        spec.setdefault("channels", nch)
        spec.setdefault("classes", {str(v): k for k, v in ds["labels"].items()})
    spec.setdefault("channels", 1)
    spec.setdefault("normalization", "zscore")
    sbs = spec.get("spacing_by_subject") or {}                         # 2D images without a header: the pixel size per brain (mm)
    for c in spec["cases"]:
        if "spacing" not in c and (c["subject"] in sbs or spec.get("spacing")):
            c["spacing"] = sbs.get(c["subject"], spec.get("spacing"))
    assert spec["normalization"] in io.NORMALIZATIONS, spec["normalization"]
    if remap_pairs:
        for c in spec["cases"]:
            c["image"] = [remap(p, remap_pairs) for p in c["image"]] if isinstance(c["image"], list) else remap(c["image"], remap_pairs)
            c["label"] = remap(c["label"], remap_pairs)
    return spec


FOLD_SEED = 2026


def make_folds(spec, seed=FOLD_SEED):
    """subject-grouped folds: K = 5 when the task has >= 10 subjects, else one fold per subject (leave one subject out); subjects in a
    seeded random order, dealt round-robin; returns ({fold: [held-out subjects]}, the rule as text)"""
    subs = sorted({c["subject"] for c in spec["cases"]})
    K = 5 if len(subs) >= 10 else len(subs)
    perm = [subs[i] for i in np.random.default_rng(seed).permutation(len(subs))]
    folds = {str(k): sorted(perm[k::K]) for k in range(K)}
    rule = (f"{len(subs)} subjects, {len(spec['cases'])} cases; " + ("5 folds" if K == 5 else f"leave one subject out ({K} folds)") +
            f": subjects in numpy default_rng({seed}).permutation order (from the sorted list), dealt round-robin; a subject's cases "
            f"are all in one fold")
    return folds, rule


def median_spacing(spec):
    """nnU-Net's rule: the per-axis median voxel size over the task's cases (NIfTI headers), the grid every fold trains on"""
    import nibabel as nib
    sp = []
    for c in spec["cases"]:
        f = c["image"][0] if isinstance(c["image"], list) else c["image"]
        sp.append([float(z) for z in nib.load(f).header.get_zooms()[:int(spec["dim"])]])
    return [float(v) for v in np.median(np.array(sp), 0)]


def split(spec, fold):
    if str(fold) == "all":
        return list(spec["cases"]), []
    held = set(spec["folds"][str(fold)])
    tr = [c for c in spec["cases"] if c["subject"] not in held]
    va = [c for c in spec["cases"] if c["subject"] in held]
    assert tr and va, f"fold {fold}: {len(tr)} training and {len(va)} validation cases"
    return tr, va


def cache_case(case, dim, cache, spacing=None, target_spacing=None, normalization="zscore"):
    """z-scored float16 image, uint8 label and foreground indices in <cache>/<id>.{img,lab,fg}.npy; returns the case's shape and spacing"""
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    fi, fl, ff, fm = (cache / f"{case['id']}.{s}" for s in ("img.npy", "lab.npy", "fg.npy", "meta.json"))
    if fm.exists():
        return json.load(open(fm))
    img, meta = io.read_image(case["image"], dim, spacing)
    lab, _ = io.read_label(case["label"], dim)
    assert lab.shape == img.shape[1:], f"{case['id']}: label {lab.shape} vs image {img.shape[1:]}"
    if target_spacing:
        shape = io.target_shape(lab.shape, meta["spacing"], target_spacing)
        img = io.resample(img, shape, "linear"); lab = io.resample_label(lab, shape)
        meta = dict(meta, spacing=list(target_spacing))
    fg = np.argwhere(lab > 0)
    if len(fg) > 20000:
        fg = fg[np.random.default_rng(0).choice(len(fg), 20000, replace=False)]
    for f, arr in ((fi, io.zscore(img, normalization).astype(np.float16)), (fl, lab), (ff, fg.astype(np.int32))):
        _save_atomic(f, arr)
    m = dict(shape=list(lab.shape), spacing=meta["spacing"], channels=int(img.shape[0]), n_fg=int((lab > 0).sum()))
    tmp = fm.with_name(fm.name + f".tmp{os.getpid()}"); json.dump(m, open(tmp, "w")); os.replace(tmp, fm)   # meta last: the case is complete
    return m


def _save_atomic(f, arr):
    """several folds of one task may cache the same case at once: write, then rename"""
    tmp = f.with_name(f.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, f)


class Patches(torch.utils.data.IterableDataset):
    """Endless random patches (a context box around each sampled center; the GPU augmentation crops the patch out of it).
    fg_fraction of the patches are centered on a foreground voxel (nnU-Net: 1/3)."""

    def __init__(self, cases, cache, ctx, fg_fraction=0.33, seed=0):
        self.ids = [c["id"] for c in cases]
        self.cache, self.ctx, self.fg_fraction, self.seed = Path(cache), list(ctx), fg_fraction, seed

    def _open(self):
        self.img = {i: np.load(self.cache / f"{i}.img.npy", mmap_mode="r") for i in self.ids}
        self.lab = {i: np.load(self.cache / f"{i}.lab.npy", mmap_mode="r") for i in self.ids}
        self.fg = {i: np.load(self.cache / f"{i}.fg.npy") for i in self.ids}

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed * 1000 + (info.id if info else 0))
        self._open()
        while True:
            i = self.ids[rng.integers(len(self.ids))]
            img, lab, fg = self.img[i], self.lab[i], self.fg[i]
            S = lab.shape
            if len(fg) and rng.random() < self.fg_fraction:
                c = fg[rng.integers(len(fg))]
            else:
                c = np.array([rng.integers(s) for s in S])
            yield crop(img, lab, c, self.ctx)


def crop(img, lab, center, size):
    """the box of `size` centered at `center`, zero-padded where it leaves the image"""
    S = lab.shape
    lo = [int(c) - s // 2 for c, s in zip(center, size)]
    src = tuple(slice(max(a, 0), min(a + s, n)) for a, s, n in zip(lo, size, S))
    dst = tuple(slice(sl.start - a, sl.stop - a) for sl, a in zip(src, lo))
    oi = np.zeros((img.shape[0],) + tuple(size), np.float16); ol = np.zeros(tuple(size), np.uint8)
    oi[(slice(None),) + dst] = img[(slice(None),) + src]; ol[dst] = lab[src]
    return torch.from_numpy(oi), torch.from_numpy(ol)


class WholeImages(torch.utils.data.IterableDataset):
    """Endless random whole images (a task with "whole_image": true, e.g. photographs at a working level): each image read from its
    file, normalized, and centered in the context box (zero-padded); the GPU augmentation crops the padded patch out of it."""

    def __init__(self, cases, dim, ctx, normalization, spacing=None, seed=0):
        self.cases, self.dim, self.ctx, self.norm, self.spacing, self.seed = cases, dim, list(ctx), normalization, spacing, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed * 1000 + (info.id if info else 0))
        while True:
            c = self.cases[rng.integers(len(self.cases))]
            img, _ = io.read_image(c["image"], self.dim, self.spacing)
            lab, _ = io.read_label(c["label"], self.dim)
            assert all(s <= k for s, k in zip(lab.shape, self.ctx)), f"{c['id']}: {lab.shape} larger than the context {self.ctx}"
            yield crop(io.zscore(img, self.norm), lab, [s // 2 for s in lab.shape], self.ctx)
