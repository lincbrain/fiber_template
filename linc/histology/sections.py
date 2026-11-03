"""Histology sections from a multi-slice OME-Zarr stack (or a JP2 per section), read through one interface.

Facts the reader relies on (LINC OME-Zarr histology, checked against the JP2 sources):
  * a stack is (c, z, y, x) uint8; each section is CENTERED on a canvas the size of the largest section, independently per level:
    offset_L = (canvas_L - ceil(section / 2**L)) // 2, and level L equals the JP2's resolution level L bit for bit;
  * z is the stack index, NOT the section number (gaps and re-scans are packed in); the map z -> section comes from a section map
    (TSV with columns z, section, num, h0, w0[, status, use]);
  * some stacks carry only levels 4..8; a missing level is block-mean reduced from the next finer present level.

`Section` mimics the subset of glymur.Jp2k the pipeline uses (.shape, .codestream.segment[2].num_res, .read(rlevel, area),
.read_bands(rlevel), glymur-style slicing). Ported from the analysis project's labels/ingest_omezarr.py.
"""
import glob as _glob
import math
import os
import re

import numpy as np
import zarr


def stain_of(path):
    return re.search(r"_stain-([A-Za-z]+)_", os.path.basename(path)).group(1)


def _has_levels(p):
    try:
        return "multiscales" in zarr.open_group(p, mode="r").attrs
    except Exception:
        return False


def image_stacks(micr_dir, subject, stain=None):
    """The image stacks of a subject in a LINC micr/ folder (not the _mask stacks nor masks/)."""
    ps = sorted(_glob.glob(os.path.join(micr_dir, f"sub-{subject}_sample-*_stain-*_DF.ome.zarr")))
    ps = [p for p in ps if _has_levels(p)]
    if stain:
        ps = [p for p in ps if f"_stain-{stain}_" in os.path.basename(p)]
    return ps


def levels(g):
    """{level: array} for the levels actually written (declared paths may be missing on disk)."""
    out = {}
    for ds in g.attrs["multiscales"][0]["datasets"]:
        k = ds["path"]
        try:
            out[int(k)] = g[k]
        except Exception:
            pass
    return out


def canvas0(g):
    """Level-0 canvas (rows, cols), from the finest present level (exact when level 0 exists)."""
    lv = levels(g)
    L = min(lv)
    return lv[L].shape[2] * 2 ** L, lv[L].shape[3] * 2 ** L


def load_map(path):
    """Section map rows {z, section, num, h0, w0, status, use, ...}; integers parsed, '' / 'NA' -> None."""
    rows = []
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            if line.strip() and not line.startswith("#"):
                r = dict(zip(head, line.rstrip("\n").split("\t")))
                for k in ("z", "num", "h0", "w0"):
                    r[k] = int(r[k]) if r.get(k, "") not in ("", "NA") else None
                rows.append(r)
    return rows


class Section:
    """One histology section from an OME-Zarr stack, read like a JP2."""

    def __init__(self, zpath, z, h0=None, w0=None):
        self.g = zarr.open_group(zpath, mode="r")
        self.lv = levels(self.g)
        self.z = z
        self.H0, self.W0 = canvas0(self.g)
        self._centred = h0 is not None
        if h0 is None:                                   # no JP2 header -> extent = non-zero bounding box
            h0, w0, self._r0, self._c0 = self._bbox0()
        self.h0, self.w0 = h0, w0
        self.shape = (h0, w0, 3)

        class _Seg:                                      # codestream.segment[2].num_res
            num_res = max(self.lv) + 1

        class _CS:
            segment = [None, None, _Seg]

        self.codestream = _CS

    def _bbox0(self):
        L = max(k for k in self.lv if k <= 6) if any(k <= 6 for k in self.lv) else min(self.lv)
        a = np.asarray(self.lv[L][:, self.z]).max(0)
        rr, cc = np.nonzero(a)
        if rr.size == 0:
            return 0, 0, 0, 0
        f = 2 ** L
        r0, r1, c0, c1 = rr.min() * f, (rr.max() + 1) * f, cc.min() * f, (cc.max() + 1) * f
        return r1 - r0, c1 - c0, r0, c0

    def level(self, L):
        """The whole section at level L as (h, w, c) uint8, cropped to its own extent."""
        if L in self.lv:
            arr, f = self.lv[L], 1
        else:
            finer = [k for k in self.lv if k < L]
            if not finer:
                raise ValueError(f"level {L} unavailable (present: {sorted(self.lv)})")
            k = max(finer)
            arr, f = self.lv[k], 2 ** (L - k)
            L = k
        s = 2 ** L
        h, w = math.ceil(self.h0 / s), math.ceil(self.w0 / s)
        if self._centred:                                # the writer centers each level on its own canvas
            r0, c0 = (arr.shape[2] - h) // 2, (arr.shape[3] - w) // 2
        else:                                            # extent from the non-zero bounding box
            r0, c0 = int(round(self._r0 / s)), int(round(self._c0 / s))
        a = np.asarray(arr[:, self.z, r0:r0 + h, c0:c0 + w]).transpose(1, 2, 0)
        if f > 1:                                        # block mean to the requested level
            H, W = a.shape[0] // f * f, a.shape[1] // f * f
            a = a[:H, :W].reshape(H // f, f, W // f, f, a.shape[2]).mean((1, 3)).round().astype(np.uint8)
        return a

    def read(self, rlevel=0, area=None):
        a = self.level(rlevel)
        if area is not None:                             # glymur: (r0, c0, r1, c1) in level-0 pixels
            s = 2 ** rlevel
            r0, c0, r1, c1 = area
            a = a[r0 // s:math.ceil(r1 / s), c0 // s:math.ceil(c1 / s)]
        return a

    read_bands = read

    def __getitem__(self, key):
        """glymur-style slicing j2k[r0:r1:2**L, c0:c1:2**L(, bands)] -> read(L, (r0, c0, r1, c1)) (level-0 coordinates)."""
        rs, cs = key[0], key[1]
        L = int(round(math.log2(rs.step or 1)))
        a = self.read(L, (rs.start or 0, cs.start or 0, rs.stop if rs.stop is not None else self.h0,
                          cs.stop if cs.stop is not None else self.w0))
        return a[(slice(None), slice(None)) + tuple(key[2:])] if len(key) > 2 else a


def sections(stack=None, section_map=None, jp2_dir=None):
    """[(section_id, num, reader_factory)] in section order. JP2s (`jp2_dir/*nDF.jp2`) when given and present, else the OME-Zarr
    `stack` with its `section_map` (rows with use = 0/no are skipped). section_id is the JP2 basename without '.jp2'
    (e.g. mr252FS_c11_s01nDF), so the `_s(\\d+)` section-number rule and the piece names work on either source."""
    out = []
    if jp2_dir and _glob.glob(os.path.join(jp2_dir, "*nDF.jp2")):
        import glymur
        for p in sorted(_glob.glob(os.path.join(jp2_dir, "*nDF.jp2"))):
            sid = os.path.basename(p)[:-4]
            out.append((sid, int(re.search(r"_s(\d+)", sid).group(1)), lambda p=p: glymur.Jp2k(p)))
        return out
    if stack is None or section_map is None:
        raise ValueError("no JP2 sections: pass the OME-Zarr stack and its section map")
    for r in load_map(section_map):
        if r.get("use", "1") in ("0", "no"):
            continue
        out.append((r["section"], r["num"], lambda r=r: Section(stack, r["z"], r["h0"], r["w0"])))
    return out
