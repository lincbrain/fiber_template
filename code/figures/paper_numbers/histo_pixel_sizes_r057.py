#!/usr/bin/env python3
"""The pixel size of the digitized histology sections as stored, and of the level this work read (paper: Methods, histology).

For the six histology brains (MM1-MM3 = MR243, MR252, MR256; MF5 = MF278, MF6 = MF275, MF8 = MF191), the first image stack of the brain
in the raw LINC dataset ($LINC_RAW/linc_000003/rawdata/sub-<ID>/micr; histology.sections.image_stacks): the level-0 pixel size (um)
from the OME-Zarr multiscales (dataset "0" scale, else the JSON sidecar's PixelSize), the levels present, and the pixel size of level 6
(= level 0 x 2^6; the level the tissue masks, the piece registrations and the tracer labels were read at).
Ported from the analysis project's paper/scripts/histo_pixel_sizes_r057.py; its description is kept verbatim as METHOD (part of the record).
-> $LINC_NUMBERS/histo_pixel_sizes_r057.json
    python code/figures/paper_numbers/histo_pixel_sizes_r057.py          (settings/env/tirl.yml, PY_TIRL: zarr 3.1.5)"""
import json
import os
import sys
from pathlib import Path

import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from shared import paths, provenance  # noqa: E402
from histology.sections import image_stacks, levels  # noqa: E402

BRAINS = {"MM1": "MR243", "MM2": "MR252", "MM3": "MR256", "MF5": "MF278", "MF6": "MF275", "MF8": "MF191"}
METHOD = [
    'For the six histology brains (MM1-MM3 = MR243, MR252, MR256; MF5 = MF278, MF6 = MF275, MF8 = MF191), the stack this work used',
    '(labels/ingest_omezarr.zarr_paths(S)[0], LINC 000003 rawdata): the level-0 pixel size (um) from the OME-Zarr multiscales (dataset "0"',
    'scale, else the JSON sidecar PixelSize), the levels present, and the pixel size of level 6 (= level 0 x 2^6; the level the tissue masks,',
    'the piece registrations and the tracer labels were read at; MF5 stores levels 0, 4 and 8 only, and its level 6 is the block mean of level 4).',
    '-> paper/numbers/histo_pixel_sizes_r057.json   (summary: l0_um_min/max, l6_um_min/max over the six)']


def main():
    raw = paths.raw_root() / "linc_000003" / "rawdata"
    out = dict(method=METHOD, brains={})
    ins = []
    for name, S in BRAINS.items():
        zp = image_stacks(str(raw / f"sub-{S}" / "micr"), S)[0]; ins.append(os.path.join(zp, ".zattrs"))
        g = zarr.open_group(zp, mode="r")
        ds = g.attrs["multiscales"][0]["datasets"]
        l0 = [float([t for t in d["coordinateTransformations"] if t["type"] == "scale"][0]["scale"][-1]) for d in ds if d["path"] == "0"]
        if not l0:
            l0 = [float(json.load(open(zp.replace(".ome.zarr", ".json")))["PixelSize"].strip("[]").split(",")[0])]
        present = sorted(levels(g))
        out["brains"][name] = dict(subject=S, stack=os.path.basename(zp), l0_um=round(l0[0], 6), levels_present=present,
                                   l6_um=round(l0[0] * 64, 3), l6_stored=6 in present)
    v0 = [b["l0_um"] for b in out["brains"].values()]; v6 = [b["l6_um"] for b in out["brains"].values()]
    out["summary"] = dict(l0_um_min=min(v0), l0_um_max=max(v0), l6_um_min=min(v6), l6_um_max=max(v6), n=len(v0))
    print(json.dumps(out["summary"]), json.dumps(out["brains"]))
    print(provenance.write_numbers("histo_pixel_sizes_r057", out, ins, md5=False, cmd="python code/figures/paper_numbers/histo_pixel_sizes_r057.py"))


if __name__ == "__main__":
    main()
