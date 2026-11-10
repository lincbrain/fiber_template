#!/usr/bin/env python3
"""The blockface camera of record, for the camera sentence of Methods ("Brains"), read from the raw photographs' EXIF.

    python code/figures/paper_numbers/bf_camera_exif.py [--raw $LINC_RAW]

You need
  the raw photographs ($LINC_RAW/<settings/brains/<ID>.yaml raw>).
What it does
  It reads every 25th photograph of each of the 21 brains: camera make and model, image size, focal length, f-number, exposure
  time, ISO and flash. The lighting set-up itself is not in the files (EXIF says only whether the flash fired).
You get
  $LINC_NUMBERS/bf_camera_exif.json. The record lists each brain's folder as it was read, so it regenerates identically only with
  LINC_RAW set as for the paper.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import yaml
from PIL import Image, ExifTags

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from shared import paths, provenance  # noqa: E402

KEYS = ("Make", "Model", "FocalLength", "FNumber", "ExposureTime", "ISOSpeedRatings", "Flash", "WhiteBalance")
NOTE = "Brains whose files carry no EXIF are grouped by image size; the D80's 3872x2592 frame is shared by every brain with EXIF of that size."                                       # the record's text, verbatim


def exif(p):
    im = Image.open(p); ex = im.getexif()
    try:
        sub = ex.get_ifd(0x8769)
    except Exception:  # noqa: BLE001
        sub = {}
    d = {**{ExifTags.TAGS.get(k, k): v for k, v in ex.items()}, **{ExifTags.TAGS.get(k, k): v for k, v in sub.items()}}
    return im.size, {k: (str(d[k]).strip() if k in ("Make", "Model") else float(d[k])) for k in KEYS if k in d}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.environ.get("LINC_RAW"), help="root of the raw photographs")
    a = ap.parse_args()
    assert a.raw, "set --raw or LINC_RAW"
    cohort = REPO / "tests" / "reference" / "bf_cohort.json"        # the cohort record of 28 Sep, shipped
    subs = [r["subject"] for r in json.loads(cohort.read_text())["data"]["rows"]]
    per = {}
    for S in subs:
        c = yaml.safe_load((REPO / "settings" / "brains" / f"{S}.yaml").read_text())
        raw = str(Path(a.raw) / c["raw"])
        fs = sorted(f for f in glob.glob(f"{raw}/*") if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".nef"))
        rows, bad = [], 0
        for f in fs[::25]:
            try:
                rows.append(exif(f))
            except Exception:  # noqa: BLE001  (zero-byte or corrupt raw files: the cohort's zero_byte_raw)
                bad += 1
        sizes = sorted({f"{w}x{h}" for (w, h), _ in rows})
        with_exif = [e for _, e in rows if "Model" in e]
        rng = lambda k: [min(e[k] for e in with_exif if k in e), max(e[k] for e in with_exif if k in e)] if any(k in e for e in with_exif) else None  # noqa: E731
        per[S] = dict(raw_dir=raw, n_files=len(fs), n_sampled=len(rows), n_unreadable_sampled=bad, n_with_exif=len(with_exif), image_size=sizes,
                      camera=sorted({f"{e.get('Make', '')} {e.get('Model', '')}".strip() for e in with_exif}),
                      focal_length_mm=rng("FocalLength"), f_number=rng("FNumber"), exposure_s=rng("ExposureTime"),
                      iso=rng("ISOSpeedRatings"), flash_fired=sorted({int(e["Flash"]) & 1 for e in with_exif if "Flash" in e}))
        print(S, per[S]["camera"], per[S]["image_size"], flush=True)
    by_cam = {}
    for S, p in per.items():
        key = p["camera"][0] if p["camera"] else f"no EXIF ({'/'.join(p['image_size'])})"
        by_cam.setdefault(key, []).append(S)
    # one entry per camera model: the brains whose EXIF names it, plus those without EXIF whose photographs have exactly its frame size
    cameras = {}
    for key, model in (("d80", "NIKON D80"), ("d100", "NIKON D100")):
        exif_brains = sorted(S for S, p in per.items() if any(c_.endswith(model) for c_ in p["camera"]))
        sizes = sorted({z for S in exif_brains for z in per[S]["image_size"]})
        assert len(sizes) == 1, (model, sizes)
        w, h = (int(v) for v in sizes[0].split("x"))
        frame_only = sorted(S for S, p in per.items() if not p["camera"] and p["image_size"] == sizes)
        cameras[key] = dict(model=model, width_px=w, height_px=h, brains_exif=exif_brains, brains_frame_size_only=frame_only,
                            n_brains=len(exif_brains) + len(frame_only))
    assert sum(c_["n_brains"] for c_ in cameras.values()) == len(per), cameras
    provenance.write_numbers("bf_camera_exif", dict(n=len(per), cameras=cameras, by_camera=by_cam, per_brain=per, note=NOTE),
                             inputs=[cohort], cmd="python code/figures/paper_numbers/bf_camera_exif.py", md5=False)


if __name__ == "__main__":
    main()
