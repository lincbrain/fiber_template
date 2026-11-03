#!/usr/bin/env python3
"""The registration of each brain's blockface to its own b0 (R1), over all 13 brains with MRI: the interior edge distance of each
registration (linc/evaluation/regeval.py `edge_mean`, computed on the registration's own products: the b0 on its 0.25 mm grid and
the registered blockface), and its median, IQR (numpy's default linear interpolation), minimum and maximum.
Paper: Results, registration of each brain's blockface to its own MRI. The products: configs/products_blockface.yaml `r1_products`.
-> $LINC_NUMBERS/r1_edge_summary.json
    python reproduce/numbers/r1_edge_summary.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402
from linc.evaluation import regeval  # noqa: E402

NOTE = "interior edge distance, registered blockface vs the brain's own b0; all 13 brains with MRI"     # the record's text, verbatim


def main():
    root, prod = paths.products_root(), paths.products("products_blockface")
    per, ins = {}, []
    for S, rel in prod["r1_products"].items():
        fixed, moved = root / rel / f"{S}_mri.nii.gz", root / rel / f"{S}_bf_nonlin.nii.gz"
        per[S] = float(regeval.evaluate(str(fixed), str(moved))["edge_mean"])
        ins += [fixed, moved]
        print(S, round(per[S], 4), flush=True)
    v = np.array(list(per.values()))
    assert len(v) == 13, len(v)
    data = dict(edge_mean=dict(n=int(len(v)), median=float(np.median(v)), iqr=[float(np.percentile(v, 25)), float(np.percentile(v, 75))],
                               min=float(v.min()), max=float(v.max()), per_brain=per, unit="mm", note=NOTE))
    provenance.write_numbers("r1_edge_summary", data, inputs=ins, cmd="python reproduce/numbers/r1_edge_summary.py", md5=False)


if __name__ == "__main__":
    main()
