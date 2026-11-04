#!/usr/bin/env python3
"""What registering the leave-one-brain-out inputs by all channels would cost on a GPU (paper: Methods, leave-one-brain-out).

You need (configs/products_mri.yaml)
  mri_logs: the leave-one-brain-out registrations' logs (lobo_t2_stream_*.log); mri_runs: the logs of the template construction and its
  four folds (<build>/out/log/out_077_nlnT_mmorf.o*). MMORF prints "REGISTRATION TOOK <s> SECONDS" in both.

What it does
  1. takes the median time of one registration by the T2 channel alone (the leave-one-brain-out registrations of record);
  2. takes the median time of one registration by all channels at step 6, iteration 3, the last step whose per-task times were kept. The
     finer steps cost more, so 32 times this median is a lower bound;
  3. adds the estimate from one finest-step task, 4.7 h on a 48 GB card (a profile of 30 Sep 2026, quoted; its files were not kept).

You get
  $LINC_NUMBERS/lobo_multichannel_cost_r061.json: the hours per input and per 32 inputs, and the ratio of the two medians. The logs are not
  released, so without them the script says so and stops; reproduce/reference holds the paper's record.

    python reproduce/numbers/lobo_multichannel_cost_r061.py
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

N_INPUTS, N_HELDOUT = 32, 8                      # 8 inputs per fold (held-out brain, its mirror, 6 included) x 4 folds; held-out + mirror only
STEP8_PROFILE_H = 4.7                            # the 30 Sep profile (files not kept); quoted
BUILDS = ("rb_final", "rb_lobo_MF283", "rb_lobo_MF284", "rb_lobo_MF285", "rb_lobo_MF286")


def took(files):
    v = []
    for f in files:
        v += [float(m.group(1)) for m in re.finditer(r"REGISTRATION TOOK ([0-9.]+) SECONDS", Path(f).read_text(errors="ignore"))]
    return np.array(v)


def main():
    P = paths.products("products_mri")
    t2 = sorted(P["mri_logs"].glob("lobo_t2_stream_*.log"))
    mc = sorted(f for b in BUILDS for f in (P["mri_runs"] / b / "out" / "log").glob("out_077_nlnT_mmorf.o*"))
    if not t2 or not mc:
        print("SKIPPED lobo_multichannel_cost_r061: the runs' logs are not released; reproduce/reference holds the paper's record")
        return
    a, b = took(t2), took(mc)
    per = dict(t2only_median=round(float(np.median(a)) / 3600, 3), t2only_n=int(len(a)), t2only_min=round(float(a.min()) / 3600, 3), t2only_max=round(float(a.max()) / 3600, 3),
               multichannel_step6_median=round(float(np.median(b)) / 3600, 3), multichannel_step6_n=int(len(b)),
               multichannel_step6_min=round(float(b.min()) / 3600, 3), multichannel_step6_max=round(float(b.max()) / 3600, 3),
               multichannel_step8_profile_estimate=STEP8_PROFILE_H)
    ch = dict(t2only_32_inputs=round(N_INPUTS * per["t2only_median"], 1),
              multichannel_step6_32_inputs=round(N_INPUTS * per["multichannel_step6_median"], 1),
              multichannel_step8_32_inputs_estimate=round(N_INPUTS * STEP8_PROFILE_H, 1),
              multichannel_step8_heldout_and_mirror_only_8_inputs_estimate=round(N_HELDOUT * STEP8_PROFILE_H, 1))
    rec = dict(n_inputs=N_INPUTS, n_heldout_and_mirror=N_HELDOUT, per_input_h=per, card_hours=ch,
               ratio_multichannel_step6_over_t2only=round(per["multichannel_step6_median"] / per["t2only_median"], 2))
    provenance.write_numbers("lobo_multichannel_cost_r061", rec, inputs=t2 + mc, md5=False, cmd="python reproduce/numbers/lobo_multichannel_cost_r061.py")


if __name__ == "__main__":
    main()
