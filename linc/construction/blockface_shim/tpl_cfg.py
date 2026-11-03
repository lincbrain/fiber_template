#!/usr/bin/env python3
"""TPL's MMORF config from a step's config: the BLOCKFACE block's lambda_scalar -> 0 at every level and estimate_bias -> 0 for
the blockface (TPL has no blockface; its MMORF call gets a stand-in image and a ones mask so MMORF's per-image normalization stays
finite). Line-based: only the first lambda_scalar line after the '; --- scalar modality BF' header changes (the two modality blocks
have identical bodies, so a text replace would change both). Usage: tpl_cfg.py <in.ini> <out.ini> [BF lambda, default 0]
Ported unchanged from the analysis project's bftemplate/d2b/dispatch/tpl_cfg.py."""
import sys
src, dst = sys.argv[1], sys.argv[2]; lam = sys.argv[3] if len(sys.argv) > 3 else "0"
L = open(src).read().splitlines()
h = [i for i, l in enumerate(L) if l.startswith("; --- scalar modality BF")]
b = [i for i, l in enumerate(L) if l.startswith("; --- scalar modality B0")]
assert len(h) == 1 and len(b) == 1 and h[0] < b[0], "unexpected config layout"
k = [i for i in range(h[0], b[0]) if L[i].startswith("lambda_scalar")]
assert len(k) == 1, "no single lambda_scalar line in the blockface block"
n = len(L[k[0]].split("=")[1].split()); L[k[0]] = "lambda_scalar           = " + " ".join([lam] * n)
e = [i for i, l in enumerate(L) if l.startswith("estimate_bias")]
assert len(e) == 1 and L[e[0]].split("=")[1].split() == ["1", "1"], L[e[0]]
L[e[0]] = "estimate_bias           = 0 1"
open(dst, "w").write("\n".join(L) + "\n")
