# Wrapper for the blockface template's build

`blockface_build.py` puts this folder first on PATH, so that mmorf here runs in place of MMORF. It runs
MMORF unchanged, except for the one b0-only member (the MRI template's b0), which has no blockface
image: an all-zero image would make MMORF's cost NaN, so that member gets a stand-in with zero weight
(`tpl_cfg.py` writes its settings). You do not run these yourself.
