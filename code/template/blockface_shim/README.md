# Wrapper for the blockface template's build

You do not run these yourself.
`blockface_build.py` puts this folder first on PATH, so that the mmorf here runs in place of MMORF.
It runs MMORF unchanged, except for the one b0-only member (the MRI template's b0), which has no blockface image.
An all-zero image would make MMORF's cost NaN, so that member gets a stand-in image with zero weight (`tpl_cfg.py` writes its settings).
