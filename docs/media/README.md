# Media

These are the animations at the top of the repository's README.

- `template_coronal.gif` shows the template's six channels (T2-weighted, b0, tensor, soma fraction, neurite fraction, blockface) in coronal planes, from the plane with the largest brain area to the back, then the front.
  `make -C code/figures template-animation` makes it with [code/figures/readme_animation.py](../../code/figures/readme_animation.py), reading the data folder (`LINC_DATA_TREE`).
- `tracers_rotation.gif` shows the charted tracer labels of the four cases of the paper's Fig. 9 in the template, MR243 (MM1), MR252 (MM2) and MR256 (MM3) on the left and MF191 (MF8) on the right, turning together. Each case is named with its injection site (area 10, pre-SMA, area 10d, area 11); MM3's site is not in these labels, so it is not drawn.
  `make -C code/figures tracers-animation PY_VIZ3D=$PY_VIZ3D` makes it with [code/figures/tracer_sections3d.py](../../code/figures/tracer_sections3d.py) `--animation`.
