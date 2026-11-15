# Our change to TIRL

The histology registration runs TIRL 3.6.3 (`settings/env/tirl.yml`).
TIRL's optimizer wrapper, `tirl/optimisation/optnl.py`, normalizes each parameter's start point to its bounds before calling nlopt.
When a parameter ends a resolution level exactly on its bound, round-off can put the next level's normalized start a hair outside [-1, 1], and nlopt's BOBYQA refuses it, which stops the registration.
Our one change clips the normalized start into [-1, 1].
It changes no result otherwise: no registration with the published settings reached the bound, the clipped TIRL reproduced the paper's registrations byte for byte, and every histology registration behind the paper's numbers ran with it.

`tirl-3.6.3_optnl_clip.edits.json` holds only our lines (TIRL is Apache-2.0, and the edit file uses the same form as `code/template/patches/`): the range of `optnl.py`'s lines it replaces, their sha256, our new lines, and the md5 of the clean and of the edited file.
`install.sh` applies it to the copy of TIRL's sources it builds.
To apply it yourself to a source tree or an installed TIRL (the folder that holds `tirl/`):

```bash
python code/template/patches/apply_edits.py code/histology/patches/tirl-3.6.3_optnl_clip.edits.json <folder>
```

The histology registration (`run_tirl` in `code/histology/register.py`) reports whether the TIRL it runs carries the change.
