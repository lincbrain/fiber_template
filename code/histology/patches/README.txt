OUR CHANGE TO TIRL

The histology registration runs TIRL 3.6.3 (settings/env/tirl.yml). Its optimizer wrapper, tirl/optimisation/optnl.py, normalizes each
parameter's start point to its bounds before calling nlopt. When a parameter ends a resolution level exactly on its bound, the next
level's normalized start can fall a hair outside [-1, 1] by round-off, and nlopt's BOBYQA refuses it: the registration stops. The one
change clips the normalized start into [-1, 1]. It changes no result otherwise: no registration of the paper's settings reached the
bound (the clipped TIRL reproduced the registrations of record byte for byte), and every histology registration of the paper's numbers
ran with it.

tirl-3.6.3_optnl_clip.edits.json holds our lines only (TIRL is Apache-2.0; the edit file follows code/template/patches/ for a uniform
form): the range of optnl.py's lines it replaces, their sha256, our new lines, and the md5 of the clean and of the edited file.
install.sh applies it to the copy of TIRL's sources it builds. By hand, on a source tree or an installed TIRL (the folder that holds
tirl/):
    python code/template/patches/apply_edits.py code/histology/patches/tirl-3.6.3_optnl_clip.edits.json <folder>
The histology registration (code/histology/register.py run_tirl) reports whether the TIRL it runs carries the change.
