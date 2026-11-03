# Frequently asked questions

*A stub; questions are added as they come up.*

**Which brain is which?** The paper names brains MF1–MF10 (*Macaca fascicularis*) and MM1–MM15 (*Macaca mulatta*). Files use the
LINC identifiers (for example `sub-MR252` for MM2). `docs/specimen_labels.json` maps one to the other.

**Which Python environment do I need?** `bash env/install.sh` builds them all: four, because their packages conflict (`linc`,
`deep`, `anatomix`, `tirl`), and two optional ones (`nnunet`, only to reproduce the earlier networks' masks; `viz3d`, for the 3D
tracer figure). `env/README.md` says which one runs each step, and each stage's page in `docs/` names its own.

**Do I need a GPU?** Feature-based registration (anatomix + FireANTs), the synthesis of missing tissue and the nnU-Net models run on a GPU. The
paper's runs used 24 to 48 GB cards. Reconstruction, label carrying, numbers and figures run on a CPU.

**I only want the template. Do I need any of this?** No. The template is a data release (see the README). This code is for
building it, using it with new data, and checking the paper.

**How do I reproduce a figure?** Set `LINC_PRODUCTS` and `LINC_NUMBERS` (docs/data_layout.md), then run its script in
`reproduce/figures/`, in the environment `docs/paper_to_code.md` names for it. With another matplotlib version than that
environment's, the text can be laid out slightly differently; the data drawn are the same.

**Why do my registrations differ slightly from the paper's?** The deformable registration (anatomix + FireANTs) is deterministic
within a GPU session, but its convolution kernels can be chosen differently in another session or on another GPU model, and the
warps then differ by floating point: a median of 0.002–0.025 mm and at most 1.9 mm at single voxels in the tests listed in
`docs/reproduce.md` (Exact reproduction). The paper's products are the registrations of record, not a rerun.

**Why does my histology registration differ slightly from the paper's?** TIRL's result depends on the thread count. Run
`linc-histology` with the default 12 threads per piece (`--threads 12`, `configs/histology.yaml`) to reproduce the paper's
registrations exactly.

**Why do some numbers records describe their method in different words from the paper?** A record keeps the free-text
description written when it was computed. The numbers in it are the ones the paper reports, and `reproduce/check_numbers.py`
checks them.

**How do I report a bug?** Attach the output of `python3 cli/linc-info`, run from the environment that failed (`docs/install.md`,
step 6).

**Why are ONPRC18 and CIVM-R images missing?** Their licenses do not allow redistribution. Only our transforms to them are
released; download the templates from their authors.

**Where are the network weights?** Not in git: they come with the data release. `models/` holds the model cards, the checksum
lists and a script that checks a downloaded copy against them (`docs/install.md`).
