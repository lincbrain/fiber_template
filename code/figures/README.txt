THE PAPER'S FIGURES AND NUMBERS

    make -C code/figures numbers           the numbers records that run on a CPU
    make -C code/figures numbers-gpu       the one record that needs a GPU
    make -C code/figures numbers-figures   the figures that also write a record
    make -C code/figures figures           every figure
    make -C code/figures check             compares the records with the paper's

check passes only when all three numbers targets have run. A target skips, with a message, when its
input is missing: your own copies of NMT v2 and CIVM-R (settings/targets.yaml), or raw data outside
the data tree. Each record and figure runs in the environment it ran in for the paper. The figures
take about 10 minutes on a CPU.

One figure at a time: $PY code/figures/<script>.py --out <file.png>

    Figure  Script                     Figure  Script
    1       template.py                7       histology.py
    2       pipeline.py                8       lgnstn.py
    3       reconstruction.py          9       alignment_comparison.py
    4       variability.py             10      tracer_sections3d.py
    5       bf_template.py             S2      s_synthesis_gallery.py
    6       link.py                    S3      s_tracer_fidelity.py
                                       S4      s_networks.py

Figure 10 of the paper is drawn differently from its script's output. Figure S1 has no script here.
s_synthesis_depth.py draws a figure the paper no longer shows (its numbers are in the Supplement's
text). figstyle.py holds the shared figure style. Fonts: Liberation Sans.

    paper_numbers/   one script per numbers record
    evaluation/      the measures the paper reports
