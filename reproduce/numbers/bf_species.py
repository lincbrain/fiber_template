#!/usr/bin/env python3
"""Species by the ID-PREFIX RULE (MF = Macaca fascicularis, MR = Macaca mulatta; a naming rule, not a specimen record; the species of
record from LINC dandiset 000003 stays in the cohort record), for the 21 blockface brains of the cohort record ($LINC_NUMBERS/bf_cohort.json,
written by reproduce/numbers/bf_cohort.py): per brain, and counted. Paper: Table 1 and Methods (the species of each brain).
The record of record also counted two groups of a superseded template build (its `blockface_template` and `utility` blocks); the paper
does not read them, so they are not regenerated (configs/numbers_exceptions.yaml).
-> $LINC_NUMBERS/bf_species.json
    python reproduce/numbers/bf_species.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from linc import paths, provenance  # noqa: E402

RULE = {"MF": ("Macaca fascicularis", "M. fascicularis", "fascicularis"), "MR": ("Macaca mulatta", "M. mulatta", "mulatta")}
RULE_TEXT = 'species by the ID prefix: MF = Macaca fascicularis, MR = Macaca mulatta (CONVENTIONS.md, Brains and names; docs/glossary.md); a naming rule, not a specimen record (species_of_record: bf_cohort.json)'   # verbatim


def species(S):
    return RULE[S[:2]]


def count(subjects):
    out = {"n": len(subjects), "fascicularis": 0, "mulatta": 0}
    for S in subjects:
        out[species(S)[2]] += 1
    return out


def main():
    cohort = paths.numbers_root() / "bf_cohort.json"
    rows = json.loads(cohort.read_text())["data"]["rows"]
    per = {r["subject"]: dict(species=species(r["subject"])[0], short=species(r["subject"])[1],
                              species_of_record=r.get("species_of_record")) for r in rows}
    provenance.write_numbers("bf_species", dict(rule=RULE_TEXT, per_brain=per, blockface_cohort=count(sorted(per))), inputs=[cohort],
                             cmd="python reproduce/numbers/bf_species.py")


if __name__ == "__main__":
    main()
