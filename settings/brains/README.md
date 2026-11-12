# One file per brain

`<ID>.yaml` holds a brain's photographs, pixel size and section step, camera jumps, missing sections and
masks of record. For a new brain, copy `TEMPLATE.yaml` to `<ID>.yaml` and fill it in: every field is
explained there, and [02, Step 6](../../docs/tutorials/02_blockface.md#step-6-adding-a-new-brain-its-settings-file) lists every field with its default and when to set it.

## What a person makes once per brain

|  |  |
|---|---|
| pixel size and section step      | vx, vz in this file; from a ruler and the microtome |
| camera jumps, missing sections   | jumps, missing in this file; by eye, from the stacked photographs |
| corrected masks (optional)       | see [code/blockface](../../code/blockface/README.md) |
| match table                      | see [code/histology](../../code/histology/README.md) |
| section map                      | which section each image in the histology stack is; checked manually |
| labels                           | drawn on the sections by an anatomist |
