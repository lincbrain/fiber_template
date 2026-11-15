# The paper's numbers

There is one script per numbers record: `<record>.py` writes `<record>.json`, and every number in the paper's text and tables comes from one of these records.
`tests/check_numbers.py` compares a rewritten record with the paper's (counts exactly, floats to 1e-9).

```bash
$PY code/figures/paper_numbers/<record>.py
$PY tests/check_numbers.py --ref tests/reference <record>
```

Some records read others, so run a record alone only after the ones it reads; `make -C code/figures` runs them in order.
A record keeps the name of the analysis that made it, even where that name sounds like a different method.
Its free-text method field is kept as it was written at the time, and its numbers are the ones the paper reports.
