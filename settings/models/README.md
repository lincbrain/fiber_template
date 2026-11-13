# Checksums of the trained networks

The weights are in this repository, under `models/` at the top: `models/segment/<network>/` (the four segmentation
networks) and `models/completion/<network>/` (the two synthesis networks and the two leave-slab-out networks of the
paper's evaluation). A file over GitHub's 100 MB limit is kept as `<file>.part1` and `<file>.part2`.
`settings/roots.sh` joins them once into `outputs/models/` (`$LINC_MODELS`) with `code/shared/join_weights.sh`, and checks
every file against the lists here. The joined files are the files of record, bit for bit.

Check the repository's copy, or a joined one:

```bash
$PY tests/verify_weights.py settings/models/segmentation.sha256 --root models/segment          # parts are checked joined
$PY tests/verify_weights.py settings/models/completion_public.sha256 --root models/completion
$PY tests/verify_weights.py settings/models/segmentation.sha256 --root $LINC_MODELS/segment
```

`completion_checksum_map.tsv` maps each synthesis file as first made to its released copy. The released
copies have no machine paths; the weights are the same.
