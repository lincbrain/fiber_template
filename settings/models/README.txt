CHECKSUMS OF THE TRAINED NETWORKS

The weights are in the data, under models/. Check a downloaded copy:

    $PY tests/verify_weights.py settings/models/segmentation.sha256 --root $LINC_MODELS/segment
    $PY tests/verify_weights.py settings/models/completion_public.sha256 --root $LINC_MODELS/completion

completion_checksum_map.tsv maps each synthesis file as first made to its released copy. The released
copies have no machine paths; the weights are the same.
