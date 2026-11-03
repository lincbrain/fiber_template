"""Run the anatomix registration CLI (neel-dey/anatomix `anatomix-register.py`, FireANTs backend) unmodified, inside its own environment.

    $ANATOMIX_PYTHON -m linc.registration.anatomix_runner <anatomix-register.py arguments ...>

Added around the CLI, in-process only (the anatomix source is never edited):
  1. --fused-ops is configured FIRST (anatomix requires it before anything imports FireANTs);
  2. FireANTs' global /tmp/fireants_image_lock is redirected to a per-user file ($FIREANTS_LOCK), so another user's lock file on a
     shared machine cannot block every image write;
  3. with LINC_UNGATED_FEATURES=1 the anatomix feature channels are NOT multiplied by the masks (the masks still reach the CLI for the
     overlap score and the moved segmentation; with a masked loss the mask channel is still appended). The registration of each
     blockface to its own b0 and direct placement ran this way ("no engine masks"); the route test's route A did not.
The CLI is found next to the installed anatomix package (anatomix/registration/anatomix-register.py) unless ANATOMIX_REGISTER names it.

Ported from the analysis project's code/anatomix_rt/run_upstream.py: the same three steps (that runner's other feature hooks were study
switches, off for every product of record, and are not ported).
"""
import os
import sys

if __name__ == "__main__":
    import anatomix

    CLI = os.environ.get("ANATOMIX_REGISTER",
                         os.path.join(os.path.dirname(anatomix.__file__), "registration", "anatomix-register.py"))

    # 1. fused ops before any FireANTs import
    from anatomix.registration.registration_infrastructure.fused_ops import configure  # noqa: E402
    configure(sys.argv[1:])

    # 2. per-user FireANTs image-write lock
    import fcntl  # noqa: E402
    from contextlib import contextmanager  # noqa: E402
    import fireants.io.image as _fimg  # noqa: E402

    _LOCK = os.environ.get("FIREANTS_LOCK", f"/tmp/fireants_image_lock_{os.getuid()}")

    @contextmanager
    def _get_lock():
        with open(_LOCK, "w") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    if hasattr(_fimg, "get_lock"):
        _fimg.get_lock = _get_lock

    # 3. feature channels not gated by the masks
    if os.environ.get("LINC_UNGATED_FEATURES", "") == "1":
        import torch  # noqa: E402
        from anatomix.registration.registration_infrastructure import pipeline as _pl  # noqa: E402
        from anatomix.registration.registration_infrastructure.features import combine_feature_channels  # noqa: E402
        from anatomix.registration.registration_infrastructure.warp_io import as_batch  # noqa: E402

        _state = {"calls": 0}

        def _call(self, image, normalized, spacing, mask, append_mask):
            primary, mind = self.prepare(normalized, spacing, self.model, **self.config)
            _state["calls"] += 1
            if _state["calls"] == 1:
                print("  [anatomix_runner] feature channels NOT gated by the masks", flush=True)
            image.array = combine_feature_channels(primary, mind, None, self.features, append_mask=False)
            if append_mask:                                             # masked losses: ungated features, the mask channel still last
                image.array = torch.cat([image.array, (mask > 0).to(device=image.array.device, dtype=image.array.dtype)], 1)
            image.channels = image.array.shape[1]
            return as_batch(image)

        _pl.FeatureExtractor.__call__ = _call

    import runpy  # noqa: E402
    sys.argv = [CLI] + sys.argv[1:]
    runpy.run_path(CLI, run_name="__main__")
