"""Run the anatomix registration program unmodified, inside its own environment. anatomix is a pretrained network that computes image
features for registration; its program, `anatomix-register.py`, registers them with FireANTs.

    $ANATOMIX_PYTHON -m blockface.registration.anatomix_runner <anatomix-register.py arguments ...>

You need
  settings/env/anatomix.yml's Python ($ANATOMIX_PYTHON) with anatomix installed; ANATOMIX_REGISTER names the program when it is not next to
  the installed package (anatomix/registration/anatomix-register.py).
What it does
  It adds three things around the program, in this process only (the anatomix source is never edited):
  1. --fused-ops is set first (anatomix needs it before anything imports FireANTs);
  2. FireANTs' global /tmp/fireants_image_lock is moved to a per-user file ($FIREANTS_LOCK), so another user's lock file on a
     shared machine cannot block every image write;
  3. with LINC_UNGATED_FEATURES=1 the feature channels are not multiplied by the masks (the masks still reach the program for the
     overlap score and the moved segmentation; with a masked loss the mask channel is still appended). The registration of each
     blockface to its own b0 and the cross-modal alignment ran this way; the blockface-based alignment did not.
  4. with LINC_FEATURE_RES_MM=<mm> as well (the blockface template's step 9, code/template/blockface_step9.py), the network's
     features are computed on the image resampled to that voxel size (trilinear), upsampled back to the grid and normalized to
     unit length; MIND-SSC stays on the grid. The channels are then assembled in host memory, which keeps a whole brain at
     0.15 mm within a 48 GB card.
You get
  the program's outputs, where its arguments say.

Ported from the analysis project's code/anatomix_rt/run_upstream.py (steps 1-3) and its step-9 copy (step 4, its exact mode).
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

    # 3. feature channels not gated by the masks; 4. the network's features at a coarser voxel size (needs 3)
    UNGATED = os.environ.get("LINC_UNGATED_FEATURES", "") == "1"
    FEATRES = float(os.environ.get("LINC_FEATURE_RES_MM", "0") or 0)
    if FEATRES and not UNGATED:
        raise SystemExit("LINC_FEATURE_RES_MM needs LINC_UNGATED_FEATURES=1 (the combination step 9 was built with)")
    if UNGATED:
        import torch  # noqa: E402
        import torch.nn.functional as F  # noqa: E402
        from anatomix.registration.registration_infrastructure import pipeline as _pl  # noqa: E402
        from anatomix.registration.registration_infrastructure.features import combine_feature_channels  # noqa: E402
        from anatomix.registration.registration_infrastructure.warp_io import as_batch  # noqa: E402

        _state = {"calls": 0}

        def _featres(self, normalized, spacing):
            f = FEATRES / float(min(spacing))
            shape = tuple(normalized.shape[-3:])
            small = F.interpolate(normalized, scale_factor=1.0 / f, mode="trilinear", align_corners=False)
            p, _ = self.prepare(small, tuple(s * f for s in spacing), self.model, **dict(self.config, features="anatomix"))
            x = F.interpolate(p.to(normalized.device), size=shape, mode="trilinear", align_corners=False)
            del p
            x.div_(x.norm(dim=1, keepdim=True).clamp_min(1e-8))       # unit length per voxel, in place
            torch.cuda.empty_cache()
            return x

        def _call(self, image, normalized, spacing, mask, append_mask):
            if FEATRES and self.model is not None:
                _, mind = self.prepare(normalized, spacing, None, **dict(self.config, features="mindssc"))
                primary = _featres(self, normalized, spacing)
            else:
                primary, mind = self.prepare(normalized, spacing, self.model, **self.config)
            _state["calls"] += 1
            if _state["calls"] == 1:
                print("  [anatomix_runner] feature channels NOT gated by the masks"
                      + (f"; network features at {FEATRES} mm, assembled in host memory" if FEATRES else ""), flush=True)
            if FEATRES and append_mask and mind is not None and primary is not None and self.features == "anatomix+mindssc":
                mk = (mask > 0).to(device=primary.device, dtype=primary.dtype)   # [features, MIND-SSC, mask] in one volume
                image.array = torch.cat([primary, mind.to(device=primary.device, dtype=primary.dtype), mk], 1)
                del primary, mind, mk
                image.array = image.array.cpu()
                torch.cuda.empty_cache()
            else:
                image.array = combine_feature_channels(primary, mind, None, self.features, append_mask=False)
                if append_mask:                                             # masked losses: ungated features, the mask channel still last
                    image.array = torch.cat([image.array, (mask > 0).to(device=image.array.device, dtype=image.array.dtype)], 1)
            image.channels = image.array.shape[1]
            return as_batch(image)

        _pl.FeatureExtractor.__call__ = _call

    import runpy  # noqa: E402
    sys.argv = [CLI] + sys.argv[1:]
    runpy.run_path(CLI, run_name="__main__")
