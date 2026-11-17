"""The LGN and STN carry's targets (code/labels/structures.py b0_source): a brain is carried into its b0 only when it has a b0
registration. A paper brain uses its variant of record; a brain without `registration_to_own_mri.variant_of_record` (a new brain,
settings/brains/TEMPLATE.yaml) uses its own `linc-register mri` entry, and without one it is carried into the template only."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def _with_data(tmp_path, fn):
    old = os.environ.get("LINC_DATA"); os.environ["LINC_DATA"] = str(tmp_path)
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("LINC_DATA", None)
        else:
            os.environ["LINC_DATA"] = old


def test_paper_brain_uses_its_variant_of_record(tmp_path):
    from labels.structures import b0_source
    assert _with_data(tmp_path, lambda: b0_source("MR243")) == ("record", "r1v3")


def test_new_brain_without_b0_registration_goes_to_the_template_only(tmp_path):
    from labels.structures import b0_source
    assert _with_data(tmp_path, lambda: b0_source("NB001")) is None          # no settings file, no variant, no entry


def test_new_brain_uses_its_own_b0_registration(tmp_path):
    from labels.structures import b0_source
    d = tmp_path / "sub-NB001" / "xfm" / "bf-native__to__b0-native"; d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"src": "bf-native", "dst": "b0-native", "subject": "NB001", "dst_grid": "b0.nii.gz",
                                             "components": []}))
    assert _with_data(tmp_path, lambda: b0_source("NB001")) == ("own", d)


def test_template_settings_leave_the_variant_out(tmp_path):
    """settings/brains/TEMPLATE.yaml, the file a new brain starts from, has no variant_of_record: b0_source must not need it."""
    import yaml
    t = yaml.safe_load((Path(__file__).resolve().parents[1] / "settings" / "brains" / "TEMPLATE.yaml").read_text()) or {}
    assert not (t.get("registration_to_own_mri") or {}).get("variant_of_record")
