"""What the registration to the brain's own b0 does with the files it finds, before any work (CPU, no data: empty toy files).

    python -m pytest tests/test_register_inputs.py      or      python tests/run.py tests/test_register_inputs.py

You need
  settings/env/linc.yml.
What it does
  It checks code/blockface/registration/to_own_mri.py own_mri_plan on toy mri/ folders: a folder with the placed b0 is ready; a
  folder that holds only the b0 as acquired and its mask (a data folder of inputs only) is told to place the b0 first; a paper
  brain whose mask of record is missing stops with one line that names the file and where it comes from, unless it is run as a
  new brain; and a folder with no b0 at all (a brain without MRI) stops with one line that points to the blockface-template route.
You get
  pytest's pass / fail per test.
"""
from blockface.registration import to_own_mri

S = "MR000"
PLACED = [f"{S}_mri.nii.gz", f"{S}_mri_mask.nii.gz", f"{S}_mri2template.mat"]
NATIVE = [f"{S}_mri_b0_native.nii.gz", f"{S}_mri_mask_native.nii.gz"]
RECORD = dict(fixed_b0=f"{S}_mri.nii.gz", b0_mask=f"{S}_mri_mask.nii.gz", start_pose_world=[[1, 0, 0, 0]])


def folder(tmp_path, names):
    for n in names:
        (tmp_path / n).write_bytes(b"")
    return tmp_path


def test_placed_b0_there_ready(tmp_path):
    assert to_own_mri.own_mri_plan(folder(tmp_path, PLACED + NATIVE), S, RECORD) == ("ready", None)


def test_inputs_only_places_first(tmp_path):
    assert to_own_mri.own_mri_plan(folder(tmp_path, NATIVE), S, RECORD) == ("place", None)
    assert to_own_mri.own_mri_plan(folder(tmp_path, NATIVE), S, None) == ("place", None)          # a new brain too


def test_place_b0_given_needs_nothing_in_the_folder(tmp_path):
    assert to_own_mri.own_mri_plan(tmp_path, S, RECORD, placing=True) == ("ready", None)


def test_mask_of_record_missing_stops_with_one_line(tmp_path):
    rec = dict(RECORD, b0_mask=f"{S}_mri_mask_of_record.nii.gz")
    for names, placing in ((NATIVE, False), (PLACED + NATIVE, False), ([], True)):
        d = tmp_path / f"case{len(names)}{placing}"; d.mkdir()
        act, why = to_own_mri.own_mri_plan(folder(d, names), S, rec, placing=placing)
        assert act == "stop" and "\n" not in why
        assert f"{S}_mri_mask_of_record.nii.gz" in why and f"brains/sub-{S}/intermediate/registration/" in why and "--new-brain" in why
    assert to_own_mri.own_mri_plan(folder(tmp_path, NATIVE), S, rec, new_brain=True) == ("place", None)   # as a new brain: no mask of record


def test_fixed_b0_of_record_in_a_subfolder(tmp_path):
    rec = dict(RECORD, fixed_b0=f"b0_n4_v2/{S}_mri_n4.nii.gz")
    act, why = to_own_mri.own_mri_plan(folder(tmp_path, PLACED + NATIVE), S, rec)
    assert act == "stop" and f"b0_n4_v2/{S}_mri_n4.nii.gz" in why
    (tmp_path / "b0_n4_v2").mkdir(); (tmp_path / "b0_n4_v2" / f"{S}_mri_n4.nii.gz").write_bytes(b"")
    assert to_own_mri.own_mri_plan(tmp_path, S, rec) == ("ready", None)


def test_no_b0_at_all_points_to_the_other_route(tmp_path):
    act, why = to_own_mri.own_mri_plan(tmp_path, S, None)
    assert act == "stop" and "\n" not in why and "--place-b0" in why and "blockface-template" in why
    act, why = to_own_mri.own_mri_plan(folder(tmp_path, NATIVE[:1]), S, RECORD)                    # the b0 without its mask: not enough
    assert act == "stop" and "--place-b0" in why
