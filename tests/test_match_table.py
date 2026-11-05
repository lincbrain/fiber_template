"""The one order in which every step finds a brain's match table (linc/histology/match.py find_table)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from linc.histology import match  # noqa: E402


def test_order(tmp_path):
    assert match.find_table(tmp_path) == (None, None)
    (tmp_path / match.PROPOSED).write_text("")
    (tmp_path / match.REVIEW).write_text("section,component,flag\n1,left,low-score\n2,left,\n")
    assert match.find_table(tmp_path)[0] == str(tmp_path / match.PROPOSED)
    assert "1 components flagged" in match.find_table(tmp_path)[1]
    (tmp_path / match.TABLE).write_text("")
    assert match.find_table(tmp_path)[0] == str(tmp_path / match.TABLE)
    (tmp_path / match.EDITED).write_text("")
    assert match.find_table(tmp_path)[0] == str(tmp_path / match.EDITED)
    assert match.find_table(tmp_path, "other.txt")[0] == "other.txt"
