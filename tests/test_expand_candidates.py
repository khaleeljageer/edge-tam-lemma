import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "expand_candidates", Path(__file__).parents[1] / "scripts/data/expand_candidates.py"
)
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)


def test_vote_status_and_tie_break():
    assert exp.vote(["அ", "அ", "அ", "அ"]) == ("அ", "agree")
    assert exp.vote(["இ", "அ", "அ", "அ"]) == ("அ", "majority")
    assert exp.vote(["இ", "அ", "அ", "உ"]) == ("அ", "conflict")  # 2 of 4 is not a majority
    assert exp.vote(["இ", "அ", "அ", "இ"]) == ("இ", "conflict")  # tie -> earlier source


def _row(word, suggested, status, lemma=None, checked="y", pos=""):
    cands = dict.fromkeys(exp.SOURCES, suggested)
    return {"word": word, "suggested": suggested, "status": status, **cands,
            "lemma": lemma or suggested, "checked": checked, "pos": pos, "count": "5"}


def test_calibration_per_status():
    rows = [_row("மரங்கள்", "மரம்", "agree"), _row("வீடுகள்", "வீடு", "agree"),
            _row("போனான்", "போனா", "conflict")]
    rows[2]["teacher"] = "போ"
    report = exp.calibration(rows, {"மரங்கள்": "மரம்", "வீடுகள்": "வீடு", "போனான்": "போ"})
    assert report["agree"]["n"] == 2 and report["agree"]["acc_suggested"] == 1.0
    assert report["conflict"]["acc_suggested"] == 0.0 and report["conflict"]["acc_teacher"] == 1.0
    assert abs(report["all"]["share"] - 1.0) < 1e-9


def test_reviewed_pairs_leakage_and_marks():
    rows = [_row("மரங்கள்", "மரம்", "agree", pos="noun"),
            _row("வீடுகள்", "வீடுகள்", "majority", lemma="வீடு"),  # corrected
            _row("கதைகள்", "கதை", "agree"),                         # held-out lemma
            _row("அஅஅ", "அஅஅ", "conflict", checked="x"),
            _row("பூக்கள்", "பூ", "agree", checked="")]
    pairs, stats = exp.reviewed_pairs(rows, held_words=set(), held_lemmas={"கதை"}, source="t")
    assert [(p.word, p.lemma, p.pos) for p in pairs] == [("மரங்கள்", "மரம்", "NOUN"),
                                                         ("வீடுகள்", "வீடு", "OTHER")]
    assert stats == {"kept": 2, "corrected": 1, "dropped_heldout": 1, "skipped": 1,
                     "unchecked": 1}
