import json
import subprocess
import sys

from tel.config import config_from_dict
from tel.data import Pair, read_tsv, write_tsv
from tel.train import train

STEMS = ["மரம்", "நாய்", "பசு", "வீடு", "கல்", "பூ", "மாடு", "ஊர்", "பால்", "தேன்"]


def test_distill_adds_teacher_words_without_heldout_lemmas(tmp_path):
    data = tmp_path / "data"
    train_pairs = [Pair(s + "கள்", s, "NOUN") for s in STEMS[:8]] + [Pair(s, s) for s in STEMS[:8]]
    held = [Pair(STEMS[8] + "கள்", STEMS[8], "NOUN")]  # dev/test lemma: பால்
    write_tsv(data / "train.tsv", train_pairs)
    for s in ("dev", "test"):
        write_tsv(data / f"{s}.tsv", held)
        (data / "splits").mkdir(parents=True, exist_ok=True)
        (data / "splits" / f"{s}.txt").write_text(STEMS[8] + "\n", encoding="utf-8")
    cfg = config_from_dict({
        "name": "t", "model": {"type": "lstm_seq2seq", "emb": 16, "hidden": 32},
        "data": {"dir": str(data)}, "train": {"epochs": 30, "batch_size": 8, "seed": 1},
    })
    train(cfg, tmp_path / "teacher", verbose=False)

    words = tmp_path / "words.tsv"
    words.write_text("word\tcount\n" + "பாலை\t9\n" + "தேன்கள்\t5\n" + "மரங்கள்கள்\t3\n"
                     + STEMS[8] + "கள்\t2\n", encoding="utf-8")
    out = tmp_path / "distill"
    subprocess.run([sys.executable, "-m", "tel.distill", "--teacher", str(tmp_path / "teacher"),
                    "--extra", str(words), "--min-token-prob", "0.0", "--out", str(out)],
                   check=True, capture_output=True)
    pairs = read_tsv(out / "train.tsv")
    stats = json.loads((out / "distill_stats.json").read_text())
    assert pairs[: len(train_pairs)] == train_pairs  # gold kept first
    extra = pairs[len(train_pairs):]
    assert all(p.source == "distill:teacher" for p in extra)
    assert STEMS[8] + "கள்" not in {p.word for p in extra}  # dev/test word excluded
    assert STEMS[8] not in {p.lemma for p in extra}  # held-out lemma never a target
    assert stats["extra_candidates"] == 3
    assert stats["extra_kept"] + stats["extra_dropped_heldout_lemma"] \
        + stats["extra_dropped_low_confidence"] == 3


def test_train_file_override(tmp_path):
    data = tmp_path / "data"
    pairs = [Pair(s + "கள்", s, "NOUN") for s in STEMS]
    for s in ("train", "dev", "test"):
        write_tsv(data / f"{s}.tsv", pairs)
    alt = tmp_path / "alt.tsv"
    write_tsv(alt, pairs[:3])
    cfg = config_from_dict({
        "name": "t", "model": {"type": "edit_classifier", "emb": 16, "channels": 16},
        "data": {"dir": str(data), "train_file": str(alt)},
        "train": {"epochs": 1, "batch_size": 8},
    })
    m = train(cfg, tmp_path / "run", verbose=False)
    assert m["test"]["acc_seen"] is not None  # seen/unseen still from gold train.tsv
