import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "build_ud", Path(__file__).parents[1] / "scripts/data/build_ud.py"
)
build_ud = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_ud)

CONLLU = """# sent_id = 1
1-2\tஉறுதியளித்துள்ளார்\t_\t_\t_\t_\t_\t_\t_\t_
1\tஉறுதியளித்த்\tஉறுதியளி\tVERB\t_\t_\t0\troot\t_\t_
2\tஉள்ளார்\tஉள்\tAUX\t_\t_\t1\taux\t_\t_
3\tவீட்டை\tவீடு\tNOUN\t_\t_\t1\tobj\t_\t_
4-5\tஅமைக்கப்பட\t_\t_\t_\t_\t_\t_\t_\t_
4\tஅமைக்க\tஅமை\tVERB\t_\t_\t1\txcomp\t_\t_
5\tப்பட\tபடு\tAUX\t_\t_\t4\taux\t_\t_
6\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_

"""


def test_read_conllu_keeps_written_words(tmp_path):
    path = tmp_path / "x.conllu"
    path.write_text(CONLLU, encoding="utf-8")
    assert build_ud.read_conllu(path) == [
        ("உறுதியளித்துள்ளார்", "உறுதியளி", "VERB"),
        ("வீட்டை", "வீடு", "NOUN"),
        ("அமைக்கப்பட", "அமை", "VERB"),
        (".", ".", "PUNCT"),
    ]
