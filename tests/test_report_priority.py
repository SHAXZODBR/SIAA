"""Provisional "priority finding" rules and CLI (scripts/report_priority.py).

Synthetic only: invented Russian conclusion sentences, patient ids >= 9000000,
dates in 2031. No model, no patient data, CPU only.

Covered:
  * every priority family positive, and the same wording negated does not fire
  * sinus / scalp / orbit context excluded; brain context keeps it
  * old vs recent infarct; external (replacement) vs internal hydrocephalus;
    microadenoma alone vs mass; Fazekas grades and severity words
  * the label gate ('normal' never, 'atrophy' only with acute / mass /
    haemorrhage / demyelination); empty text
  * the join (two sections on one date, two dates, other modality / body part,
    non-brain and unmatched rows skipped)
  * CLI exit codes, atomic output, and stdout never carrying text or ids
"""
from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO

SCRIPT = REPO / "scripts" / "report_priority.py"
_spec = importlib.util.spec_from_file_location("report_priority", SCRIPT)
rp = importlib.util.module_from_spec(_spec)
sys.modules["report_priority"] = rp
_spec.loader.exec_module(rp)


def pri(text: str, label: str | None = None) -> tuple[str, tuple[str, ...]]:
    p, reasons, _, _ = rp.classify(text, label)
    return p, reasons


# ----------------------------------------------------------------------------- families

POSITIVE = [
    ("mass", "МРТ-картина объёмного образования левой лобной доли."),
    ("mass", "Менингиома в области серпа мозга."),
    ("haemorrhage", "Внутримозговая гематома правой височной доли."),
    ("haemorrhage", "Признаки субарахноидального кровоизлияния (САК)."),
    ("recent_infarct", "МР-картина острого ишемического инфаркта в бассейне левой СМА."),
    ("recent_infarct", "Ограничение диффузии в очаге правой теменной доли."),
    ("recent_infarct", "МР-признаки ОНМК по ишемическому типу в бассейне ПМА."),
    ("demyelination", "МР-картина демиелинизирующего заболевания головного мозга."),
    ("hydrocephalus", "Внутренняя окклюзионная гидроцефалия."),
    ("mass_effect", "Смещение срединных структур вправо на 6 мм."),
    ("mass_effect", "Перифокальный вазогенный отек вокруг очага."),
    ("vascular_lesion", "Аневризма правой СМА размером 4 мм."),
    ("vascular_lesion", "Артериовенозная мальформация левой теменной доли."),
    ("vascular_lesion", "Тромбоз верхнего сагиттального синуса."),
    ("infection", "МР-картина абсцесса левой лобной доли."),
    ("infection", "Изменения могут соответствовать энцефалиту."),
    ("urgent_referral", "Рекомендована срочная консультация нейрохирурга."),
    ("contrast_advised", "Рекомендовано МРТ с контрастным усилением."),
]

NEGATED = [
    "Данных за объёмное образование не получено.",
    "Данных за объемное образование головного мозга не получено.",
    "Признаков кровоизлияния не выявлено.",
    "Очагов ограничения диффузии не выявлено.",
    "Острых ишемических изменений не определяется.",
    "Признаков демиелинизации не выявлено.",
    "Гидроцефалии не выявлено.",
    "Смещения срединных структур не отмечается.",
    "МР-признаков аневризм, мальформаций не обнаружено.",
    "Абсцесс отсутствует.",
]


@pytest.mark.parametrize("reason,text", POSITIVE)
def test_family_positive(reason, text):
    p, reasons = pri(text)
    assert p == "yes"
    assert reason in reasons


@pytest.mark.parametrize("text", NEGATED)
def test_family_negated_does_not_fire(text):
    assert pri(text) == ("no", ())
    assert rp.classify(text)[3] == "no"


def test_negation_is_per_sentence():
    text = "Данных за объёмное образование не получено. Гематома левой височной доли."
    assert pri(text) == ("yes", ("haemorrhage",))


# rules-v2 (review 2026-11-14): common negation cues that rules-v1 missed.
NEGATED_V2 = [
    "Объемных образований головного мозга нет.",
    "Очагов с ограничением диффузии нет.",
    "Кровоизлияние не выявляется.",
    "Очагов ограничения диффузии не наблюдается.",
    "Объемное образование исключено.",
    "Объемных образований, кровоизлияний не выявлено.",                 # a negated list stays negated
    "Признаков объемного образования, кровоизлияния, острой ишемии не выявлено.",
]


@pytest.mark.parametrize("text", NEGATED_V2)
def test_negation_cues_added_in_rules_v2(text):
    assert pri(text, "other") == ("no", ())


def test_no_restricted_diffusion_is_not_an_acute_lesion():
    for text in ("Очаги глиоза в белом веществе, без ограничения диффузии.",
                 "Очаги глиоза без ограничения диффузии."):
        p, reasons, _, focal = rp.classify(text, "other")
        assert (p, reasons, focal) == ("no", (), "yes")


@pytest.mark.parametrize("text", [
    "Нельзя исключить демиелинизирующее заболевание.",
    "Не исключается демиелинизирующее заболевание.",
    "Не исключена демиелинизация.",
    "Для исключения демиелинизирующего заболевания рекомендована консультация.",
])
def test_possible_or_suspected_is_not_a_negation(text):
    assert "demyelination" in pri(text, "other")[1]


@pytest.mark.parametrize("text,reason", [
    ("Объемное образование левой лобной доли, признаков масс-эффекта нет.", "mass"),
    ("Объемное образование левой лобной доли, без признаков кровоизлияния.", "mass"),
    ("Объемное образование левой лобной доли без признаков кровоизлияния.", "mass"),
    ("Гематома левой височной доли, без смещения срединных структур.", "haemorrhage"),
])
def test_negated_side_clause_keeps_the_positive_finding(text, reason):
    p, reasons = pri(text, "tumor")
    assert p == "yes" and reason in reasons
    assert "mass_effect" not in reasons and ("haemorrhage" not in reasons or reason == "haemorrhage")


@pytest.mark.parametrize("text", [
    "ОНМК в анамнезе.",
    "Состояние после ОНМК.",
    "Давний инсульт в бассейне левой СМА.",
    "Рекомендована профилактика повторного ОНМК.",
    "Мальформация Киари I типа.",
    "Кефалогематома теменной области.",
    "Подапоневротическая гематома теменной области.",
    "Кистозное образование шишковидной железы.",
    "Киста эпифиза.",
])
def test_rules_v2_false_priority_reasons_removed(text):
    assert pri(text, "other") == ("no", ())


def test_vascular_malformation_still_counts_next_to_chiari_wording():
    assert "vascular_lesion" in pri("Артериовенозная мальформация мозжечка, мальформация Киари.", "other")[1]
    assert "mass" in pri("Пинеалома, кистозное образование шишковидной железы.", "tumor")[1]


@pytest.mark.parametrize("text,fires", [
    ("Динамическое наблюдение. Очаги глиоза. МРТ без контраста.", False),
    ("МРТ выполнена без контрастного усиления, рекомендовано наблюдение.", False),
    ("Образование с накоплением контраста, рекомендована консультация.", False),
    ("После введения контраста усиления нет, рекомендовано наблюдение.", False),
    ("Рекомендовано МРТ с контрастным усилением.", True),
    ("Рекомендовано:\nМРТ с контрастным усилением.", True),
    ("Рекомендовано: консультация невролога. МРТ с контрастным усилением.", True),
    ("Контрастное усиление не показано.", False),
])
def test_contrast_advised_needs_a_recommendation_in_the_same_sentence(text, fires):
    assert ("contrast_advised" in pri(text, "other")[1]) is fires


def test_reasons_are_stable_ascii_in_fixed_order():
    p, reasons = pri("Опухоль правой лобной доли со смещением срединных структур. Рекомендовано МРТ с контрастом.")
    assert p == "yes"
    assert list(reasons) == [r for r in rp.REASONS if r in reasons]
    assert set(reasons) == {"mass", "mass_effect", "contrast_advised"}
    assert all(r.isascii() and r == r.lower() and ";" not in r for r in rp.REASONS)


def test_yo_and_e_spellings_agree():
    assert pri("Объёмное образование мозжечка.") == pri("Объемное образование мозжечка.")


# ----------------------------------------------------------------------------- context

@pytest.mark.parametrize("text", [
    "Кистозное образование правой верхнечелюстной пазухи.",
    "Полип левой гайморовой пазухи.",
    "Липома мягких тканей теменной области, образование подкожно.",
    "Образование правой орбиты.",
    "Гематома мягких тканей затылочной области.",
])
def test_sinus_scalp_orbit_context_excluded(text):
    assert pri(text) == ("no", ())


def test_brain_context_overrides_extracranial_words():
    assert pri("Образование правой лобной доли с распространением в орбиту.")[0] == "yes"


def test_sinus_cyst_is_not_a_brain_cyst_but_arachnoid_cyst_is():
    assert rp.families("Киста левой височной кости, клеток сосцевидного отростка (мастоидит).")["brain_cyst"] is False
    f = rp.families("Арахноидальная киста левой височной области.")
    assert f["brain_cyst"] is True
    assert rp.classify("Арахноидальная киста левой височной области.")[0] == "no"   # focal, not priority
    assert rp.classify("Арахноидальная киста левой височной области.")[3] == "yes"


# ----------------------------------------------------------------------------- infarct / hydrocephalus / adenoma

def test_old_infarct_is_focal_not_priority():
    for text in ("Последствия перенесённого ОНМК в бассейне левой СМА.",
                 "Постишемические кистозно-глиозные изменения правой теменной доли."):
        p, reasons, _, focal = rp.classify(text)
        assert p == "no", text
        assert focal == "yes", text
    assert pri("Хронический инсульт, кистозная трансформация.") == ("no", ())


def test_recent_stroke_without_acute_word_is_priority():
    assert pri("МР-картина ОНМК по ишемическому типу в левой гемисфере мозжечка.") == ("yes", ("recent_infarct",))


def test_external_replacement_hydrocephalus_is_not_priority():
    for text in ("Наружная заместительная гидроцефалия.", "Заместительная гидроцефалия.",
                 "Умеренная наружная гидроцефалия."):
        assert pri(text) == ("no", ()), text
    assert pri("Наружная и внутренняя гидроцефалия.") == ("yes", ("hydrocephalus",))
    assert pri("Тривентрикулярная гидроцефалия.") == ("yes", ("hydrocephalus",))


def test_microadenoma_alone_vs_mass():
    p, reasons, _, focal = rp.classify("МР-картина микроаденомы гипофиза.")
    assert (p, focal) == ("no", "yes")
    assert pri("Микроаденома гипофиза, образование 5 мм.") == ("no", ())
    assert pri("Объёмное образование гипофиза (макроаденома).")[1] == ("mass",)
    # mass effect makes it priority; a microadenoma-only sentence never fires 'mass' itself
    assert pri("Микроаденома гипофиза с масс-эффектом на хиазму.") == ("yes", ("mass_effect",))
    assert pri("Микроаденома гипофиза, образование с масс-эффектом на хиазму.") == ("yes", ("mass", "mass_effect"))
    assert pri("Микроаденома гипофиза. Опухоль левой лобной доли.") == ("yes", ("mass",))


# ----------------------------------------------------------------------------- grade

@pytest.mark.parametrize("text,grade", [
    ("Микроангиопатия (Fazekas 1).", 1),
    ("Микроангиопатия Fazekas II.", 2),
    ("Признаки микроангиопатии головного мозга, Fazekas III.", 3),
    ("Микроангиопатия (Fazekas I), выраженная атрофия.", 1),        # a written Fazekas grade wins
    ("Незначительные проявления атрофии.", 1),
    ("Начальные проявления дисциркуляторной энцефалопатии.", 1),
    ("Умеренная атрофия головного мозга.", 2),
    ("Умеренно выраженная атрофия коры.", 2),
    ("Выраженная атрофия головного мозга.", 3),
    ("Невыраженная атрофия головного мозга.", 0),
    ("Слабовыраженная атрофия.", 1),
    ("Выраженное образование левой лобной доли.", 0),
    ("Микроангиопатия Fazekas I-II.", 2),                            # a range counts as its upper end
    ("Микроангиопатия Fazekas II-III.", 3),
    ("Микроангиопатия Fazekas 1/2.", 2),
    ("Микроангиопатия по шкале Фазекас 2.", 2),
    ("Незначительно выраженные признаки микроангиопатии.", 1),
    ("Нерезко выраженная атрофия.", 1),
    ("Маловыраженная атрофия.", 1),                # no white-matter / atrophy wording
    ("Объёмное образование не выявлено.", 0),
])
def test_wm_grade(text, grade):
    assert rp.classify(text)[2] == grade


# ----------------------------------------------------------------------------- label gate / empty

def test_label_normal_is_never_priority():
    p, reasons, grade, focal = rp.classify("Опухоль правой лобной доли.", "normal")
    assert (p, reasons, focal) == ("no", (), "yes")


def test_label_atrophy_gate():
    text_contrast = "Умеренная атрофия. Рекомендовано МРТ с контрастным усилением."
    assert pri(text_contrast) == ("yes", ("contrast_advised",))
    assert pri(text_contrast, "atrophy") == ("no", ())
    assert pri("Атрофия. Аневризма правой СМА.", "atrophy") == ("no", ())
    assert pri("Атрофия. Аневризма правой СМА.", "other")[0] == "yes"
    assert pri("Атрофия. Острый инфаркт левой теменной доли.", "atrophy") == ("yes", ("recent_infarct",))
    assert pri("Атрофия. Гемосидерин в правой лобной доле.", "atrophy") == ("yes", ("haemorrhage",))
    assert pri("Атрофия. Демиелинизирующее заболевание.", "atrophy") == ("yes", ("demyelination",))
    # the gate uses the raw mass family: a microadenoma + 'образование' counts for atrophy
    assert pri("Атрофия. Микроаденома гипофиза, образование 5 мм.", "atrophy") == ("yes", ("mass",))


@pytest.mark.parametrize("text", [None, "", "   ", "\n"])
def test_empty_text(text):
    assert rp.classify(text, "stroke") == ("", (), None, "")


# ----------------------------------------------------------------------------- join + CLI

MATCHED_COLS = ["study_folder", "patient_id", "modality", "study_date", "match_quality", "body_part_dicom",
                "is_brain", "report_matched", "report_body_part", "label", "abnormal", "label_source",
                "label_confidence", "needs_review", "report_modality", "report_study_date"]
LABEL_COLS = ["patient_id", "modality", "body_part", "study_date", "label", "conclusion", "description",
              "source_file", "section_index", "section_header", "abnormal", "label_confidence", "label_source",
              "needs_review"]

TEXT_TUMOR = "Объёмное образование левой лобной доли с перифокальным отеком."
TEXT_ATROPHY = "Умеренная атрофия головного мозга."
TEXT_SPINE = "Грыжа диска L5-S1."
TEXT_NORMAL = "МРТ-признаков структурных изменений головного мозга не выявлено."


def _write(path: Path, cols: list[str], rows: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


def _m(folder, pid, label, date="2031-03-02", mod="MR", brain="yes", matched="yes", bp="brain"):
    return {"study_folder": folder, "patient_id": pid, "modality": mod, "study_date": date.replace("-", ""),
            "match_quality": "date", "is_brain": brain, "report_matched": matched, "report_body_part": bp,
            "label": label, "report_modality": mod if matched == "yes" else "",
            "report_study_date": date if matched == "yes" else ""}


def _l(pid, date, label, text, section, mod="MR", bp="brain"):
    return {"patient_id": pid, "modality": mod, "body_part": bp, "study_date": date, "label": label,
            "conclusion": text, "section_index": str(section), "source_file": "synthetic.pdf"}


@pytest.fixture
def batch(tmp_path):
    matched = [
        _m("S_A", "9000001", "tumor"),                                  # two sections on one date
        _m("S_B", "9000002", "atrophy", date="2031-05-06"),             # two dates for the patient
        _m("S_C", "9000003", "normal"),                                 # two brain sections, label tie -> lowest index
        _m("S_D", "9000004", "", brain="no", bp="spine"),               # not a brain study
        _m("S_E", "9000005", "", matched="no"),                         # no report
        _m("S_F", "9000006", "tumor", mod="CT"),                        # other modality
        _m("S_G", "9000007", "stroke"),                                 # report row missing from labels
        _m("S_A", "9000001", "tumor"),                                  # duplicate folder
    ]
    labels = [
        _l("9000001", "2031-03-02", "", TEXT_SPINE, 1, bp="spine"),
        _l("9000001", "2031-03-02", "atrophy", TEXT_ATROPHY, 2),
        _l("9000001", "2031-03-02", "tumor", TEXT_TUMOR, 3),
        _l("9000002", "2031-03-02", "tumor", TEXT_TUMOR, 1),
        _l("9000002", "2031-05-06", "atrophy", TEXT_ATROPHY, 1),
        _l("9000003", "2031-03-02", "normal", TEXT_NORMAL, 4),
        _l("9000003", "2031-03-02", "normal", TEXT_TUMOR, 2),
        _l("9000006", "2031-03-02", "tumor", TEXT_TUMOR, 1, mod="CT"),
        _l("9000006", "2031-03-02", "tumor", TEXT_ATROPHY, 2, mod="MR"),
    ]
    return (_write(tmp_path / "m.csv", MATCHED_COLS, matched), _write(tmp_path / "l.csv", LABEL_COLS, labels),
            tmp_path / "out" / "priority.csv")


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True,
                          cwd=str(REPO), timeout=60)


def _read(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == list(rp.OUTPUT_COLUMNS)
    return {r["study_folder"]: r for r in rows}


def test_join_and_cli(batch):
    m, l, out = batch
    res = _run("--matched", m, "--labels", l, "--out", out)
    assert res.returncode == 0, res.stderr
    rows = _read(out)
    assert set(rows) == {"S_A", "S_B", "S_C", "S_G"}
    a, b, c, g = rows["S_A"], rows["S_B"], rows["S_C"], rows["S_G"]
    # same label beats the lower section_index; the spine section is never a candidate
    assert (a["priority"], a["reasons"], a["focal"]) == ("yes", "mass;mass_effect", "yes")
    # the report of the matched date is used, not the other date's tumour report
    assert (b["priority"], b["reasons"], b["wm_grade"], b["focal"]) == ("no", "", "2", "no")
    assert b["report_study_date"] == "2031-05-06"
    # label 'normal' on both sections -> lowest section_index (the tumour text), but the gate says no
    assert (c["priority"], c["focal"]) == ("no", "yes")
    # report row missing -> unknown
    assert (g["priority"], g["reasons"], g["wm_grade"], g["focal"]) == ("", "", "", "")
    assert rp.RULES_VERSION == "rules-v2"
    assert all(r["rules_version"] == rp.RULES_VERSION for r in rows.values())
    assert a["patient_id"] == "9000001"
    # stdout: counts only, never ids, folders or report text
    for secret in ("9000001", "9000002", "9000003", "9000007", "S_A", "S_B", "2031",
                   "образование", "атрофия", "Грыжа", "МРТ-признаков"):
        assert secret not in res.stdout, secret
    assert "priority yes 1" in res.stdout
    assert not out.with_name(out.name + ".tmp").exists()


def test_modality_option_selects_ct(batch):
    m, l, out = batch
    res = _run("--matched", m, "--labels", l, "--out", out, "--modality", "CT")
    assert res.returncode == 0, res.stderr
    rows = _read(out)
    assert set(rows) == {"S_F"}
    assert rows["S_F"]["priority"] == "yes"            # the CT section, not the MR one


def test_no_qualifying_study_writes_header_only(tmp_path):
    m = _write(tmp_path / "m.csv", MATCHED_COLS, [_m("S_X", "9000009", "", brain="no", bp="spine")])
    l = _write(tmp_path / "l.csv", LABEL_COLS, [_l("9000009", "2031-03-02", "", TEXT_SPINE, 1, bp="spine")])
    out = tmp_path / "p.csv"
    res = _run("--matched", m, "--labels", l, "--out", out)
    assert res.returncode == 0, res.stderr
    assert out.read_text(encoding="utf-8").strip() == ",".join(rp.OUTPUT_COLUMNS)


def test_bom_in_input_csv_is_accepted(batch, tmp_path):
    """A csv re-saved by Excel starts with a UTF-8 BOM; the first column must still be found."""
    m, l, out = batch
    for src in (m, l):
        src.write_bytes(b"\xef\xbb\xbf" + src.read_bytes())
    res = _run("--matched", m, "--labels", l, "--out", out)
    assert res.returncode == 0, res.stderr
    assert _read(out)["S_A"]["priority"] == "yes"


@pytest.mark.parametrize("case", ["missing_matched", "missing_labels", "matched_cols", "labels_cols", "modality"])
def test_unusable_input_exit_1_nothing_written(batch, tmp_path, case):
    m, l, out = batch
    extra: list[str] = []
    if case == "missing_matched":
        m = tmp_path / "nope.csv"
    elif case == "missing_labels":
        l = tmp_path / "nope.csv"
    elif case == "matched_cols":
        m = _write(tmp_path / "bad_m.csv", ["study_folder", "patient_id"], [{"study_folder": "S", "patient_id": "9000001"}])
    elif case == "labels_cols":
        l = _write(tmp_path / "bad_l.csv", ["patient_id", "study_date"], [{"patient_id": "9000001", "study_date": "2031-03-02"}])
    else:
        extra = ["--modality", "XR"]
    res = _run("--matched", m, "--labels", l, "--out", out, *extra)
    assert res.returncode == 1
    assert res.stderr.strip()
    assert "9000001" not in res.stderr + res.stdout
    assert not out.exists()
