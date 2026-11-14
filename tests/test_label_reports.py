"""Tests for scripts/label_reports.py (stage 2 "label" of the release pipeline).

All fixtures are SYNTHETIC Russian conclusions written for the tests; the one
test that touches the real batch-1 reference (data/matched_clean.csv +
data/hospital_reports_labels.csv) skips when those files are absent and never
prints anything from them.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))

import label_reports as L  # noqa: E402

PY = sys.executable
SCRIPT = REPO / 'scripts' / 'label_reports.py'


def lab(text: str):
    a = L.analyse_conclusion(text)
    return a.label, a.abnormal


# --------------------------------------------------------------------- negation
def test_global_negation_is_normal():
    a = L.analyse_conclusion('МРТ-признаков наличия структурных изменений головного мозга не выявлено.')
    assert (a.label, a.abnormal) == ('normal', 'no')
    assert a.confidence == 'high' and a.needs_review == 'no'


def test_comma_list_negation_covers_all_keywords():
    assert lab('Очагов ишемии, кровоизлияний, объемных образований не выявлено.') == ('normal', 'no')
    assert lab('МРТ- признаки окклюзии, аневризм магистральных артерий, АВМ не выявлено.') == ('normal', 'no')


def test_local_negation_does_not_hide_the_finding():
    assert lab('Объемное образование левой лобной доли, срединные структуры не смещены.') == ('tumor', 'yes')
    assert lab('Ишемический инсульт в бассейне левой СМА. Признаков кровоизлияния не выявлено.') == ('stroke', 'yes')


def test_hedge_is_not_a_negation():
    assert lab('Единичные очаги в белом веществе обоих полушарий (не исключается демиелинизирующий процесс).') == ('ms_wml', 'yes')
    assert lab('Неоднородность структуры гипофиза (не исключается микроаденома).') == ('tumor', 'yes')


# ----------------------------------------- review regressions (negation scope)
# Verbatim reviewer examples (review_findings.json, scripts/label_reports.py:375/384/135):
# each used to come out normal / abnormal=no / high confidence / no review.
@pytest.mark.parametrize('text,expected', [
    ('МР-картина объемного образования левой лобной доли, перифокальный отек не выражен.', 'tumor'),
    ('Внутримозговая гематома правой теменной доли, отек не выражен.', 'hemorrhage'),
    ('Объемное образование правой височной доли, смещения срединных структур не выявлено.', 'tumor'),
    ('Ишемический инсульт в бассейне правой позвоночной артерии.', 'stroke'),
    # further phrasings quoted in the same findings
    ('МР-признаки арахноидальной кисты левой височной области, очаговых изменений не выявлено.', 'cyst'),
    ('Внутримозговая гематома левой лобной доли, прорыва крови в желудочки не выявлено.', 'hemorrhage'),
    ('Ишемический инсульт в бассейне левой СМА, геморрагической трансформации не выявлено.', 'stroke'),
    ('Киста левой височной доли, перифокального отека не выявлено.', 'cyst'),
    ('Острый инфаркт в вертебро-базилярном бассейне.', 'stroke'),
])
def test_negated_tail_never_swallows_the_asserted_finding(text, expected):
    a = L.analyse_conclusion(text)
    assert (a.label, a.abnormal) == (expected, 'yes'), (a.label, a.negated)


def test_genitive_negated_lists_are_still_fully_negated():
    # every item of a list governed by one "не выявлено" is negated, side-effect words included
    for t in ('Очагов ишемии, кровоизлияний, объемных образований не выявлено.',
              'Очагов ишемии, кровоизлияний, отека, смещения срединных структур не выявлено.',
              'МРТ-признаков окклюзий, стенозов, аневризм не выявлено.'):
        a = L.analyse_conclusion(t)
        assert (a.label, a.abnormal) == ('normal', 'no'), t
    # the short list tail may reach over "признаки окклюзии" (vascular list) but flags it for review
    a = L.analyse_conclusion('МРТ- признаки окклюзии, аневризм магистральных артерий, АВМ не выявлено.')
    assert (a.label, a.abnormal, a.needs_review) == ('normal', 'no', 'yes') and a.confidence != 'high'
    # ...and never after a side-effect tail: post-operative state stays abnormal
    assert lab('Состояние после удаления опухоли, отек не выражен.') == ('other', 'yes')


@pytest.mark.parametrize('text', [
    'Нельзя исключить объемное образование правой височной доли.',
    'Не исключается объемное образование правой височной доли.',
    'Не исключено объемное образование правой височной доли.',
    'Объемное образование правой височной доли под вопросом.',
    'Объемное образование (?) правой височной доли.',
    'Объемное образование правой височной доли?',
])
def test_hedged_finding_is_kept_and_sent_to_review(text):
    a = L.analyse_conclusion(text)
    assert (a.label, a.abnormal) == ('tumor', 'yes'), (text, a.negated)
    assert a.confidence in ('medium', 'low') and a.needs_review == 'yes'


def test_hedge_in_parenthesis_does_not_demote_a_certain_finding():
    a = L.analyse_conclusion('Единичные очаги в белом веществе обоих полушарий (не исключается демиелинизирующий процесс).')
    assert a.label == 'ms_wml' and a.confidence == 'high' and a.needs_review == 'no'
    # "исключено" / "исключается" are still negations
    assert lab('Объемное образование исключено.') == ('normal', 'no')
    assert lab('Ишемия исключается.') == ('normal', 'no')


def test_spine_is_non_brain_but_vertebral_artery_territory_is_brain():
    assert lab('Остеохондроз шейного отдела позвоночника. Грыжа межпозвонкового диска C5-C6.') == ('normal', 'no')
    assert lab('Компрессионный перелом тела позвонка Th12.') == ('normal', 'no')
    assert not L.is_non_brain_clause('ишемический инсульт в бассейне правой позвоночной артерии')
    assert L.is_non_brain_clause('гемангиома тела позвонка th12')


def test_clause_head_case():
    for c in ('очагов ишемии', 'кровоизлияний', 'объемных образований', 'окклюзии', 'аневризм магистральных артерий',
              'смещения срединных структур', 'стенозов'):
        assert L.clause_head_case(c) == 'gen', c
    for c in ('объемное образование правой височной доли', 'внутримозговая гематома', 'ишемический инсульт',
              'киста левой височной доли', 'мр-картина объемного образования', 'головной мозг обычной формы'):
        assert L.clause_head_case(c) == 'nom', c


def test_boilerplate_is_stripped_before_scoring():
    text = ('МРТ-признаков структурных изменений головного мозга не выявлено. Рекомендовано консультация '
            'невролога Заключение не является диагнозом и требует консультации. Результат подготовил(а): '
            'Ф.И.О. врача клинико-патологическая корреляция необходима')
    a = L.analyse_conclusion(text)
    assert (a.label, a.abnormal) == ('normal', 'no')
    assert 'подготовил' not in a.text and 'клинико-патол' not in a.text


# ----------------------------------------------------------- sinus/scalp → normal
@pytest.mark.parametrize('text', [
    'Двусторонний гайморит. Киста левой гайморовой пазухи.',
    'Утолщение (отек) слизистой оболочки основной пазухи. Искривление носовой перегородки.',
    'Подкожная гематома теменной области справа.',
    'Липома мягких тканей головы лобной области.',
    'МРТ-признаков структурных изменений головного мозга не выявлено. Косвенные признаки внутричерепной гипертензии. Пансинусит.',
    'Гипоплазия интракраниального сегмента правой позвоночной артерии.',
])
def test_non_brain_findings_are_normal(text):
    assert lab(text) == ('normal', 'no')


def test_sinus_cyst_is_not_a_brain_cyst_but_pineal_cyst_is():
    assert lab('Киста правой гайморовой пазухи.') == ('normal', 'no')
    assert lab('Арахноидальная киста средней черепной ямки слева.') == ('cyst', 'yes')
    assert lab('МРТ-признаков структурных изменений головного мозга не выявлено. Киста шишковидной железы.') == ('cyst', 'yes')


# -------------------------------------------------------------------- classes
@pytest.mark.parametrize('text,expected', [
    ('МРТ-признаки менингиомы левой лобной области с перифокальным отеком.', 'tumor'),
    ('МРТ-признаки внутримозгового образования височной доли справа.', 'tumor'),
    ('Множественное метастатическое поражение головного мозга.', 'tumor'),
    ('Внутримозговое кровоизлияние в правой височной доле.', 'hemorrhage'),
    ('Хроническая субдуральная гематома слева.', 'hemorrhage'),
    ('Лакунарный инфаркт в области базальных ядер слева.', 'stroke'),
    ('МРТ-признаки участков и очагов ишемии (в острой стадии) в левой лобно-теменной области.', 'stroke'),
    ('Выраженная внутренняя гидроцефалия.', 'hydrocephalus'),
    ('Демиелинизирующее заболевание головного мозга (рассеянный склероз).', 'ms_wml'),
    ('Единичные очаги патологической интенсивности в белом веществе обоих полушарий (сосудистого генеза).', 'ms_wml'),
    ('МРТ – признаки церебральной микроангиопатии – Fazekas Grade I (Small Vessel Disease).', 'atrophy'),
    ('Умеренная атрофия лобно-теменных областей с обеих сторон.', 'atrophy'),
    ('МРТ-признаки дисциркуляторной энцефалопатии.', 'atrophy'),
    ('МРТ-признаки гипоксическо-ишемической энцефалопатии.', 'atrophy'),
    ('Артериовенозная мальформация левой теменной доли.', 'other'),
    ('Аномалия Киари 1 типа. Эктопия миндалин мозжечка.', 'other'),
    ('Состояние после удаления объемного образования левой лобной доли. Кистозно-глиозные изменения.', 'other'),
])
def test_single_finding_classes(text, expected):
    a = L.analyse_conclusion(text)
    assert a.label == expected and a.abnormal == 'yes'


# ------------------------------------------------------------- severity order
def test_severity_picks_the_single_most_significant_finding():
    assert lab('Менингиома правой лобной области. Церебральная микроангиопатия Fazekas I. Атрофия.') == ('tumor', 'yes')
    assert lab('Ишемический инсульт левой СМА. Единичные очаги в белом веществе. Гайморит.') == ('stroke', 'yes')
    assert lab('Внутримозговая гематома. Лакунарный инфаркт.') == ('hemorrhage', 'yes')
    assert lab('Умеренная гидроцефалия. Атрофия лобных долей.') == ('hydrocephalus', 'yes')
    assert L.SEVERITY == ('tumor', 'hemorrhage', 'stroke', 'hydrocephalus', 'ms_wml', 'cyst', 'atrophy', 'other', 'normal')


def test_multi_finding_confidence_and_review():
    one = L.analyse_conclusion('МРТ – признаки церебральной микроангиопатии – Fazekas Grade I.')
    assert one.confidence == 'high' and one.needs_review == 'no'
    multi = L.analyse_conclusion('Церебральная микроангиопатия Fazekas I. Арахноидальная киста задней черепной ямки.')
    assert multi.label == 'cyst' and multi.confidence == 'medium' and multi.needs_review == 'no'
    other = L.analyse_conclusion('Артериовенозная мальформация левой теменной доли.')
    assert other.label == 'other' and other.needs_review == 'yes'
    unmatched = L.analyse_conclusion('Особенности строения, требующие сопоставления с анамнестическими данными пациента и клиникой.')
    assert unmatched.confidence == 'low' and unmatched.needs_review == 'yes'
    assert L.analyse_conclusion('').needs_review == 'yes'


# ------------------------------------------------------------- row labelling
def _rows():
    return [
        {'patient_id': '9000001', 'modality': 'MR', 'body_part': 'brain', 'study_date': '2031-07-01',
         'section_index': '0', 'conclusion': 'МРТ-признаков структурных изменений головного мозга не выявлено.'},
        {'patient_id': '9000002', 'modality': 'MR', 'body_part': 'brain', 'study_date': '2031-07-02',
         'section_index': '0', 'conclusion': 'Артериовенозная мальформация левой теменной доли.'},
        {'patient_id': '9000003', 'modality': 'MR', 'body_part': 'spine', 'study_date': '2031-07-03',
         'section_index': '0', 'conclusion': 'Грыжа диска L5-S1.'},
        {'patient_id': '9000002', 'modality': 'CT', 'body_part': 'brain', 'study_date': '2031-08-09',
         'section_index': '0', 'conclusion': 'Внутримозговое кровоизлияние в правой височной доле.'},
    ]


def test_label_row_leaves_non_brain_rows_empty():
    rows = [L.label_row(r) for r in _rows()]
    assert rows[0]['label'] == 'normal' and rows[0]['abnormal'] == 'no' and rows[0]['label_source'] == 'rules'
    assert rows[2]['label'] == '' and rows[2]['abnormal'] == '' and rows[2]['needs_review'] == ''
    assert rows[3]['label'] == 'hemorrhage'


def _jsonl(path: Path, objs) -> Path:
    path.write_text(''.join(json.dumps(o, ensure_ascii=False) + '\n' for o in objs), encoding='utf-8')
    return path


def test_export_import_round_trip(tmp_path: Path):
    rows = [L.label_row(r) for r in _rows()]
    out = tmp_path / 'review.jsonl'
    n = L.export_for_llm(rows, out)                       # only needs_review=yes rows
    lines = [json.loads(l) for l in out.read_text(encoding='utf-8').splitlines()]
    assert n == 1 and lines[0] == {'patient_id': '9000002', 'study_date': '2031-07-02', 'modality': 'MR',
                                   'section_index': '0',
                                   'conclusion': 'артериовенозная мальформация левой теменной доли.'}
    assert L.export_for_llm(rows, tmp_path / 'all.jsonl', all_rows=True) == 3   # brain rows only

    # what training/relabel_workflow.js returns → --import-llm-labels (legacy lines without
    # modality/section_index still match through the unique (patient_id, study_date) fallback)
    llm = _jsonl(tmp_path / 'llm.jsonl', [
        {'patient_id': '9000002', 'study_date': '2031-07-02', 'modality': 'MR', 'section_index': '0',
         'label': 'other', 'abnormal': True},
        {'patient_id': '9000002', 'study_date': '20310809', 'label': 'hemorrhage', 'abnormal': 'yes'},
        {'patient_id': '9999999', 'study_date': '', 'label': 'tumor', 'abnormal': True},
        {'patient_id': '9000001', 'study_date': '2031-07-01', 'label': 'bogus', 'abnormal': True},
    ])
    res = L.import_llm_labels(rows, L.read_jsonl(llm))
    assert (res.applied, res.skipped, res.ambiguous) == (2, 2, 0)
    assert rows[1]['label_source'] == 'import' and rows[1]['needs_review'] == 'no' and rows[1]['label'] == 'other'
    assert rows[3]['label_source'] == 'import' and rows[3]['abnormal'] == 'yes'
    assert rows[0]['label_source'] == 'rules'

    # a whole workflow result pasted as one line is accepted too
    one_line = tmp_path / 'wf.jsonl'
    one_line.write_text(json.dumps({'labels': [{'patient_id': '9000001', 'study_date': '2031-07-01',
                                                'label': 'normal', 'abnormal': False}]}), encoding='utf-8')
    assert tuple(L.import_llm_labels(rows, L.read_jsonl(one_line))) == (1, 0, 0)


# ---------------------------------------- import key: CT + MR brain on one day
def _same_day_rows():
    return [L.label_row(r) for r in [
        {'patient_id': '9000111', 'modality': 'CT', 'body_part': 'brain', 'study_date': '2031-07-21',
         'section_index': '0', 'conclusion': 'КТ-признаков патологии головного мозга не выявлено.'},
        {'patient_id': '9000111', 'modality': 'MR', 'body_part': 'brain', 'study_date': '2031-07-21',
         'section_index': '1', 'conclusion': 'Особенности строения, требующие сопоставления с клиникой.'},
    ]]


@pytest.mark.parametrize('order', [(0, 1), (1, 0)])
def test_import_keys_on_modality_and_section(tmp_path: Path, order):
    """Review finding line 661: CT-brain and MR-brain on the same day must not overwrite each other."""
    rows = _same_day_rows()
    exp = tmp_path / 'review.jsonl'
    assert L.export_for_llm(rows, exp, all_rows=True) == 2
    exported = [json.loads(l) for l in exp.read_text(encoding='utf-8').splitlines()]
    assert [(e['modality'], e['section_index']) for e in exported] == [('CT', '0'), ('MR', '1')]
    # the workflow echoes the keys unchanged and adds label/abnormal
    answers = [{**{k: e[k] for k in ('patient_id', 'study_date', 'modality', 'section_index')},
                'label': lab, 'abnormal': lab != 'normal'} for e, lab in zip(exported, ['normal', 'stroke'])]
    res = L.import_llm_labels(rows, [answers[i] for i in order])
    assert (res.applied, res.skipped, res.ambiguous) == (2, 0, 0)
    assert (rows[0]['label'], rows[0]['abnormal'], rows[0]['label_source']) == ('normal', 'no', 'import')
    assert (rows[1]['label'], rows[1]['abnormal'], rows[1]['label_source']) == ('stroke', 'yes', 'import')


def test_import_fallbacks_are_unique_only():
    rows = _same_day_rows()
    before = [dict(r) for r in rows]
    # (patient_id, study_date) matches 2 rows → ambiguous, nothing applied
    res = L.import_llm_labels(rows, [{'patient_id': '9000111', 'study_date': '2031-07-21',
                                      'label': 'stroke', 'abnormal': True}])
    assert (res.applied, res.skipped, res.ambiguous) == (0, 0, 1)
    assert rows == before
    # (patient_id, study_date, modality) is unique → applied to the MR row only
    res = L.import_llm_labels(rows, [{'patient_id': '9000111', 'study_date': '2031-07-21', 'modality': 'mr',
                                      'label': 'stroke', 'abnormal': True}])
    assert tuple(res) == (1, 0, 0) and rows[1]['label'] == 'stroke' and rows[0]['label_source'] == 'rules'
    # unknown section_index falls back to the unique modality level; no date → only a single-row patient
    rows = _same_day_rows()
    res = L.import_llm_labels(rows, [
        {'patient_id': '9000111', 'study_date': '2031-07-21', 'modality': 'CT', 'section_index': '7',
         'label': 'normal', 'abnormal': False},
        {'patient_id': '9000111', 'study_date': '', 'label': 'tumor', 'abnormal': True},
    ])
    assert tuple(res) == (1, 0, 1) and rows[0]['label_source'] == 'import' and rows[1]['label_source'] == 'rules'


def test_import_never_applies_two_labels_to_one_row():
    rows = _same_day_rows()
    key = {'patient_id': '9000111', 'study_date': '2031-07-21', 'modality': 'MR', 'section_index': '1'}
    res = L.import_llm_labels(rows, [{**key, 'label': 'stroke', 'abnormal': True},
                                     {**key, 'label': 'stroke', 'abnormal': True},      # duplicate
                                     {**key, 'label': 'normal', 'abnormal': False}])    # conflict
    assert tuple(res) == (1, 1, 1)
    assert rows[1]['label'] == 'stroke' and rows[1]['needs_review'] == 'yes'   # conflict → back to review
    assert rows[0]['label_source'] == 'rules'


def _two_rows_one_patient(date):
    return [L.label_row(r) for r in [
        {'patient_id': '9000401', 'modality': 'MR', 'body_part': 'brain', 'study_date': date,
         'section_index': '1', 'conclusion': 'МРТ-признаков структурных изменений головного мозга не выявлено.'},
        {'patient_id': '9000401', 'modality': 'CT', 'body_part': 'brain', 'study_date': date,
         'section_index': '1', 'conclusion': 'Объемное образование левой лобной доли (глиома).'},
    ]]


@pytest.mark.parametrize('date', ['', '2031-07-22'])
def test_import_exact_key_with_an_empty_study_date(date):
    """Review 2026-11-11 #6.1: the parser can emit study_date=''. A collector line carries all four key
    fields, so it must match its row exactly even when the date is empty — not fall back to the patient
    id alone, where two rows make it ambiguous and the readers' split / overrule is silently lost."""
    rows = _two_rows_one_patient(date)
    assert [(r['label'], r['needs_review']) for r in rows] == [('normal', 'no'), ('tumor', 'no')]
    key = {'patient_id': '9000401', 'study_date': date, 'section_index': '1'}
    res = L.import_llm_labels(rows, [
        {**key, 'modality': 'MR', 'label': 'tumor', 'abnormal': True, 'needs_review': 'yes'},   # readers split
        {**key, 'modality': 'CT', 'label': 'normal', 'abnormal': False, 'needs_review': 'no'},  # overruled
    ])
    assert tuple(res) == (2, 0, 0)
    assert [(r['label'], r['abnormal'], r['needs_review'], r['label_source']) for r in rows] == [
        ('tumor', 'yes', 'yes', 'import'), ('normal', 'no', 'no', 'import')]


def test_import_undated_line_with_modality_only_uses_the_three_part_key():
    rows = _two_rows_one_patient('')
    res = L.import_llm_labels(rows, [{'patient_id': '9000401', 'study_date': '', 'modality': 'CT',
                                      'label': 'normal', 'abnormal': False}])
    assert tuple(res) == (1, 0, 0) and rows[1]['label'] == 'normal' and rows[0]['label_source'] == 'rules'


@pytest.mark.parametrize('line', [{'patient_id': '9000401'}, {'patient_id': '9000401', 'study_date': ''}])
def test_import_legacy_patient_only_line_is_unchanged(line):
    """A legacy line without modality/section_index still falls back to the patient id alone."""
    rows = _two_rows_one_patient('')
    before = [dict(r) for r in rows]
    assert tuple(L.import_llm_labels(rows, [{**line, 'label': 'tumor', 'abnormal': True}])) == (0, 0, 1)
    assert rows == before
    one = rows[:1]
    assert tuple(L.import_llm_labels(one, [{**line, 'label': 'tumor', 'abnormal': True}])) == (1, 0, 0)
    assert one[0]['label'] == 'tumor' and one[0]['label_source'] == 'import'


def test_cli_end_to_end(tmp_path: Path):
    src = tmp_path / 'reports.csv'
    cols = ['patient_id', 'modality', 'body_part', 'study_date', 'conclusion', 'description',
            'source_file', 'section_index', 'label']
    with open(src, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in _rows():
            w.writerow({**{c: '' for c in cols}, **r})
    out = tmp_path / 'labels.csv'
    exp = tmp_path / 'review.jsonl'
    rc = L.main(['--reports', str(src), '--out', str(out), '--export-for-llm', str(exp)])
    assert rc == 0
    with open(out, newline='', encoding='utf-8') as fh:
        r = csv.DictReader(fh)
        got = list(r)
        assert all(c in r.fieldnames for c in L.OUTPUT_COLS)
        assert [x['label'] for x in got] == ['normal', 'other', '', 'hemorrhage']
        assert [x['abnormal'] for x in got] == ['no', 'yes', '', 'yes']
    exported = [json.loads(l) for l in exp.read_text(encoding='utf-8').splitlines()]
    assert len(exported) == 1 and exported[0]['modality'] == 'MR' and exported[0]['section_index'] == '0'
    # round trip through the CLI import: the counts line reports applied / ambiguous / skipped
    llm = _jsonl(tmp_path / 'llm.jsonl', [
        {**{k: exported[0][k] for k in ('patient_id', 'study_date', 'modality', 'section_index')},
         'label': 'other', 'abnormal': True},
        {'patient_id': '9000002', 'study_date': '', 'label': 'tumor', 'abnormal': True},   # 2 brain rows → ambiguous
    ])
    proc = subprocess.run([PY, str(SCRIPT), '--reports', str(src), '--out', str(out),
                           '--import-llm-labels', str(llm)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert '1 applied, 1 ambiguous, 0 skipped' in proc.stdout
    # the Ollama path must degrade gracefully when no server answers
    rc = L.main(['--reports', str(src), '--out', str(out), '--llm', 'ollama:test-model',
                 '--ollama-url', 'http://127.0.0.1:9/api/generate'])
    assert rc == 0


# ------------------------------------------------------------------ benchmark
def _write_reference(tmp_path: Path, rows):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ref = tmp_path / 'matched_clean.csv'
    rep = tmp_path / 'hospital_reports_labels.csv'
    with open(ref, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['study_folder', 'patient_id', 'label', 'abnormal'])
        w.writeheader()
        for pid, label, abn, _ in rows:
            w.writerow({'study_folder': 'x', 'patient_id': pid, 'label': label, 'abnormal': abn})
        w.writerow({'study_folder': 'y', 'patient_id': '9000077', 'label': '', 'abnormal': ''})   # unlabelled
    with open(rep, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['patient_id', 'modality', 'body_part', 'study_date', 'label', 'conclusion'])
        w.writeheader()
        for pid, _, _, text in rows:
            w.writerow({'patient_id': pid, 'modality': 'МРТ', 'body_part': 'brain', 'study_date': '',
                        'label': '', 'conclusion': text + ' Результат подготовил(а): Имя Врача'})
        w.writerow({'patient_id': '9000077', 'modality': 'МРТ', 'body_part': 'brain', 'study_date': '', 'label': '',
                    'conclusion': 'x'})
    return ref, rep


SYNTHETIC_REF = [
    ('9000001', 'normal', 'no', 'МРТ-признаков структурных изменений головного мозга не выявлено. Гайморит.'),
    ('9000002', 'tumor', 'yes', 'МРТ-признаки менингиомы левой лобной области.'),
    ('9000003', 'stroke', 'yes', 'Лакунарный инфаркт в области базальных ядер.'),
    ('9000004', 'atrophy', 'yes', 'Церебральная микроангиопатия Fazekas Grade I.'),
    ('9000005', 'cyst', 'yes', 'Арахноидальная киста задней черепной ямки.'),
]


def test_benchmark_on_synthetic_reference(tmp_path: Path):
    ref, rep = _write_reference(tmp_path, SYNTHETIC_REF)
    reference = L.load_reference(ref, rep)
    assert [r['patient_id'] for r in reference] == ['9000001', '9000002', '9000003', '9000004', '9000005']
    assert all('подготовил' not in r['conclusion'] for r in reference)
    res = L.benchmark(reference)
    assert res['n'] == 5 and res['abnormal_agreement'] == 1.0 and res['label_agreement'] == 1.0
    assert res['confusion'][('tumor', 'tumor')] == 1 and res['worst'] == []

    # a disagreeing reference row is counted and surfaces in `worst` (id only, no names)
    ref2, rep2 = _write_reference(tmp_path / 'b', SYNTHETIC_REF + [('9000006', 'tumor', 'yes', 'Киста левой гайморовой пазухи.')])
    res2 = L.benchmark(L.load_reference(ref2, rep2))
    assert res2['abnormal_agreement'] == pytest.approx(5 / 6)
    assert res2['worst'][0]['patient_id'] == '9000006' and res2['per_class']['tumor'] == (0.5, 2)


def test_benchmark_cli_exit_code(tmp_path: Path):
    ref, rep = _write_reference(tmp_path, SYNTHETIC_REF + [('9000006', 'tumor', 'yes', 'Киста левой гайморовой пазухи.')])
    base = [PY, str(SCRIPT), '--benchmark', '--reference', str(ref), '--reference-reports', str(rep), '--worst', '5']
    ok = subprocess.run(base + ['--min-agreement', '0.8'], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert 'agreement on abnormal' in ok.stdout and 'Имя Врача' not in ok.stdout
    bad = subprocess.run(base + ['--min-agreement', '0.95'], capture_output=True, text=True)
    assert bad.returncode == 1


REAL_REF = REPO / 'data' / 'matched_clean.csv'
REAL_REP = REPO / 'data' / 'hospital_reports_labels.csv'


@pytest.mark.skipif(not (REAL_REF.exists() and REAL_REP.exists()), reason='batch-1 reference not on this machine')
def test_real_reference_agreement_floor():
    reference = L.load_reference(REAL_REF, REAL_REP)
    assert len(reference) >= 800
    res = L.benchmark(reference)
    assert res['abnormal_agreement'] >= 0.85, res['abnormal_agreement']
    assert res['label_agreement'] >= 0.80, res['label_agreement']
    assert res['n_review'] / res['n'] <= 0.15


# ------------------------------------------- training/relabel_workflow.js keys
WORKFLOW = REPO / 'training' / 'relabel_workflow.js'
_NODE_HARNESS = r"""
const fs = require('fs')
let src = fs.readFileSync(process.argv[1], 'utf8').replace('export const meta', 'const meta')
const rows = JSON.parse(process.argv[2])
const run = new Function('args', 'phase', 'agent', 'log', 'pipeline', 'return (async () => {' + src + '})()')
const agent = async (prompt, opts) => {
  if (opts.label === 'read-jsonl') return { rows }
  const lines = prompt.split('\n').filter(l => l.startsWith('{')).map(l => JSON.parse(l))
  // a sloppy labeller: mangles the echoed modality and returns one duplicate
  const out = lines.map((l, i) => ({ row: l.row, patient_id: l.patient_id, study_date: l.study_date,
    modality: 'XX', section_index: l.section_index, label: i ? 'stroke' : 'normal', abnormal: i > 0 }))
  return { labels: out.concat([out[0]]) }
}
run({ file: 'x.jsonl' }, () => {}, agent, () => {}, async (items, fn) => Promise.all(items.map((it, i) => fn(it, it, i))))
  .then(r => process.stdout.write(JSON.stringify(r)))
"""


def test_workflow_schema_carries_modality_and_section():
    js = WORKFLOW.read_text(encoding='utf-8')
    assert js.count("modality: { type: 'string' }") == 2 and js.count("section_index: { type: 'string' }") == 2
    assert "r.modality, r.section_index" in js          # de-duplication key is the 4-part import key
    assert 'patient_id|study_date' not in js and '`${l.patient_id}|${l.study_date}`' not in js


@pytest.mark.skipif(not __import__('shutil').which('node'), reason='node not installed')
def test_workflow_echoes_keys_for_same_day_ct_and_mr(tmp_path: Path):
    rows = [{'patient_id': '9000111', 'study_date': '2031-07-21', 'modality': 'CT', 'section_index': '0',
             'conclusion': 'a'},
            {'patient_id': '9000111', 'study_date': '2031-07-21', 'modality': 'MR', 'section_index': '1',
             'conclusion': 'b'}]
    proc = subprocess.run(['node', '-e', _NODE_HARNESS, str(WORKFLOW), json.dumps(rows)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    res = json.loads(proc.stdout)
    assert res['n_out'] == 2 and res['missing'] == []
    got = [(l['modality'], l['section_index'], l['label']) for l in res['labels']]
    assert got == [('CT', '0', 'normal'), ('MR', '1', 'stroke')]
    # ...and the output imports one label per row
    labelled = _same_day_rows()
    assert tuple(L.import_llm_labels(labelled, res['labels'])) == (2, 0, 0)
    assert [r['label'] for r in labelled] == ['normal', 'stroke']


# ------------------------------------------------------------ review round 2
@pytest.mark.parametrize('text', [
    'Последствия ОНМК в бассейне СМА слева.',
    'Последствия ОНМК в бассейне правой позвоночной артерии.',
    'Энцефаломаляция в бассейне правой позвоночной артерии.',
    'Острое нарушение мозгового кровообращения в бассейне левой СМА.',
    'Последствия ОНМК в виде кистозно-атрофической зоны в бассейне СМА справа.',
    'Последствия ОНМК в бассейне ЗМА справа.',
])
def test_vascular_territory_never_makes_a_stroke_clause_non_brain(text):
    """A brain vascular territory (СМА / ЗМА / позвоночная артерия) must not
    route the clause into the non-brain filter (round 2, line 219)."""
    a = L.analyse_conclusion(text)
    assert (a.label, a.abnormal) == ('stroke', 'yes'), (text, a.label, a.ignored)
    assert not a.ignored


def test_aneurysm_of_a_brain_artery_is_abnormal():
    a = L.analyse_conclusion('Аневризма левой СМА.')
    assert a.abnormal == 'yes' and a.label == 'other' and not a.ignored


def test_non_brain_vessel_and_spine_still_ignored():
    assert lab('Остеохондроз шейного отдела позвоночника.') == ('normal', 'no')
    assert lab('Извитость правой позвоночной артерии.') == ('normal', 'no')


@pytest.mark.parametrize('text', [
    'Признаки энцефалита, других изменений не выявлено.',
    'Признаки глиоза в левой лобной доле, других изменений не выявлено.',
    'Картина энцефалита, других очагов не выявлено.',
])
def test_short_tail_never_negates_a_non_vascular_finding(text):
    """The short list-tail override is for vascular 'other' findings only
    (round 2, line 502): an asserted encephalitis/gliosis stays abnormal."""
    a = L.analyse_conclusion(text)
    assert a.abnormal == 'yes' and a.label == 'other', (text, a.negated)


@pytest.mark.parametrize('text, expected', [
    ('Кровоизлияний нет.', ('normal', 'no')),
    ('Объемных образований нет.', ('normal', 'no')),
    ('Объемных образований нет', ('normal', 'no')),
    ('Острый ишемический инсульт в бассейне левой СМА, кровоизлияний нет.', ('stroke', 'yes')),
    ('Кистозное образование, кровоизлияния в кисту нет.', ('cyst', 'yes')),
])
def test_sentence_final_net_is_a_negation(text, expected):
    """'нет' at the end of a sentence (its '.' stripped) negates (round 2, line 140)."""
    assert lab(text) == expected


# ------------------------------------------- import: needs_review on a label line
_KEY = {'patient_id': '9000111', 'study_date': '2031-07-21', 'modality': 'MR', 'section_index': '1'}


def test_import_result_is_a_three_field_tuple():
    assert L.ImportResult._fields == ('applied', 'skipped', 'ambiguous')
    applied, skipped, ambiguous = L.import_llm_labels(_same_day_rows(), [])
    assert (applied, skipped, ambiguous) == (0, 0, 0)


@pytest.mark.parametrize('flag', ['yes', 'YES', ' true ', '1', True, 'да', 'y'])
def test_import_needs_review_applies_but_keeps_row_in_review(flag):
    rows = _same_day_rows()
    res = L.import_llm_labels(rows, [{**_KEY, 'label': 'stroke', 'abnormal': True, 'needs_review': flag}])
    assert tuple(res) == (1, 0, 0)
    assert (rows[1]['label'], rows[1]['abnormal'], rows[1]['label_source']) == ('stroke', 'yes', 'import')
    assert rows[1]['needs_review'] == 'yes' and rows[1]['label_confidence'] == 'low'
    assert rows[0]['label_source'] == 'rules'


@pytest.mark.parametrize('extra', [{}, {'needs_review': 'no'}, {'needs_review': False}, {'needs_review': ''},
                                   {'needs_review': 'maybe'}, {'needs_review': None}])
def test_import_without_needs_review_leaves_review(extra):
    rows = _same_day_rows()
    assert rows[1]['needs_review'] == 'yes'                       # the rules sent it to review
    res = L.import_llm_labels(rows, [{**_KEY, 'label': 'stroke', 'abnormal': True, **extra}])
    assert tuple(res) == (1, 0, 0)
    assert (rows[1]['label'], rows[1]['needs_review'], rows[1]['label_confidence']) == ('stroke', 'no', 'high')


def test_import_needs_review_with_another_source():
    rows = _same_day_rows()
    L.import_llm_labels(rows, [{**_KEY, 'label': 'stroke', 'abnormal': True}], source='llm')
    assert (rows[1]['label_confidence'], rows[1]['needs_review']) == ('medium', 'no')
    rows = _same_day_rows()
    L.import_llm_labels(rows, [{**_KEY, 'label': 'stroke', 'abnormal': True, 'needs_review': 'yes'}], source='llm')
    assert (rows[1]['label_confidence'], rows[1]['needs_review'], rows[1]['label_source']) == ('low', 'yes', 'llm')


@pytest.mark.parametrize('first,second', [('yes', 'no'), ('no', 'yes'), ('yes', 'yes')])
def test_import_needs_review_with_identical_duplicate_stays_in_review(first, second):
    """An identical duplicate is skipped either way; a review request on either line wins."""
    rows = _same_day_rows()
    res = L.import_llm_labels(rows, [{**_KEY, 'label': 'stroke', 'abnormal': True, 'needs_review': first},
                                     {**_KEY, 'label': 'stroke', 'abnormal': True, 'needs_review': second}])
    assert tuple(res) == (1, 1, 0)
    assert (rows[1]['label'], rows[1]['needs_review'], rows[1]['label_confidence']) == ('stroke', 'yes', 'low')


def test_import_needs_review_with_conflicting_duplicate_and_ambiguous_key():
    rows = _same_day_rows()
    res = L.import_llm_labels(rows, [{**_KEY, 'label': 'stroke', 'abnormal': True, 'needs_review': 'no'},
                                     {**_KEY, 'label': 'tumor', 'abnormal': True, 'needs_review': 'yes'}])
    assert tuple(res) == (1, 0, 1)
    assert (rows[1]['label'], rows[1]['needs_review'], rows[1]['label_confidence']) == ('stroke', 'yes', 'low')
    # an ambiguous key is never applied, needs_review or not
    rows = _same_day_rows()
    before = [dict(r) for r in rows]
    res = L.import_llm_labels(rows, [{'patient_id': '9000111', 'study_date': '2031-07-21', 'label': 'stroke',
                                      'abnormal': True, 'needs_review': 'yes'}])
    assert tuple(res) == (0, 0, 1) and rows == before
    # an invalid label is skipped, needs_review or not
    res = L.import_llm_labels(rows, [{**_KEY, 'label': 'bogus', 'abnormal': True, 'needs_review': 'yes'}])
    assert tuple(res) == (0, 1, 0) and rows == before


def _cli_import(tmp_path: Path, labels):
    src = tmp_path / 'reports.csv'
    cols = ['patient_id', 'modality', 'body_part', 'study_date', 'conclusion', 'section_index']
    with open(src, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in _rows():
            w.writerow(r)
    llm = _jsonl(tmp_path / 'llm.jsonl', labels)
    proc = subprocess.run([PY, str(SCRIPT), '--reports', str(src), '--out', str(tmp_path / 'labels.csv'),
                           '--import-llm-labels', str(llm)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    with open(tmp_path / 'labels.csv', newline='', encoding='utf-8') as fh:
        return proc.stdout, list(csv.DictReader(fh))


def test_cli_prints_how_many_imported_labels_stay_in_review(tmp_path: Path):
    k1 = {'patient_id': '9000001', 'study_date': '2031-07-01', 'modality': 'MR', 'section_index': '0'}
    k2 = {'patient_id': '9000002', 'study_date': '2031-07-02', 'modality': 'MR', 'section_index': '0'}
    k3 = {'patient_id': '9000002', 'study_date': '2031-08-09', 'modality': 'CT', 'section_index': '0'}
    out, rows = _cli_import(tmp_path, [{**k1, 'label': 'normal', 'abnormal': False, 'needs_review': 'yes'},
                                       {**k2, 'label': 'other', 'abnormal': True, 'needs_review': 'yes'},
                                       {**k3, 'label': 'hemorrhage', 'abnormal': True, 'needs_review': 'no'}])
    assert '3 applied, 0 ambiguous, 0 skipped' in out
    assert '2 imported label(s) stay in review' in out
    assert [(r['label_source'], r['needs_review'], r['label_confidence']) for r in rows] == [
        ('import', 'yes', 'low'), ('import', 'yes', 'low'), ('', '', ''), ('import', 'no', 'high')]
    out, _ = _cli_import(tmp_path, [{**k1, 'label': 'normal', 'abnormal': False}])
    assert '1 applied' in out and 'stay in review' not in out


def test_cli_ambiguous_hint_names_the_real_cause(tmp_path: Path):
    out, rows = _cli_import(tmp_path, [{'patient_id': '9000002', 'label': 'tumor', 'abnormal': True}])
    assert '0 applied, 1 ambiguous, 0 skipped' in out
    assert 'ambiguous labels were NOT applied' in out and 'incomplete or matches no row' in out
    assert 'export again' not in out
    assert all(r['label_source'] != 'import' for r in rows)


# --------------------------------------- training/relabel_workflow.js direct mode
_NODE_DIRECT = r"""
const fs = require('fs')
let src = fs.readFileSync(process.argv[1], 'utf8').replace('export const meta', 'const meta')
const spec = JSON.parse(process.argv[2])
const calls = [], logs = []
const run = new Function('args', 'phase', 'agent', 'log', 'pipeline', 'return (async () => {' + src + '})()')
const agent = async (prompt, opts) => {
  const m = prompt.match(/NR>=(\d+) && NR<=(\d+) \{print NR-1 "\\t" \$0\}' '([^']+)'/)
  const w = prompt.match(/write your labels to '([^']+)'/)
  const a = m ? +m[1] : null, z = m ? +m[2] : null
  calls.push({ label: opts.label, phase: opts.phase, model: opts.model === undefined ? null : opts.model,
               schema_required: opts.schema ? opts.schema.properties.labels.items.required : null,
               a, z, file: m ? m[3] : null, part: w ? w[1] : null })
  const how = (spec.answers || {})[opts.label]
  if (how === undefined || how === 'ok') {
    const labels = []
    for (let r = a - 1; r <= z - 1; r++) labels.push({ row: r, patient_id: String(9000300 + r), label: spec.label || 'normal' })
    return { labels }
  }
  if (how === 'null') return null
  return how                                          // a literal (possibly garbage) answer
}
run(spec.args, () => {}, agent, m => logs.push(String(m)), async (items, fn) => Promise.all(items.map((it, i) => fn(it, it, i))))
  .then(r => process.stdout.write(JSON.stringify({ result: r, calls, logs })),
        e => process.stdout.write(JSON.stringify({ error: String((e && e.message) || e), calls, logs })))
"""
_HAS_NODE = __import__('shutil').which('node')


def _direct(spec):
    proc = subprocess.run(['node', '-e', _NODE_DIRECT, str(WORKFLOW), json.dumps(spec)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_workflow_direct_mode_rules_is_the_rules_only_file():
    """Review 2026-11-11 #6.3: the collector needs the rules-only CSV, not the import target."""
    js = WORKFLOW.read_text(encoding='utf-8')
    assert '--rules data/<batch>_labels_rules.csv' in js
    assert '--rules data/<batch>_labels.csv' not in js
    assert '--out data/<batch>_labels_rules.csv --export-for-llm' in js


def test_workflow_direct_mode_static():
    js = WORKFLOW.read_text(encoding='utf-8')
    assert 'const N_LINES = args && args.n_lines' in js
    assert "required: ['row', 'patient_id', 'label']" in js
    assert '/v${job.v + 1}-b${job.b + 1}.tsv' in js
    assert "return { mode: 'direct', file, out_dir: outDir, n_in: N_LINES, votes: VOTES, batch: BATCH, " \
           "n_out: tsv.length, missing, rows: tsv }" in js
    # the direct branch returns before the reader agent is ever started
    assert js.index("return { mode: 'direct'") < js.index("phase('Read')")
    # the existing key pins still hold with the direct schema added
    assert js.count("modality: { type: 'string' }") == 2 and js.count("section_index: { type: 'string' }") == 2


@pytest.mark.skipif(not _HAS_NODE, reason='node not installed')
def test_workflow_direct_mode_batches_votes_and_parts():
    f = '/synthetic/b_review_all.jsonl'
    out = _direct({'args': {'file': f, 'n_lines': 7, 'batch': 3, 'votes': 2, 'model': 'opus'}})
    assert 'error' not in out, out
    calls = out['calls']
    assert [c['label'] for c in calls] == ['label-v1-b1', 'label-v1-b2', 'label-v1-b3',
                                           'label-v2-b1', 'label-v2-b2', 'label-v2-b3']
    assert all(c['phase'] == 'Label' and c['model'] == 'opus' and c['file'] == f for c in calls)
    assert all(c['schema_required'] == ['row', 'patient_id', 'label'] for c in calls)
    assert [(c['a'], c['z']) for c in calls] == [(1, 3), (4, 6), (7, 7)] * 2
    assert [c['part'] for c in calls] == [f'{f}.labels/v{v}-b{b}.tsv' for v in (1, 2) for b in (1, 2, 3)]
    res = out['result']
    assert {k: res[k] for k in res if k != 'rows'} == {
        'mode': 'direct', 'file': f, 'out_dir': f'{f}.labels', 'n_in': 7, 'votes': 2, 'batch': 3,
        'n_out': 14, 'missing': []}
    assert res['rows'][:2] == ['1\t0\t9000300\tnormal', '1\t1\t9000301\tnormal']
    assert res['rows'][7] == '2\t0\t9000300\tnormal' and len(res['rows']) == 14


@pytest.mark.skipif(not _HAS_NODE, reason='node not installed')
def test_workflow_direct_mode_defaults_and_out_dir():
    out = _direct({'args': {'file': 'x.jsonl', 'n_lines': 80}})
    assert [(c['a'], c['z']) for c in out['calls']] == [(1, 75), (76, 80)]           # batch 75, 1 vote
    assert all(c['model'] is None for c in out['calls'])                            # session model
    assert out['result']['votes'] == 1 and out['result']['batch'] == 75 and out['result']['n_out'] == 80
    out = _direct({'args': {'file': 'x.jsonl', 'n_lines': 2, 'batch': 0, 'votes': -1, 'model': '',
                            'out_dir': '/synthetic/parts'}})
    assert [c['part'] for c in out['calls']] == ['/synthetic/parts/v1-b1.tsv'] and out['calls'][0]['model'] is None
    assert out['result']['out_dir'] == '/synthetic/parts' and out['result']['batch'] == 75


@pytest.mark.skipif(not _HAS_NODE, reason='node not installed')
@pytest.mark.parametrize('n_lines', [0, -3, 2.5, '5'])
def test_workflow_direct_mode_rejects_bad_n_lines(n_lines):
    out = _direct({'args': {'file': 'x.jsonl', 'n_lines': n_lines}})
    assert 'args.n_lines must be' in out['error'] and out['calls'] == []


@pytest.mark.skipif(not _HAS_NODE, reason='node not installed')
def test_workflow_direct_mode_null_garbage_and_missing():
    garbage = {'labels': [None, 7, 'x', {'row': '2'}, {'row': 2.5, 'patient_id': '9000302', 'label': 'cyst'},
                          {'row': 0, 'patient_id': '9000300', 'label': 'tumor'},          # outside its slice
                          {'row': 3, 'patient_id': '9000303', 'label': 'tumor'},
                          {'row': 3, 'patient_id': '9000303', 'label': 'stroke'}]}        # duplicate: first wins
    out = _direct({'args': {'file': 'x.jsonl', 'n_lines': 6, 'batch': 2, 'votes': 2},
                   'answers': {'label-v1-b1': 'null', 'label-v1-b2': garbage, 'label-v1-b3': 'not an object',
                               'label-v2-b3': {'labels': 'nope'}}})
    assert 'error' not in out, out
    res = out['result']
    assert res['rows'] == ['1\t3\t9000303\ttumor', '2\t0\t9000300\tnormal', '2\t1\t9000301\tnormal',
                           '2\t2\t9000302\tnormal', '2\t3\t9000303\tnormal']
    assert res['n_out'] == 5
    assert res['missing'] == [{'vote': 1, 'row': r} for r in (0, 1, 2, 4, 5)] + \
                             [{'vote': 2, 'row': r} for r in (4, 5)]
    assert any('7 (vote,row) pair(s) came back without a label' in l for l in out['logs'])
    assert any('direct mode: 5/12 labels' in l for l in out['logs'])


@pytest.mark.skipif(not _HAS_NODE, reason='node not installed')
def test_workflow_direct_mode_result_feeds_the_collector(tmp_path: Path):
    """The return value is a valid --result for scripts/collect_llm_labels.py."""
    import collect_llm_labels as C
    export = tmp_path / 'e.jsonl'
    _jsonl(export, [{'patient_id': str(9000300 + i), 'study_date': '2031-07-01', 'modality': 'MR',
                     'section_index': '0', 'conclusion': 'МР-признаков патологии не выявлено.'} for i in range(3)])
    rules = tmp_path / 'labels.csv'
    with open(rules, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['patient_id', 'study_date', 'modality', 'section_index', 'body_part', 'label'])
        for i in range(3):
            w.writerow([9000300 + i, '2031-07-01', 'MR', '0', 'brain', 'normal'])
    out = _direct({'args': {'file': str(export), 'n_lines': 3, 'batch': 2}})
    (tmp_path / 'result.json').write_text(json.dumps({'result': out['result']}), encoding='utf-8')
    rc = C.main(['--export', str(export), '--rules', str(rules), '--result', str(tmp_path / 'result.json'),
                 '--out', str(tmp_path / 'llm.jsonl'), '--disputed', str(tmp_path / 'd.jsonl')])
    assert rc == 0
    got = L.read_jsonl(tmp_path / 'llm.jsonl')
    assert [(g['patient_id'], g['label'], g['decided_by']) for g in got] == [
        (str(9000300 + i), 'normal', 'agree') for i in range(3)]
