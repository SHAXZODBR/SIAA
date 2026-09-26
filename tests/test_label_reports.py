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
