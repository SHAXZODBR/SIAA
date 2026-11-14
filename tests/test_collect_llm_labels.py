"""Tests for scripts/collect_llm_labels.py (stage 2b "second read" of the labels).

Everything is SYNTHETIC and lives under pytest's tmp_path: short Russian
conclusions written for the tests, patient ids >= 9000000, dates in 2031.
Nothing under data/ or models/ is read or written.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))

import collect_llm_labels as C  # noqa: E402
import label_reports as L  # noqa: E402

PY = sys.executable
SCRIPT = REPO / 'scripts' / 'collect_llm_labels.py'

NORMAL = 'МРТ-признаков структурных изменений головного мозга не выявлено.'
NORMAL_CT = 'КТ-признаков патологии головного мозга не выявлено.'
HEMORRHAGE = 'Внутримозговое кровоизлияние в правой височной доле.'
ATROPHY = 'Умеренная атрофия лобно-теменных областей с обеих сторон.'
CYST = 'Арахноидальная киста средней черепной ямки слева.'


# --------------------------------------------------------------------- helpers
def _exp(pid, date='2031-03-01', modality='MR', section='0', conclusion='x'):
    return {'patient_id': pid, 'study_date': date, 'modality': modality, 'section_index': section,
            'conclusion': conclusion}


def _write_export(path: Path, rows) -> Path:
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    return path


def _write_part(directory: Path, vote: int, batch: int, lines) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f'v{vote}-b{batch}.tsv'
    p.write_text(''.join('\t'.join(str(c) for c in line) + '\n' for line in lines), encoding='utf-8')
    return p


RULE_COLS = ['patient_id', 'modality', 'body_part', 'study_date', 'section_index', 'conclusion',
             'label', 'abnormal', 'label_confidence', 'label_source', 'needs_review']


def _write_rules(path: Path, rows) -> Path:
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=RULE_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({**{c: '' for c in RULE_COLS}, 'body_part': 'brain', **r})
    return path


def _rule(pid, label, date='2031-03-01', modality='MR', section='0', **kw):
    return {'patient_id': pid, 'study_date': date, 'modality': modality, 'section_index': section,
            'label': label, 'abnormal': 'no' if label == 'normal' else 'yes', **kw}


def _read_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]


# ---------------------------------------------------------------------- decide
@pytest.mark.parametrize('votes,rule,expected', [
    ([], 'normal', None),
    # < 3 votes: a second read can only CONFIRM the rule label
    (['tumor'], 'tumor', C.Decision('tumor', 'agree', False)),
    (['normal', 'normal'], 'normal', C.Decision('normal', 'agree', False)),
    (['tumor'], 'stroke', None),
    (['tumor', 'stroke'], 'tumor', None),
    (['tumor'], '', None),                                         # no rule label → nothing to confirm
    # 3 votes
    (['stroke'] * 3, 'normal', C.Decision('stroke', 'unanimous', False)),
    (['tumor', 'tumor', 'stroke'], 'stroke', C.Decision('tumor', 'majority', False)),
    (['normal', 'normal', 'tumor'], 'tumor', C.Decision('normal', 'majority', True)),     # split → review
    (['tumor', 'tumor', 'normal'], 'normal', C.Decision('tumor', 'majority', True)),
    (['tumor', 'stroke', 'normal'], 'stroke', C.Decision('stroke', 'tie_rules', True)),
    (['tumor', 'stroke', 'normal'], 'normal', C.Decision('tumor', 'tie_severity', True)),  # rule outside the tie
    (['cyst', 'atrophy', 'other'], 'atrophy', C.Decision('atrophy', 'tie_rules', False)),
    (['cyst', 'atrophy', 'other'], '', C.Decision('cyst', 'tie_severity', False)),       # rule label absent
    (['hydrocephalus', 'ms_wml', 'cyst'], 'tumor', C.Decision('hydrocephalus', 'tie_severity', False)),
    # 4 votes
    (['ms_wml'] * 4, '', C.Decision('ms_wml', 'unanimous', False)),
    (['tumor', 'tumor', 'tumor', 'normal'], 'normal', C.Decision('tumor', 'majority', True)),
    (['atrophy', 'atrophy', 'stroke', 'stroke'], 'atrophy', C.Decision('atrophy', 'tie_rules', False)),
    (['atrophy', 'atrophy', 'stroke', 'stroke'], 'cyst', C.Decision('stroke', 'tie_severity', False)),
    # even split on normal-vs-abnormal: the rule label's side wins, always review
    (['normal', 'normal', 'tumor', 'stroke'], 'tumor', C.Decision('tumor', 'tie_rules', True)),
    (['normal', 'normal', 'tumor', 'stroke'], 'cyst', C.Decision('tumor', 'tie_severity', True)),
    (['normal', 'normal', 'tumor', 'stroke'], 'normal', C.Decision('normal', 'majority', True)),
    (['normal', 'normal', 'tumor', 'tumor'], 'tumor', C.Decision('tumor', 'majority', True)),
    (['normal', 'normal', 'tumor', 'stroke'], '', None),         # even split and no rule label
])
def test_decide(votes, rule, expected):
    assert C.decide(votes, rule) == expected


def test_decide_covers_every_decided_by_value():
    seen = {C.decide(v, r).how for v, r in [
        (['tumor'], 'tumor'), (['tumor'] * 3, ''), (['tumor', 'tumor', 'cyst'], ''),
        (['tumor', 'cyst', 'stroke'], 'cyst'), (['tumor', 'cyst', 'stroke'], '')]}
    assert seen == {'agree', 'unanimous', 'majority', 'tie_rules', 'tie_severity'}
    assert C.MAJORITY_VOTES == 3


# ----------------------------------------------------------------- Votes.add
def _votes(n=3):
    return C.Votes([_exp(str(9000100 + i)) for i in range(n)])


@pytest.mark.parametrize('row,pid,label,reason', [
    ('x', '9000100', 'normal', 'malformed'),
    ('1.0', '9000101', 'normal', 'malformed'),
    ('', '9000100', 'normal', 'malformed'),
    ('3', '9000103', 'normal', 'row_out_of_range'),
    ('-1', '9000100', 'normal', 'row_out_of_range'),
    ('0', '9000101', 'normal', 'patient_mismatch'),          # the neighbour's id: a shifted answer
    ('0', '', 'normal', 'patient_mismatch'),
    ('0', '9000100', 'bogus', 'bad_label'),
    ('0', '9000100', '', 'bad_label'),
])
def test_votes_add_rejects(row, pid, label, reason):
    v = _votes()
    v.add('run', 1, row, pid, label)
    assert v.seen == 1 and v.n_rejected == 1 and v.rejected[reason] == 1
    assert dict(v.by_vote) == {} and all(v.for_row(i) == [] for i in range(3))
    assert set(C.REJECT_REASONS) >= {reason}


def test_votes_add_accepts_normalised_and_handles_duplicates():
    v = _votes()
    v.add('run', 1, ' 0 ', '9000100 ', ' Normal ')
    v.add('run', 1, '0', '9000100', 'normal')               # identical duplicate: ignored, not rejected
    assert v.for_row(0) == ['normal'] and v.n_rejected == 0 and v.seen == 2
    v.add('run', 1, '1', '9000101', 'tumor')
    v.add('run', 1, '1', '9000101', 'stroke')               # conflicting duplicate: both answers rejected
    assert v.rejected['conflicting_duplicate'] == 2 and v.for_row(1) == []
    v.add('run', 1, '1', '9000101', 'tumor')                # ...and a third answer cannot revive the row
    assert v.for_row(1) == [] and v.n_rejected == 2 and v.seen == 5
    # an integer patient_id in the export still matches
    w = C.Votes([{'patient_id': 9000100, 'study_date': '2031-03-01'}])
    w.add('run', 1, 0, '9000100', 'cyst')
    assert w.for_row(0) == ['cyst']


def test_votes_for_row_collects_one_vote_per_set_in_set_order():
    v = _votes()
    v.add(('parts', 0), 2, '0', '9000100', 'stroke')
    v.add(('parts', 0), 1, '0', '9000100', 'tumor')
    v.add(('parts', 1), 1, '0', '9000100', 'normal')
    assert v.for_row(0) == ['tumor', 'stroke', 'normal'] and v.n_votes == 3
    assert v.for_row(1) == []


def test_cross_check():
    v = _votes()
    for i, lab in enumerate(['normal', 'tumor']):
        v.add('p', 1, str(i), str(9000100 + i), lab)
    v.add('p', 2, '1', '9000101', 'stroke')
    v.add('other', 1, '1', '9000101', 'cyst')
    ref = _votes()
    ref.add('check', 1, '0', '9000100', 'normal')           # agrees
    ref.add('check', 1, '1', '9000101', 'stroke')           # disagrees with vote 1 of run 'p'
    ref.add('check', 1, '2', '9000102', 'cyst')             # only in the reference: NOT added
    ref.add('check', 2, '1', '9000101', 'stroke')           # agrees with vote 2
    v.cross_check('p', ref, 'check')
    assert v.rejected == {'source_mismatch': 1}
    assert v.for_row(0) == ['normal'] and v.for_row(1) == ['cyst', 'stroke'] and v.for_row(2) == []
    # a vote already rejected is not counted twice
    v.cross_check('p', ref, 'check')
    assert v.n_rejected == 1


# ------------------------------------------------------------------ read_export
def test_read_export_ok(tmp_path: Path):
    rows = [_exp('9000100', modality='CT', section='0'), _exp('9000100', modality='MR', section='1')]
    p = tmp_path / 'e.jsonl'
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\r\n' for r in rows), encoding='utf-8')
    got, raw = C.read_export(p)
    assert got == rows and raw == [json.dumps(r, ensure_ascii=False) for r in rows]


@pytest.mark.parametrize('content,msg', [
    ('{"patient_id": "9000100"}\n\n{"patient_id": "9000101"}\n', 'line 2 is blank'),
    ('{"patient_id": "9000100"}\n   \n', 'line 2 is blank'),
    ('{"patient_id": "9000100"}\nnot json\n', 'line 2 is not JSON'),
    ('{"patient_id": ""}\n', 'line 1 has no patient_id'),
    ('{"study_date": "2031-03-01"}\n', 'line 1 has no patient_id'),
    ('["9000100"]\n', 'line 1 has no patient_id'),
    # the same import key twice, also after normalisation (date format, modality case, 0 vs "0")
    ('{"patient_id": "9000100", "study_date": "2031-03-01", "modality": "MR", "section_index": "0"}\n'
     '{"patient_id": "9000100", "study_date": "20310301", "modality": "mr", "section_index": 0}\n',
     'occur more than once'),
])
def test_read_export_errors(tmp_path: Path, content, msg):
    p = tmp_path / 'e.jsonl'
    p.write_text(content, encoding='utf-8')
    with pytest.raises(C.InputError, match=msg):
        C.read_export(p)


def test_read_export_not_utf8(tmp_path: Path):
    p = tmp_path / 'e.jsonl'
    p.write_bytes(b'{"patient_id": "9000100", "conclusion": "\xff\xfe"}\n')
    with pytest.raises(C.InputError, match='not UTF-8'):
        C.read_export(p)


# ------------------------------------------------------------------ read_parts
def test_read_parts_numeric_order_and_malformed_lines(tmp_path: Path):
    v = C.Votes([_exp(str(9000100 + i)) for i in range(4)])
    d = tmp_path / 'e.jsonl.labels'
    _write_part(d, 1, 10, [(2, 9000102, 'cyst')])
    _write_part(d, 1, 2, [(1, 9000101, 'stroke')])
    _write_part(d, 1, 1, [(0, 9000100, 'normal')])
    _write_part(d, 2, 1, [(0, 9000100, 'tumor')])
    (d / 'notes.txt').write_text('0\t9000100\tstroke\n', encoding='utf-8')            # ignored
    (d / 'v1-b1.tsv.bak').write_text('0\t9000100\tstroke\n', encoding='utf-8')        # ignored
    (d / 'v3-b1.tsv').write_text('row\tpatient_id\tlabel\n'                           # header → malformed
                                 '\n'                                                 # blank: skipped
                                 '3\t9000103\n'                                       # 2 cells
                                 '3\t9000103\tcyst\textra\n'                          # 4 cells
                                 '3\t9000103\tatrophy\r\n', encoding='utf-8')
    assert C.read_parts(v, 'run', d) == 5
    assert list(v.by_vote) == [('run', 1), ('run', 2), ('run', 3)]
    assert list(v.by_vote[('run', 1)]) == [0, 1, 2]          # b1, b2, b10 — not b1, b10, b2
    assert v.for_row(0) == ['normal', 'tumor'] and v.for_row(3) == ['atrophy']
    assert v.rejected == {'malformed': 3} and v.seen == 8


# ----------------------------------------------------------------- read_result
def _result(rows, out_dir='', **kw):
    return {'mode': 'direct', 'file': 'x.jsonl', 'out_dir': out_dir, 'n_in': 2, 'votes': 1, 'batch': 75,
            'n_out': len(rows), 'missing': [], 'rows': rows, **kw}


@pytest.mark.parametrize('wrap', [False, True])
def test_read_result_both_shapes(tmp_path: Path, wrap):
    res = _result(['1\t0\t9000100\tnormal'])
    p = tmp_path / 'r.json'
    p.write_text(json.dumps({'result': res} if wrap else res), encoding='utf-8')
    assert C.read_result(p) == res


@pytest.mark.parametrize('content', ['not json', '[]', '{"rows": "1\\t0"}', '{"result": "done"}',
                                     '{"result": {"labels": []}}'])
def test_read_result_errors(tmp_path: Path, content):
    p = tmp_path / 'r.json'
    p.write_text(content, encoding='utf-8')
    with pytest.raises(C.InputError):
        C.read_result(p)


def test_add_result():
    v = _votes()
    C.add_result(v, 'r', _result(['1\t0\t9000100\tnormal', '2\t0\t9000100\ttumor', '1\t1\t9000101\tstroke',
                                  '1\t2\t9000102', 'x\t2\t9000102\tcyst', '1\t2\t9000199\tcyst']))
    assert v.for_row(0) == ['normal', 'tumor'] and v.for_row(1) == ['stroke'] and v.for_row(2) == []
    assert v.rejected == {'malformed': 2, 'patient_mismatch': 1} and v.seen == 6


# ------------------------------------------------------------------ load_rules
def test_load_rules_brain_only_and_duplicates_unusable(tmp_path: Path):
    p = _write_rules(tmp_path / 'labels.csv', [
        _rule('9000100', 'normal'),
        _rule('9000101', 'tumor', body_part='Brain'),
        _rule('9000102', 'cyst', body_part='spine'),
        _rule('9000103', 'stroke'),
        _rule('9000103', 'atrophy', date='20310301', modality='mr'),     # same key once normalised
        _rule('9000104', 'stroke', section='1'),
    ])
    rules = C.load_rules(p)
    assert sorted(k[0] for k in rules) == ['9000100', '9000101', '9000104']
    assert rules[('9000104', '2031-03-01', 'MR', '1')]['label'] == 'stroke'


def test_load_rules_refuses_a_labels_file_that_holds_imported_labels(tmp_path: Path):
    """The import step may overwrite the labels CSV; a second read compared with THAT file would
    'agree' with earlier LLM labels instead of the rules."""
    rows = [_rule('9000100', 'normal'), _rule('9000101', 'tumor'), _rule('9000102', 'cyst', body_part='spine')]
    rows[1]['label_source'] = 'import'
    p = _write_rules(tmp_path / 'labels.csv', rows)
    with pytest.raises(C.InputError, match='already carry imported'):
        C.load_rules(p)
    rows[1]['label_source'] = 'llm'
    with pytest.raises(C.InputError, match='already carry imported'):
        C.load_rules(_write_rules(tmp_path / 'labels2.csv', rows))
    rows[1]['label_source'] = 'rules'
    rows[2]['label_source'] = 'import'                        # a non-brain row is none of its business
    assert len(C.load_rules(_write_rules(tmp_path / 'labels3.csv', rows))) == 2


def test_collector_refuses_rules_overwritten_by_the_import(batch, capsys):
    """End to end: import into the SAME labels file, then try a second read against it → exit 2."""
    t = batch['dir']
    _round1_parts(batch)
    assert C.main(['--export', str(batch['export']), '--rules', str(batch['labels']),
                   '--out', str(t / 'final.jsonl')]) == 0
    assert L.main(['--reports', str(batch['reports']), '--out', str(batch['labels']),
                   '--import-llm-labels', str(t / 'final.jsonl')]) == 0
    capsys.readouterr()
    rc = C.main(['--export', str(batch['export']), '--rules', str(batch['labels']), '--out', str(t / 'again.jsonl')])
    assert rc == 2 and not (t / 'again.jsonl').exists()
    assert 'already carry imported' in capsys.readouterr().out


def test_load_rules_unreadable(tmp_path: Path):
    p = tmp_path / 'labels.csv'
    p.write_bytes(b'patient_id,label\n\xff\xfe,normal\n')
    with pytest.raises(C.InputError):
        C.load_rules(p)


# --------------------------------------------------------------------- collect
def _collect_fixture():
    export = [_exp(str(9000100 + i)) for i in range(6)]
    raw = [json.dumps(e, ensure_ascii=False) for e in export]
    rules = {L.row_key(r): r for r in [_rule('9000100', 'normal'), _rule('9000101', 'normal'),
                                        _rule('9000102', 'tumor'), _rule('9000103', 'cyst'),
                                        _rule('9000105', 'atrophy')]}
    v = C.Votes(export)
    for i, lab in [(0, 'normal'), (1, 'tumor'), (2, 'stroke'), (4, 'normal'), (5, 'atrophy')]:
        v.add('p', 1, str(i), str(9000100 + i), lab)
    v.add('p', 2, '5', '9000105', 'normal')                  # two votes that differ
    return export, raw, v, rules


def test_collect_not_final_writes_disputed_lines_verbatim():
    export, raw, v, rules = _collect_fixture()
    res = C.collect(export, raw, v, rules, final=False)
    assert [o['patient_id'] for o in res['out']] == ['9000100']
    assert res['out'][0] == {'patient_id': '9000100', 'study_date': '2031-03-01', 'modality': 'MR',
                             'section_index': '0', 'label': 'normal', 'abnormal': False, 'needs_review': 'no',
                             'decided_by': 'agree', 'votes': ['normal'], 'rule': 'normal'}
    assert res['disputed'] == raw[1:]
    s = res['stats']
    assert s['undecided_why'] == {'abnormal_bit_differs': 1, 'finding_differs': 1, 'no_vote': 1,
                                  'no_rule_label': 1, 'votes_differ': 1}
    assert s['decided'] == 1 and s['disputed_written'] == 5 and s['final'] is False
    assert s['no_rule_label'] == 1 and s['votes_per_row'] == {0: 1, 1: 4, 2: 1}
    assert s['compared'] == 4 and s['abnormal_agreement'] == 0.75 and s['label_agreement'] == 0.5
    assert s['abnormal_flips'] == {'normal→abnormal': 1}
    assert s['top_disagreements'] == [['normal', 'tumor', 1], ['tumor', 'stroke', 1]]


def test_collect_final_keeps_rule_label_in_review_or_drops():
    export, raw, v, rules = _collect_fixture()
    res = C.collect(export, raw, v, rules, final=True)
    assert res['disputed'] == []
    by_pid = {o['patient_id']: o for o in res['out']}
    assert sorted(by_pid) == ['9000100', '9000101', '9000102', '9000103', '9000105']     # 9000104: no rule, no decision
    for pid, lab in [('9000101', 'normal'), ('9000102', 'tumor'), ('9000103', 'cyst'), ('9000105', 'atrophy')]:
        o = by_pid[pid]
        assert (o['label'], o['decided_by'], o['needs_review']) == (lab, 'unresolved', 'yes'), pid
    assert by_pid['9000101']['votes'] == ['tumor'] and by_pid['9000103']['votes'] == []
    s = res['stats']
    assert s['unresolved_kept_rule_label'] == 4 and s['dropped_no_label'] == 1 and s['final'] is True
    assert s['labels'] == {'normal': 2, 'tumor': 1, 'cyst': 1, 'atrophy': 1}


def test_collect_keys_come_from_the_export():
    export = [{'patient_id': 9000100, 'study_date': '2031-03-01', 'modality': 'CT', 'section_index': 0,
               'conclusion': NORMAL_CT}]
    v = C.Votes(export)
    v.add('p', 1, '0', '9000100', 'normal')
    v.add('p', 2, '0', '9000100', 'normal')
    v.add('p', 3, '0', '9000100', 'normal')
    out = C.collect(export, ['x'], v, {}, final=True)['out']
    assert [{k: o[k] for k in C.KEY_FIELDS} for o in out] == [
        {'patient_id': '9000100', 'study_date': '2031-03-01', 'modality': 'CT', 'section_index': '0'}]
    assert out[0]['decided_by'] == 'unanimous' and out[0]['rule'] == ''


# -------------------------------------------------------------- main, 2 rounds
REPORT_ROWS = [
    ('9000201', 'MR', 'brain', '2031-04-01', '0', NORMAL),
    ('9000202', 'MR', 'brain', '2031-04-02', '0', HEMORRHAGE),
    ('9000203', 'MR', 'brain', '2031-04-03', '0', ATROPHY),
    ('9000204', 'MR', 'brain', '2031-04-04', '0', CYST),
    ('9000204', 'CT', 'brain', '2031-04-04', '1', NORMAL_CT),
    ('9000205', 'MR', 'spine', '2031-04-05', '0', 'Грыжа диска L5-S1.'),
]


@pytest.fixture
def batch(tmp_path: Path):
    """Stage-2 output of a synthetic batch: rule labels CSV + an --all export."""
    src = tmp_path / 'b_reports.csv'
    cols = ['patient_id', 'modality', 'body_part', 'study_date', 'section_index', 'conclusion']
    with open(src, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(REPORT_ROWS)
    labels, export = tmp_path / 'b_labels.csv', tmp_path / 'b_review_all.jsonl'
    assert L.main(['--reports', str(src), '--out', str(labels), '--export-for-llm', str(export), '--all']) == 0
    rules = {r['patient_id'] + r['modality']: r['label'] for r in L.read_csv(labels)[0] if r['body_part'] == 'brain'}
    assert rules == {'9000201MR': 'normal', '9000202MR': 'hemorrhage', '9000203MR': 'atrophy',
                     '9000204MR': 'cyst', '9000204CT': 'normal'}
    ex = [json.loads(l) for l in export.read_text(encoding='utf-8').splitlines()]
    assert [(e['patient_id'], e['modality']) for e in ex] == [
        ('9000201', 'MR'), ('9000202', 'MR'), ('9000203', 'MR'), ('9000204', 'MR'), ('9000204', 'CT')]
    return {'dir': tmp_path, 'reports': src, 'labels': labels, 'export': export}


def _round1_parts(batch, row2='normal'):
    d = Path(f'{batch["export"]}.labels')
    _write_part(d, 1, 1, [(0, 9000201, 'normal'), (1, 9000202, 'hemorrhage'), (2, 9000203, row2)])
    _write_part(d, 1, 2, [(3, 9000204, 'tumor'), (4, 9000204, 'normal')])
    return d


def _round1(batch, *extra):
    t = batch['dir']
    args = ['--export', str(batch['export']), '--rules', str(batch['labels']), '--out', str(t / 'r1.jsonl'),
            '--disputed', str(t / 'disputed.jsonl'), '--summary', str(t / 's1.json'), *extra]
    return C.main(args)


def test_main_two_rounds_and_import(batch, capsys):
    t = batch['dir']
    _round1_parts(batch)
    assert _round1(batch) == 0
    r1 = _read_jsonl(t / 'r1.jsonl')
    assert [(o['patient_id'], o['modality'], o['label'], o['decided_by']) for o in r1] == [
        ('9000201', 'MR', 'normal', 'agree'), ('9000202', 'MR', 'hemorrhage', 'agree'),
        ('9000204', 'CT', 'normal', 'agree')]
    export_lines = batch['export'].read_text(encoding='utf-8').splitlines()
    disputed = t / 'disputed.jsonl'
    assert disputed.read_text(encoding='utf-8').splitlines() == export_lines[2:4]
    s1 = json.loads((t / 's1.json').read_text(encoding='utf-8'))
    assert s1['decided'] == 3 and s1['disputed_written'] == 2 and s1['part_files'] == 2 and s1['prior'] == 0
    assert s1['undecided_why'] == {'abnormal_bit_differs': 1, 'finding_differs': 1} and s1['final'] is False
    out1 = capsys.readouterr().out
    assert 'n_lines = 2, votes = 3' in out1 and f'--export {disputed}' in out1

    # round 2: three reads of the two disputed rows, FINAL
    d2 = Path(f'{disputed}.labels')
    _write_part(d2, 1, 1, [(0, 9000203, 'atrophy'), (1, 9000204, 'tumor')])
    _write_part(d2, 2, 1, [(0, 9000203, 'normal'), (1, 9000204, 'cyst')])
    _write_part(d2, 3, 1, [(0, 9000203, 'atrophy'), (1, 9000204, 'stroke')])
    rc = C.main(['--export', str(disputed), '--rules', str(batch['labels']), '--prior', str(t / 'r1.jsonl'),
                 '--out', str(t / 'final.jsonl'), '--summary', str(t / 's2.json')])
    assert rc == 0
    final = _read_jsonl(t / 'final.jsonl')
    assert final[:3] == r1                                      # prior lines first, unchanged
    assert [(o['patient_id'], o['label'], o['decided_by'], o['needs_review'], o['votes']) for o in final[3:]] == [
        ('9000203', 'atrophy', 'majority', 'yes', ['atrophy', 'normal', 'atrophy']),
        ('9000204', 'cyst', 'tie_rules', 'no', ['tumor', 'cyst', 'stroke'])]
    s2 = json.loads((t / 's2.json').read_text(encoding='utf-8'))
    assert s2['decided_by'] == {'majority': 1, 'tie_rules': 1} and s2['review'] == 1
    assert s2['prior'] == 3 and s2['total_out'] == 5 and s2['part_files'] == 3 and s2['final'] is True
    assert s2['rejected'] == {} and s2['votes_seen'] == 6 and s2['n_vote_sets'] == 3
    assert not (t / 'disputed.jsonl.labels' / 'disputed.jsonl').exists()

    # the final JSONL imports one label per row, the split row stays in review
    rows, _ = L.read_csv(batch['labels'])
    rows = [L.label_row(r) for r in rows]
    export_keys = {L.row_key(json.loads(l)) for l in export_lines}
    assert {L.row_key(o) for o in final} == export_keys
    assert tuple(L.import_llm_labels(rows, final)) == (5, 0, 0)
    got = {(r['patient_id'], r['modality']): (r['label'], r['needs_review'], r['label_source'])
           for r in rows if r['body_part'] == 'brain'}
    assert got == {('9000201', 'MR'): ('normal', 'no', 'import'), ('9000202', 'MR'): ('hemorrhage', 'no', 'import'),
                   ('9000203', 'MR'): ('atrophy', 'yes', 'import'), ('9000204', 'MR'): ('cyst', 'no', 'import'),
                   ('9000204', 'CT'): ('normal', 'no', 'import')}
    # ...and through the label_reports CLI, which reports the row kept in review
    proc = subprocess.run([PY, str(REPO / 'scripts' / 'label_reports.py'), '--reports', str(batch['reports']),
                           '--out', str(t / 'imported.csv'), '--import-llm-labels', str(t / 'final.jsonl')],
                          capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert '5 applied, 0 ambiguous, 0 skipped' in proc.stdout
    assert '1 imported label(s) stay in review' in proc.stdout


def test_main_final_round_without_prior_keeps_rule_labels(batch):
    t = batch['dir']
    _round1_parts(batch)
    rc = C.main(['--export', str(batch['export']), '--rules', str(batch['labels']), '--out', str(t / 'o.jsonl'),
                 '--summary', str(t / 's.json')])
    assert rc == 0
    out = {(o['patient_id'], o['modality']): o for o in _read_jsonl(t / 'o.jsonl')}
    assert len(out) == 5
    assert (out[('9000203', 'MR')]['label'], out[('9000203', 'MR')]['decided_by'],
            out[('9000203', 'MR')]['needs_review']) == ('atrophy', 'unresolved', 'yes')
    assert out[('9000204', 'MR')]['label'] == 'cyst' and out[('9000204', 'MR')]['needs_review'] == 'yes'
    s = json.loads((t / 's.json').read_text(encoding='utf-8'))
    assert s['unresolved_kept_rule_label'] == 2 and s['dropped_no_label'] == 0


# ---------------------------------------------------------------- exit codes
def test_exit_2_usage(batch):
    with pytest.raises(SystemExit) as e:
        C.main(['--rules', str(batch['labels']), '--out', 'x.jsonl'])
    assert e.value.code == 2


def _assert_exit_2(batch, args, capsys, msg):
    out = batch['dir'] / 'o.jsonl'
    rc = C.main(['--rules', str(batch['labels']), '--out', str(out), *args])
    assert rc == 2
    assert msg in capsys.readouterr().out
    assert not out.exists()


def test_exit_2_missing_inputs(batch, capsys):
    t = batch['dir']
    _assert_exit_2(batch, ['--export', str(t / 'nope.jsonl')], capsys, 'not found')
    _round1_parts(batch)
    _assert_exit_2(batch, ['--export', str(batch['export']), '--prior', str(t / 'nope.jsonl')], capsys, 'not found')
    _assert_exit_2(batch, ['--export', str(batch['export']), '--result', str(t / 'nope.json')], capsys, 'not found')


def test_exit_2_blank_line_and_duplicate_keys(batch, capsys):
    t = batch['dir']
    lines = batch['export'].read_text(encoding='utf-8').splitlines()
    blank = _write_export(t / 'blank.jsonl', [])
    blank.write_text(lines[0] + '\n\n' + lines[1] + '\n', encoding='utf-8')
    _assert_exit_2(batch, ['--export', str(blank)], capsys, 'blank')
    dup = t / 'dup.jsonl'
    dup.write_text(lines[0] + '\n' + lines[0] + '\n', encoding='utf-8')
    _assert_exit_2(batch, ['--export', str(dup)], capsys, 'more than once')
    empty = t / 'empty.jsonl'
    empty.write_text('', encoding='utf-8')
    _assert_exit_2(batch, ['--export', str(empty)], capsys, 'empty')


def test_exit_2_no_votes(batch, capsys):
    # no part-file directory
    _assert_exit_2(batch, ['--export', str(batch['export'])], capsys, 'no part-file directory')
    # an explicit --parts directory that does not exist
    _assert_exit_2(batch, ['--export', str(batch['export']), '--parts', str(batch['dir'] / 'nope')], capsys,
                   'no part-file directory')
    # a directory without part files
    Path(f'{batch["export"]}.labels').mkdir()
    _assert_exit_2(batch, ['--export', str(batch['export'])], capsys, 'no votes found')


def test_every_vote_rejected_is_exit_3_and_writes_no_labels(batch, capsys):
    """Vote lines exist but none is acceptable: that is a mis-aligned run (exit 3), not 'no votes'."""
    t = batch['dir']
    _write_part(Path(f'{batch["export"]}.labels'), 1, 1, [(0, 9000299, 'normal')])
    out, disputed, summary = t / 'o.jsonl', t / 'd.jsonl', t / 's.json'
    for final in (True, False):
        args = ['--export', str(batch['export']), '--rules', str(batch['labels']), '--out', str(out),
                '--summary', str(summary)] + ([] if final else ['--disputed', str(disputed)])
        assert C.main(args) == 3
        printed = capsys.readouterr().out
        assert '1 of 1 votes were rejected' in printed and 'NOTHING was written' in printed
        assert 'no votes found' not in printed
        assert not out.exists() and not disputed.exists()
        s = json.loads(summary.read_text(encoding='utf-8'))
        assert s['written'] is False and s['rejected'] == {'patient_mismatch': 1} and s['rejected_share'] == 1.0


def test_exit_2_overlap_with_prior(batch, capsys):
    t = batch['dir']
    _round1_parts(batch)
    assert _round1(batch) == 0
    # round 2 run by mistake on the FULL export: rows decided in round 1 are decided again
    _assert_exit_2(batch, ['--export', str(batch['export']), '--prior', str(t / 'r1.jsonl')], capsys,
                   'labelled more than once')
    # the same prior file twice overlaps too
    _assert_exit_2(batch, ['--export', str(batch['export']), '--disputed', str(t / 'd.jsonl'),
                           '--prior', str(t / 'r1.jsonl'), '--prior', str(t / 'r1.jsonl')], capsys,
                   'labelled more than once')


@pytest.mark.parametrize('content,msg', [
    ('{"patient_id": "9000201"\n', 'not a readable JSONL'),
    ('{"patient_id": "9000201", "label": "bogus"}\n', 'no valid label'),
    ('"normal"\n', 'no valid label'),
])
def test_exit_2_bad_prior(batch, capsys, content, msg):
    _round1_parts(batch)
    bad = batch['dir'] / 'bad_prior.jsonl'
    bad.write_text(content, encoding='utf-8')
    _assert_exit_2(batch, ['--export', str(batch['export']), '--prior', str(bad)], capsys, msg)


def test_exit_2_bad_result(batch, capsys):
    _round1_parts(batch)
    bad = batch['dir'] / 'r.json'
    bad.write_text('{"result": {"mode": "direct"}}', encoding='utf-8')
    _assert_exit_2(batch, ['--export', str(batch['export']), '--result', str(bad)], capsys, 'no "rows" list')


def test_exit_3_too_many_rejected_votes(batch, capsys):
    t = batch['dir']
    d = _round1_parts(batch)
    _write_part(d, 1, 3, [(1, 9000201, 'tumor')])            # shifted answer: patient_mismatch
    assert _round1(batch) == 3                                 # 1 of 6 votes > 2 %
    out = capsys.readouterr().out
    assert '1 of 6 votes were rejected' in out
    # a mis-aligned run writes NO labels: an agent shifted by one row passes the patient check on a
    # same-patient CT/MR pair, so even its accepted votes are not trusted. Only the summary is kept.
    assert 'NOTHING was written to --out / --disputed' in out
    assert not (t / 'r1.jsonl').exists() and not (t / 'disputed.jsonl').exists()
    s = json.loads((t / 's1.json').read_text(encoding='utf-8'))
    assert s['rejected'] == {'patient_mismatch': 1} and s['votes_seen'] == 6
    assert s['written'] is False and s['rejected_share'] == round(1 / 6, 4)
    # the threshold is a strict ">": 1/6 passes at --max-rejected 1/6 and above
    assert _round1(batch, '--max-rejected', str(1 / 6)) == 0
    r1 = _read_jsonl(t / 'r1.jsonl')                          # ...and then the rejected vote labelled nothing
    assert [o['patient_id'] for o in r1] == ['9000201', '9000202', '9000204']
    assert all(o['votes'] == [o['label']] for o in r1)
    assert len((t / 'disputed.jsonl').read_text(encoding='utf-8').splitlines()) == 2
    assert json.loads((t / 's1.json').read_text(encoding='utf-8'))['written'] is True
    before = (t / 'r1.jsonl').read_bytes()
    assert _round1(batch, '--max-rejected', '0.1') == 3
    assert (t / 'r1.jsonl').read_bytes() == before, 'a refused run must leave an earlier --out untouched'


def test_exit_3_names_stale_part_files_of_an_earlier_run(batch, capsys):
    """Review 2026-11-11 #6.2: part files of an earlier, larger run stay in <export>.labels and are all
    row_out_of_range. The exit-3 message must say which files and that deleting the directory fixes it
    (re-running the workflow alone rewrites only the batches it needs)."""
    t = batch['dir']
    d = _round1_parts(batch)
    _write_part(d, 1, 3, [(5, 9000206, 'normal'), (6, 9000207, 'tumor')])     # an earlier run's batch 3
    _write_part(d, 2, 3, [(5, 9000206, 'normal')])
    assert _round1(batch) == 3
    out = capsys.readouterr().out
    assert '3 of 8 votes were rejected' in out and 'NOTHING was written' in out
    assert 'v1-b3.tsv: 2 rejected' in out and 'v2-b3.tsv: 1 rejected' in out
    assert 'v1-b1.tsv' not in out and 'v1-b2.tsv' not in out                   # clean files are not listed
    assert 'earlier run' in out and 'delete' in out.lower() and str(d) in out
    for pid in ('9000201', '9000206', '9000207'):
        assert pid not in out                                                  # counts and file names only
    assert not (t / 'r1.jsonl').exists() and not (t / 'disputed.jsonl').exists()
    # what is accepted / rejected does not change
    s = json.loads((t / 's1.json').read_text(encoding='utf-8'))
    assert s['rejected'] == {'row_out_of_range': 3} and s['votes_seen'] == 8 and s['part_files'] == 4
    # a patient_mismatch alone is still reported per file but without the stale-run advice
    for p in d.glob('v*-b3.tsv'):
        p.unlink()
    _write_part(d, 1, 3, [(1, 9000201, 'tumor')])
    assert _round1(batch) == 3
    out = capsys.readouterr().out
    assert 'v1-b3.tsv: 1 rejected' in out and 'earlier run' not in out


# ------------------------------------------------------------------- --result
def _result_rows(lines):
    return [f'{v}\t{r}\t{p}\t{lab}' for v, r, p, lab in lines]


ROUND1_RESULT = [(1, 0, 9000201, 'normal'), (1, 1, 9000202, 'hemorrhage'), (1, 2, 9000203, 'normal'),
                 (1, 3, 9000204, 'tumor'), (1, 4, 9000204, 'normal')]


@pytest.mark.parametrize('wrap', [False, True])
@pytest.mark.parametrize('empty_dir', [False, True])
def test_result_used_directly_without_part_files(batch, wrap, empty_dir):
    t = batch['dir']
    d = Path(f'{batch["export"]}.labels')
    if empty_dir:                                              # the agents made the directory but wrote nothing
        d.mkdir()
    res = _result(_result_rows(ROUND1_RESULT), out_dir=str(d))
    rp = t / 'result.json'
    rp.write_text(json.dumps({'result': res} if wrap else res), encoding='utf-8')
    assert _round1(batch, '--result', str(rp)) == 0
    assert [o['patient_id'] for o in _read_jsonl(t / 'r1.jsonl')] == ['9000201', '9000202', '9000204']
    s = json.loads((t / 's1.json').read_text(encoding='utf-8'))
    assert s['part_files'] == 0 and s['votes_seen'] == 5 and s['disputed_written'] == 2


def test_result_cross_checks_the_part_files(batch):
    t = batch['dir']
    d = _round1_parts(batch)
    lines = list(ROUND1_RESULT)
    lines[1] = (1, 1, 9000202, 'stroke')                      # the part file says hemorrhage
    lines.append((1, 5, 9000299, 'tumor'))                     # junk in the result is not added
    rp = t / 'result.json'
    rp.write_text(json.dumps(_result(_result_rows(lines), out_dir=str(d))), encoding='utf-8')
    rc = _round1(batch, '--result', str(rp), '--max-rejected', '0.5')
    assert rc == 0
    r1 = _read_jsonl(t / 'r1.jsonl')
    assert [o['patient_id'] for o in r1] == ['9000201', '9000204']          # 9000202 lost its vote
    assert len((t / 'disputed.jsonl').read_text(encoding='utf-8').splitlines()) == 3
    s = json.loads((t / 's1.json').read_text(encoding='utf-8'))
    assert s['rejected'] == {'source_mismatch': 1} and s['votes_seen'] == 5 and s['n_vote_sets'] == 1
    # default --max-rejected: 1 of 5 rejected → exit 3
    assert _round1(batch, '--result', str(rp)) == 3


# --------------------------------------------------------------- script / docs
def test_script_runs_from_repo_root(batch):
    t = batch['dir']
    _round1_parts(batch)
    proc = subprocess.run([PY, 'scripts/collect_llm_labels.py', '--export', str(batch['export']),
                           '--rules', str(batch['labels']), '--out', str(t / 'r1.jsonl'),
                           '--disputed', str(t / 'disputed.jsonl')],
                          capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'LLM SECOND READ' in proc.stdout


def test_docstring_commands_use_real_options():
    doc = C.__doc__
    help_out = subprocess.run([PY, str(SCRIPT), '--help'], capture_output=True, text=True, cwd=REPO)
    assert help_out.returncode == 0
    opts = set(re.findall(r'--[a-z][a-z-]+', help_out.stdout))
    lr_help = subprocess.run([PY, str(REPO / 'scripts' / 'label_reports.py'), '--help'],
                             capture_output=True, text=True, cwd=REPO).stdout
    lr_opts = set(re.findall(r'--[a-z][a-z-]+', lr_help))
    cmds = re.findall(r'\$PY (scripts/\S+)(.*?)(?=\n\s*(?:#|\$PY|\n|Part files))', doc, flags=re.S)
    assert len(cmds) == 4
    for script, rest in cmds:
        assert (REPO / script).is_file(), script
        used = set(re.findall(r'--[a-z][a-z-]+', rest))
        assert used and used <= (opts if script.endswith('collect_llm_labels.py') else lr_opts), (script, used)
    for flag in ('--export', '--rules', '--out', '--disputed', '--prior'):
        assert flag in doc
