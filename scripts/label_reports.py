#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — STAGE 2 "label": RULE-BASED LABELLING OF BRAIN REPORT CONCLUSIONS
================================================================================
  WHAT
    Reads the per-study CSV written by scripts/parse_hospital_reports.py
    (stage 1) and adds, to every BRAIN row, the fixed study-level label policy
    the validated triage model embodies (docs/dev/release_pipeline_contract.md):

        label    ∈ normal | tumor | stroke | atrophy | hemorrhage | ms_wml |
                   hydrocephalus | cyst | other
        abnormal ∈ yes | no        (no ONLY when the brain itself is normal)
      + label_confidence (high|medium|low), label_source (rules|llm|import),
        needs_review (yes|no).   Non-brain rows get label='' and abnormal=''.

  WHY
    A new hospital batch arrives with ~5,000 Russian ЗАКЛЮЧЕНИЕ texts. The
    rules here label them deterministically in seconds, benchmarked against
    the 894-patient LLM-labelled reference of batch 1 (data/matched_clean.csv
    joined to data/hospital_reports_labels.csv). Only the rows the rules are
    unsure about (needs_review=yes) go to an LLM: either the local Ollama
    server (--llm ollama:<model>) or the Claude Code Workflow
    training/relabel_workflow.js (--export-for-llm → workflow → --import-llm-labels).

  HOW (design)
    1. boilerplate is stripped ("Заключение не является диагнозом…",
       "Результат подготовил…", recommendations);
    2. the text is split into sentences ('.', ';', newline) and each sentence
       into comma clauses;
    3. per clause negation is detected: existential negations ("не выявлено",
       "отсутствуют", "не определяется") negate their own clause and extend
       backwards over the preceding comma clauses of the same sentence until a
       clause that asserts something ("признаки", "определяется", …) — and never
       over a clause that NAMES a finding in the nominative ("Внутримозговая
       гематома …, отек не выражен" keeps the haematoma; only genitive list
       items "очагов ишемии, кровоизлияний … не выявлено" are negated). Local
       negations ("не смещены", "без масс-эффекта") negate only their clause.
       Hedges ("нельзя исключить", "не исключается", "под вопросом", "(?)")
       are not negations: the finding is kept, confidence ≤ medium, review=yes;
    4. finding keywords are matched in the non-negated clauses; a clause that
       talks only about sinus / scalp / orbit / bone / spine is ignored
       (those are NOT brain findings → normal, abnormal=no);
    5. the SINGLE most significant finding wins by severity
       tumor > hemorrhage > stroke > hydrocephalus > ms_wml > cyst > atrophy > other > normal.

  USAGE
    python scripts/label_reports.py --reports data/<batch>_reports.csv --out data/<batch>_labels.csv
    python scripts/label_reports.py --benchmark [--min-agreement 0.90] [--worst 20]
    python scripts/label_reports.py --reports … --out … --export-for-llm data/<batch>_review.jsonl [--all]
    python scripts/label_reports.py --reports … --out … --import-llm-labels data/<batch>_llm_labels.jsonl
        (keys: patient_id + study_date + modality + section_index; see import_llm_labels
         for the unique-only fallbacks; prints applied / ambiguous / skipped)
    python scripts/label_reports.py --reports … --out … --llm ollama:gemma3:12b   (optional, local only)

  PHI: only the numeric hospital patient_id is ever printed or written; the
  conclusion snippets in the benchmark output are boilerplate-stripped so the
  radiologist's name never appears either.
================================================================================
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, NamedTuple

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'

REPO = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------------
# Label policy constants
# ----------------------------------------------------------------------------
LABELS = ('normal', 'tumor', 'stroke', 'atrophy', 'hemorrhage', 'ms_wml',
          'hydrocephalus', 'cyst', 'other')
# most significant first
SEVERITY = ('tumor', 'hemorrhage', 'stroke', 'hydrocephalus', 'ms_wml',
            'cyst', 'atrophy', 'other', 'normal')
SEVERITY_RANK = {name: i for i, name in enumerate(SEVERITY)}

# Rubric — used VERBATIM by training/relabel_workflow.js and the Ollama prompt.
RUBRIC = (
    "You are a neuroradiologist labelling brain MRI/CT report conclusions (Russian ЗАКЛЮЧЕНИЕ). "
    "For EACH report choose the SINGLE most significant BRAIN finding: normal = no significant brain "
    "pathology; tumor = опухоль / новообразование / глиома / менингиома / аденома / mts / объёмное "
    "образование; stroke = инсульт / инфаркт / ишемия / очаг ишемии / лакунарный; atrophy = атрофия / "
    "церебральная микроангиопатия / Fazekas / расширение борозд; hemorrhage = внутримозговое "
    "кровоизлияние / гематома (NOT scalp/подкожная); ms_wml = демиелинизация / рассеянный склероз / "
    "очаги белого вещества; hydrocephalus = гидроцефалия / выраженное расширение желудочков; cyst = "
    "киста ГОЛОВНОГО МОЗГА / арахноидальная (NOT киста пазухи); other = other genuine brain finding. "
    "CRITICAL: many conclusions describe ONLY sinus/nose (гайморит, синусит, пазух, носовой) or scalp "
    "findings while the BRAIN is normal — label those normal, abnormal=false. abnormal=false ONLY when "
    "the brain itself is normal."
)

OUTPUT_COLS = ('label', 'abnormal', 'label_confidence', 'label_source', 'needs_review')


def _rx(p: str) -> re.Pattern:
    return re.compile(p)

# ----------------------------------------------------------------------------
# Text normalisation and boilerplate
# ----------------------------------------------------------------------------
# Everything from these cues to the end of the text is letterhead / signature.
BOILERPLATE_CUES = (
    'заключение не является диагнозом',
    'результат подготовил',
    'уважаемые пациенты',
    'при необходимости последующей динамической',
    'результаты исследований и снимки',
    'пожалуйста, заключение коррелируйте',
    'заключение коррелируйте клинически',
    'клинико-патологическая корреляция',
    'врач:',
)
# Sentences that are advice, not findings.
RECOMMENDATION_CUES = ('рекоменд', 'консультац', 'в динамике', 'дообследован',
                       'целесообразн', 'показано проведение', 'контроль через')
# A comma clause with one of these is advice / a condition, never a finding
# ("консультация нейрохирурга при появлении эпизодов кровоизлияния").
CONDITIONAL_CUES = ('рекоменд', 'консультац', 'при появлении', 'при возникновении', 'в случае',
                    'при ухудшении', 'для исключ', 'с целью', 'для уточн', 'при отсутствии',
                    'дифференцировать', 'дифференциальн', 'сочетать с', 'коррелир', 'в динамике',
                    'контрол', 'наблюдени')

# Existential negation: negates its clause and the preceding clauses of the
# same sentence back to the last asserting clause.
NEG_EXISTENTIAL = (
    'не выявлен', 'не обнаружен', 'не определя', 'не отмеча', 'не визуализ',
    'не прослежива', 'не наблюда', 'не установлен', 'не получено', 'не выражен',
    'отсутств', 'не характерн', 'нет данных', 'данных за',
    # bare "нет" as a word: a sentence-final "Кровоизлияний нет." loses its
    # '.' in split_sentences, so a literal 'нет ' (trailing space) never fired
    # there (review round 2, line 140).
    _rx(r'\bнет\b'),
    'патологических изменений не', 'патологии не', 'не найден', 'не зафиксир',
    'не убедительн',
    # "исключено / исключается / исключен" = excluded. The hedges "нельзя
    # исключить", "не исключается", "не исключено", "исключить …", "под
    # вопросом", "(?)" are NOT negations: they are cut out of the clause
    # (HEDGE_RX) before any cue is tested.
    _rx(r'(?<!не )исключ(ен|ён|ает|ают|ено|ена|ены)'),
    # genitive "признаков наличия/структурных изменений …" = negated existence
    # (the verb "не выявлено" may sit in the next clause or be truncated away)
    _rx(r'\bпризнаков (наличия|патологических|структурных|очаговых|объемных|объемного|образований)'),
)
# Local negation: negates only its own comma clause.
NEG_LOCAL = (
    'без патолог', 'без особенност', 'без признаков', 'без масс', 'без отек',
    'без отёк', 'без перифок', 'без динамик', 'без структурн', 'без очагов',
    'без изменений', 'без смещ', 'без деформ', 'без нарушен', 'без расшир',
    'не изменен', 'не расширен', 'не смещен', 'не сужен', 'не наруш',
    'не деформир', 'не утолщ', 'не компримир', 'не дислоц', 'не увелич',
    'не уменьш', 'не пораж', 'не вовлеч',
    'в пределах нормы', 'в пределах возрастной нормы', 'вариант нормы',
    'соответствует норме', 'норма.', 'нормальн', _rx(r'\bобычн\w* (размер|форм|расположен|конфигурац|структур)'),
    _rx(r'\bсохранен[ыао]?\b'), _rx(r'\bсимметричны\b'), 'не изменён', 'не расширён', 'не смещён', 'интактн',
    'без структурных', 'не выражена', 'не выраженн',
)
# Hedges: the finding is KEPT (it is not negated) but the label is uncertain
# → label_confidence ≤ medium and needs_review=yes (review finding, line 135).
HEDGE_RX = _rx(r'(нельзя|невозможно|не может быть|не могут быть|трудно) исключ\w*'
               r'|исключ\w* нельзя|не исключ\w*|(?<!для )(?<!целью )\bисключить\b'
               r'|под вопросом|\(\s*\?\s*\)')
# "(?)" / "менингиома?" are rewritten to this before sentence splitting ('?' is
# a sentence separator).
_QMARK_RX = re.compile(r'\(\s*\?\s*\)|(?<=\w)\?')

# Side-effect tails: a negated clause that only denies a COMPLICATION of the
# finding named before it ("…, перифокальный отек не выражен", "…, смещения
# срединных структур не выявлено", "…, геморрагической трансформации не
# выявлено", "…, прорыва крови в желудочки нет"). Such a tail negates itself
# only — it never reaches back into the finding it qualifies.
SIDE_EFFECT_RX = _rx(
    r'(геморрагическ\w* )?трансформац\w*|кровоизлия\w* в (опухол|образован|строму|кист)\w*'
    r'|прорыв\w*( крови)?( в желудоч\w*)?|(перифокальн\w* )?от[её]к\w*|смещени\w*|смещен\w*'
    r'|дислокац\w*|масс-?эффект\w*|компресси\w*|сдавлени\w*|вклинени\w*|деформаци\w*')

# Vascular words: the short list-tail override ("…, аневризм, АВМ не выявлено")
# may only reach over an asserting clause that names a VASCULAR 'other' finding
# ("признаки окклюзии, …"); an asserted encephalitis / gliosis is never negated
# by a trailing "других изменений не выявлено" (review round 2, line 502).
VASCULAR_TERMS = (
    'окклюз', 'стеноз', 'аневризм', 'мальформац', _rx(r'\bавм\b'), 'тромбоз', 'артери', 'сосуд',
    'каверн', 'ангиом', 'ангиопат', 'гипоплаз', 'аплаз', 'извитост', 'вен', 'синус',
    _rx(r'\b(вса|пма|сма|зма)\b'),
)

# A clause that asserts a finding — stops the backward negation scope.
POSITIVE_MARKERS = (
    'признаки ', 'признаки:', 'картина', 'проявлен', 'определяется', 'определяются',
    'отмечается', 'отмечаются', 'визуализируется', 'визуализируются', 'имеется',
    'имеются', 'наличие ', 'состояние после', 'состояния после', 'сохраняется',
    'сохраняются', 'выявлены', 'выявлен ', 'выявлена', 'выявлено ', 'обнаружен',
    'прослеживается', 'наблюдается', 'расценива', 'соответствует', 'может соответствовать',
    'могут соответствовать', 'характерн', 'участок', 'участки', 'зона', 'зоны', 'очаг ', 'очаги',
)
# Global "brain is normal" statements → used for confidence only.
NORMAL_CUES = (
    'структурных изменений', 'патологических изменений', 'патологии', 'без патолог',
    'без особенност', 'очаговых', 'очаговой патологии', 'объемных образований',
    'объемного образования', 'образований гипофиза', 'в пределах нормы',
    'вариант нормы', 'норма', 'без изменений', 'изменений головного мозга',
    'изменений вещества', 'изменений со стороны', 'изменения не выявлены',
    'не выявлено', 'не выявлены', 'не обнаружено', 'не определяется',
)

# Non-brain anatomy: a clause mentioning these and NOT the brain is ignored.
NON_BRAIN_CONTEXT = (
    'пазух', 'гаймор', 'сфеноид', 'этмоид', 'решетчат', 'синусит', 'сфеноидит',
    'фронтит', 'этмоидит', 'пансинусит', 'полисинусит', 'носов', 'носоглот',
    'носа', 'перегородк', 'раковин', 'мягких тканей', 'мягкие ткани', 'подкожн',
    'кожи', 'кожн', 'скальп', 'апоневро', 'орбит', 'глазниц', 'глазн', 'зуб',
    'челюст', 'сосцевидн', 'ячеек', 'ячейк', 'среднего уха', 'наружного слухов',
    'внутреннего уха', 'миндалин', 'аденоид', 'щитовидн', 'слюнн', 'околоушн',
    'шейного отдела', 'шейн', 'позвон', 'диск', 'кост', 'костн', 'свода черепа',
    'черепа', 'теменной кости', 'височной кости', 'затылочной кости', 'лобной пазух',
    'лобных пазух', 'клиновидн', 'мастоид', 'отит', 'слизист', 'ринит', 'сосудов шеи',
    'полости носа', 'лимфоуз', 'лимфатическ',
    'липом', 'атером', 'экзостоз', 'остеом', 'мягких ткан', 'мягкотка', 'хрусталик', 'катаракт',
    'легк', 'легочн', 'бронх', 'плевр', 'средостен', 'печен', 'почк', 'почек', 'надпочечн',
    'сустав', 'синовит', 'мениск', 'связк', 'матк', 'яичник', 'простат', 'кишечн', 'желчн',
    'селезен', 'поджелуд', 'мочев', 'пузыр', 'молочн', 'сухожил', 'мышц', 'мыщц',
    'артери', 'вса', 'пма', 'сма', 'зма', 'извитост', 'атеросклероз',
)
# Organs that prove the clause is NOT about the brain even when brain words
# ("доли", "сегмент") also appear. The spine is matched by explicit spine
# forms only (позвоночник / позвонок / "отдел позвоночника" / межпозвонковый):
# a bare 'позвон' would also hit "позвоночной артерии", the standard way to
# name a posterior-circulation stroke territory (review finding, line 384).
STRONG_NON_BRAIN = (
    'легк', 'легочн', 'бронх', 'плевр', 'печен', 'почк', 'почек', 'надпочечн', 'сустав',
    'синовит', 'мениск', 'матк', 'яичник', 'простат', 'кишечн', 'желчн', 'селезен',
    'поджелуд', 'щитовидн', 'молочн', 'позвоночник', 'позвонк', 'позвонок', 'межпозвон',
    _rx(r'отдел\w* позвоночн'), _rx(r'позвоночн\w* (столб|канал)'),
)
# Vascular-territory phrases name BRAIN territories ("в бассейне правой
# позвоночной артерии", "в бассейне СМА слева", "вертебро-базилярный бассейн");
# they are cut out of a clause before BOTH non-brain tests (STRONG_NON_BRAIN and
# NON_BRAIN_CONTEXT) so they can never make a clause non-brain (review round 2,
# line 219: "Последствия ОНМК в бассейне СМА слева." was dropped as non-brain).
_VASCULAR_TERRITORY_RX = _rx(
    r'позвоночн(ой|ых|ая|ую|ые|ым)\s+артер\w*|вертебр\w*|базилярн\w*'
    r'|бассейн\w*(\s+(право\w*|лево\w*|обеих|обоих|\w+их))?(\s+(передн\w*|средн\w*|задн\w*))?'
    r'(\s+(мозгов\w*|внутренн\w*))?(\s+(сонн\w*))?\s+(артери\w*|вса|пма|сма|зма|па|ба)\b(\s+(справа|слева|с обеих сторон))?'
    r'|бассейн\w*\s+(вса|пма|сма|зма|па|ба)\b')
# Words that prove a clause talks about the brain even if non-brain words are present.
BRAIN_CONTEXT = (
    'мозг', 'мозж', 'доли', 'доле', 'долей', 'полушар', 'желудоч', 'ствол', 'гипофиз',
    'церебр', 'интракран', 'внутричерепн', 'внутримозгов', 'субарахн', 'субдурал',
    'эпидурал', 'перивентрик', 'белого вещества', 'белом веществе', 'базальн', 'таламус',
    'стриат', 'капсул', 'варолиев', 'подкорков', 'кортик', 'инсуляр', 'островк',
    'мозолист', 'цистерн', 'миндалин мозж', 'миндалина мозж',
    'турецк', 'хиазм', 'селляр', 'гиппокамп', 'паренхим', 'серого вещества',
    'мозговой ткани', 'вещества мозга', 'вещество мозга', 'сильвиев', 'борозд',
    'демиелин', 'лакунар', 'ишеми', 'инсульт', 'инфаркт', 'fazekas', 'микроангиопат',
    'энцефалопат', 'мостомозжечков', 'черепной ямк', 'семиовальн', 'сосудистого генеза',
    'лейкоареоз', 'лейкоэнцефалопат', 'глиоз', 'кровоизлия',
    # stroke / vascular words that never describe anything but the brain here
    # (review round 2, line 219)
    'онмк', 'энцефаломаляц', 'постишем', 'постинфаркт', 'кистозно-глиоз', 'кистозно-атроф',
    'аневризм', 'мальформац', _rx(r'\bавм\b'), _rx(r'нарушени\w* мозгового'),
)

# Finding keyword tables (matched on lowercase, ё→е, whitespace-collapsed text).
# Each entry: (label, [keywords]) — a keyword may be a plain substring or a
# compiled regex.
FINDINGS: list[tuple[str, list]] = [
    ('tumor', [
        'опухол', 'новообразов', 'глиом', 'менингиом', 'аденом', 'mts', 'метастаз',
        _rx(r'объемн\w* образован'), _rx(r'объемн\w* процесс'), 'шванном', 'неврином',
        'гемангиобластом', 'лимфом', 'краниофарингиом', 'эпендимом', 'медуллобластом',
        'астроцитом', 'бластом', 'neoplas', 'tumor', 'дермоид', 'эпидермоид',
        'хордом', 'гермином', 'папиллом', 'параганглиом', 'гломусн', 'пинеалом',
        _rx(r'солидн\w* образован'), _rx(r'кистозно-солидн\w* образован'),
        _rx(r'внемозгов\w* образован'), 'опухолев', 'бластоматоз', 'неопластич',
        'макроаденом', 'микроаденом', 'канцероматоз', 'лимфопролифер', 'метастат',
        _rx(r'внутримозгов\w* образован'), _rx(r'внемозгов\w* образован'), 'dnet', 'глиобластом',
        _rx(r'образован\w*[^.;,]{0,40}(доли|полушар|мозжечк|желудочк|гипофиз)'),
    ]),
    ('hemorrhage', [
        'кровоизлия', 'гематом', 'геморраг', 'внутрижелудочков', 'сак ', 'субарахноидальное кров',
        'кровотеч', 'геморрагическ',
    ]),
    ('stroke', [
        'инсульт', 'инфаркт', 'ишеми', 'лакунар', 'онмк', 'нмк', 'энцефаломаляц',
        'постишем', 'постинфаркт', 'острого нарушения мозгового',
        _rx(r'нарушени\w* мозгового кровообращ'),
    ]),
    ('hydrocephalus', [
        'гидроцефал', 'вентрикуломегал', 'окклюзионн', _rx(r'выраженн\w* расширени\w* (боковых |обоих |желудочк)'),
        'гидроцефальн',
    ]),
    ('ms_wml', [
        'демиелин', 'рассеянн', _rx(r'очаг\w*[^.;,]{0,60}бел\w* веществ'),
        _rx(r'бел\w* веществ\w*[^.;]{0,60}очаг'), _rx(r'очаг\w*[^.;]{0,80}сосудистого генеза'),
        _rx(r'очаг\w*[^.;]{0,60}патологическ\w* (интенсивност|сигнал|плотност)'),
        _rx(r'(участ|глиоз)\w*[^.;]{0,90}бел\w* веществ'),
        _rx(r'очагов\w* изменени'), 'лейкоэнцефалопат', _rx(r'очаг\w*[^.;]{0,40}(полушар|доли|доле)'),
        'васкулит', 'очаги глиоза', 'очагов глиоза', 'глиозные очаги',
        'сосудистого характера', 'дисмиелин', 'очагового поражения', 'мультифокальн',
        'очаги ', 'очагов ', 'очаг ',
    ]),
    ('cyst', [
        'кист', _rx(r'(?<!суб)арахноидальн'), 'ликворн',
    ]),
    ('atrophy', [
        'атроф', 'микроангиопат', 'fazekas', 'фазекас', 'лейкоареоз', 'small vessel',
        _rx(r'расширени\w* (субарахноидальн|конвекситальн|борозд|наружных|ликворных)'),
        _rx(r'расширени\w* (боковых |обоих )?желудочк'), _rx(r'расширени\w* ликворн'),
        'инволютив', 'заместительн', 'гидроцефалия ex vacuo', 'уменьшение объема',
        'церебральн склероз', 'энцефалопат', 'дисциркулят', 'хроническ ишеми', 'хроническая ишеми',
        'хронической ишеми',
    ]),
    ('other', [
        'мальформац', 'аневризм', 'глиоз', 'кавернозн', 'кавернома', 'отек', 'отёк',
        'эктопи', 'нейроваскулярн', _rx(r'\bспо\b'), 'гиппокамп', 'ангиом', 'кистозн дегенерац',
        _rx(r'кистозн\w* дегенерац'), 'дегенерац', 'рубцов', 'посттравмат', 'энцефаломаляц',
        'киари', 'арнольд', 'гипоплаз', 'аплаз', 'дисплаз', 'гетеротоп', 'порэнцефал',
        'энцефалит', 'абсцесс', 'менингит', 'воспалит', 'контузи', 'ушиб', 'сотрясен',
        'состояние после', 'состояния после', 'послеоперац', 'после удаления',
        'после операц', 'резекци', 'трепанац', 'шунт', 'кистозно-глиозн', 'кистозно-атроф',
        'гигром', 'пустое турецкое', 'пустого турецкого', 'формирующееся пустое',
        'участок патологической', 'участки патологической', 'зона патологической',
        'диффузн', 'токсич', 'метаболич', 'нейродегенерат', 'болезнь', 'синдром',
        'гипоксич', 'лейкомаляц', 'кальцинат', 'обызвествл', 'миграци', 'агенез',
        'дизгенез', 'липома', 'мегалэнцефал', 'микроцефал', 'сирингом', 'тромбоз',
        'стеноз', 'окклюз', 'ангиопат', 'сдавлен', 'компресс', 'дислокац', 'смещение срединных',
        'перифокальн', 'патологическ', 'изменени', 'структур', 'образован', 'дегенерат',
        'кальцификат', 'вазогенн', 'расширение большой цистерны', 'тройничн', 'конфликт',
    ]),
]
# Clauses whose only brain finding is one of these ("other" lookalikes) are
# NOT abnormal by the reference rubric — incidental/normal variants.
NORMAL_VARIANT_CUES = (
    'внутричерепной гипертензи', 'интракраниальной гипертензи', 'вчг',
    'megacisterna', 'мега цистерн', 'большой цистерны', 'mega cisterna',
    'киста шишковидной', 'кисты шишковидной', 'киста эпифиза',
    'киста ратке', 'кисты ратке', 'кармана ратке',
    'полость прозрачной перегородки', 'полости прозрачной перегородки', 'cavum',
    'asymmetr', 'асимметри', 'вариант развития', 'вариант строения',
    'кальцинат', 'обызвествл', 'физиологическ', 'возрастн',
    'киста хороидального', 'кисты хороидального', 'кист хороидального',
    'периваскулярн', 'вирхов', 'робин', 'ликворн', 'дилатац', 'гиперплазии гипофиза',
)


def normalise(text: str) -> str:
    t = (text or '').replace('ё', 'е').replace('Ё', 'Е').lower()
    t = re.sub(r'\s+', ' ', t).strip()
    t = t.lstrip('. ').strip()
    return t


def strip_boilerplate(text: str) -> str:
    """Cut the letterhead / signature / disclaimer tail off a conclusion.

    The stage-1 parser already does this; the batch-1 reference file
    (data/hospital_reports_labels.csv) does not, so the labeller strips it
    again. Recommendation sentences are dropped at clause level instead.
    """
    t = normalise(text)
    cut = len(t)
    for cue in BOILERPLATE_CUES:
        i = t.find(cue)
        if i > 0:
            cut = min(cut, i)
        elif i == 0:
            return ''
    return t[:cut].strip()


_SENT_SPLIT = re.compile(r'(?<!\d)[.;!?\n]+(?!\d)|\n+')


def split_sentences(text: str) -> list[str]:
    parts = [p.strip(' ,-–—:') for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if p]


def split_clauses(sentence: str) -> list[str]:
    """Split on commas that are NOT inside parentheses ("(вероятно, аденома)" stays whole)."""
    parts, depth, cur = [], 0, []
    for ch in sentence:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        if ch == ',' and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append(''.join(cur))
    parts = [p.strip(' -–—:') for p in parts]
    return [p for p in parts if p]


def is_recommendation(sentence: str) -> bool:
    return any(c in sentence for c in RECOMMENDATION_CUES)


def _contains(text: str, kws: Iterable) -> bool:
    for k in kws:
        if isinstance(k, re.Pattern):
            if k.search(text):
                return True
        elif k in text:
            return True
    return False


def _has_positive_marker(clause: str) -> bool:
    return any(m in clause for m in POSITIVE_MARKERS)


# ------------------------------------------------------------ grammatical case
# Words whose case ending is misleading for the heuristic below.
_NOM_WORDS = frozenset(('картина', 'признаки', 'признак', 'зона', 'зоны', 'участок', 'участки',
                        'очаг', 'очаги', 'киста', 'кисты', 'опухоль', 'состояние', 'головной'))
_SOFT_ADJ_NOM = frozenset(('средний', 'внутренний', 'задний', 'нижний', 'верхний', 'поздний',
                           'ранний', 'передний', 'крайний', 'последний'))
_CASE_SKIP = frozenset(('мрт', 'мр', 'кт', 'мскт', 'мра', 'мскта', 'мрта', 'ct', 'mr', 'mri'))


def _word_case(words: list[str], k: int = 0) -> str:
    """'gen' | 'nom' for words[k] (Russian ending heuristic, finding nouns/adjectives)."""
    if k >= len(words):
        return 'nom'
    w = words[k]
    if w in _NOM_WORDS or w in _SOFT_ADJ_NOM:
        return 'nom'
    if not re.fullmatch(r'[а-я]+', w):
        return 'nom'                                   # latin / abbreviation: treat as asserted
    if w.endswith(('ов', 'ев', 'ых', 'их', 'ого', 'его', 'ии', 'ей', 'изм', 'ом', 'ист', 'ния')):
        return 'gen'                                   # очагов, объемных, окклюзии, аневризм, гематом, кист, кровоизлияния
    if w.endswith('ий'):
        return 'nom' if w[-3:-2] in 'кгхжшчщ' else 'gen'   # ишемический / кровоизлияний, изменений
    if w.endswith('ой'):
        nxt = words[k + 1] if k + 1 < len(words) else ''
        # "головной мозг" (masc. nom.) vs "левой лобной доли" (fem. gen.)
        return 'nom' if nxt and re.search(r'[бвгджзклмнпрстфхцчшщ]$', nxt) else (
            _word_case(words, k + 1) if nxt.endswith('ой') else 'gen')
    if w.endswith(('ма', 'ста')):
        return 'nom'                                   # гематома, аневризма, киста
    if w.endswith('а'):
        return 'gen'                                   # очага, отека, инсульта, инфаркта
    return 'nom'                                       # -ое/-ая/-ый/-ие/-ия/-ые/-ь/-и/-ы/consonant


def clause_head_case(clause: str) -> str:
    """Grammatical case of the word that opens a clause ('gen' or 'nom').

    Inside a negated comma list every item is governed by the negation verb and
    stands in the GENITIVE ("очагов ишемии, кровоизлияний, объемных образований
    не выявлено"); an asserted finding opens in the NOMINATIVE ("Объемное
    образование …", "Внутримозговая гематома …", "Ишемический инсульт …").
    """
    words = [w for w in re.findall(r'[а-яa-z]+', clause) if w not in _CASE_SKIP and len(w) > 1]
    return _word_case(words, 0) if words else 'nom'


def is_side_effect_tail(clause: str) -> bool:
    """True when a negated clause only denies a complication (oedema, shift, …)."""
    if not SIDE_EFFECT_RX.search(clause):
        return False
    rest = SIDE_EFFECT_RX.sub(' ', clause)
    return not (_finding_hits(rest) - {'other'})


def _strip_hedges(clause: str) -> str:
    return HEDGE_RX.sub(' ', clause)


def negation_scope(sentence: str) -> tuple[list[str], list[bool], list[bool]]:
    """Return (clauses, negated, review) for one sentence.

    Rules (module docstring step 3, hardened after review):
      * a clause is negated by its own local/existential cue (hedges removed first);
      * an existential negation extends BACKWARDS over the preceding comma
        clauses of the sentence, but
          - a clause that names a finding (any class) is negated by the
            extension ONLY when it is a genitive list item without an asserting
            marker ("очагов ишемии, кровоизлияний … не выявлено"); a clause that
            asserts its finding in the nominative ("Объемное образование …,
            смещения срединных структур не выявлено") STOPS the extension;
          - a short list-final tail ("…, аневризм, АВМ не выявлено") may still
            reach over an asserting clause, but only when that clause names
            nothing but an 'other'-class vascular finding, never after a
            side-effect tail, and the row is then flagged for review;
          - a side-effect tail (отек / смещение / дислокация / прорыв /
            трансформация / кровоизлияние в опухоль) never overrides a marker.
    `review[j]` marks clauses whose negation came from that short-tail override.
    """
    clauses = split_clauses(sentence)
    neg = [False] * len(clauses)
    review = [False] * len(clauses)
    for i, c in enumerate(clauses):
        probe = _strip_hedges(c)
        if _contains(probe, NEG_LOCAL):
            neg[i] = True
        if not _contains(probe, NEG_EXISTENTIAL):
            continue
        neg[i] = True
        list_tail = len(c.split()) <= 4 and not is_side_effect_tail(c)
        for j in range(i - 1, -1, -1):
            if neg[j]:
                continue            # already inside a negation (own cue or an earlier scope)
            cj = clauses[j]
            if HEDGE_RX.search(cj):
                break               # a hedged clause is an assertion ("нельзя исключить X, …")
            marker = _has_positive_marker(cj)
            hits = _finding_hits(cj)
            if hits:
                if not marker and clause_head_case(cj) == 'gen':
                    neg[j] = True
                    continue
                if marker and list_tail and hits <= {'other'} and _contains(cj, VASCULAR_TERMS):
                    neg[j] = review[j] = True
                    continue
                break               # an asserted finding: never negated from outside
            if marker and not list_tail:
                break
            neg[j] = True
    return clauses, neg, review


def _unhedged_part(clause: str) -> str:
    """The part of a clause asserted WITHOUT a hedge.

    A hedge inside parentheses qualifies only the parenthetical
    ("очаги в белом веществе (не исключается демиелинизация)" → the очаги are
    certain); a hedge outside parentheses governs the whole clause.
    """
    t = re.sub(r'\([^()]*\)', lambda m: ' ' if HEDGE_RX.search(m.group(0)) else m.group(0), clause)
    return '' if HEDGE_RX.search(t) else t


def negated_clauses(sentence: str) -> list[tuple[str, bool]]:
    """Return [(clause, negated)] for one sentence (see negation_scope)."""
    clauses, neg, _ = negation_scope(sentence)
    return list(zip(clauses, neg))


def is_non_brain_clause(clause: str) -> bool:
    # Brain vascular territories ("в бассейне левой СМА", "позвоночной
    # артерии") are removed before BOTH anatomy tests: they name where a brain
    # finding sits, not a non-brain organ.
    probe = _VASCULAR_TERRITORY_RX.sub(' ', clause)
    if _contains(probe, STRONG_NON_BRAIN) and 'мозг' not in clause:
        return True
    return _contains(probe, NON_BRAIN_CONTEXT) and not _contains(clause, BRAIN_CONTEXT)


def _finding_hits(clause: str) -> set[str]:
    hits: set[str] = set()
    for label, kws in FINDINGS:
        if _contains(clause, kws):
            hits.add(label)
    return _refine_hits(clause, hits)


def _refine_hits(clause: str, hits: set[str]) -> set[str]:
    """Class-specific disambiguation on ONE clause."""
    if not hits:
        return hits
    # cysts: sinus / scalp / Rathke / pineal / choroid cysts are not brain cysts;
    # "кистозно-глиозные" changes are post-stroke → other.
    if 'cyst' in hits:
        if _contains(clause, ('кистозно-глиоз', 'кистозно-атроф', 'глиозно-кист', 'кистозно-солидн',
                              'кистозно-слизист', 'постишем', 'постинфаркт', 'энцефаломаляц')):
            hits.discard('cyst')
            if 'tumor' not in hits:
                hits.add('other')
        if _contains(clause, ('прозрачной перегородки', 'cavum', 'верге', 'большой цистерны',
                              'megacisterna', 'mega cisterna', 'кистозн дегенерац', 'кистозной дегенерац',
                              'кистозной дегенрац')) or re.search(r'кистозн\w* дегенерац', clause):
            hits.discard('cyst')
            hits.add('other')
        if 'ликворн' in clause and 'кист' not in clause and 'арахноид' not in clause:
            hits.discard('cyst')
    # "кистозное образование" is a cyst unless something says tumour explicitly
    if 'tumor' in hits and 'cyst' in hits and re.search(r'кистозн\w* (\(?образован)', clause) \
            and not _contains(clause, ('солидн', 'опухол', 'менингиом', 'глиом', 'аденом', 'метастат',
                                       'метастаз', 'новообразов', 'объемн', 'бластом', 'неврином', 'шванном')):
        hits.discard('tumor')
    # cavernoma: haemosiderin / "blooming" is not a bleed unless a haemorrhage is stated
    if 'hemorrhage' in hits and _contains(clause, ('каверн', 'гемосидерин', 'blooming')) \
            and not _contains(clause, ('кровоизлия', 'геморраг', 'гематом')):
        hits.discard('hemorrhage')
    # stroke: "гипоксическо-ишемическая / хроническая ишемия" is an encephalopathy
    # (→ atrophy by the reference), not an infarct.
    if 'stroke' in hits and not _contains(clause, ('инсульт', 'инфаркт', 'лакунар', 'онмк', 'энцефаломаляц',
                                                    'постишем', 'постинфаркт',
                                                    _rx(r'нарушени\w* мозгового'))):
        ischemic = re.sub(r'(гипокси\w*[- ]|хроническ\w* |сосудисто-)ишеми\w*', '', clause)
        if 'ишеми' not in ischemic:
            hits.discard('stroke')
            if 'энцефалопат' in clause or 'хроническ' in clause:
                hits.add('atrophy')
    # "расширение желудочков" without гидроцефалия is an atrophic/ex-vacuo pattern
    if 'hydrocephalus' in hits and not _contains(clause, ('гидроцефал', 'вентрикуломегал', 'окклюзионн')):
        if not _contains(clause, ('выраженн',)):
            hits.discard('hydrocephalus')
            hits.add('atrophy')
    # hemorrhage: scalp/subcutaneous haematoma is not a brain bleed
    if 'hemorrhage' in hits and _contains(clause, ('подкожн', 'мягких тканей', 'скальп', 'кожи',
                                                    'подапоневрот', 'кефалогематом')) \
            and not _contains(clause, ('внутримозгов', 'субдурал', 'эпидурал', 'субарахн', 'паренхим',
                                       'желудочк', 'мозга', 'полушар')):
        hits.discard('hemorrhage')
    # tumor: post-operative state after tumour removal → other; "образование" in
    # sinus handled by non-brain filter already.
    if 'tumor' in hits and _contains(clause, ('после удаления', 'состояние после', 'состояния после',
                                               'послеоперац', 'после операц', 'после резекц',
                                               'после хирург')):
        if not _contains(clause, ('рецидив', 'продолженн', 'остаточн', 'резидуальн')):
            hits.discard('tumor')
            hits.add('other')
    # ms_wml: an encephalopathy clause without real focal words is not WML
    if 'ms_wml' in hits and 'энцефалопат' in clause and not _contains(
            clause, ('очаг', 'демиелин', 'рассеянн', 'белого вещества', 'белом веществе', 'лейкоэнцефалопат')):
        hits.discard('ms_wml')
    # ms_wml: "очаг" that is clearly a tumour/hemorrhage/stroke descriptor is not WML
    if 'ms_wml' in hits and (hits & {'tumor', 'hemorrhage', 'stroke'}):
        if not _contains(clause, ('демиелин', 'рассеянн', 'белого вещества', 'белом веществе')):
            hits.discard('ms_wml')
    # vessel anatomy variants (hypoplastic A1, tortuous ICA, stenosis on angiography) are
    # not brain-parenchymal findings by the reference rubric
    if 'other' in hits and _contains(clause, ('гипоплаз', 'аплаз', 'стеноз', 'окклюз', 'извитост',
                                               'атеросклероз', 'сдавлен', 'компресс')) \
            and _contains(clause, ('артери', 'сегмент', 'вса', 'пма', 'сма', 'зма', 'сосуд', 'синус',
                                   'вены', 'вена')) \
            and not _contains(clause, ('мальформац', 'аневризм', 'тромбоз', 'инфаркт', 'ишеми')):
        hits.discard('other')
    # other: "изменени/структур/образован/патологическ" are generic — keep 'other'
    # only when nothing more specific matched in the clause, and drop it for the
    # normal-variant phrases the rubric treats as normal.
    if 'other' in hits:
        if hits - {'other'}:
            hits.discard('other')
        elif _contains(clause, NORMAL_VARIANT_CUES) and not _contains(
                clause, ('мальформац', 'аневризм', 'глиоз', 'каверн', 'киари', 'энцефалит', 'абсцесс',
                         'отек', 'после', 'гигром', 'тромбоз', 'стеноз', 'окклюз', 'дислокац')):
            hits.discard('other')
    return hits


@dataclass
class Analysis:
    label: str = 'normal'
    abnormal: str = 'no'
    confidence: str = 'low'
    needs_review: str = 'yes'
    findings: list[tuple[str, str]] = field(default_factory=list)   # (label, clause)
    negated: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)                # non-brain clauses
    normal_cue: bool = False
    positive_text_len: int = 0
    text: str = ''
    certain: set = field(default_factory=set)     # classes asserted without a hedge
    hedged: set = field(default_factory=set)      # classes named only under a hedge
    scope_review: bool = False                    # a short-tail negation reached over an asserting clause


def analyse_conclusion(conclusion: str) -> Analysis:
    """Apply the label policy to ONE conclusion text."""
    a = Analysis()
    text = strip_boilerplate(conclusion)
    a.text = text
    if not text:
        a.label, a.abnormal, a.confidence, a.needs_review = 'normal', 'no', 'low', 'yes'
        return a
    # "(?)" / "менингиома?" is a hedge; '?' would otherwise split the sentence.
    text = _QMARK_RX.sub(' под вопросом', text)
    classes: set[str] = set()
    for sentence in split_sentences(text):
        if is_recommendation(sentence) and not _finding_hits(sentence):
            continue
        advice = is_recommendation(sentence)
        clauses, negs, reviews = negation_scope(sentence)
        for clause, neg, rev in zip(clauses, negs, reviews):
            if advice and not _has_positive_marker(clause):
                continue          # "консультация … при эпизодах кровоизлияния" is not a finding
            if neg:
                a.negated.append(clause)
                a.scope_review |= rev
                if _contains(clause, NORMAL_CUES) or 'мозг' in clause:
                    a.normal_cue = True
                continue
            if is_non_brain_clause(clause):
                a.ignored.append(clause)
                continue
            if _contains(clause, CONDITIONAL_CUES) and not _has_positive_marker(clause):
                continue
            hits = _finding_hits(clause)
            if hits:
                for h in hits:
                    a.findings.append((h, clause))
                classes |= hits
                if HEDGE_RX.search(clause):
                    sure = _finding_hits(_unhedged_part(clause))
                    a.certain |= hits & sure
                    a.hedged |= hits - sure
                else:
                    a.certain |= hits
            else:
                a.positive_text_len += len(clause)
    if 'normal' in classes:
        classes.discard('normal')
    if classes:
        a.label = min(classes, key=lambda c: SEVERITY_RANK[c])
        a.abnormal = 'yes'
    else:
        a.label, a.abnormal = 'normal', 'no'
    # confidence
    n_classes = len(classes)
    if n_classes == 0:
        if a.normal_cue or a.ignored:
            a.confidence = 'high' if a.positive_text_len < 40 else 'medium'
            a.needs_review = 'no' if a.positive_text_len < 80 else 'yes'
        else:
            # nothing matched and nothing said "normal": review if there is real text
            a.confidence = 'low' if a.positive_text_len >= 30 else 'medium'
            a.needs_review = 'yes' if a.positive_text_len >= 30 else 'no'
    elif n_classes == 1:
        a.confidence = 'high'
        a.needs_review = 'no'
        if a.label == 'other':
            a.confidence, a.needs_review = 'medium', 'yes'
        elif a.normal_cue and a.label in ('atrophy', 'ms_wml', 'cyst'):
            a.confidence = 'medium'
    else:
        a.confidence = 'medium'
        a.needs_review = 'no'
        # the pairs the reference rubric itself is least consistent about
        if {'tumor', 'cyst'} <= classes or {'stroke', 'ms_wml'} <= classes:
            a.confidence, a.needs_review = 'low', 'yes'
    # a hedged winning finding ("нельзя исключить объемное образование") keeps
    # its label but is never high-confidence and always goes to review
    if classes and a.label in a.hedged and a.label not in a.certain:
        if a.confidence == 'high':
            a.confidence = 'medium'
        a.needs_review = 'yes'
    if a.scope_review:
        if a.confidence == 'high':
            a.confidence = 'medium'
        a.needs_review = 'yes'
    return a


def label_row(row: dict) -> dict:
    """Fill the five label columns of one stage-1 row (in place, returned)."""
    if (row.get('body_part') or '').strip().lower() != 'brain':
        for c in OUTPUT_COLS:
            row[c] = ''
        return row
    a = analyse_conclusion(row.get('conclusion', ''))
    row['label'] = a.label
    row['abnormal'] = a.abnormal
    row['label_confidence'] = a.confidence
    row['label_source'] = 'rules'
    row['needs_review'] = a.needs_review
    return row


# ----------------------------------------------------------------------------
# CSV / JSONL IO
# ----------------------------------------------------------------------------
def read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, newline='', encoding='utf-8') as fh:
        r = csv.DictReader(fh)
        rows = list(r)
        return rows, list(r.fieldnames or [])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})


def _norm_abnormal(v) -> str:
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    s = str(v).strip().lower()
    if s in ('yes', 'true', '1', 'y', 'да'):
        return 'yes'
    if s in ('no', 'false', '0', 'n', 'нет'):
        return 'no'
    return ''


def _norm_date(v: str) -> str:
    s = (v or '').strip()
    m = re.match(r'(\d{4})-?(\d{2})-?(\d{2})$', s)
    if m:
        return f'{m.group(1)}-{m.group(2)}-{m.group(3)}'
    m = re.match(r'(\d{2})\.(\d{2})\.(\d{4})$', s)
    if m:
        return f'{m.group(3)}-{m.group(2)}-{m.group(1)}'
    return s


def _norm_modality(v) -> str:
    return str(v or '').strip().upper()


def _norm_section(v) -> str:
    s = str(v if v is not None else '').strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else s
    except ValueError:
        return s


def row_key(r: dict) -> tuple[str, str, str, str]:
    """(patient_id, study_date, modality, section_index) — identifies ONE stage-1 row.

    One patient can have a CT brain and an MR brain on the same day (stroke
    work-up), and the stage-1 parser emits one row per report section, so the
    patient/date pair alone is not unique.
    """
    return (str(r.get('patient_id', '')).strip(), _norm_date(str(r.get('study_date', '') or '')),
            _norm_modality(r.get('modality', '')), _norm_section(r.get('section_index', '')))


def export_for_llm(rows: list[dict], path: Path, all_rows: bool = False) -> int:
    """Write {"patient_id","study_date","modality","section_index","conclusion"} lines for review rows.

    modality + section_index make every line map back to exactly one row on
    --import-llm-labels; training/relabel_workflow.js echoes them unchanged.
    """
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        for r in rows:
            if (r.get('body_part') or '').lower() != 'brain':
                continue
            if not all_rows and r.get('needs_review') != 'yes':
                continue
            fh.write(json.dumps({'patient_id': r.get('patient_id', ''),
                                 'study_date': r.get('study_date', ''),
                                 'modality': r.get('modality', '') or '',
                                 'section_index': str(r.get('section_index', '') or ''),
                                 'conclusion': strip_boilerplate(r.get('conclusion', ''))},
                                ensure_ascii=False) + '\n')
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and isinstance(obj.get('labels'), list):
                out.extend(obj['labels'])         # a whole workflow result pasted as one line
            else:
                out.append(obj)
    return out


class ImportResult(NamedTuple):
    applied: int
    skipped: int        # invalid label, or no row matches the key
    ambiguous: int      # the key matches >1 row, or a second, different label for an already-labelled row


def import_llm_labels(rows: list[dict], labels: list[dict], source: str = 'import') -> ImportResult:
    """Merge LLM labels into the brain rows, ONE label → ONE row.

    Matching, most specific first (each level only when the label carries the
    fields it needs; a level that matches >1 row makes the label AMBIGUOUS and
    it is not applied — it never falls through to a coarser level):
      1. (patient_id, study_date, modality, section_index)   — what --export-for-llm writes
      2. (patient_id, study_date, modality)                   — only when unique
      3. (patient_id, study_date)                             — only when unique
      4. patient_id alone — only when the label has NO date and the patient
         has exactly one brain row.
    A row already labelled by an earlier line of the same import is never
    relabelled: an identical duplicate is skipped, a conflicting one is counted
    ambiguous and the row is sent back to review (needs_review=yes).
    """
    brain = [r for r in rows if (r.get('body_part') or '').lower() == 'brain']
    idx4: dict[tuple, list[dict]] = defaultdict(list)
    idx3: dict[tuple, list[dict]] = defaultdict(list)
    idx2: dict[tuple, list[dict]] = defaultdict(list)
    idx1: dict[str, list[dict]] = defaultdict(list)
    for r in brain:
        pid, date, mod, sec = row_key(r)
        idx4[(pid, date, mod, sec)].append(r)
        idx3[(pid, date, mod)].append(r)
        idx2[(pid, date)].append(r)
        idx1[pid].append(r)
    applied = skipped = ambiguous = 0
    done: dict[int, tuple[str, str]] = {}          # id(row) → (label, abnormal) applied by THIS import
    for lab in labels:
        label = str(lab.get('label', '')).strip().lower()
        abn = _norm_abnormal(lab.get('abnormal', ''))
        if label not in LABELS or abn == '':
            skipped += 1
            continue
        if label == 'normal' and abn == 'yes':
            abn = 'no'
        if label != 'normal' and abn == 'no':
            abn = 'yes'
        pid, date, mod, sec = row_key(lab)
        levels = []
        if date and mod and sec:
            levels.append(idx4.get((pid, date, mod, sec), []))
        if date and mod:
            levels.append(idx3.get((pid, date, mod), []))
        if date:
            levels.append(idx2.get((pid, date), []))
        else:
            levels.append(idx1.get(pid, []))
        cands: list[dict] = []
        for cands in levels:
            if cands:
                break
        if not cands:
            skipped += 1
            continue
        if len(cands) > 1:
            ambiguous += 1
            continue
        target = cands[0]
        prev = done.get(id(target))
        if prev is not None:
            if prev == (label, abn):
                skipped += 1
            else:
                ambiguous += 1
                target['needs_review'] = 'yes'
                target['label_confidence'] = 'low'
            continue
        target['label'] = label
        target['abnormal'] = abn
        target['label_confidence'] = 'high' if source == 'import' else 'medium'
        target['label_source'] = source
        target['needs_review'] = 'no'
        done[id(target)] = (label, abn)
        applied += 1
    return ImportResult(applied, skipped, ambiguous)


# ----------------------------------------------------------------------------
# Optional local LLM refinement (Ollama)
# ----------------------------------------------------------------------------
def ollama_label(conclusion: str, model: str, url: str, timeout: float = 120.0) -> dict | None:
    prompt = (RUBRIC + '\n\nReturn ONLY a JSON object {"label": <one of ' + '|'.join(LABELS) +
              '>, "abnormal": true|false}.\n\nЗАКЛЮЧЕНИЕ:\n' + strip_boilerplate(conclusion))
    body = json.dumps({'model': model, 'prompt': prompt, 'stream': False, 'format': 'json',
                       'options': {'temperature': 0}}).encode('utf-8')
    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode('utf-8'))
    try:
        obj = json.loads(payload.get('response', ''))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def ollama_available(url: str) -> bool:
    base = url.split('/api/')[0]
    try:
        with urllib.request.urlopen(base + '/api/tags', timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def refine_with_ollama(rows: list[dict], model: str, url: str) -> tuple[int, int]:
    todo = [r for r in rows if (r.get('body_part') or '').lower() == 'brain' and r.get('needs_review') == 'yes']
    if not todo:
        return 0, 0
    if not ollama_available(url):
        print(f'  {Y}⚠ Ollama not reachable at {url} — skipping LLM refinement '
              f'({len(todo)} rows keep needs_review=yes){NC}')
        return 0, len(todo)
    labels = []
    failed = 0
    for i, r in enumerate(todo):
        try:
            obj = ollama_label(r.get('conclusion', ''), model, url)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f'  {Y}⚠ Ollama call failed ({e}); stopping refinement at {i}/{len(todo)}{NC}')
            failed += len(todo) - i
            break
        if not obj:
            failed += 1
            continue
        labels.append({'patient_id': r.get('patient_id'), 'study_date': r.get('study_date'),
                       'modality': r.get('modality', ''), 'section_index': r.get('section_index', ''),
                       'label': obj.get('label', ''), 'abnormal': obj.get('abnormal', '')})
        if (i + 1) % 25 == 0:
            print(f'  …{i + 1}/{len(todo)} refined')
    res = import_llm_labels(rows, labels, source='llm')
    return res.applied, failed + res.skipped + res.ambiguous


# ----------------------------------------------------------------------------
# Benchmark against the batch-1 reference
# ----------------------------------------------------------------------------
def load_reference(ref_csv: Path, reports_csv: Path) -> list[dict]:
    """Join matched_clean (patient_id,label,abnormal) with the conclusions.

    Returns one dict per reference patient: patient_id, ref_label, ref_abnormal,
    conclusion (boilerplate-stripped). Brain rows only.
    """
    ref: dict[str, tuple[str, str]] = {}
    with open(ref_csv, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            lab, abn = (r.get('label') or '').strip(), (r.get('abnormal') or '').strip()
            if lab in LABELS and abn in ('yes', 'no'):
                ref.setdefault(str(r['patient_id']).strip(), (lab, abn))
    out = []
    with open(reports_csv, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            pid = str(r.get('patient_id', '')).strip()
            if pid not in ref:
                continue
            if (r.get('body_part') or '').lower() != 'brain':
                continue
            lab, abn = ref.pop(pid)
            out.append({'patient_id': pid, 'ref_label': lab, 'ref_abnormal': abn,
                        'conclusion': strip_boilerplate(r.get('conclusion', ''))})
    return out


def benchmark(reference: list[dict]) -> dict:
    """Score analyse_conclusion() against a reference list (see load_reference)."""
    n = len(reference)
    abn_ok = lab_ok = 0
    confusion: Counter = Counter()
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    disagreements = []
    for item in reference:
        a = analyse_conclusion(item['conclusion'])
        item['pred_label'], item['pred_abnormal'] = a.label, a.abnormal
        item['confidence'], item['needs_review'] = a.confidence, a.needs_review
        confusion[(item['ref_label'], a.label)] += 1
        per_class[item['ref_label']][1] += 1
        if a.label == item['ref_label']:
            lab_ok += 1
            per_class[item['ref_label']][0] += 1
        if a.abnormal == item['ref_abnormal']:
            abn_ok += 1
        else:
            disagreements.append(item)
    label_dis = [i for i in reference if i['pred_label'] != i['ref_label'] and i['pred_abnormal'] == i['ref_abnormal']]

    def sev_dist(i):
        return abs(SEVERITY_RANK[i['pred_label']] - SEVERITY_RANK[i['ref_label']])
    worst = sorted(disagreements, key=lambda i: (-sev_dist(i), i['confidence'] != 'high', i['patient_id']))
    worst += sorted(label_dis, key=lambda i: (-sev_dist(i), i['patient_id']))
    return {
        'n': n,
        'abnormal_agreement': abn_ok / n if n else 0.0,
        'label_agreement': lab_ok / n if n else 0.0,
        'per_class': {k: (v[0] / v[1] if v[1] else 0.0, v[1]) for k, v in per_class.items()},
        'confusion': confusion,
        'worst': worst,
        'n_review': sum(1 for i in reference if i['needs_review'] == 'yes'),
    }


def print_benchmark(res: dict, n_worst: int) -> None:
    print('\n' + '=' * 72)
    print(f'  {B}LABEL RULES vs REFERENCE ({res["n"]} brain patients){NC}')
    print('=' * 72)
    col = G if res['abnormal_agreement'] >= 0.90 else R
    print(f'  agreement on abnormal : {col}{res["abnormal_agreement"]:.3f}{NC}')
    print(f'  agreement on label    : {res["label_agreement"]:.3f}')
    print(f'  needs_review rows     : {res["n_review"]} ({res["n_review"] / max(res["n"], 1):.1%})')
    print(f'\n  {B}per-class agreement (reference class → recall, n):{NC}')
    for k in SEVERITY:
        if k in res['per_class']:
            acc, cnt = res['per_class'][k]
            print(f'    {k:<14} {acc:.2f}  (n={cnt})')
    print(f'\n  {B}confusion (rows = reference, cols = rules):{NC}')
    short = {'normal': 'norm', 'tumor': 'tum', 'stroke': 'strk', 'atrophy': 'atro', 'hemorrhage': 'hem',
             'ms_wml': 'wml', 'hydrocephalus': 'hydr', 'cyst': 'cyst', 'other': 'oth'}
    print('    ' + ' ' * 14 + ''.join(f'{short[c]:>6}' for c in SEVERITY))
    for r in SEVERITY:
        row = ''.join(f'{res["confusion"].get((r, c), 0):>6}' for c in SEVERITY)
        print(f'    {r:<14}{row}')
    print(f'\n  {B}worst {n_worst} disagreements (patient_id | ref → rules | conclusion[:160]):{NC}')
    for i in res['worst'][:n_worst]:
        print(f'    {i["patient_id"]:>9} | {i["ref_label"]}/{i["ref_abnormal"]} → '
              f'{i["pred_label"]}/{i["pred_abnormal"]} ({i["confidence"]}) | {i["conclusion"][:160]}')
    print(f'\n  {Y}Known limitation of the reference: conclusions in hospital_reports_labels.csv '
          f'are truncated at 600 chars and carried boilerplate (stripped here).{NC}')


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Stage 2: label brain report conclusions (rules + optional LLM).')
    ap.add_argument('--reports', help='stage-1 CSV (parse_hospital_reports.py output)')
    ap.add_argument('--out', help='labelled CSV to write')
    ap.add_argument('--benchmark', action='store_true', help='score the rules against the batch-1 reference')
    ap.add_argument('--reference', default=str(REPO / 'data' / 'matched_clean.csv'))
    ap.add_argument('--reference-reports', default=str(REPO / 'data' / 'hospital_reports_labels.csv'))
    ap.add_argument('--min-agreement', type=float, default=0.90, help='benchmark floor on abnormal agreement')
    ap.add_argument('--worst', type=int, default=20, help='how many disagreements to print')
    ap.add_argument('--benchmark-out', help='optional CSV with every reference row + prediction')
    ap.add_argument('--llm', help='ollama:<model> — refine needs_review rows via local Ollama')
    ap.add_argument('--ollama-url', default='http://localhost:11434/api/generate')
    ap.add_argument('--import-llm-labels',
                    help='JSONL of {patient_id,study_date,modality,section_index,label,abnormal} '
                         '(modality/section_index optional: unique fallback on patient_id+study_date)')
    ap.add_argument('--export-for-llm',
                    help='write JSONL {patient_id,study_date,modality,section_index,conclusion} for review rows')
    ap.add_argument('--all', action='store_true', help='with --export-for-llm: export every brain row')
    args = ap.parse_args(argv)

    if args.benchmark:
        ref_csv, rep_csv = Path(args.reference), Path(args.reference_reports)
        if not ref_csv.exists() or not rep_csv.exists():
            print(f'{R}✗ reference files missing: {ref_csv} / {rep_csv}{NC}')
            return 2
        reference = load_reference(ref_csv, rep_csv)
        if not reference:
            print(f'{R}✗ reference join is empty{NC}')
            return 2
        res = benchmark(reference)
        print_benchmark(res, args.worst)
        if args.benchmark_out:
            write_csv(Path(args.benchmark_out), reference,
                      ['patient_id', 'ref_label', 'ref_abnormal', 'pred_label', 'pred_abnormal',
                       'confidence', 'needs_review', 'conclusion'])
            print(f'  ✓ wrote {args.benchmark_out}')
        if res['abnormal_agreement'] < args.min_agreement:
            print(f'\n  {R}✗ abnormal agreement {res["abnormal_agreement"]:.3f} < {args.min_agreement}{NC}')
            return 1
        print(f'\n  {G}✓ abnormal agreement {res["abnormal_agreement"]:.3f} ≥ {args.min_agreement}{NC}')
        if not args.reports:
            return 0

    if not args.reports or not args.out:
        print(f'{R}✗ --reports and --out are required (or --benchmark){NC}')
        return 2
    src = Path(args.reports)
    if not src.exists():
        print(f'{R}✗ not found: {src}{NC}')
        return 2
    rows, fieldnames = read_csv(src)
    for c in OUTPUT_COLS:
        if c not in fieldnames:
            fieldnames.append(c)
    for r in rows:
        label_row(r)
    brain = [r for r in rows if (r.get('body_part') or '').lower() == 'brain']
    n_rules = len(brain)

    imp = ImportResult(0, 0, 0)
    if args.import_llm_labels:
        p = Path(args.import_llm_labels)
        if not p.exists():
            print(f'{R}✗ not found: {p}{NC}')
            return 2
        imp = import_llm_labels(rows, read_jsonl(p), source='import')

    llm_applied = llm_skipped = 0
    if args.llm:
        if not args.llm.startswith('ollama:'):
            print(f'{R}✗ --llm expects ollama:<model>{NC}')
            return 2
        llm_applied, llm_skipped = refine_with_ollama(rows, args.llm.split(':', 1)[1], args.ollama_url)

    write_csv(Path(args.out), rows, fieldnames)

    n_export = 0
    if args.export_for_llm:
        n_export = export_for_llm(rows, Path(args.export_for_llm), all_rows=args.all)

    labels = Counter(r['label'] for r in brain)
    print('\n' + '=' * 72)
    print(f'  {B}LABELLED {len(rows)} rows — {n_rules} brain rows{NC}')
    print('=' * 72)
    for k in SEVERITY:
        if labels.get(k):
            print(f'    {labels[k]:>5}  {k}')
    abn = Counter(r['abnormal'] for r in brain)
    print(f'  abnormal: yes={abn.get("yes", 0)}  no={abn.get("no", 0)}')
    print(f'  source  : {dict(Counter(r["label_source"] for r in brain))}')
    print(f'  confidence: {dict(Counter(r["label_confidence"] for r in brain))}')
    n_rev = sum(1 for r in brain if r['needs_review'] == 'yes')
    print(f'  needs_review: {Y if n_rev else G}{n_rev}{NC}')
    if args.import_llm_labels:
        col = Y if (imp.skipped or imp.ambiguous) else G
        print(f'  imported LLM labels: {col}{imp.applied} applied, {imp.ambiguous} ambiguous, '
              f'{imp.skipped} skipped{NC}')
        if imp.ambiguous:
            print(f'    {Y}⚠ ambiguous labels were NOT applied (key matched >1 row, or conflicting labels '
                  f'for one row) — export again with modality/section_index and re-run the workflow{NC}')
    if args.llm:
        print(f'  ollama refinement  : {llm_applied} applied, {llm_skipped} skipped')
    if args.export_for_llm:
        print(f'  ✓ exported {n_export} rows for LLM review → {args.export_for_llm}')
        print(f'    next: Workflow training/relabel_workflow.js (args.file) → --import-llm-labels <result.jsonl>')
    print(f'  ✓ {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
