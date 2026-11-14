#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — "PRIORITY FINDING" TARGET FROM THE RADIOLOGY REPORT CONCLUSION
================================================================================
  WHAT
    For every brain study of a batch that has a matched report, decide from the
    report's conclusion (Russian free text) whether it carries a finding that
    should move the read up the worklist — a "priority finding": mass, recent
    infarct, haemorrhage, demyelination, true (internal) hydrocephalus, mass
    effect, vascular lesion (occlusion / aneurysm / malformation / thrombosis),
    infection, urgent referral, or "contrast advised". Also a white-matter /
    atrophy grade (0-3) and whether ANY focal lesion is described.

  WHY
    On 2026-11-14 we found that the shipped triage target ("report abnormal vs
    normal") is matched by patient AGE alone: most "abnormal" reports describe
    only age-typical change (atrophy, microangiopathy). A model can look good on
    that target by learning age. Only about 1 study in 10 carries a finding that
    should change the reading order. scripts/eval_prospective.py (--priority)
    reports how the shipped flag does on THIS target for never-seen patients, so
    that every new batch shows whether the flag catches what matters and how
    much else it flags.

  PROVISIONAL — READ THIS BEFORE USING THE NUMBERS
    rules-v2 are wording rules written on 2026-11-14 from 558 reports of ONE
    clinic (batch new1), rules-v1 plus the fixes of the same day's review;
    on manual reading of rules-v1 about 1 in 10 positives was
    debatable. They are for REPORTING, not for a release gate and not as a
    training target, until radiologists have confirmed them with one-click
    labels. Change RULES_VERSION whenever a rule changes so that numbers from
    different rule sets are never compared silently.

  RULES (rules-v2; rules-v1 was a port of the independent reviewer's provisional rules)
    * Text: ё→е; split into sentences at '.', ';' and new lines, and each
      sentence into clauses at ',', ';', ':' and before 'без'. Negation is
      scoped per clause (see NEGATION): a cue that governs what follows it
      ('признаков X', 'данных за X', 'без X', 'Не выявлено X', 'Нет X')
      drops the rest of the sentence; a cue that governs what precedes it
      ('X не выявлено', 'X нет', 'X отсутствует', 'X исключено') drops the
      sentence up to and including its clause (a negated list stays
      negated). So 'Образование доли, без признаков кровоизлияния' keeps
      the mass. 'Нельзя исключить', 'не исключается', 'не исключена' and
      'для исключения' are NOT negations.
    * Context: mass / haemorrhage / chronic-focal / mass-effect wording in a
      sentence about soft tissue, scalp, orbit, teeth, neck or the paranasal
      sinuses / mastoid does not count unless the sentence also names the
      brain (мозг, полушарие, доля, желудочек, гипофиз, мозжечок). A "cyst"
      next to sinus words is not a brain cyst. External / replacement
      hydrocephalus (наружная, заместительная) is atrophy, not hydrocephalus.
      A microadenoma alone is not a mass (unless mass effect or tumour wording
      — менингиома, глиома, метастаз, опухоль, объёмное — is present).
    * 'Acute' needs acute / subacute / restricted-diffusion wording AND a
      lesion word in the same sentence. A stroke (ОНМК / инсульт) sentence
      counts as recent unless it says sequelae / past / history / chronic /
      cystic / prevention or risk ('в анамнезе', 'после', 'профилактика').
      Vascular lesions count only when specific (occlusion or aneurysm of a
      named artery side/segment, malformation other than Chiari, AVM,
      thrombosis) or any aneurysm. A cephalhaematoma is extracranial; a
      pineal (epiphysis) cyst is not a mass.
    * 'Contrast advised' needs a recommendation word and contrast wording in
      the same sentence, or a contrast procedure item ('МРТ с контрастным
      усилением.') after a sentence that starts with a recommendation word;
      'без контраста', contrast already given ('накопление контраста',
      'после введения') and 'не показано' do not count.
    * Label gate (the study's report label, when known): label 'normal' is
      never priority; label 'atrophy' is priority only when an acute, mass,
      haemorrhage or demyelination family fired (wording like 'contrast
      advised' alone in an atrophy report is routine and does not count).
    * wm_grade: the highest Fazekas (Фазекас) grade written, a range counting
      as its upper end ('I-II' = 2); without one, the severity word next to
      microangiopathy / encephalopathy / atrophy wording
      (незначительн/легк/начальн/минимальн/слабо/нерезко/мало выраженн = 1,
      умеренн = 2, выраженн = 3).
    * focal: any focal lesion family (chronic focal / lacunes / gliosis,
      white-matter foci, demyelination, brain cyst, mass, microadenoma,
      haemorrhage, acute lesion, true hydrocephalus) — not label-gated.

  OUTPUT
    --out CSV, one row per study folder of --matched that is a brain study
    (is_brain = yes) with a matched report (report_matched = yes) of the given
    modality and body part:
      study_folder,patient_id,report_study_date,priority,reasons,wm_grade,focal,rules_version
    priority yes | no | '' (report row not found or empty conclusion);
    reasons ';'-joined (only when priority = yes); wm_grade 0-3 or '';
    focal yes | no | ''. Written atomically. stdout carries COUNTS ONLY — no
    ids, no study folders, no report text.

    Joining a study to its report row reproduces scripts/match_images_to_reports.py:
    the labels row of the same patient_id whose study_date equals the matched
    row's report_study_date, whose modality equals report_modality and whose
    body part is the requested one; among several sections, the one whose label
    equals the matched row's label wins, then the lowest section_index.

  USAGE
    python scripts/report_priority.py --matched data/<batch>_matched.csv \
        --labels data/<batch>_labels.csv --out data/<batch>_priority.csv \
        [--modality MR] [--body-part brain]

  EXIT CODES
    0  written (also when no study qualifies: a header-only file)
    1  unusable input (file missing / unreadable / required columns missing);
       message on stderr, nothing written
================================================================================
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))      # scripts/ is not a package
from match_images_to_reports import norm_date, norm_modality    # noqa: E402

RULES_VERSION = 'rules-v2'
OUTPUT_COLUMNS = ('study_folder', 'patient_id', 'report_study_date', 'priority', 'reasons',
                  'wm_grade', 'focal', 'rules_version')
MATCHED_REQUIRED = ('study_folder', 'patient_id', 'is_brain', 'report_matched', 'report_body_part',
                    'report_modality', 'report_study_date', 'label')
LABELS_REQUIRED = ('patient_id', 'modality', 'body_part', 'study_date', 'label', 'conclusion',
                   'section_index')

# Stable reason names, in the order they are reported.
REASONS = ('mass', 'haemorrhage', 'recent_infarct', 'demyelination', 'hydrocephalus', 'mass_effect',
           'vascular_lesion', 'infection', 'urgent_referral', 'contrast_advised')

_I = re.IGNORECASE


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, _I)


# Negation is scoped per CLAUSE, not per sentence (rules-v2): Russian conclusions often attach
# a negated side clause to a positive finding ('Образование левой лобной доли, признаков
# масс-эффекта нет.'), and dropping the whole sentence would throw the finding away.
#   * PREFIX cues govern what FOLLOWS them ('признаков X', 'данных за X', 'без X', and any cue
#     at the very start of a clause: 'Не выявлено X', 'Нет X'): the rest of the clause and every
#     later clause of the sentence are dropped (an enumeration continues the negation).
#   * POSTFIX cues govern what PRECEDES them ('X не выявлено', 'X нет'): the clause and every
#     earlier clause of the sentence are dropped, because 'Образований, кровоизлияний не
#     выявлено' negates the whole list. This also drops 'Образование доли, кровоизлияния не
#     выявлено' — a deliberate, conservative miss, as before.
#   * 'исключено / исключена' is a negation; 'не исключен(а|о)', 'нельзя исключить', 'не
#     исключается' say "possible" and 'для исключения X' says "suspected": NOT negations.
NEGATION = _rx(r'не выявл|не определя|не обнаруж|не визуализир|не отмеча|не наблюда|отсутств|не получен'
               r'|данных за|признаков\s+(наличия\s+)?\w|\bнет\b|(?<!не )исключен[аоы]?\b'
               r'|\bбез\b')
PREFIX_ONLY = _rx(r'^(данных за|признак|без\b)')
# Lead-in words that may stand before a prefix cue at the start of a clause.
CLAUSE_LEAD = _rx(r'^(\s|также|при этом|убедительн\w*|достоверн\w*|четк\w*|явн\w*|мр[- ]?|кт[- ]?)*$')
SENTENCE_SPLIT = re.compile(r'(?<=[.;])\s+|\n')
CLAUSE_SPLIT = re.compile(r'[,;:]|\s+(?=без\b)', re.IGNORECASE)

# Context that is outside the brain parenchyma.
EXTRA_CRANIAL = _rx(r'мягк\w* ткан|подкожн|кефало|подапоневрот|орбит|челюст|резц|зуб|хрустал|глазн|шейн|щитовид|слюнн|околоушн')
SINUS_CONTEXT = _rx(r'пазух|гаймор|синус|сфеноид|этмоид|носоглот|аденоид|ринит|носов|мастоид|ЛОР|отит|полип'
                    r'|клет\w* решетч')
BRAIN_CONTEXT = _rx(r'мозг|полушари|дол[иья]\b|желудоч|гипофиз|мозжеч')

FAMILY_RX = {
    'chronic_focal': _rx(r'лакунар|кистозн\w* дегенерац|глиоз|энцефаломаляц|постишем|последстви'
                         r'|перенес\w* (инфаркт|онмк)|онмк|старо\w* инфаркт|порэнцефал'),
    'wm_foci': _rx(r'очаг\w*.{0,60}бел\w* веществ|бел\w* веществ.{0,40}очаг|очагов\w* (изменени|поражени)'
                   r'|неспецифическ\w* очаг|единичн\w* очаг'),
    'demyelination': _rx(r'рассеянн\w* склероз|демиелиниз|\bРС\b'),
    'brain_cyst': _rx(r'арахноидальн\w* кист|ликворн\w* кист|кист(а|ы|у|ой|ами|ах)?\b.{0,40}'
                      r'(височн|лобн|теменн|затылочн|черепн\w* ямк|сосудист\w* сплет|полушари|мозжеч|прозрачн'
                      r'|шишковидн|эпифиз)|кистозн\w* образовани\w* (шишковидн|эпифиз)'),
    'mass': _rx(r'образовани|опухол|новообразов|менингиом|глиом|метастаз|\bmts\b|аденом|лимфом|невринома'
                r'|шваннома|контрастирующ'),
    'haemorrhage': _rx(r'кровоизлиян|гематом|геморраг|\bСАК\b|гемосидерин|каверном'),
    'hydrocephalus': _rx(r'(?<!наружной )(?<!заместительной )(внутренн\w* |окклюзионн\w* |тривентрикуляр\w* '
                         r'|сообщающ\w* )?гидроцефал'),
    'mass_effect': _rx(r'масс-?эффект|смещени\w* срединн|дислокац|перифокальн\w* (вазогенн\w* )?отек'
                       r'|компресси\w* (ствол|желудоч)'),
    'vascular_any': _rx(r'окклюзи|аневризм|мальформац|\bАВМ\b|стеноз|тромбоз'),
    'infection': _rx(r'абсцесс|энцефалит|менингит|вентрикулит'),
    'urgent_referral': _rx(r'нейрохирург|срочн|экстренн|неотложн'),
}
# Families whose hit in an extracranial / sinus sentence is ignored unless the brain is named.
_CONTEXT_CHECKED = ('mass', 'haemorrhage', 'chronic_focal', 'mass_effect')

MICROADENOMA = _rx(r'микроаденом')
MICROADENOMA_ONLY = _rx(r'^(?!.*(образовани|опухол|менингиом|глиом|метастаз|mts|контрастирующ)).*микроаденом')
TUMOUR_WORDS = _rx(r'менингиом|глиом|метастаз|mts|опухол|объемн')
# A pineal (epiphysis) cyst is a common incidental finding, not a mass.
PINEAL_CYST = _rx(r'кист\w*.{0,30}(шишковидн|эпифиз)|(шишковидн|эпифиз)\w*.{0,30}кист')
PINEAL_TUMOUR = _rx(r'менингиом|глиом|метастаз|mts|опухол|пинеалом|пинеобласт|солидн')
CYST_KEEP = _rx(r'арахноид|ликворн')
EXTERNAL_HYDRO = _rx(r'наружн\w* (\w+ )?гидроцефал|заместительн\w* гидроцефал')
INTERNAL_HYDRO = _rx(r'внутренн|окклюзион|сообщающ')
ACUTE_WORD = _rx(r'\bостр|подостр|ограничени\w* диффузи')
ACUTE_LESION = _rx(r'инфаркт|ишеми|онмк|кровоизлиян|гематом|очаг|инсульт')
ACUTE_ISCHAEMIC = _rx(r'инфаркт|ишеми|онмк|очаг|инсульт')
STROKE = _rx(r'ОНМК|инсульт')
# A stroke sentence that is about the past, or about preventing / the risk of a stroke, is not a
# recent stroke ('ОНМК в анамнезе', 'Состояние после ОНМК', 'профилактика повторного ОНМК').
STROKE_OLD = _rx(r'последстви|перенес|постинсульт|хронич|кистозн|анамнез|после|давн|резидуальн'
                 r'|профилакт|риск')
ANEURYSM = _rx(r'аневризм')
VASCULAR_SPECIFIC = _rx(r'окклюзи\w* (прав|лев|внутрен|средн|базил)'
                        r'|аневризм\w* (прав|лев|перед|средн|задн|базил|внутрен)|\bАВМ\b|тромбоз')
# 'Мальформация' is vascular unless it is a Chiari malformation (a hindbrain anomaly) and no
# vascular word is in the sentence.
MALFORMATION = _rx(r'мальформац')
CHIARI = _rx(r'киари|chiari')
VASCULAR_MALFORMATION = _rx(r'артериовен|сосудист|венозн|кавернозн|\bАВМ\b')
# 'Contrast advised' (rules-v2) needs a recommendation word AND contrast wording in the SAME
# sentence (rules-v1 searched 200 characters across sentences and fired on 'МРТ без контраста'
# after an unrelated 'Динамическое наблюдение'). A header line ending in ':' ('Рекомендовано:')
# is joined to the line after it. Contrast that was already given, or said to be not needed,
# does not count.
CONTRAST_WORD = _rx(r'контраст(?!ирующ)')
RECOMMEND = _rx(r'рекоменд|показан|целесообраз|необходим')
# Inside the recommendation block (from a sentence that STARTS with a recommendation word to the
# end of the conclusion) a following item sentence such as 'МРТ с контрастным усилением.' is a
# recommendation too: 'Рекомендовано: консультация невролога. МРТ с контрастным усилением.'
RECOMMEND_LEAD = _rx(r'^\W*(рекоменд|показан|целесообраз|необходим)')
CONTRAST_PROCEDURE = _rx(r'с\s+(\w+\s+)?контраст|контрастн\w* (усилен|исследован)|в/в\s+контраст|контрастирован')
CONTRAST_NOT_ADVISED = _rx(r'без\s+(\w+\s+)?контраст|накоплени\w* (\w+ )?контраст|после (\w+ )?введени'
                           r'|не показан|не требу|нет необходим|не целесообраз|не рекоменд|не необходим')

GRADE_CONTEXT = _rx(r'микроангиопат|fazekas|фазекас|энцефалопат|атрофи|лейкоареоз|small vessel'
                    r'|расширени\w* субарахноид')
# A range ('Fazekas I-II') counts as its upper end.
FAZEKAS = _rx(r'(?:fazekas|фазекас)\w*\s*(?:grade|степен\w*)?\s*(III|II|I|3|2|1)'
              r'(?:\s*[-–/]\s*(III|II|I|3|2|1))?(?![\w])')
_FAZEKAS_VALUE = {'I': 1, '1': 1, 'II': 2, '2': 2, 'III': 3, '3': 3}
SEVERE = _rx(r'выраженн')
NOT_SEVERE = _rx(r'(не|слабо|умеренно|незначительно|нерезко|мало)[ -]?выраженн')
MODERATE = _rx(r'умеренн')
MILD = _rx(r'незначительн|легк|начальн|минимальн|слабо|нерезк|мало[ -]?выраженн')

# Label gate: in an 'atrophy' report only these families make it a priority study.
ATROPHY_OVERRIDE = ('acute', 'mass', 'haemorrhage', 'demyelination')


def normalise(text: Optional[str]) -> str:
    return (text or '').replace('ё', 'е').replace('Ё', 'Е')


def _is_prefix_cue(clause: str, m: re.Match) -> bool:
    """A cue that governs the words AFTER it: an inherently prefix cue ('признаков', 'данных за',
    'без'), or any cue standing at the start of its clause ('Не выявлено очагов', 'Нет данных')."""
    return bool(PREFIX_ONLY.match(m.group(0))) or bool(CLAUSE_LEAD.match(clause[:m.start()]))


def positive_part(sentence: str) -> str:
    """The non-negated clauses of one sentence, joined by ', ' (see NEGATION for the scope rules).

    WHY per clause: a negated side clause ('…, без признаков кровоизлияния') must not throw away
    the positive finding in front of it, while a negated list must stay negated as a whole."""
    clauses = CLAUSE_SPLIT.split(sentence)
    keep: list[Optional[str]] = [c.strip() for c in clauses]
    for i, clause in enumerate(clauses):
        if keep[i] is None:
            continue                                  # already inside a prefix negation
        m = NEGATION.search(clause)
        if not m:
            continue
        if _is_prefix_cue(clause, m):
            keep[i] = clause[:m.start()].strip()      # text before the cue stays
            for j in range(i + 1, len(clauses)):
                keep[j] = None
        else:
            for j in range(0, i + 1):
                keep[j] = None
    return ', '.join(k for k in keep if k)


def positive_sentences(text: str) -> list[str]:
    """Sentences of the conclusion with their negated clauses removed (empty ones dropped)."""
    out = []
    for s in SENTENCE_SPLIT.split(text):
        p = positive_part(s.strip()) if s.strip() else ''
        if p:
            out.append(p)
    return out


def contrast_advised(text: str) -> bool:
    """A sentence (or a 'Рекомендовано:' header plus its next line, or an item of the
    recommendation block) that recommends contrast."""
    parts = [p.strip() for p in SENTENCE_SPLIT.split(text) if p.strip()]
    joined: list[str] = []
    for p in parts:
        if joined and joined[-1].endswith(':'):
            joined[-1] = joined[-1] + ' ' + p
        else:
            joined.append(p)
    in_block = False
    for s in joined:
        if RECOMMEND_LEAD.search(s):
            in_block = True
        if CONTRAST_NOT_ADVISED.search(s) or not CONTRAST_WORD.search(s):
            continue
        if RECOMMEND.search(s) or (in_block and CONTRAST_PROCEDURE.search(s)):
            return True
    return False


def _vascular_specific(sentence: str) -> bool:
    if VASCULAR_SPECIFIC.search(sentence):
        return True
    return bool(MALFORMATION.search(sentence)) and (not CHIARI.search(sentence)
                                                    or bool(VASCULAR_MALFORMATION.search(sentence)))


def _family_hit(name: str, sentence: str) -> bool:
    if not FAMILY_RX[name].search(sentence):
        return False
    if name == 'brain_cyst' and SINUS_CONTEXT.search(sentence) and not CYST_KEEP.search(sentence):
        return False
    if name == 'hydrocephalus' and EXTERNAL_HYDRO.search(sentence) and not INTERNAL_HYDRO.search(sentence):
        return False
    if name == 'mass' and MICROADENOMA_ONLY.search(sentence):
        return False                      # a microadenoma alone is handled by its own flag
    if name == 'mass' and PINEAL_CYST.search(sentence) and not PINEAL_TUMOUR.search(sentence):
        return False                      # a pineal cyst ('кистозное образование эпифиза') is not a mass
    if name in _CONTEXT_CHECKED and (EXTRA_CRANIAL.search(sentence) or SINUS_CONTEXT.search(sentence)) \
            and not BRAIN_CONTEXT.search(sentence):
        return False
    return True


def families(conclusion: Optional[str]) -> dict[str, bool]:
    """Every rule family on one conclusion (no label gate). Exposed for tests and audits."""
    text = normalise(conclusion)
    sents = positive_sentences(text)
    out = {name: any(_family_hit(name, s) for s in sents) for name in FAMILY_RX}
    out['microadenoma'] = any(MICROADENOMA.search(s) for s in sents)
    acute = [s for s in sents if not EXTRA_CRANIAL.search(s) and ACUTE_WORD.search(s) and ACUTE_LESION.search(s)]
    out['acute'] = bool(acute)
    out['acute_ischaemic'] = any(ACUTE_ISCHAEMIC.search(s) for s in acute)
    out['recent_stroke'] = any(STROKE.search(s) and not STROKE_OLD.search(s) for s in sents)
    out['aneurysm'] = any(ANEURYSM.search(s) for s in sents)
    out['vascular_specific'] = any(_vascular_specific(s) for s in sents)
    out['tumour_words'] = any(TUMOUR_WORDS.search(s) for s in sents)
    out['contrast_advised'] = contrast_advised(text)
    return out


def wm_grade(conclusion: Optional[str]) -> int:
    """0-3: the highest Fazekas grade written; without one, the severity word of the
    microangiopathy / encephalopathy / atrophy sentence(s)."""
    sents = positive_sentences(normalise(conclusion))
    fazekas = 0
    by_words = 0
    for s in sents:
        if not GRADE_CONTEXT.search(s):
            continue
        for m in FAZEKAS.finditer(s):
            for g in m.groups():
                if g:
                    fazekas = max(fazekas, _FAZEKAS_VALUE[g.upper()])
        if SEVERE.search(s) and not NOT_SEVERE.search(s):
            by_words = max(by_words, 3)
        elif MODERATE.search(s):
            by_words = max(by_words, 2)
        elif MILD.search(s):
            by_words = max(by_words, 1)
    return fazekas if fazekas > 0 else by_words


def _reasons(f: dict[str, bool]) -> list[str]:
    """Priority reason names that fired (before the label gate), in REASONS order."""
    mass = f['mass'] and not (f['microadenoma'] and not f['mass_effect'] and not f['tumour_words'])
    hit = {
        'mass': mass,
        'haemorrhage': f['haemorrhage'] or (f['acute'] and not f['acute_ischaemic']),
        'recent_infarct': (f['acute'] and f['acute_ischaemic']) or f['recent_stroke'],
        'demyelination': f['demyelination'],
        'hydrocephalus': f['hydrocephalus'],
        'mass_effect': f['mass_effect'],
        'vascular_lesion': f['vascular_specific'] or f['aneurysm'],
        'infection': f['infection'],
        'urgent_referral': f['urgent_referral'],
        'contrast_advised': f['contrast_advised'],
    }
    return [r for r in REASONS if hit[r]]


def classify(conclusion: Optional[str], label: Optional[str] = None
             ) -> tuple[str, tuple[str, ...], Optional[int], str]:
    """(priority, reasons, wm_grade, focal) for one report conclusion.

    priority 'yes' | 'no' | '' (no usable text); reasons only when priority is 'yes';
    wm_grade 0-3 (None without text); focal 'yes' | 'no' | ''.
    `label` is the study's report label (label_reports.py); None / '' = unknown, no gate.
    """
    if not normalise(conclusion).strip():
        return '', (), None, ''
    f = families(conclusion)
    reasons = _reasons(f)
    lab = (label or '').strip().lower()
    if lab == 'normal':
        reasons = []
    elif lab == 'atrophy':
        if any(f[k] for k in ATROPHY_OVERRIDE):
            if f['mass'] and 'mass' not in reasons:     # the gate uses the raw mass family
                reasons = sorted(reasons + ['mass'], key=REASONS.index)
        else:
            reasons = []
    focal = any(f[k] for k in ('chronic_focal', 'wm_foci', 'demyelination', 'brain_cyst', 'mass',
                               'haemorrhage', 'acute', 'hydrocephalus', 'microadenoma'))
    return ('yes' if reasons else 'no'), tuple(reasons), wm_grade(conclusion), ('yes' if focal else 'no')


# ----------------------------------------------------------------------------- CLI

class InputError(Exception):
    """Unusable input: message for stderr (paths and column names only, never data)."""


def read_table(path: Path, required: tuple[str, ...], what: str) -> list[dict]:
    if not path.is_file():
        raise InputError(f'{what} file not found: {path}')
    try:
        with open(path, encoding='utf-8-sig', newline='') as fh:     # -sig: a csv re-saved by Excel
            reader = csv.DictReader(fh)
            cols = set(reader.fieldnames or [])
            missing = [c for c in required if c not in cols]
            if missing:
                raise InputError(f'{what} file {path} lacks column(s): {", ".join(missing)}')
            return list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise InputError(f'{what} file {path} is unreadable: {type(exc).__name__}') from None


def _section(v: Optional[str]) -> int:
    try:
        return int(str(v).strip())
    except ValueError:
        return 10 ** 6


def pick_section(cands: list[dict], label: str) -> Optional[dict]:
    """Same-label section first, then the lowest section_index."""
    if not cands:
        return None
    lab = (label or '').strip()
    return min(cands, key=lambda r: (0 if (r.get('label') or '').strip() == lab else 1,
                                     _section(r.get('section_index'))))


def build_rows(matched: list[dict], labels: list[dict], modality: str, body_part: str
               ) -> tuple[list[dict], Counter]:
    want_mod = norm_modality(modality)
    want_bp = body_part.strip().lower()
    by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in labels:
        if (r.get('body_part') or '').strip().lower() != want_bp:
            continue
        key = ((r.get('patient_id') or '').strip(), norm_date(r.get('study_date') or ''),
               norm_modality(r.get('modality') or ''))
        by_key[key].append(r)
    rows: list[dict] = []
    stats: Counter = Counter()
    seen: set[str] = set()
    for m in matched:
        folder = (m.get('study_folder') or '').strip()
        if not folder or folder in seen:
            continue
        if (m.get('is_brain') or '').strip().lower() != 'yes' \
                or (m.get('report_matched') or '').strip().lower() != 'yes' \
                or (m.get('report_body_part') or '').strip().lower() != want_bp \
                or norm_modality(m.get('report_modality') or '') != want_mod:
            continue
        seen.add(folder)
        pid = (m.get('patient_id') or '').strip()
        label = (m.get('label') or '').strip()
        key = (pid, norm_date(m.get('report_study_date') or ''), want_mod)
        rep = pick_section(by_key.get(key, []), label)
        if rep is None:
            stats['report_row_not_found'] += 1
        if rep is not None and len(by_key[key]) > 1:
            stats['several_sections'] += 1
        priority, reasons, grade, focal = classify(rep.get('conclusion') if rep else None, label)
        rows.append({'study_folder': folder, 'patient_id': pid,
                     'report_study_date': (m.get('report_study_date') or '').strip(),
                     'priority': priority, 'reasons': ';'.join(reasons),
                     'wm_grade': '' if grade is None else str(grade), 'focal': focal,
                     'rules_version': RULES_VERSION})
        stats[f'label:{label or "?"}:{priority or "unknown"}'] += 1
    return rows, stats


def write_csv_atomic(out: Path, rows: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(OUTPUT_COLUMNS))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)


def print_summary(rows: list[dict], stats: Counter, out: Path) -> None:
    """Counts only: no ids, no study folders, no report text."""
    pr = Counter(r['priority'] or 'unknown' for r in rows)
    print(f'report_priority ({RULES_VERSION}, PROVISIONAL — reporting only, not a gate)')
    print(f'  brain studies with a matched report: {len(rows)}')
    print(f'  priority yes {pr["yes"]}   no {pr["no"]}   unknown (no usable conclusion) {pr["unknown"]}')
    if stats['report_row_not_found']:
        print(f'  report row not found in the labels file: {stats["report_row_not_found"]}')
    if stats['several_sections']:
        print(f'  studies whose report date had several sections (label, then section_index chose): '
              f'{stats["several_sections"]}')
    reasons = Counter(x for r in rows for x in r['reasons'].split(';') if x)
    if reasons:
        print('  by reason: ' + ', '.join(f'{k} {reasons[k]}' for k in REASONS if reasons[k]))
    by_label: dict[str, Counter] = defaultdict(Counter)
    for k, v in stats.items():
        if k.startswith('label:'):
            _, lab, p = k.split(':', 2)
            by_label[lab][p] += v
    if by_label:
        print('  priority yes by report label: ' + ', '.join(
            f'{lab} {c["yes"]}/{sum(c.values())}' for lab, c in sorted(by_label.items())))
    focal = Counter(r['focal'] for r in rows)
    grades = Counter(r['wm_grade'] for r in rows if r['wm_grade'] != '')
    print(f'  focal lesion described: {focal["yes"]}   wm_grade 0/1/2/3: '
          + '/'.join(str(grades[str(g)]) for g in range(4)))
    print(f'  written: {out}')


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Provisional "priority finding" target from report conclusions '
                                             '(counts only on stdout).')
    ap.add_argument('--matched', required=True, help='data/<batch>_matched.csv (match_images_to_reports.py)')
    ap.add_argument('--labels', required=True, help='data/<batch>_labels.csv (label_reports.py)')
    ap.add_argument('--out', required=True, help='data/<batch>_priority.csv')
    ap.add_argument('--modality', default='MR')
    ap.add_argument('--body-part', default='brain')
    args = ap.parse_args(argv)
    if not norm_modality(args.modality):
        print(f'unknown --modality {args.modality!r} (use MR or CT)', file=sys.stderr)
        return 1
    try:
        matched = read_table(Path(args.matched), MATCHED_REQUIRED, 'matched')
        labels = read_table(Path(args.labels), LABELS_REQUIRED, 'labels')
    except InputError as exc:
        print(f'report_priority: {exc}; nothing written', file=sys.stderr)
        return 1
    rows, stats = build_rows(matched, labels, args.modality, args.body_part)
    out = Path(args.out)
    write_csv_atomic(out, rows)
    print_summary(rows, stats, out)
    if not rows:
        print('  note: no study qualified — header-only file written')
    return 0


if __name__ == '__main__':
    sys.exit(main())
