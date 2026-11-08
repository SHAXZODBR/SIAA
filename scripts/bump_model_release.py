#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — BUMP EVERY PINNED VALIDATION NUMBER AFTER A MODEL RELEASE
================================================================================
  WHAT
    Stage 9 ("bump-docs") of the release pipeline. Reads the validated numbers
    of the model release (models/brain_triage_finetuned/study_level_eval.json +
    MANIFEST.json, through src/inference/validated_metrics.py) and rewrites every
    place that quotes them by hand: the hospital-facing markdown documents in
    three languages, the two READMEs, the Windows laptop checklists, and the
    FALLBACK block of validated_metrics.py itself (what a frozen installer
    without the model dir shows).

  HOW
    A table of (file, regex, template) entries, each anchored to the exact
    phrase that surrounds a number today ("… on 178 held-out local patients"),
    with the digits as wildcards. Every entry must match EXACTLY ONCE — zero or
    several matches means a document was reworded and the script stops naming
    the file and the pattern, so nothing is silently skipped. Running it twice
    changes nothing (idempotent); the numbers themselves are never used as
    anchors, so the same table works before and after a bump.

    Derived prose — words that follow from the numbers ("about half of normal
    studies are flagged", "the small number of normal studies … imprecise") —
    is templated too (derived_prose()), so a release with spec 0.72 on 480
    normals never reads "about half" or "small number".

    The runtime strings (registry card notes, report templates, brain_analysis
    label) are NOT in this table: they are generated from validated_metrics at
    import time and cannot go stale.

  USAGE
    python scripts/bump_model_release.py                 # rewrite in place, print what changed
    python scripts/bump_model_release.py --dry-run       # unified diff, write nothing
    python scripts/bump_model_release.py --check         # exit 1 when any file is out of sync
    python scripts/bump_model_release.py --model models/brain_triage_candidate_x --dry-run
    (--root <dir> rewrites a copy of the repo — used by tests)

  EXIT CODES
    0 ok / nothing to do        1 anchor missing or ambiguous, or --check found drift
    2 metrics could not be read (no study_level_eval.json / MANIFEST.json in --model)
================================================================================
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.inference import validated_metrics as vm  # noqa: E402  (torch-free)

G, Y, R, B, NC = '\033[0;32m', '\033[1;33m', '\033[0;31m', '\033[0;34m', '\033[0m'

EXIT_DRIFT = 1
EXIT_NO_METRICS = 2

# regex fragments for the wildcards
D = r'\d[.,]\d{2}'          # 0.90 / 0,90
I = r'\d+'                  # 178
H = r'[0-9a-f]{12}'         # 0d559766ce58
DT = r'\d{4}-\d{2}-\d{2}'   # 2026-09-07
FL = r'[0-9]+\.[0-9]+(?:e-?\d+)?'   # python float repr


def ru_form(n: int, singular: str, plural: str) -> str:
    """Noun after a numeral in an oblique case: 1/21/131 → singular, else plural
    (11–14 → plural)."""
    return singular if (n % 10 == 1 and n % 100 != 11) else plural


def build_context(m: vm.Metrics) -> dict[str, str]:
    """All the tokens the templates may use, pre-formatted per language."""
    miss = vm.miss_one_in(m)
    ctx: dict[str, str] = {
        'tp': str(m.tp), 'fn': str(m.fn), 'tn': str(m.tn), 'fp': str(m.fp),
        'n': str(m.n_studies), 'n_abn': str(m.n_abnormal), 'n_norm': str(m.n_normal),
        'n_slices': str(m.n_slices), 'date': m.evaluated_on, 'identity': m.identity,
        'miss': str(miss),
        # python reprs for the FALLBACK block
        'sens_raw': repr(float(m.sens)), 'spec_raw': repr(float(m.spec)), 'acc_raw': repr(float(m.acc)),
        'min_sens_raw': repr(float(m.min_sens)), 'min_spec_raw': repr(float(m.min_spec)),
        'thr_raw': repr(float(m.threshold)),
        # Russian noun forms (Uzbek and English do not decline after numerals)
        'abn_gen_ru': ru_form(m.n_abnormal, 'патологического исследования', 'патологических исследований'),
        'norm_gen_ru': ru_form(m.n_normal, 'нормального исследования', 'нормальных исследований'),
        'miss_abn_ru': ru_form(miss, 'патологического исследования', 'патологических исследований'),
        'patients_prep_ru': ru_form(m.n_studies, 'отложенном пациенте', 'отложенных пациентах'),
        'patients_local_prep_ru': ru_form(m.n_studies, 'отложенном локальном пациенте', 'отложенных локальных пациентах'),
        'patients_pilot_ru': ru_form(m.n_studies, 'локальном пациенте', 'локальных пациентах'),
        'slices_gen_ru': ru_form(m.n_slices, 'аксиального среза', 'аксиальных срезов'),
    }
    for lang in ('en', 'ru'):
        ctx[f'sens_{lang}'] = vm.dec(m.sens, lang)
        ctx[f'spec_{lang}'] = vm.dec(m.spec, lang)
        ctx[f'acc_{lang}'] = vm.dec(m.acc, lang)
        ctx[f'min_sens_{lang}'] = vm.dec(m.min_sens, lang)
        ctx[f'min_spec_{lang}'] = vm.dec(m.min_spec, lang)
        ctx[f'thr_{lang}'] = vm.dec(m.threshold, lang)
        ctx[f'spec_ci_{lang}'] = vm.ci(m.spec_ci, lang)
        ctx[f'sens_ci_{lang}'] = vm.ci(m.sens_ci, lang)
    ctx.update(derived_prose(m))
    return ctx


# ---- derived prose: sentences whose WORDS (not only digits) follow from the numbers ----------
SMALL_NORMALS = 100     # below this many normal studies the specificity CI is called "imprecise"


def normals_flagged_in_10(spec: float) -> int:
    """'about k in 10 normal studies are flagged': k = round((1 − spec) × 10), half-up,
    clamped to 1..10 (a gate-passing model has spec ≥ 0.40, so k ≤ 6 in practice)."""
    k = int((1.0 - float(spec)) * 10 + 0.5)
    return min(10, max(1, k))


def derived_prose(m: vm.Metrics) -> dict[str, str]:
    """Wording that is DERIVED from the numbers and must change with them:

      user manual   'about half of normal studies are flagged'  (k = 5)
                    'about {k} in 10 normal studies are flagged' (any other k)
      intended use  'Note the small number of normal studies (N): … imprecise (95% CI roughly …)'
                    while N < SMALL_NORMALS, else 'With N normal studies the specificity
                    estimate has a 95% CI of …' — en / ru / uz variants.

    k == 5 keeps today's 'half' wording, so a bump back to the current release reproduces the
    documents byte for byte."""
    k = normals_flagged_in_10(m.spec)
    n = m.n_normal
    ci_en, ci_ru = vm.ci(m.spec_ci, 'en'), vm.ci(m.spec_ci, 'ru')
    out = {
        'normals_flagged_en': 'about half of' if k == 5 else f'about {k} in 10',
        'normals_flagged_ru': ('около половины нормальных **отмечаются**' if k == 5 else
                               f'около {k} из 10 нормальных **{"отмечается" if k == 1 else "отмечаются"}**'),
        'normals_flagged_uz': ('normal tekshiruvlarning esa taxminan yarmi' if k == 5 else
                               f'har 10 ta normal tekshiruvdan esa taxminan {k} tasi'),
    }
    if n < SMALL_NORMALS:
        out['spec_note_en'] = (f'Note the small number of normal studies ({n}): the specificity estimate is '
                               f'imprecise (95% CI roughly {ci_en})')
        out['spec_note_ru'] = (f'Обратите внимание на малое число нормальных исследований ({n}): оценка '
                               f'специфичности неточна (95 % ДИ примерно {ci_ru})')
        out['spec_note_uz'] = (f'Normal tekshiruvlar sonining kamligiga ({n}) e’tibor bering: spetsifiklik '
                               f'bahosi noaniq (95 % ishonch oralig‘i taxminan {ci_ru})')
    else:
        out['spec_note_en'] = f'With {n} normal studies the specificity estimate has a 95% CI of {ci_en}'
        out['spec_note_ru'] = (f'При {n} {ru_form(n, "нормальном исследовании", "нормальных исследованиях")} '
                               f'95 % ДИ специфичности составляет {ci_ru}')
        out['spec_note_uz'] = f'Normal tekshiruvlar soni {n} ta: spetsifiklikning 95 % ishonch oralig‘i {ci_ru}'
    return out


# --------------------------------------------------------------------------- the table
# (file relative to root, regex that must match exactly once, str.format template)

IDENTITY_ELLIPSIS = (r'`' + H + r'…`', '`{identity}…`')

TABLE: list[tuple[str, str, str]] = [
    # ---------------- intended use (performance table + "how it was measured")
    ('docs/intended_use_en.md',
     r'about 1 in ' + I + r' abnormal studies in local validation was not flagged',
     'about 1 in {miss} abnormal studies in local validation was not flagged'),
    ('docs/intended_use_en.md',
     r'\| \*\*' + D + r'\*\* \(' + I + r' of ' + I + r' abnormal studies flagged\) \|',
     '| **{sens_en}** ({tp} of {n_abn} abnormal studies flagged) |'),
    ('docs/intended_use_en.md',
     r'\| \*\*' + D + r'\*\* \(' + I + r' of ' + I + r' normal studies not flagged\) \|',
     '| **{spec_en}** ({tn} of {n_norm} normal studies not flagged) |'),
    ('docs/intended_use_en.md', r'\| Accuracy \| ' + D + r' \|', '| Accuracy | {acc_en} |'),
    ('docs/intended_use_en.md',
     r'tested on \*\*' + I + r' held-out patients\*\* \(' + I + r' abnormal, ' + I + r' normal\) not used in training — ' + I + r' axial slices',
     'tested on **{n} held-out patients** ({n_abn} abnormal, {n_norm} normal) not used in training — {n_slices} axial slices'),
    ('docs/intended_use_en.md', r'flagged when the mean is ≥ ' + D + r' — the same protocol',
     'flagged when the mean is ≥ {thr_en} — the same protocol'),
    ('docs/intended_use_en.md', r'\(evaluated ' + DT + r'\)', '(evaluated {date})'),
    ('docs/intended_use_en.md', r'falls below sensitivity ' + D + r' or specificity ' + D,
     'falls below sensitivity {min_sens_en} or specificity {min_spec_en}'),
    ('docs/intended_use_en.md',                       # derived prose: see derived_prose()
     r'(?:Note the small number of normal studies \(' + I + r'\): the specificity estimate is imprecise '
     r'\(95% CI roughly ' + D + r'–' + D + r'\)|With ' + I + r' normal studies the specificity estimate has a '
     r'95% CI of ' + D + r'–' + D + r')',
     '{spec_note_en}'),
    ('docs/intended_use_en.md', *IDENTITY_ELLIPSIS),

    ('docs/intended_use_ru.md',
     r'примерно 1 из ' + I + r' патологическ(?:ого исследования|их исследований) не отмечалось',
     'примерно 1 из {miss} {miss_abn_ru} не отмечалось'),
    ('docs/intended_use_ru.md',
     r'\| \*\*' + D + r'\*\* \(отмечено ' + I + r' из ' + I + r' патологическ(?:ого исследования|их исследований)\) \|',
     '| **{sens_ru}** (отмечено {tp} из {n_abn} {abn_gen_ru}) |'),
    ('docs/intended_use_ru.md',
     r'\| \*\*' + D + r'\*\* \(не отмечено ' + I + r' из ' + I + r' нормальн(?:ого исследования|ых исследований)\) \|',
     '| **{spec_ru}** (не отмечено {tn} из {n_norm} {norm_gen_ru}) |'),
    ('docs/intended_use_ru.md', r'\| Точность \(accuracy\) \| ' + D + r' \|', '| Точность (accuracy) | {acc_ru} |'),
    ('docs/intended_use_ru.md',
     r'проверена на \*\*' + I + r' отложенн(?:ом пациенте|ых пациентах)\*\* \(' + I + r' с патологией, ' + I + r' без\), не участвовавших в обучении — ' + I + r' аксиальн(?:ого среза|ых срезов)',
     'проверена на **{n} {patients_prep_ru}** ({n_abn} с патологией, {n_norm} без), не участвовавших в обучении — {n_slices} {slices_gen_ru}'),
    ('docs/intended_use_ru.md', r'отмечается при среднем ≥ ' + D + r' — тот же протокол',
     'отмечается при среднем ≥ {thr_ru} — тот же протокол'),
    ('docs/intended_use_ru.md', r'\(оценка от ' + DT + r'\)', '(оценка от {date})'),
    ('docs/intended_use_ru.md', r'чувствительность падает ниже ' + D + r' или специфичность ниже ' + D,
     'чувствительность падает ниже {min_sens_ru} или специфичность ниже {min_spec_ru}'),
    ('docs/intended_use_ru.md',
     r'(?:Обратите внимание на малое число нормальных исследований \(' + I + r'\): оценка специфичности неточна '
     r'\(95 % ДИ примерно ' + D + r'–' + D + r'\)|При ' + I + r' нормальн(?:ом исследовании|ых исследованиях) '
     r'95 % ДИ специфичности составляет ' + D + r'–' + D + r')',
     '{spec_note_ru}'),
    ('docs/intended_use_ru.md', *IDENTITY_ELLIPSIS),

    ('docs/intended_use_uz.md',
     r'har ' + I + r' ta patologik tekshiruvdan taxminan 1 tasi belgilanmagan',
     'har {miss} ta patologik tekshiruvdan taxminan 1 tasi belgilanmagan'),
    ('docs/intended_use_uz.md',
     r'\| \*\*' + D + r'\*\* \(' + I + r' ta patologik tekshiruvdan ' + I + r' tasi belgilangan\) \|',
     '| **{sens_ru}** ({n_abn} ta patologik tekshiruvdan {tp} tasi belgilangan) |'),
    ('docs/intended_use_uz.md',
     r'\| \*\*' + D + r'\*\* \(' + I + r' ta normal tekshiruvdan ' + I + r' tasi belgilanmagan\) \|',
     '| **{spec_ru}** ({n_norm} ta normal tekshiruvdan {tn} tasi belgilanmagan) |'),
    ('docs/intended_use_uz.md', r'\| Aniqlik \(accuracy\) \| ' + D + r' \|', '| Aniqlik (accuracy) | {acc_ru} |'),
    ('docs/intended_use_uz.md',
     r'\*\*' + I + r' ta ajratib qo‘yilgan bemorda\*\* \(' + I + r' patologik, ' + I + r' normal\) sinovdan o‘tkazilgan: ' + I + r' ta aksial kesim',
     '**{n} ta ajratib qo‘yilgan bemorda** ({n_abn} patologik, {n_norm} normal) sinovdan o‘tkazilgan: {n_slices} ta aksial kesim'),
    ('docs/intended_use_uz.md', r'o‘rtacha qiymat ≥ ' + D + r' bo‘lsa tekshiruv belgilanadi — bu dasturiy',
     'o‘rtacha qiymat ≥ {thr_ru} bo‘lsa tekshiruv belgilanadi — bu dasturiy'),
    ('docs/intended_use_uz.md', r'\(baholash sanasi ' + DT + r'\)', '(baholash sanasi {date})'),
    ('docs/intended_use_uz.md', r'Sezgirlik ' + D + r' dan yoki spetsifiklik ' + D + r' dan past bo‘lsa',
     'Sezgirlik {min_sens_ru} dan yoki spetsifiklik {min_spec_ru} dan past bo‘lsa'),
    ('docs/intended_use_uz.md',
     r'(?:Normal tekshiruvlar sonining kamligiga \(' + I + r'\) e’tibor bering: spetsifiklik bahosi noaniq '
     r'\(95 % ishonch oralig‘i taxminan ' + D + r'–' + D + r'\)|Normal tekshiruvlar soni ' + I + r' ta: '
     r'spetsifiklikning 95 % ishonch oralig‘i ' + D + r'–' + D + r')',
     '{spec_note_uz}'),
    ('docs/intended_use_uz.md', *IDENTITY_ELLIPSIS),

    # ---------------- user manual (detector table row)
    ('docs/user_manual_en.md', r'flags the study when the mean is ≥ ' + D + r'\.',
     'flags the study when the mean is ≥ {thr_en}.'),
    ('docs/user_manual_en.md',
     r'Study-level sensitivity ' + D + r' and specificity ' + D + r' on ' + I + r' held-out local patients\. In other words: about 1 in ' + I + r' abnormal studies',
     'Study-level sensitivity {sens_en} and specificity {spec_en} on {n} held-out local patients. In other words: about 1 in {miss} abnormal studies'),
    ('docs/user_manual_en.md', r'about (?:half of|\d+ in 10) normal studies \*\*are\*\* flagged',
     '{normals_flagged_en} normal studies **are** flagged'),
    ('docs/user_manual_ru.md', r'отмечает исследование при среднем ≥ ' + D + r'\.',
     'отмечает исследование при среднем ≥ {thr_ru}.'),
    ('docs/user_manual_ru.md',
     r'Чувствительность ' + D + r' и специфичность ' + D + r' на уровне исследования на ' + I + r' отложенн(?:ом локальном пациенте|ых локальных пациентах)\. Иначе говоря: примерно 1 из ' + I + r' патологическ(?:ого исследования|их исследований)',
     'Чувствительность {sens_ru} и специфичность {spec_ru} на уровне исследования на {n} {patients_local_prep_ru}. Иначе говоря: примерно 1 из {miss} {miss_abn_ru}'),
    ('docs/user_manual_ru.md', r'около (?:половины|\d+ из 10) нормальных \*\*(?:отмечаются|отмечается)\*\*',
     '{normals_flagged_ru}'),
    ('docs/user_manual_uz.md', r'o‘rtacha qiymat ≥ ' + D + r' bo‘lsa tekshiruvni belgilaydi',
     'o‘rtacha qiymat ≥ {thr_ru} bo‘lsa tekshiruvni belgilaydi'),
    ('docs/user_manual_uz.md',
     r'\| ' + I + r' ta ajratib qo‘yilgan mahalliy bemorda tekshiruv darajasida sezgirlik \(sensitivity\) ' + D + r' va spetsifiklik \(specificity\) ' + D + r'\. Boshqacha aytganda: har ' + I + r' ta patologik tekshiruvdan',
     '| {n} ta ajratib qo‘yilgan mahalliy bemorda tekshiruv darajasida sezgirlik (sensitivity) {sens_ru} va spetsifiklik (specificity) {spec_ru}. Boshqacha aytganda: har {miss} ta patologik tekshiruvdan'),
    ('docs/user_manual_uz.md',
     r'(?:normal tekshiruvlarning esa taxminan yarmi|har 10 ta normal tekshiruvdan esa taxminan \d+ tasi) \*\*belgilanadi\*\*',
     '{normals_flagged_uz} **belgilanadi**'),

    # ---------------- pilot evaluation protocol (vendor figures + gate floors)
    ('docs/pilot_evaluation_protocol_en.md',
     r'\(sensitivity ' + D + r', specificity ' + D + r' on ' + I + r' local patients\) and against the release-gate floors \(' + D + r' / ' + D + r'\)',
     '(sensitivity {sens_en}, specificity {spec_en} on {n} local patients) and against the release-gate floors ({min_sens_en} / {min_spec_en})'),
    ('docs/pilot_evaluation_protocol_en.md',
     r'Triage sensitivity point estimate ≥ ' + D + r' \*\*and\*\* specificity ≥ ' + D,
     'Triage sensitivity point estimate ≥ {min_sens_en} **and** specificity ≥ {min_spec_en}'),
    ('docs/pilot_evaluation_protocol_ru.md',
     r'\(чувствительность ' + D + r', специфичность ' + D + r' на ' + I + r' локальн(?:ом пациенте|ых пациентах)\) и с порогами выпуска \(' + D + r' / ' + D + r'\)',
     '(чувствительность {sens_ru}, специфичность {spec_ru} на {n} {patients_pilot_ru}) и с порогами выпуска ({min_sens_ru} / {min_spec_ru})'),
    ('docs/pilot_evaluation_protocol_ru.md',
     r'Точечная оценка чувствительности триажа ≥ ' + D + r' \*\*и\*\* специфичности ≥ ' + D,
     'Точечная оценка чувствительности триажа ≥ {min_sens_ru} **и** специфичности ≥ {min_spec_ru}'),
    ('docs/pilot_evaluation_protocol_uz.md',
     r'\(' + I + r' ta mahalliy bemorda sezgirlik ' + D + r', spetsifiklik ' + D + r'\) va reliz darvozasi pollari \(' + D + r' / ' + D + r'\)',
     '({n} ta mahalliy bemorda sezgirlik {sens_ru}, spetsifiklik {spec_ru}) va reliz darvozasi pollari ({min_sens_ru} / {min_spec_ru})'),
    ('docs/pilot_evaluation_protocol_uz.md',
     r'nuqtaviy bahosi ≥ ' + D + r' \*\*va\*\* spetsifiklik ≥ ' + D,
     'nuqtaviy bahosi ≥ {min_sens_ru} **va** spetsifiklik ≥ {min_spec_ru}'),

    # ---------------- admin guide (identity + release gate)
    ('docs/admin_install_guide_en.md', r'Current release: `' + H + r'…`', 'Current release: `{identity}…`'),
    ('docs/admin_install_guide_en.md',
     r'if study-level sensitivity < ' + D + r' or specificity < ' + D + r' at the production operating point \(mean over 5 central slices, threshold ' + D + r'\)',
     'if study-level sensitivity < {min_sens_en} or specificity < {min_spec_en} at the production operating point (mean over 5 central slices, threshold {thr_en})'),
    ('docs/admin_install_guide_ru.md', r'Текущий выпуск: `' + H + r'…`', 'Текущий выпуск: `{identity}…`'),
    ('docs/admin_install_guide_ru.md',
     r'если чувствительность на уровне исследования < ' + D + r' или специфичность < ' + D + r' в рабочей точке продукта \(среднее по 5 центральным срезам, порог ' + D + r'\)',
     'если чувствительность на уровне исследования < {min_sens_ru} или специфичность < {min_spec_ru} в рабочей точке продукта (среднее по 5 центральным срезам, порог {thr_ru})'),
    ('docs/admin_install_guide_uz.md', r'Joriy reliz: `' + H + r'…`', 'Joriy reliz: `{identity}…`'),
    ('docs/admin_install_guide_uz.md',
     r'\(5 ta markaziy kesim bo‘yicha o‘rtacha, chegara ' + D + r'\) tekshiruv darajasidagi sezgirlik < ' + D + r' yoki spetsifiklik < ' + D + r' bo‘lsa',
     '(5 ta markaziy kesim bo‘yicha o‘rtacha, chegara {thr_ru}) tekshiruv darajasidagi sezgirlik < {min_sens_ru} yoki spetsifiklik < {min_spec_ru} bo‘lsa'),

    # ---------------- attributions (EN table row + RU summary)
    ('docs/attributions.md',
     r'study-level sensitivity ' + D + r' / specificity ' + D + r' on ' + I + r' held-out local patients\. `models/brain_triage_finetuned/`, pinned in `MANIFEST.json` \(SHA-256 `' + H + r'…`\)',
     'study-level sensitivity {sens_en} / specificity {spec_en} on {n} held-out local patients. `models/brain_triage_finetuned/`, pinned in `MANIFEST.json` (SHA-256 `{identity}…`)'),
    ('docs/attributions.md', r'SHA-256 `' + H + r'…`\) обучена и проверена', 'SHA-256 `{identity}…`) обучена и проверена'),
    ('docs/attributions.md',
     r'\(чувствительность ' + D + r' / специфичность ' + D + r' на ' + I + r' отложенн(?:ом пациенте|ых пациентах)\)',
     '(чувствительность {sens_ru} / специфичность {spec_ru} на {n} {patients_prep_ru})'),

    # ---------------- READMEs
    ('docs/README.md',
     r'study-level sensitivity ' + D + r' / specificity ' + D + r' on ' + I + r' held-out local patients \(mean over 5 central axial slices, threshold ' + D + r'\)',
     'study-level sensitivity {sens_en} / specificity {spec_en} on {n} held-out local patients (mean over 5 central axial slices, threshold {thr_en})'),
    ('docs/README.md', r'`MANIFEST.json` SHA-256 `' + H + r'…`', '`MANIFEST.json` SHA-256 `{identity}…`'),
    ('README.md',
     r'validated on \*\*' + I + r' held-out Tashkent patients\*\* \(study-level sensitivity ' + D + r' / specificity ' + D + r'\)',
     'validated on **{n} held-out Tashkent patients** (study-level sensitivity {sens_en} / specificity {spec_en})'),

    # ---------------- Windows laptop checklists (identity printed on the PDF)
    ('docs/windows_laptop_test_en.md', r'the model identity `' + H + r'`', 'the model identity `{identity}`'),
    ('docs/windows_laptop_test_ru.md', r'идентификатор модели `' + H + r'`', 'идентификатор модели `{identity}`'),

    # ---------------- validated_metrics FALLBACK block (frozen installer without the model)
    ('src/inference/validated_metrics.py', r'^FALLBACK_SENS = ' + FL + r'$', 'FALLBACK_SENS = {sens_raw}'),
    ('src/inference/validated_metrics.py', r'^FALLBACK_SPEC = ' + FL + r'$', 'FALLBACK_SPEC = {spec_raw}'),
    ('src/inference/validated_metrics.py', r'^FALLBACK_ACC = ' + FL + r'$', 'FALLBACK_ACC = {acc_raw}'),
    ('src/inference/validated_metrics.py',
     r'^FALLBACK_TP, FALLBACK_FN, FALLBACK_TN, FALLBACK_FP = ' + I + r', ' + I + r', ' + I + r', ' + I + r'$',
     'FALLBACK_TP, FALLBACK_FN, FALLBACK_TN, FALLBACK_FP = {tp}, {fn}, {tn}, {fp}'),
    ('src/inference/validated_metrics.py', r'^FALLBACK_N_STUDIES = ' + I + r'$', 'FALLBACK_N_STUDIES = {n}'),
    ('src/inference/validated_metrics.py', r'^FALLBACK_N_SLICES = ' + I + r'$', 'FALLBACK_N_SLICES = {n_slices}'),
    ('src/inference/validated_metrics.py', r"^FALLBACK_EVALUATED_ON = '" + DT + r"'$", "FALLBACK_EVALUATED_ON = '{date}'"),
    ('src/inference/validated_metrics.py', r"^FALLBACK_IDENTITY = '" + H + r"'$", "FALLBACK_IDENTITY = '{identity}'"),
    ('src/inference/validated_metrics.py', r'^FALLBACK_MIN_SENS = ' + FL + r'$', 'FALLBACK_MIN_SENS = {min_sens_raw}'),
    ('src/inference/validated_metrics.py', r'^FALLBACK_MIN_SPEC = ' + FL + r'$', 'FALLBACK_MIN_SPEC = {min_spec_raw}'),
    ('src/inference/validated_metrics.py', r'^FALLBACK_THRESHOLD = ' + FL + r'$', 'FALLBACK_THRESHOLD = {thr_raw}'),
]

# ---------------- sales documents (docs/sales/)
# Copied verbatim from docs/sales/test_sales_docs.py (BUMP_ENTRIES), which proves they match
# exactly once, render the current release byte for byte and move every figure for another one.
# Same D / I fragments and build_context keys as the table above.
_EN_PATIENTS = (r"\*\*" + I + r" held-out patients\*\*", "**{n} held-out patients**")
_EN_FIGURES = (
    r"\*\*sensitivity " + D + r"\*\* \(" + I + r" of " + I + r" abnormal studies flagged\) and \*\*specificity "
    + D + r"\*\* \(" + I + r" of " + I + r" normal studies not flagged; 95% CI " + D + r"–" + D + r"\)",
    "**sensitivity {sens_en}** ({tp} of {n_abn} abnormal studies flagged) and **specificity {spec_en}** "
    "({tn} of {n_norm} normal studies not flagged; 95% CI {spec_ci_en})",
)
_RU_PATIENTS = (r"\*\*" + I + r" отложенн(?:ом пациенте|ых пациентах)\*\*", "**{n} {patients_prep_ru}**")
_RU_FIGURES = (
    r"\*\*чувствительность " + D + r"\*\* \(отмечено " + I + r" из " + I
    + r" патологическ(?:ого исследования|их исследований)\) и \*\*специфичность " + D + r"\*\* \(не отмечено "
    + I + r" из " + I + r" нормальн(?:ого исследования|ых исследований); 95 % ДИ " + D + r"–" + D + r"\)",
    "**чувствительность {sens_ru}** (отмечено {tp} из {n_abn} {abn_gen_ru}) и **специфичность {spec_ru}** "
    "(не отмечено {tn} из {n_norm} {norm_gen_ru}; 95 % ДИ {spec_ci_ru})",
)

_SALES_ENTRIES: list[tuple[str, str, str]] = [
    ("docs/sales/one_pager_en.md", *_EN_PATIENTS),
    ("docs/sales/one_pager_en.md", *_EN_FIGURES),
    ("docs/sales/one_pager_en.md",
     r"about 1 in " + I + r" abnormal studies is not flagged, and (?:about half of|about \d+ in 10) normal studies "
     r"are flagged as well",
     "about 1 in {miss} abnormal studies is not flagged, and {normals_flagged_en} normal studies are flagged as well"),

    ("docs/sales/one_pager_ru.md", *_RU_PATIENTS),
    ("docs/sales/one_pager_ru.md", *_RU_FIGURES),
    ("docs/sales/one_pager_ru.md",
     r"примерно 1 из " + I + r" патологическ(?:ого исследования|их исследований) не отмечается, а около "
     r"(?:половины|\d+ из 10) нормальных \*\*(?:отмечаются|отмечается)\*\*",
     "примерно 1 из {miss} {miss_abn_ru} не отмечается, а {normals_flagged_ru}"),

    ("docs/sales/one_pager_uz.md", r"\*\*" + I + r" ta ajratib qo‘yilgan bemorda\*\*",
     "**{n} ta ajratib qo‘yilgan bemorda**"),
    ("docs/sales/one_pager_uz.md",
     r"\*\*sezgirlik " + D + r"\*\* \(" + I + r" ta patologik tekshiruvdan " + I + r" tasi belgilangan\) va "
     r"\*\*spetsifiklik " + D + r"\*\* \(" + I + r" ta normal tekshiruvdan " + I + r" tasi belgilanmagan; "
     r"95 % ishonch oralig‘i " + D + r"–" + D + r"\)",
     "**sezgirlik {sens_ru}** ({n_abn} ta patologik tekshiruvdan {tp} tasi belgilangan) va **spetsifiklik "
     "{spec_ru}** ({n_norm} ta normal tekshiruvdan {tn} tasi belgilanmagan; 95 % ishonch oralig‘i {spec_ci_ru})"),
    ("docs/sales/one_pager_uz.md",
     r"har " + I + r" ta patologik tekshiruvdan taxminan 1 tasi belgilanmaydi, (?:normal tekshiruvlarning esa "
     r"taxminan yarmi|har 10 ta normal tekshiruvdan esa taxminan \d+ tasi) \*\*belgilanadi\*\*",
     "har {miss} ta patologik tekshiruvdan taxminan 1 tasi belgilanmaydi, {normals_flagged_uz} **belgilanadi**"),

    ("docs/sales/pilot_proposal_en.md", *_EN_PATIENTS),
    ("docs/sales/pilot_proposal_en.md", *_EN_FIGURES),
    ("docs/sales/pilot_proposal_en.md", r"release-gate floors \(" + D + r" / " + D + r"\)",
     "release-gate floors ({min_sens_en} / {min_spec_en})"),
    ("docs/sales/pilot_proposal_en.md", r"Triage sensitivity point estimate ≥ " + D + r" \*\*and\*\* specificity ≥ " + D,
     "Triage sensitivity point estimate ≥ {min_sens_en} **and** specificity ≥ {min_spec_en}"),

    ("docs/sales/pilot_proposal_ru.md", *_RU_PATIENTS),
    ("docs/sales/pilot_proposal_ru.md", *_RU_FIGURES),
    ("docs/sales/pilot_proposal_ru.md", r"порогами выпуска \(" + D + r" / " + D + r"\)",
     "порогами выпуска ({min_sens_ru} / {min_spec_ru})"),
    ("docs/sales/pilot_proposal_ru.md",
     r"Точечная оценка чувствительности триажа ≥ " + D + r" \*\*и\*\* специфичности ≥ " + D,
     "Точечная оценка чувствительности триажа ≥ {min_sens_ru} **и** специфичности ≥ {min_spec_ru}"),
]

TABLE += _SALES_ENTRIES

FILES: list[str] = list(dict.fromkeys(f for f, _, _ in TABLE))   # unique, in table order


# --------------------------------------------------------------------------- engine

class AnchorError(Exception):
    """A pattern matched zero or several times — the document was reworded."""


def rewrite_text(text: str, entries: list[tuple[str, str]], ctx: dict[str, str], fname: str,
                 errors: list[str]) -> str:
    for pattern, template in entries:
        matches = list(re.finditer(pattern, text, re.M))
        if len(matches) != 1:
            errors.append(f'{fname}: pattern matched {len(matches)}× (need exactly 1): {pattern}')
            continue
        m = matches[0]
        text = text[:m.start()] + template.format(**ctx) + text[m.end():]
    return text


def compute(root: Path, metrics: vm.Metrics) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """{rel_path: (old_text, new_text)} for every file in the table + anchor errors."""
    ctx = build_context(metrics)
    by_file: dict[str, list[tuple[str, str]]] = {}
    for f, pat, tpl in TABLE:
        by_file.setdefault(f, []).append((pat, tpl))
    results: dict[str, tuple[str, str]] = {}
    errors: list[str] = []
    for rel, entries in by_file.items():
        p = root / rel
        if not p.is_file():
            errors.append(f'{rel}: file not found under {root}')
            continue
        old = p.read_text(encoding='utf-8')
        new = rewrite_text(old, entries, ctx, rel, errors)
        results[rel] = (old, new)
    return results, errors


def unified_diff(rel: str, old: str, new: str) -> str:
    return ''.join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                        fromfile=f'a/{rel}', tofile=f'b/{rel}'))


def run(root: Path, metrics: vm.Metrics, *, dry_run: bool, check: bool) -> int:
    print(f'{B}Validated numbers:{NC} {vm.summary_line(metrics)}')
    if metrics.source == 'fallback':
        print(f'{Y}WARNING: no study_level_eval.json/MANIFEST.json under {metrics.model_dir} — '
              f'using the FALLBACK constants of the last release{NC}')
    results, errors = compute(root, metrics)
    if errors:
        print(f'{R}Anchor errors ({len(errors)}) — a document was reworded; fix the table or the doc:{NC}')
        for e in errors:
            print(f'  {R}✗{NC} {e}')
        return EXIT_DRIFT
    changed = {rel: (o, n) for rel, (o, n) in results.items() if o != n}
    if not changed:
        print(f'{G}✓ all {len(results)} files already in sync with the release{NC}')
        return 0
    for rel, (old, new) in changed.items():
        if dry_run or check:
            print(unified_diff(rel, old, new), end='')
    if check:
        print(f'{R}✗ {len(changed)} file(s) out of sync with the release — run scripts/bump_model_release.py:{NC}')
        for rel in changed:
            print(f'  {R}•{NC} {rel}')
        return EXIT_DRIFT
    if dry_run:
        print(f'{Y}dry run: {len(changed)} file(s) would change (nothing written){NC}')
        return 0
    for rel, (_, new) in changed.items():
        (root / rel).write_text(new, encoding='utf-8')
        print(f'  {G}✓{NC} {rel}')
    print(f'{G}✓ rewrote {len(changed)} file(s); {len(results) - len(changed)} already current{NC}')
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Rewrite every pinned validation number/identity/date from the model release.')
    ap.add_argument('--model', default=None,
                    help='model dir holding study_level_eval.json + MANIFEST.json '
                         '(default: SENTINEL_MODEL_DIR_OVERRIDE_BRAIN_TRIAGE, else models/brain_triage_finetuned)')
    ap.add_argument('--root', default=str(REPO), help='repo root to rewrite (default: this checkout)')
    ap.add_argument('--dry-run', action='store_true', help='print a unified diff, write nothing')
    ap.add_argument('--check', action='store_true', help='exit 1 when any file would change (used by the docs test)')
    args = ap.parse_args(argv)

    if args.model:
        try:
            metrics = vm.load(Path(args.model), strict=True)
        except Exception as e:   # noqa: BLE001 — report and exit non-zero
            print(f'{R}✗ cannot read validated metrics from {args.model}: {e}{NC}')
            return EXIT_NO_METRICS
    else:
        metrics = vm.METRICS
    problem = release_problem(metrics)
    if problem:
        print(f'{R}✗ {metrics.model_dir} is not a validated release — {problem}. The docs and the app '
              f'would call it "validated" with these numbers; promote a model that passed the gate '
              f'(scripts/promote_model.py) first.{NC}')
        return EXIT_NO_METRICS
    return run(Path(args.root).resolve(), metrics, dry_run=args.dry_run, check=args.check)


def release_problem(metrics) -> str:
    """'' when `metrics` may be pinned as the validated release: its eval passed the gate (and
    the comparison, if any) and promote_model.py did not --force it (review round 2 — an
    --allow-unvalidated bundle must never get matching 'validated' docs + identity)."""
    if not getattr(metrics, 'validated', True):
        return getattr(metrics, 'not_validated_reason', '') or 'eval says not validated'
    log_path = Path(metrics.model_dir) / 'promotion_log.json'
    if log_path.is_file():
        try:
            log = json.loads(log_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            return f'{log_path} unreadable ({e}) — cannot prove the promotion was not forced'
        if isinstance(log, dict) and log.get('forced') is True:
            return f'{log_path} records a --force promotion'
    return ''


if __name__ == '__main__':
    sys.exit(main())
