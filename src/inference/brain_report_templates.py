"""
================================================================================
  SENTINEL — BRAIN-SPECIFIC RADIOLOGY REPORT TEMPLATES
================================================================================

  WHAT
    Deterministic RU / UZ / EN draft-report builder for brain MR and head CT
    studies. Used for every brain report (the small LLM is never trusted with
    clinical wording) and as the local fallback when MedGemma / Gemma is down.
    Covers the classes the registry can emit:

      Triage      → abnormal / normal (local, validated brain_triage panel)
      Tumor       → glioma_tumor / meningioma_tumor / pituitary_tumor / no_tumor
      Hemorrhage  → any / epidural / subdural / subarachnoid /
                    intraparenchymal / intraventricular (head CT)
      Dementia    → Very_Mild / Mild / Moderate / Non_Demented
      Tumor seg   → enhancing_tumor / necrotic_core / edema (BraTS classes)

  WORDING RULES (finish-line contract, decision D4)
    * The report tells the truth about what the model did. Public (unvalidated)
      detectors are CLASSIFIERS: their sentences say "the classifier assigned
      class X; to be confirmed by the radiologist" and never describe an
      observation the model did not make (edema, margins, infiltration, …).
    * The technique line lists only the sequences actually detected in the
      study; intravenous contrast is mentioned ONLY when a post-contrast T1
      ('T1ce') series is present. Without sequence information the line makes
      neither a sequence nor a contrast claim. CT gets a CT technique line.
    * The impression is human wording in each language — never raw class keys.
    * The product never certifies a study as normal: negative detector outputs
      (no_tumor, triage 'normal', Non_Demented) produce non-certifying text and
      the no-flag impression says explicitly that this is NOT a normal read.
    * Findings explicitly marked `positive: False` (e.g. every class a
      multi-label head-CT model reports below threshold) are NOT described as
      present — only the known negative classes above produce a sentence.

  TRIAGE NUMBERS
    triage_description() quotes sensitivity / specificity read from
    models/brain_triage_finetuned/study_level_eval.json via validated_metrics
    (torch-free) at report time, so a re-trained or imported model never gets
    stale figures; a finding whose status is not 'validated' (or a release that
    did not pass the gate) is described as NOT validated, without figures.
================================================================================
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Union

from loguru import logger

LANGS: tuple[str, ...] = ('ru', 'uz', 'en')

# ---------------------------------------------------------------------------
# Short human names per class (used in the description AND the impression)
# ---------------------------------------------------------------------------

TUMOR_LABELS: dict[str, dict[str, str]] = {
    'glioma_tumor':     {'ru': 'глиома',           'uz': 'glioma',          'en': 'glioma'},
    'meningioma_tumor': {'ru': 'менингиома',       'uz': 'meningioma',      'en': 'meningioma'},
    'pituitary_tumor':  {'ru': 'опухоль гипофиза', 'uz': "gipofiz o'smasi", 'en': 'pituitary tumor'},
}

HEMORRHAGE_LABELS: dict[str, dict[str, str]] = {
    'any':              {'ru': 'внутричерепное кровоизлияние', 'uz': "kalla ichi qon quyilishi",
                         'en': 'intracranial hemorrhage'},
    'epidural':         {'ru': 'эпидуральная гематома', 'uz': 'epidural gematoma',
                         'en': 'epidural hematoma'},
    'subdural':         {'ru': 'субдуральная гематома', 'uz': 'subdural gematoma',
                         'en': 'subdural hematoma'},
    'subarachnoid':     {'ru': 'субарахноидальное кровоизлияние', 'uz': "subaraxnoidal qon quyilishi",
                         'en': 'subarachnoid hemorrhage'},
    'intraparenchymal': {'ru': 'внутримозговое кровоизлияние', 'uz': "miya ichi qon quyilishi",
                         'en': 'intraparenchymal hemorrhage'},
    'intraventricular': {'ru': 'внутрижелудочковое кровоизлияние', 'uz': "qorinchalararo qon quyilishi",
                         'en': 'intraventricular hemorrhage'},
}

DEMENTIA_LABELS: dict[str, dict[str, str]] = {
    'Very_Mild_Demented': {'ru': 'очень лёгкая деменция', 'uz': 'juda yengil demensiya',
                           'en': 'very mild dementia'},
    'Mild_Demented':      {'ru': 'лёгкая деменция', 'uz': 'yengil demensiya', 'en': 'mild dementia'},
    'Moderate_Demented':  {'ru': 'умеренная деменция', 'uz': "o'rtacha demensiya",
                           'en': 'moderate dementia'},
}

BRATS_LABELS: dict[str, dict[str, str]] = {
    'enhancing_tumor': {'ru': 'контрастируемая часть опухоли', 'uz': "kontrast yutadigan o'sma qismi",
                        'en': 'enhancing tumor'},
    'necrotic_core':   {'ru': 'некротическое ядро', 'uz': 'nekrotik yadro', 'en': 'necrotic core'},
    'edema':           {'ru': 'перифокальный отёк', 'uz': 'perifokal shish', 'en': 'peritumoral edema'},
}

# ---------------------------------------------------------------------------
# Classification statements — what the model DID, nothing more
# ---------------------------------------------------------------------------

_CLASSIFIER_STATEMENT: dict[str, dict[str, str]] = {
    'tumor': {
        'ru': ('Классификатор опухолей (публичная модель, не валидирована на данных клиники) '
               'отнёс изображение к классу «{label}».'),
        'uz': ("O'smalar klassifikatori (ommaviy model, klinika ma'lumotlarida validatsiya qilinmagan) "
               "tasvirni «{label}» sinfiga kiritdi."),
        'en': ("The tumor classifier (public model, not validated on this clinic's data) "
               'assigned the image to class "{label}".'),
    },
    'hemorrhage': {
        'ru': ('Классификатор внутричерепных кровоизлияний (публичная модель, не валидирована на '
               'данных клиники) отнёс изображение к классу «{label}».'),
        'uz': ("Kalla ichi qon quyilishlari klassifikatori (ommaviy model, klinika ma'lumotlarida "
               "validatsiya qilinmagan) tasvirni «{label}» sinfiga kiritdi."),
        'en': ("The intracranial hemorrhage classifier (public model, not validated on this clinic's "
               'data) assigned the image to class "{label}".'),
    },
    'dementia': {
        'ru': ('Классификатор атрофии (публичная модель, не валидирована на данных клиники) '
               'отнёс изображение к классу «{label}».'),
        'uz': ("Atrofiya klassifikatori (ommaviy model, klinika ma'lumotlarida validatsiya qilinmagan) "
               "tasvirni «{label}» sinfiga kiritdi."),
        'en': ("The atrophy classifier (public model, not validated on this clinic's data) "
               'assigned the image to class "{label}".'),
    },
    'segmentation': {
        'ru': ('Модель сегментации (публичная модель, не валидирована на данных клиники) '
               'выделила область класса «{label}».'),
        'uz': ("Segmentatsiya modeli (ommaviy model, klinika ma'lumotlarida validatsiya qilinmagan) "
               "«{label}» sinfiga oid sohani ajratdi."),
        'en': ("The segmentation model (public model, not validated on this clinic's data) "
               'marked a region of class "{label}".'),
    },
}

_CONFIRMATION: dict[str, dict[str, str]] = {
    'tumor': {
        'ru': 'Признаки объёмного образования требуют подтверждения рентгенологом.',
        'uz': 'Hajmli hosil belgilari rentgenolog tomonidan tasdiqlanishi kerak.',
        'en': 'Features of a space-occupying lesion must be confirmed by the radiologist.',
    },
    'hemorrhage': {
        'ru': 'Наличие кровоизлияния требует срочного подтверждения рентгенологом.',
        'uz': "Qon quyilishi mavjudligi rentgenolog tomonidan shoshilinch tasdiqlanishi kerak.",
        'en': 'The presence of hemorrhage must be urgently confirmed by the radiologist.',
    },
    'dementia': {
        'ru': 'Атрофические изменения требуют оценки рентгенологом с учётом возраста и клинической картины.',
        'uz': "Atrofik o'zgarishlar rentgenolog tomonidan yosh va klinik manzarani hisobga olgan holda baholanishi kerak.",
        'en': 'Atrophic change must be assessed by the radiologist in the context of age and clinical findings.',
    },
    'segmentation': {
        'ru': 'Границы и характер выделенной области требуют проверки рентгенологом.',
        'uz': "Ajratilgan sohaning chegaralari va tabiati rentgenolog tomonidan tekshirilishi kerak.",
        'en': 'The extent and nature of the marked region must be verified by the radiologist.',
    },
}


def _statement(family: str, label: dict[str, str]) -> dict[str, str]:
    """Classification statement + confirmation sentence for one class, per language."""
    return {
        lang: (_CLASSIFIER_STATEMENT[family][lang].format(label=label[lang])
               + ' ' + _CONFIRMATION[family][lang])
        for lang in LANGS
    }


# ---------------------------------------------------------------------------
# TUMOR FINDINGS (2D classifier output)
# ---------------------------------------------------------------------------

TUMOR_DESCRIPTIONS: dict[str, dict[str, str]] = {
    cls: _statement('tumor', label) for cls, label in TUMOR_LABELS.items()
}
# Negative output: never a certificate — the classifier only knows three tumor classes.
TUMOR_DESCRIPTIONS['no_tumor'] = {
    'ru': ('Классификатор опухолей (публичная модель, не валидирована на данных клиники) не отнёс '
           'изображение ни к одному из классов опухолей. Это не исключает объёмное образование — '
           'требуется просмотр рентгенологом.'),
    'uz': ("O'smalar klassifikatori (ommaviy model, klinika ma'lumotlarida validatsiya qilinmagan) "
           "tasvirni o'sma sinflarining birortasiga kiritmadi. Bu hajmli hosilni istisno qilmaydi — "
           "rentgenolog tomonidan ko'rib chiqish talab qilinadi."),
    'en': ("The tumor classifier (public model, not validated on this clinic's data) did not assign "
           'the image to any tumor class. This does not exclude a mass lesion — radiologist review '
           'is required.'),
}


# ---------------------------------------------------------------------------
# BRATS SEGMENTATION (3D model output)
# ---------------------------------------------------------------------------

BRATS_DESCRIPTIONS: dict[str, dict[str, str]] = {
    cls: _statement('segmentation', label) for cls, label in BRATS_LABELS.items()
}


# ---------------------------------------------------------------------------
# HEMORRHAGE FINDINGS (Head CT)
# ---------------------------------------------------------------------------

HEMORRHAGE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    cls: _statement('hemorrhage', label) for cls, label in HEMORRHAGE_LABELS.items()
}


# ---------------------------------------------------------------------------
# DEMENTIA FINDINGS
# ---------------------------------------------------------------------------

DEMENTIA_DESCRIPTIONS: dict[str, dict[str, str]] = {
    cls: _statement('dementia', label) for cls, label in DEMENTIA_LABELS.items()
}
DEMENTIA_DESCRIPTIONS['Non_Demented'] = {
    'ru': ('Классификатор атрофии (публичная модель, не валидирована на данных клиники) не отнёс '
           'изображение к классам деменции. Это не исключает атрофические изменения.'),
    'uz': ("Atrofiya klassifikatori (ommaviy model, klinika ma'lumotlarida validatsiya qilinmagan) "
           "tasvirni demensiya sinflariga kiritmadi. Bu atrofik o'zgarishlarni istisno qilmaydi."),
    'en': ("The atrophy classifier (public model, not validated on this clinic's data) did not "
           'assign the image to a dementia class. This does not exclude atrophic change.'),
}


# ---------------------------------------------------------------------------
# LOCAL TRIAGE (brain_triage panel detector — the only locally validated one)
# ---------------------------------------------------------------------------

# The validated numbers ("чувствительность 90%, специфичность 47%") are read
# from models/brain_triage_finetuned/study_level_eval.json by validated_metrics
# (torch-free) AT REPORT TIME — POST /models/import reloads the metrics without a
# restart, so nothing here may be frozen at import. A triage finding whose status
# is not 'validated' (or a release whose eval did not pass the gate) is described
# as NOT validated, without any figures (D4: the report tells the truth).
from src.inference import validated_metrics as _vm  # noqa: E402


def triage_description(cls: str, lang: str, *, validated: bool = True) -> str:
    """The triage sentence for class 'abnormal' / 'normal' in `lang`, built
    from the CURRENT release metrics. validated=False (the finding's status is
    not 'validated') → the not-validated wording; the release's own
    validated flag is honoured as well (validated_metrics.triage_note)."""
    lang = lang if lang in LANGS else 'ru'
    if cls == 'abnormal':
        note = _vm.triage_note(lang, validated=validated)
        return {
            'ru': (f'Локальная модель триажа ({note}) отнесла исследование к категории '
                   '«патологические изменения» — приоритетный просмотр рентгенологом.'),
            'uz': (f"Mahalliy triaj modeli ({note}) tekshiruvni «patologik o'zgarishlar» "
                   "toifasiga kiritdi — rentgenolog tomonidan ustuvor ko'rib chiqish."),
            'en': (f'The local triage model ({note}) flagged this study as ABNORMAL — '
                   'prioritize for radiologist review.'),
        }[lang]
    if cls == 'normal':
        note = _vm.sensitivity_note(lang, validated=validated)
        return {
            'ru': f'Локальная модель триажа не отметила исследование ({note}) — это НЕ заключение о норме.',
            'uz': f"Mahalliy triaj modeli tekshiruvni belgilamadi ({note}) — bu norma xulosasi EMAS.",
            'en': f'The local triage model did not flag this study ({note}) — this is NOT a normal read.',
        }[lang]
    raise ValueError(f'not a triage class: {cls!r}')


def _refresh_triage_descriptions() -> dict[str, dict[str, str]]:
    """Rebuild TRIAGE_DESCRIPTIONS IN PLACE from the current metrics (callers
    holding a reference to the dict see the new figures after a reload)."""
    for cls in ('abnormal', 'normal'):
        TRIAGE_DESCRIPTIONS.setdefault(cls, {}).update(
            {lang: triage_description(cls, lang) for lang in LANGS})
    return TRIAGE_DESCRIPTIONS


# The validated-wording sentences for the CURRENT release (kept for callers and
# tests that read the dict; build_brain_report refreshes it on every call and
# uses triage_description() with the finding's status).
TRIAGE_DESCRIPTIONS: dict[str, dict[str, str]] = {}
_refresh_triage_descriptions()

# Panel-honest wording when nothing was flagged: the product never certifies
# a scan as normal, so the impression must say so explicitly.
NO_FLAG_IMPRESSION = {
    'ru': ('Панель ИИ не отметила находок. Это НЕ заключение о норме — панель выявляет только '
           'перечисленные типы патологии; требуется полный просмотр рентгенологом.'),
    'uz': ("AI paneli topilmalarni belgilamadi. Bu norma xulosasi EMAS — panel faqat sanab o'tilgan "
           "patologiya turlarini aniqlaydi; rentgenolog tomonidan to'liq ko'rib chiqish talab qilinadi."),
    'en': ('No finding flagged by the AI panel. This is NOT a normal read — the panel only detects '
           'the listed finding types; full radiologist review is required.'),
}


# ---------------------------------------------------------------------------
# IMPRESSION — human wording per language, never raw class keys
# ---------------------------------------------------------------------------

_IMPRESSION_PREFIX = {'ru': 'Отметки ИИ: ', 'uz': 'AI belgilari: ', 'en': 'AI flags: '}

_CLASSIFIER_IMPRESSION = {
    'ru': 'классификатор: {label} (не валидирован)',
    'uz': 'klassifikator: {label} (validatsiya qilinmagan)',
    'en': 'classifier: {label} (not validated)',
}
_SEGMENTATION_IMPRESSION = {
    'ru': 'сегментация: {label} (не валидирована)',
    'uz': 'segmentatsiya: {label} (validatsiya qilinmagan)',
    'en': 'segmentation: {label} (not validated)',
}
_OTHER_IMPRESSION = {
    'ru': 'отметка ИИ: {label}',
    'uz': 'AI belgisi: {label}',
    'en': 'AI flag: {label}',
}
_NOT_VALIDATED_SUFFIX = {'ru': ' (не валидирован)', 'uz': ' (validatsiya qilinmagan)', 'en': ' (not validated)'}


def _impression_dict() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {
        'triage_abnormal': {
            'ru': 'признаки патологических изменений по данным локальной модели триажа — '
                  'требуется приоритетный просмотр',
            'uz': "mahalliy triaj modeli ma'lumotlariga ko'ra patologik o'zgarishlar belgilari — "
                  "ustuvor ko'rib chiqish talab qilinadi",
            'en': 'signs of abnormality according to the local triage model — priority review required',
        },
    }
    for labels in (TUMOR_LABELS, HEMORRHAGE_LABELS, DEMENTIA_LABELS):
        for cls, label in labels.items():
            out[cls] = {lang: _CLASSIFIER_IMPRESSION[lang].format(label=label[lang]) for lang in LANGS}
    for cls, label in BRATS_LABELS.items():
        out[cls] = {lang: _SEGMENTATION_IMPRESSION[lang].format(label=label[lang]) for lang in LANGS}
    return out


IMPRESSION_LABELS: dict[str, dict[str, str]] = _impression_dict()


# ---------------------------------------------------------------------------
# CLINICAL INDICATION / TECHNIQUE
# ---------------------------------------------------------------------------

CLINICAL_INDICATION_BRAIN = {
    'ru': 'МР-исследование головного мозга. Исключение объёмного процесса.',
    'uz': 'Bosh miya MR-tekshiruvi. Hajmli jarayonni istisno qilish.',
    'en': 'Brain MRI for evaluation of intracranial pathology.',
}

CLINICAL_INDICATION_HEAD_CT = {
    'ru': 'КТ-исследование головного мозга. Оценка внутричерепной патологии.',
    'uz': 'Bosh miya KT-tekshiruvi. Kalla ichi patologiyasini baholash.',
    'en': 'Head CT for evaluation of intracranial pathology.',
}

# Canonical sequence names (src/pipeline/brain_mri_preprocessor.detect_sequence)
# in the order they are listed in the technique line.
CANONICAL_SEQUENCES: tuple[str, ...] = ('T1', 'T1ce', 'T2', 'FLAIR', 'DWI', 'ADC')
CONTRAST_SEQUENCE = 'T1ce'

# Accepted spellings (lower-case, separators stripped) → canonical name.
_SEQUENCE_ALIASES: dict[str, str] = {
    't1': 'T1', 't1w': 'T1',
    't1ce': 'T1ce', 't1c': 'T1ce', 't1gd': 'T1ce', 't1gad': 'T1ce', 't1post': 'T1ce',
    't1contrast': 'T1ce', 't1postcontrast': 'T1ce',
    't2': 'T2', 't2w': 'T2',
    'flair': 'FLAIR', 't2flair': 'FLAIR',
    'dwi': 'DWI', 'diffusion': 'DWI',
    'adc': 'ADC',
}

TECHNIQUE_MR_WITH_SEQUENCES = {
    'ru': ('Выполнено МР-исследование головного мозга{contrast} в импульсных последовательностях '
           '(определены автоматически по описаниям серий): {sequences}.'),
    'uz': ("Bosh miya MR-tekshiruvi{contrast} quyidagi impuls ketma-ketliklarida bajarildi "
           "(seriya tavsiflari bo'yicha avtomatik aniqlangan): {sequences}."),
    'en': ('Brain MRI was performed{contrast} using the following pulse sequences '
           '(identified automatically from the series descriptions): {sequences}.'),
}
TECHNIQUE_CONTRAST_PHRASE = {
    'ru': ' с внутривенным контрастированием',
    'uz': ' vena ichi kontrast bilan',
    'en': ' with intravenous contrast',
}
TECHNIQUE_MR_GENERIC = {
    'ru': ('Выполнено МР-исследование головного мозга; перечень импульсных последовательностей '
           'автоматически не определён.'),
    'uz': ("Bosh miya MR-tekshiruvi bajarildi; impuls ketma-ketliklari ro'yxati avtomatik "
           "aniqlanmadi."),
    'en': 'Brain MRI was performed; the list of pulse sequences could not be determined automatically.',
}
TECHNIQUE_CT = {
    'ru': 'Выполнена компьютерная томография головного мозга.',
    'uz': 'Bosh miya kompyuter tomografiyasi bajarildi.',
    'en': 'CT of the head was performed.',
}
# Back-compat alias: the generic MR line (no sequence, no contrast claim).
TECHNIQUE_BRAIN = TECHNIQUE_MR_GENERIC


# ---------------------------------------------------------------------------
# RECOMMENDATIONS
# ---------------------------------------------------------------------------

RECOMMENDATION_URGENT_HEMORRHAGE = {
    'ru': ('СРОЧНЫЙ просмотр рентгенологом для подтверждения кровоизлияния; при подтверждении — '
           'срочная консультация нейрохирурга / невролога, КТ-ангиография по показаниям.'),
    'uz': ("Qon quyilishini tasdiqlash uchun rentgenolog tomonidan SHOSHILINCH ko'rib chiqish; "
           "tasdiqlansa — shoshilinch neyroxirurg / nevrolog maslahati, ko'rsatmaga ko'ra KT-angiografiya."),
    'en': ('URGENT radiologist review to confirm the hemorrhage; if confirmed, urgent neurosurgical / '
           'neurology consultation and CT angiography as indicated.'),
}
RECOMMENDATION_TUMOR_CONFIRM = {
    'ru': ('Приоритетный просмотр рентгенологом для подтверждения или исключения объёмного образования; '
           'при подтверждении — консультация нейрохирурга.'),
    'uz': ("Hajmli hosilni tasdiqlash yoki istisno qilish uchun rentgenolog tomonidan ustuvor ko'rib "
           "chiqish; tasdiqlansa — neyroxirurg maslahati."),
    'en': ('Priority radiologist review to confirm or exclude a space-occupying lesion; '
           'if confirmed, neurosurgical consultation.'),
}
RECOMMENDATION_TRIAGE = {
    'ru': ('Приоритетный просмотр рентгенологом; дальнейшая тактика — по заключению рентгенолога и '
           'клиническим показаниям.'),
    'uz': ("Rentgenolog tomonidan ustuvor ko'rib chiqish; keyingi taktika — rentgenolog xulosasi va "
           "klinik ko'rsatmalarga ko'ra."),
    'en': ("Priority radiologist review; further management per the radiologist's conclusion and "
           'clinical indications.'),
}
RECOMMENDATION_DEFAULT_MR = {
    'ru': 'Динамическое МР-наблюдение через 3–6 месяцев. Консультация невролога.',
    'uz': '3–6 oy ichida dinamik MR-nazorat. Nevrolog maslahati.',
    'en': 'Follow-up MRI in 3–6 months. Neurology consultation.',
}
RECOMMENDATION_DEFAULT_CT = {
    'ru': 'Дальнейшая тактика по клиническим показаниям. Консультация невролога.',
    'uz': "Keyingi taktika klinik ko'rsatmalarga ko'ra. Nevrolog maslahati.",
    'en': 'Further management per clinical indications. Neurology consultation.',
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def normalize_sequences(sequences: Optional[Union[str, Iterable[object]]]) -> list[str]:
    """Map detected sequence names to their canonical spelling, in canonical order.

    Accepts the preprocessor's own names ('T1', 'T1ce', 'T2', 'FLAIR', 'DWI', 'ADC')
    and common spellings ('t1c', 'T1-Gd', 'T2_FLAIR', …). Unrecognised entries
    ('unknown', None, non-strings) are dropped — the technique line must only
    name sequences we actually identified. Duplicates collapse.
    """
    if sequences is None:
        return []
    if isinstance(sequences, str):
        sequences = [sequences]
    found: set[str] = set()
    dropped: list[str] = []
    for raw in sequences:
        if not isinstance(raw, str):
            continue
        key = re.sub(r'[^a-z0-9]', '', raw.lower())
        canon = _SEQUENCE_ALIASES.get(key)
        if canon:
            found.add(canon)
        elif key:
            dropped.append(raw)
    if dropped:
        logger.debug(f"technique line: ignoring unrecognised sequence names {dropped!r}")
    return [s for s in CANONICAL_SEQUENCES if s in found]


def _is_ct(modality: str) -> bool:
    return 'CT' in (modality or '').upper()


def build_technique(language: str, modality: str = 'MR',
                    sequences: Optional[Union[str, Iterable[object]]] = None) -> str:
    """Technique sentence built from what is actually in the study.

    CT → CT sentence (sequences ignored, no contrast claim). MR with recognised
    sequences → lists them; adds the intravenous-contrast phrase ONLY when a
    post-contrast T1 ('T1ce') is present. MR without sequence info → a generic
    sentence that makes neither a sequence nor a contrast claim.
    """
    lang = language if language in LANGS else 'ru'
    if _is_ct(modality):
        return TECHNIQUE_CT[lang]
    canon = normalize_sequences(sequences)
    if not canon:
        return TECHNIQUE_MR_GENERIC[lang]
    contrast = TECHNIQUE_CONTRAST_PHRASE[lang] if CONTRAST_SEQUENCE in canon else ''
    return TECHNIQUE_MR_WITH_SEQUENCES[lang].format(contrast=contrast, sequences=', '.join(canon))


def _humanize(cls: str, finding: dict) -> str:
    """Readable label for a class with no template: the finding's own text if it
    has one, else the key with separators replaced — never a raw '_' key."""
    text = finding.get('finding') or finding.get('label') or cls
    return re.sub(r'[_]+', ' ', str(text)).strip()


def _is_triage_finding(finding: dict) -> bool:
    """'abnormal' / 'normal' belong to the local triage model only. Other
    detectors (e.g. the head-CT hemorrhage model) also emit a 'normal' class
    and must not be described as the triage model's output."""
    det = finding.get('detector')
    return det in (None, '', 'triage')


def _is_validated(finding: dict) -> bool:
    """Only a finding explicitly marked status='validated' may be described as
    a validated model's output; a missing status is treated as NOT validated."""
    return str(finding.get('status') or '').strip().lower() == 'validated'


def _conf_str(confidence: float, lang: str) -> str:
    """'(оценка модели: 97%)' — the model's own score, not a measured accuracy."""
    try:
        pct = round(float(confidence or 0) * 100)
    except (TypeError, ValueError):
        pct = 0
    label = {'ru': 'оценка модели', 'uz': 'model bahosi', 'en': 'model score'}[lang]
    return f"({label}: {pct}%)"


# ---------------------------------------------------------------------------
# COMPOSITE TEMPLATE BUILDER
# ---------------------------------------------------------------------------

def build_brain_report(findings: list[dict], language: str = 'ru',
                       modality: str = 'MR',
                       sequences: Optional[Union[str, Iterable[object]]] = None) -> dict[str, str]:
    """Build a complete brain MR / head CT draft report from a findings list.

    Args:
        findings: dicts with 'class_name' (or 'className'), 'confidence' and
            optionally 'positive', 'status', 'detector', 'finding'.
        language: 'ru' | 'uz' | 'en' (anything else → 'ru').
        modality: 'MR' / 'CT' / … — CT gets CT indication / technique wording.
        sequences: sequences detected in the study (e.g. the study route's
            `sequences_present`); drives the technique line. None/empty →
            generic technique sentence without any sequence or contrast claim.

    Returns:
        dict with sections clinical_indication, technique, description,
        impression, recommendation (all non-empty strings).
    """
    lang = language if language in LANGS else 'ru'
    is_ct = _is_ct(modality)
    _refresh_triage_descriptions()       # figures of the release loaded NOW (no import-time freeze)

    descriptions: list[str] = []
    impression_items: list[str] = []
    has_hemorrhage = False
    has_tumor = False
    has_triage = False

    def _flag(item: str) -> None:
        if item not in impression_items:
            impression_items.append(item)

    for f in findings or []:
        if not isinstance(f, dict):
            continue
        cls = str(f.get('class_name') or f.get('className') or '')
        conf = f.get('confidence', 0) or 0
        positive = f.get('positive')          # None = legacy caller, treat as reported

        # Known negative outputs → non-certifying sentence, never an impression flag
        if cls == 'no_tumor':
            descriptions.append(TUMOR_DESCRIPTIONS['no_tumor'][lang])
            continue
        if cls == 'Non_Demented':
            descriptions.append(DEMENTIA_DESCRIPTIONS['Non_Demented'][lang])
            continue
        if cls == 'normal' and _is_triage_finding(f):
            descriptions.append(triage_description('normal', lang, validated=_is_validated(f)))
            continue

        # Any other class the caller explicitly marked negative is not "present"
        if positive is False:
            continue

        if cls in TUMOR_LABELS:
            descriptions.append(f"{TUMOR_DESCRIPTIONS[cls][lang]} {_conf_str(conf, lang)}")
            _flag(IMPRESSION_LABELS[cls][lang])
            has_tumor = True
        elif cls in HEMORRHAGE_LABELS:
            descriptions.append(f"{HEMORRHAGE_DESCRIPTIONS[cls][lang]} {_conf_str(conf, lang)}")
            _flag(IMPRESSION_LABELS[cls][lang])
            has_hemorrhage = True
        elif cls in DEMENTIA_LABELS:
            descriptions.append(f"{DEMENTIA_DESCRIPTIONS[cls][lang]} {_conf_str(conf, lang)}")
            _flag(IMPRESSION_LABELS[cls][lang])
        elif cls in BRATS_LABELS:
            descriptions.append(BRATS_DESCRIPTIONS[cls][lang])
            _flag(IMPRESSION_LABELS[cls][lang])
            has_tumor = True
        elif cls == 'abnormal' and _is_triage_finding(f):
            descriptions.append(f"{triage_description('abnormal', lang, validated=_is_validated(f))} "
                                f"{_conf_str(conf, lang)}")
            _flag(IMPRESSION_LABELS['triage_abnormal'][lang])
            has_triage = True
        elif cls or f.get('finding'):
            # A flagged class with no template must never vanish from the report
            # (that would read as "nothing flagged").
            label = _humanize(cls, f)
            suffix = '' if f.get('status') == 'validated' else _NOT_VALIDATED_SUFFIX[lang]
            descriptions.append({
                'ru': f'Детектор ИИ отметил находку «{label}»{suffix}. Требуется подтверждение рентгенологом.',
                'uz': f"AI detektori «{label}» topilmasini belgiladi{suffix}. Rentgenolog tasdig'i talab qilinadi.",
                'en': f'An AI detector flagged "{label}"{suffix}. Radiologist confirmation is required.',
            }[lang] + f" {_conf_str(conf, lang)}")
            _flag(_OTHER_IMPRESSION[lang].format(label=label) + suffix)

    # Nothing described at all — panel-honest, never a normal certificate
    if not descriptions:
        descriptions.append(NO_FLAG_IMPRESSION[lang])
    description = ' '.join(descriptions)

    # Impression: a negative-only result (no_tumor, triage normal) is NOT normal.
    if not impression_items:
        impression = NO_FLAG_IMPRESSION[lang]
    else:
        impression = _IMPRESSION_PREFIX[lang] + '; '.join(impression_items) + '.'

    if has_hemorrhage:
        recommendation = RECOMMENDATION_URGENT_HEMORRHAGE[lang]
    elif has_tumor:
        recommendation = RECOMMENDATION_TUMOR_CONFIRM[lang]
    elif has_triage:
        recommendation = RECOMMENDATION_TRIAGE[lang]
    else:
        recommendation = (RECOMMENDATION_DEFAULT_CT if is_ct else RECOMMENDATION_DEFAULT_MR)[lang]

    return {
        'clinical_indication': (CLINICAL_INDICATION_HEAD_CT if is_ct else CLINICAL_INDICATION_BRAIN)[lang],
        'technique': build_technique(lang, modality, sequences),
        'description': description,
        'impression': impression,
        'recommendation': recommendation,
    }
