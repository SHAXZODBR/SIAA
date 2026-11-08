"""
Regression tests for the brain report templates (finish-line contract D4).

The draft report must tell the truth about what the model did:
  * tumor sentences are classification statements — no claimed observations
    (edema, margins, infiltration) the public classifier never made;
  * the technique line lists only detected sequences and mentions contrast only
    when a post-contrast T1 ('T1ce') is present; without sequence info it makes
    neither claim; CT gets a CT technique line;
  * the impression is human wording, never raw class keys;
  * the engine (GemmaReportEngine.generate) threads `sequences` through to the
    templates on every brain path, without Ollama.
Synthetic findings only — no patient data.
"""
from __future__ import annotations

import inspect

import pytest

from src.inference import brain_report_templates as brt
from src.inference.brain_report_templates import (
    NO_FLAG_IMPRESSION, TRIAGE_DESCRIPTIONS, build_brain_report, build_technique,
    normalize_sequences,
)
from src.inference.gemma_report_engine import GemmaReportEngine

LANGS = ('ru', 'uz', 'en')
TUMOR_CLASSES = ('glioma_tumor', 'meningioma_tumor', 'pituitary_tumor')
FORBIDDEN_OBSERVATIONS = ('отёк', 'отек', 'edema', 'shish', 'инфильтрат', 'infiltrat',
                          'контур', 'kontur', 'margin', 'perifokal', 'перифокальн')
CONTRAST_WORDS = ('контраст', 'kontrast', 'contrast')
CLASSIFIER_WORDING = {
    'ru': 'Классификатор опухолей (публичная модель, не валидирована на данных клиники) отнёс изображение к классу',
    'uz': "O'smalar klassifikatori",
    'en': 'The tumor classifier (public model, not validated',
}
CONFIRM_WORDING = {
    'ru': 'Признаки объёмного образования требуют подтверждения рентгенологом.',
    'uz': 'rentgenolog tomonidan tasdiqlanishi kerak',
    'en': 'must be confirmed by the radiologist',
}
IMPRESSION_TRIAGE = {
    'ru': 'признаки патологических изменений по данным локальной модели триажа — требуется приоритетный просмотр',
    'uz': "mahalliy triaj modeli ma'lumotlariga ko'ra patologik o'zgarishlar belgilari",
    'en': 'signs of abnormality according to the local triage model',
}
IMPRESSION_CLASSIFIER = {
    'ru': 'классификатор: {} (не валидирован)',
    'uz': 'klassifikator: {} (validatsiya qilinmagan)',
    'en': 'classifier: {} (not validated)',
}


def _panel_findings(tumor_cls: str) -> list[dict]:
    """What the brain panel emits for a flagged study (synthetic values)."""
    return [
        {'class_name': 'abnormal', 'confidence': 0.83, 'positive': True,
         'status': 'validated', 'detector': 'triage', 'location': ''},
        {'class_name': tumor_cls, 'confidence': 0.99, 'positive': True,
         'status': 'pending', 'detector': 'tumor_class', 'location': ''},
    ]


def _full_text(sec: dict) -> str:
    return ' '.join(sec.values())


# --------------------------------------------------------------------------- (a) tumor wording

@pytest.mark.parametrize('lang', LANGS)
@pytest.mark.parametrize('tumor_cls', TUMOR_CLASSES)
def test_tumor_and_triage_report_claims_no_unmade_observation(lang, tumor_cls):
    sec = build_brain_report(_panel_findings(tumor_cls), language=lang, modality='MR',
                             sequences=['T1', 'T2', 'FLAIR'])
    assert set(sec) == {'clinical_indication', 'technique', 'description', 'impression', 'recommendation'}
    assert all(v.strip() for v in sec.values())
    text = _full_text(sec).lower()
    for word in FORBIDDEN_OBSERVATIONS:
        assert word.lower() not in text, (lang, tumor_cls, word)
    assert CLASSIFIER_WORDING[lang] in sec['description']
    assert CONFIRM_WORDING[lang] in sec['description']
    label = brt.TUMOR_LABELS[tumor_cls][lang]
    assert f'«{label}»' in sec['description'] or f'"{label}"' in sec['description']
    # impression: human wording only
    assert '_' not in sec['impression']
    assert tumor_cls not in sec['impression'] and 'triage_abnormal' not in sec['impression']
    assert IMPRESSION_TRIAGE[lang] in sec['impression']
    assert IMPRESSION_CLASSIFIER[lang].format(label) in sec['impression']
    # the triage numbers still come from validated_metrics (unchanged sentence)
    assert TRIAGE_DESCRIPTIONS['abnormal'][lang] in sec['description']


def test_ru_glioma_sentence_is_exactly_the_contract_wording():
    assert brt.TUMOR_DESCRIPTIONS['glioma_tumor']['ru'] == (
        'Классификатор опухолей (публичная модель, не валидирована на данных клиники) отнёс '
        'изображение к классу «глиома». Признаки объёмного образования требуют подтверждения рентгенологом.')
    sec = build_brain_report(_panel_findings('glioma_tumor'), language='ru')
    assert 'классификатор: глиома (не валидирован)' in sec['impression']
    assert sec['impression'].startswith('Отметки ИИ: признаки патологических изменений по данным '
                                        'локальной модели триажа — требуется приоритетный просмотр')


@pytest.mark.parametrize('lang', LANGS)
def test_every_description_dict_is_free_of_observation_claims_for_tumor(lang):
    for cls in (*TUMOR_CLASSES, 'no_tumor'):
        text = brt.TUMOR_DESCRIPTIONS[cls][lang].lower()
        for word in FORBIDDEN_OBSERVATIONS:
            assert word.lower() not in text, (cls, lang, word)


@pytest.mark.parametrize('lang', LANGS)
def test_no_tumor_is_non_certifying_and_impression_is_no_flag(lang):
    sec = build_brain_report([{'class_name': 'no_tumor', 'confidence': 0.95, 'positive': False,
                               'detector': 'tumor_class'}], language=lang)
    assert sec['impression'] == NO_FLAG_IMPRESSION[lang]
    desc = sec['description']
    assert {'ru': 'не исключает', 'uz': 'istisno qilmaydi', 'en': 'does not exclude'}[lang] in desc
    # the old certifying sentence is gone
    assert 'обычно сформированы' not in desc and 'appear normal' not in desc
    assert 'odatdagidek shakllangan' not in desc


@pytest.mark.parametrize('lang', LANGS)
def test_no_flag_impression_unchanged(lang):
    assert build_brain_report([], language=lang)['impression'] == NO_FLAG_IMPRESSION[lang]
    only_normal = [{'class_name': 'normal', 'confidence': 0.7, 'positive': False, 'detector': 'triage',
                    'status': 'validated'}]
    sec = build_brain_report(only_normal, language=lang)
    assert sec['impression'] == NO_FLAG_IMPRESSION[lang]
    assert TRIAGE_DESCRIPTIONS['normal'][lang] in sec['description']


# --------------------------------------------------------------------------- (c) impression keys

@pytest.mark.parametrize('lang', LANGS)
def test_impression_never_contains_raw_keys_for_any_known_class(lang):
    classes = [*brt.TUMOR_LABELS, *brt.HEMORRHAGE_LABELS, *brt.DEMENTIA_LABELS, *brt.BRATS_LABELS]
    for cls in classes:
        sec = build_brain_report([{'class_name': cls, 'confidence': 0.9, 'positive': True}],
                                 language=lang, modality='CT' if cls in brt.HEMORRHAGE_LABELS else 'MR')
        assert '_' not in sec['impression'], (cls, lang, sec['impression'])
        assert sec['impression'] != NO_FLAG_IMPRESSION[lang], cls


@pytest.mark.parametrize('lang', LANGS)
def test_unknown_positive_class_is_surfaced_humanized(lang):
    sec = build_brain_report([{'class_name': 'acute_infarct', 'confidence': 0.77, 'positive': True,
                               'status': 'experimental', 'detector': 'stroke'}], language=lang)
    assert sec['impression'] != NO_FLAG_IMPRESSION[lang]
    assert 'acute infarct' in sec['impression'] and '_' not in sec['impression']
    assert 'acute infarct' in sec['description']


def test_negative_marked_findings_are_not_described_as_present():
    """Multi-label head-CT returns every class; below-threshold ones carry positive=False."""
    findings = [
        {'class_name': 'normal', 'confidence': 0.91, 'positive': False, 'detector': 'head_ct'},
        {'class_name': 'epidural', 'confidence': 0.03, 'positive': False, 'detector': 'head_ct'},
        {'class_name': 'subdural', 'confidence': 0.02, 'positive': False, 'detector': 'head_ct'},
    ]
    sec = build_brain_report(findings, language='en', modality='CT')
    assert sec['impression'] == NO_FLAG_IMPRESSION['en']
    assert 'hematoma' not in sec['description'].lower()
    assert 'URGENT' not in sec['recommendation']
    # head_ct's 'normal' class is not the local triage model's output
    assert 'triage' not in sec['description'].lower()


def test_hemorrhage_flag_keeps_urgent_recommendation_tumor_does_not():
    hem = build_brain_report([{'class_name': 'subdural', 'confidence': 0.9, 'positive': True,
                               'detector': 'head_ct'}], language='en', modality='CT')
    assert hem['recommendation'].startswith('URGENT')
    assert 'classifier: subdural hematoma (not validated)' in hem['impression']
    tum = build_brain_report(_panel_findings('glioma_tumor'), language='en')
    assert 'URGENT' not in tum['recommendation']
    assert 'confirm or exclude' in tum['recommendation']


def test_legacy_findings_without_positive_key_still_reported():
    sec = build_brain_report([{'class_name': 'glioma_tumor', 'confidence': 0.87,
                               'location': 'left frontal'}], language='en')
    assert 'classifier: glioma (not validated)' in sec['impression']


def test_unsupported_language_falls_back_to_russian():
    sec = build_brain_report(_panel_findings('glioma_tumor'), language='de')
    assert 'классификатор: глиома' in sec['impression']


# --------------------------------------------------------------------------- (b) technique

@pytest.mark.parametrize('lang', LANGS)
def test_technique_without_post_contrast_t1_has_no_contrast_claim(lang):
    tech = build_brain_report([], language=lang, modality='MR', sequences=['T1', 'T2', 'FLAIR'])['technique']
    assert 'T1, T2, FLAIR' in tech
    for w in CONTRAST_WORDS:
        assert w not in tech.lower(), (lang, tech)


@pytest.mark.parametrize('lang', LANGS)
def test_technique_with_t1ce_mentions_intravenous_contrast(lang):
    tech = build_brain_report([], language=lang, modality='MR',
                              sequences=['T1ce', 'T1', 'T2', 'FLAIR'])['technique']
    assert brt.TECHNIQUE_CONTRAST_PHRASE[lang].strip() in tech
    assert {'ru': 'с внутривенным контрастированием', 'uz': 'vena ichi kontrast bilan',
            'en': 'with intravenous contrast'}[lang] in tech
    assert 'T1, T1ce, T2, FLAIR' in tech


@pytest.mark.parametrize('lang', LANGS)
@pytest.mark.parametrize('seqs', [None, [], ['unknown'], [None, 42]])
def test_technique_without_sequence_info_makes_neither_claim(lang, seqs):
    tech = build_brain_report([], language=lang, modality='MR', sequences=seqs)['technique']
    assert tech == brt.TECHNIQUE_MR_GENERIC[lang]
    low = tech.lower()
    for w in CONTRAST_WORDS:
        assert w not in low
    for s in ('t1', 't2', 'flair', 'dwi'):
        assert s not in low


@pytest.mark.parametrize('lang', LANGS)
def test_ct_technique(lang):
    sec = build_brain_report([], language=lang, modality='CT', sequences=['T1ce'])
    assert sec['technique'] == brt.TECHNIQUE_CT[lang]
    for w in CONTRAST_WORDS:
        assert w not in sec['technique'].lower()
    assert sec['clinical_indication'] == brt.CLINICAL_INDICATION_HEAD_CT[lang]
    assert sec['recommendation'] == brt.RECOMMENDATION_DEFAULT_CT[lang]


def test_sequence_normalisation_canonical_names_and_order():
    assert normalize_sequences(['FLAIR', 't1c', 'T2_FLAIR', 'unknown', 'T1', 'adc', 'T1']) == \
        ['T1', 'T1ce', 'FLAIR', 'ADC']
    assert normalize_sequences('T1-Gd') == ['T1ce']
    assert normalize_sequences(None) == []
    assert build_technique('en', 'MR', ['T2']) == brt.TECHNIQUE_MR_WITH_SEQUENCES['en'].format(
        contrast='', sequences='T2')


# --------------------------------------------------------------------------- (d) engine pass-through

@pytest.fixture
def template_engine(monkeypatch):
    """Engine with no LLM: 'auto' against a closed local port → template backend."""
    monkeypatch.delenv('SENTINEL_ALLOW_CLOUD_LLM', raising=False)
    eng = GemmaReportEngine(backend='auto', ollama_url='http://127.0.0.1:9')
    assert eng.backend == 'template'
    return eng


def test_generate_signature_accepts_sequences():
    sig = inspect.signature(GemmaReportEngine.generate)
    assert 'sequences' in sig.parameters and sig.parameters['sequences'].default is None


@pytest.mark.parametrize('lang', LANGS)
def test_generate_end_to_end_threads_sequences(template_engine, lang):
    findings = _panel_findings('glioma_tumor')
    with_ce = template_engine.generate(findings, language=lang, modality='MR', body_part='BRAIN',
                                       sequences=['T1', 'T1ce', 'T2', 'FLAIR'])
    assert brt.TECHNIQUE_CONTRAST_PHRASE[lang].strip() in with_ce
    assert brt.TUMOR_DESCRIPTIONS['glioma_tumor'][lang].split('«')[0].split('"')[0].strip() in with_ce
    no_seq = template_engine.generate(findings, language=lang, modality='MR', body_part='BRAIN')
    assert brt.TECHNIQUE_MR_GENERIC[lang] in no_seq
    for w in CONTRAST_WORDS:
        assert w not in no_seq.lower()
    low = with_ce.lower()
    for word in FORBIDDEN_OBSERVATIONS:
        assert word.lower() not in low, (lang, word)
    titles = {'ru': 'ЗАКЛЮЧЕНИЕ:', 'uz': 'XULOSA:', 'en': 'IMPRESSION:'}[lang]
    impression_line = next(l for l in with_ce.splitlines() if l.startswith(titles))
    assert '_' not in impression_line


def test_generate_ct_head_uses_ct_technique(template_engine):
    text = template_engine.generate([], language='en', modality='CT', body_part='HEAD')
    assert 'TECHNIQUE: ' + brt.TECHNIQUE_CT['en'] in text


def test_fallback_template_path_threads_sequences(template_engine):
    text = template_engine._fallback_template_report(
        {'findings': [], 'modality': 'MR', 'body_part': 'BRAIN', 'sequences': ['T1ce', 'FLAIR']}, 'en')
    assert 'with intravenous contrast' in text and 'T1ce, FLAIR' in text
    text = template_engine._fallback_template_report(
        {'findings': [], 'modality': 'MR', 'body_part': 'BRAIN'}, 'en')
    assert brt.TECHNIQUE_MR_GENERIC['en'] in text


def test_generate_brain_template_failure_falls_back_without_crash(template_engine, monkeypatch):
    """If the brain template raises, generate() still returns a report (template fallback)."""
    def boom(*a, **k):
        raise RuntimeError('synthetic failure')
    monkeypatch.setattr(brt, 'build_brain_report', boom)
    text = template_engine.generate([], language='en', modality='MR', body_part='BRAIN')
    assert text.strip()


@pytest.mark.parametrize('lang', LANGS)
def test_generic_no_finding_report_never_certifies_normal(template_engine, lang):
    """Chest/generic fallback with nothing flagged (or only negative findings) is non-certifying."""
    for findings in ([], [{'class_name': 'Pneumonia', 'confidence': 0.1, 'positive': False}]):
        text = template_engine.generate(findings, language=lang, modality='CR', body_part='CHEST')
        assert {'ru': 'НЕ заключение о норме', 'uz': 'norma xulosasi EMAS',
                'en': 'NOT a normal read'}[lang] in text
        assert {'ru': 'ЗАКЛЮЧЕНИЕ: Норма', 'uz': 'XULOSA: Norma', 'en': 'IMPRESSION: Normal'}[lang] not in text
        assert 'Pneumonia' not in text


def test_brain_template_failure_never_yields_a_normal_certificate(template_engine, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError('synthetic failure')
    monkeypatch.setattr(brt, 'build_brain_report', boom)
    text = template_engine.generate([], language='en', modality='MR', body_part='BRAIN')
    assert 'IMPRESSION: Normal' not in text and 'NOT a normal read' in text
