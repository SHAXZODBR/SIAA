"""
================================================================================
  SENTINEL — BRAIN-SPECIFIC RADIOLOGY REPORT TEMPLATES
================================================================================

  Used as the local fallback when MedGemma / Gemma is unavailable.
  Covers the brain pathologies the registry can detect:

    Tumor       → glioma_tumor / meningioma_tumor / pituitary_tumor
    Hemorrhage  → epidural / subdural / subarachnoid / intraparenchymal /
                  intraventricular
    Dementia    → mild / moderate / very_mild / non
    Tumor seg   → enhancing / necrotic / edema (BraTS classes)

  All 3 languages (RU / UZ / EN) — clinical phrasing reviewed against
  Russian-language radiology textbook conventions.
================================================================================
"""

from __future__ import annotations
from typing import Optional


# ---------------------------------------------------------------------------
# TUMOR FINDINGS (2D classifier output)
# ---------------------------------------------------------------------------

TUMOR_DESCRIPTIONS = {
    'glioma_tumor': {
        'ru': ('В веществе головного мозга визуализируется объёмное образование с '
               'неоднородной структурой, перифокальным отёком, признаками инфильтративного '
               'роста — рентгенологическая картина соответствует глиоме.'),
        'uz': ('Bosh miya to\'qimasida heterogen tuzilishli, perifokal shish bilan, '
               'infiltrativ o\'sish belgilari aniqlanadigan hajmli hosil ko\'rinadi — '
               'rentgenologik ko\'rinish gliomaga mos keladi.'),
        'en': ('A space-occupying lesion is identified in the brain parenchyma with '
               'heterogeneous internal architecture, perilesional edema, and signs of '
               'infiltrative growth — imaging findings consistent with glioma.'),
    },
    'meningioma_tumor': {
        'ru': ('Определяется экстра-аксиальное образование с чёткими ровными контурами, '
               'однородной структурой, прилежащее к твёрдой мозговой оболочке — '
               'характерные признаки менингиомы.'),
        'uz': ('Aniq tekis konturli, bir jinsli tuzilishli, qattiq miya pardasiga yondosh '
               'ekstra-aksial hosil aniqlanadi — meningiomaning xarakterli belgilari.'),
        'en': ('An extra-axial lesion with smooth defined margins and homogeneous '
               'architecture, abutting the dura, is identified — characteristic of meningioma.'),
    },
    'pituitary_tumor': {
        'ru': ('В области турецкого седла определяется образование, исходящее из гипофиза, '
               'с возможным супраселлярным распространением — признаки аденомы гипофиза.'),
        'uz': ('Turk egari sohasida gipofizdan kelib chiqayotgan, suprasellar tarqalish '
               'bilan ehtimoliy hosil aniqlanadi — gipofiz adenomasi belgilari.'),
        'en': ('A sellar mass arising from the pituitary gland with possible suprasellar '
               'extension is identified — findings consistent with pituitary adenoma.'),
    },
    'no_tumor': {
        'ru': 'Объёмных образований головного мозга не выявлено. Структуры головного мозга обычно сформированы.',
        'uz': 'Bosh miyada hajmli hosillar aniqlanmadi. Bosh miya tuzilmalari odatdagidek shakllangan.',
        'en': 'No intracranial mass lesion identified. Brain structures appear normal in development.',
    },
}


# ---------------------------------------------------------------------------
# BRATS SEGMENTATION (3D model output)
# ---------------------------------------------------------------------------

BRATS_DESCRIPTIONS = {
    'enhancing_tumor': {
        'ru': ('Контрастируемая часть опухоли — активно накапливающая контраст, '
               'отражает наиболее агрессивный компонент.'),
        'uz': ('Kontrast yutadigan o\'sma qismi — eng faol komponentni aks ettiradi.'),
        'en': ('Enhancing tumor component — represents the most metabolically active region.'),
    },
    'necrotic_core': {
        'ru': ('Некротическое ядро — центральный некроз опухоли без накопления контраста.'),
        'uz': ('Nekrotik yadro — markaziy nekroz, kontrast yutmaydi.'),
        'en': ('Necrotic core — central tumor necrosis without contrast enhancement.'),
    },
    'edema': {
        'ru': ('Перифокальный отёк (вазогенный) распространяется по белому веществу.'),
        'uz': ('Perifokal (vazogen) shish oq moddaga tarqalgan.'),
        'en': ('Peritumoral vasogenic edema extending into the white matter.'),
    },
}


# ---------------------------------------------------------------------------
# HEMORRHAGE FINDINGS (Head CT)
# ---------------------------------------------------------------------------

HEMORRHAGE_DESCRIPTIONS = {
    'epidural': {
        'ru': ('Эпидуральная гематома — биконвексная гиперденсная зона между костью '
               'черепа и твёрдой мозговой оболочкой. Часто связана с переломом и '
               'разрывом средней менингеальной артерии. ТРЕБУЕТ ЭКСТРЕННОГО ВМЕШАТЕЛЬСТВА.'),
        'uz': ('Epidural gematoma — kalla suyagi va qattiq miya pardasi orasida '
               'bikonveks giperdens zona. Odatda kalla suyagi sinishi bilan bog\'liq. '
               'SHOSHILINCH JARROHLIK ARALASHUVNI TALAB QILADI.'),
        'en': ('Epidural hematoma — biconvex hyperdense collection between the skull '
               'and dura. Frequently associated with skull fracture and middle meningeal '
               'artery laceration. REQUIRES URGENT NEUROSURGICAL EVALUATION.'),
    },
    'subdural': {
        'ru': ('Субдуральная гематома — серповидная гиперденсная зона между твёрдой '
               'мозговой оболочкой и веществом мозга. Возможно сдавление прилегающего '
               'полушария. Требует нейрохирургической оценки.'),
        'uz': ('Subdural gematoma — qattiq miya pardasi va miya to\'qimasi orasida '
               'oroqsimon giperdens zona. Yondosh yarim sharning siqilishi mumkin. '
               'Neyroxirurg bahosini talab qiladi.'),
        'en': ('Subdural hematoma — crescentic hyperdense collection between dura and '
               'brain parenchyma. Possible compression of the underlying hemisphere. '
               'Neurosurgical evaluation indicated.'),
    },
    'subarachnoid': {
        'ru': ('Субарахноидальное кровоизлияние — кровь в субарахноидальном пространстве, '
               'визуализируется в бороздах и базальных цистернах. Исключить аневризму '
               'мозговых артерий (рекомендована КТ-ангиография).'),
        'uz': ('Subaraxnoidal qon quyilishi — subaraxnoidal bo\'shliqda qon, egatlar va '
               'bazal sisternalarda ko\'rinadi. Miya arteriya anevrizmasini istisno qilish '
               'lozim (KT-angiografiya tavsiya etiladi).'),
        'en': ('Subarachnoid hemorrhage — blood within the subarachnoid space, visible '
               'in sulci and basal cisterns. Aneurysmal source must be excluded — CT '
               'angiography recommended.'),
    },
    'intraparenchymal': {
        'ru': ('Внутримозговое кровоизлияние — гиперденсный очаг в паренхиме головного '
               'мозга. Указать локализацию (полушарие, доля), объём, наличие масс-эффекта '
               'и прорыва в желудочковую систему.'),
        'uz': ('Miya ichi qon quyilishi — miya parenximasida giperdens o\'choq. Joylashuv '
               '(yarim shar, ulush), hajm, mass-effekt va qorincha tizimiga yorib o\'tish '
               'borligini ko\'rsatish kerak.'),
        'en': ('Intraparenchymal hemorrhage — hyperdense focus within brain parenchyma. '
               'Specify location (hemisphere, lobe), volume, mass effect, and any '
               'intraventricular extension.'),
    },
    'intraventricular': {
        'ru': ('Внутрижелудочковое кровоизлияние — кровь в желудочковой системе. '
               'Возможно развитие острой гидроцефалии — динамический контроль.'),
        'uz': ('Qorinchalararo qon quyilishi — qorincha tizimida qon. O\'tkir gidrosefaliya '
               'rivojlanishi mumkin — dinamik nazorat.'),
        'en': ('Intraventricular hemorrhage — blood within the ventricular system. Risk '
               'of acute hydrocephalus — close monitoring required.'),
    },
}


# ---------------------------------------------------------------------------
# DEMENTIA FINDINGS
# ---------------------------------------------------------------------------

DEMENTIA_DESCRIPTIONS = {
    'Mild_Demented': {
        'ru': ('Признаки умеренной кортикальной атрофии, преимущественно в височных и '
               'теменных долях. Расширение боковых желудочков. Картина соответствует '
               'лёгкой степени деменции.'),
        'uz': ('O\'rtacha kortikal atrofiya belgilari, asosan chakka va tepa ulushlarida. '
               'Yon qorinchalarning kengayishi. Manzara yengil demensiya darajasiga '
               'mos keladi.'),
        'en': ('Findings of moderate cortical atrophy, most pronounced in temporal and '
               'parietal lobes. Mild ventricular enlargement. Findings consistent with '
               'mild dementia.'),
    },
    'Moderate_Demented': {
        'ru': ('Выраженная кортикальная атрофия, преимущественно медиальных височных '
               'отделов (гиппокамп). Значительное расширение желудочковой системы и '
               'субарахноидальных пространств.'),
        'uz': ('Aniq kortikal atrofiya, ayniqsa medial chakka qismlarida (gippokamp). '
               'Qorincha tizimi va subaraxnoidal bo\'shliqlarning sezilarli darajada '
               'kengayishi.'),
        'en': ('Marked cortical atrophy, particularly involving the medial temporal '
               'structures (hippocampus). Significant ventricular and sulcal enlargement.'),
    },
    'Very_Mild_Demented': {
        'ru': ('Минимальные признаки атрофии — на грани возрастной нормы. Рекомендуется '
               'клиническая корреляция и динамический контроль.'),
        'uz': ('Atrofiyaning minimal belgilari — yosh normasi chegarasida. Klinik '
               'korrelyatsiya va dinamik nazorat tavsiya etiladi.'),
        'en': ('Minimal atrophy at the borderline of age-expected change. Clinical '
               'correlation and follow-up recommended.'),
    },
    'Non_Demented': {
        'ru': 'Возрастные изменения головного мозга в пределах нормы. Признаков деменции не выявлено.',
        'uz': 'Bosh miyaning yoshga oid o\'zgarishlari norma chegarasida. Demensiya belgilari aniqlanmadi.',
        'en': 'Age-appropriate brain morphology. No imaging features of dementia.',
    },
}


# ---------------------------------------------------------------------------
# LOCAL TRIAGE (brain_triage panel detector — the only locally validated one)
# ---------------------------------------------------------------------------

TRIAGE_DESCRIPTIONS = {
    'abnormal': {
        'ru': ('Локальная модель триажа (валидирована на данных клиники: чувствительность 90%, '
               'специфичность 47%) отнесла исследование к категории «патологические изменения» — '
               'приоритетный просмотр рентгенологом.'),
        'uz': ("Mahalliy triaj modeli (klinika ma'lumotlarida validatsiya qilingan: sezuvchanlik 90%, "
               "spetsifiklik 47%) tekshiruvni «patologik o'zgarishlar» toifasiga kiritdi — "
               "rentgenolog tomonidan ustuvor ko'rib chiqish."),
        'en': ('The local triage model (validated on this clinic\'s data: sensitivity 90%, '
               'specificity 47%) flagged this study as ABNORMAL — prioritize for radiologist review.'),
    },
    'normal': {
        'ru': ('Локальная модель триажа не отметила исследование (чувствительность 90%) — '
               'это НЕ заключение о норме.'),
        'uz': ("Mahalliy triaj modeli tekshiruvni belgilamadi (sezuvchanlik 90%) — "
               "bu norma xulosasi EMAS."),
        'en': ('The local triage model did not flag this study (sensitivity 90%) — '
               'this is NOT a normal read.'),
    },
}

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
# COMPOSITE TEMPLATE BUILDERS
# ---------------------------------------------------------------------------

CLINICAL_INDICATION_BRAIN = {
    'ru': 'МР-исследование головного мозга. Исключение объёмного процесса.',
    'uz': 'Bosh miya MR-tekshiruvi. Hajmli jarayonni istisno qilish.',
    'en': 'Brain MRI for evaluation of intracranial pathology.',
}

TECHNIQUE_BRAIN = {
    'ru': 'Выполнено МР-исследование головного мозга в стандартных импульсных последовательностях (T1, T2, FLAIR) до и после внутривенного контрастирования.',
    'uz': 'Bosh miya MR-tekshiruvi standart impuls ketma-ketliklarida (T1, T2, FLAIR) vena ichi kontrast yuborishdan oldin va keyin bajarildi.',
    'en': 'Brain MRI was performed using standard pulse sequences (T1, T2, FLAIR) before and after intravenous contrast administration.',
}


def build_brain_report(findings: list[dict], language: str = 'ru',
                         modality: str = 'MR') -> dict:
    """Build a complete brain MRI report from findings list.

    Returns dict with sections: clinical_indication, technique, description,
    impression, recommendation.
    """
    lang = language if language in ('ru', 'uz', 'en') else 'ru'

    descriptions = []
    classes_found = []
    has_severe = False

    for f in findings:
        cls = f.get('class_name') or f.get('className') or ''
        conf = f.get('confidence', 0)

        # Tumor class
        if cls in TUMOR_DESCRIPTIONS:
            tpl = TUMOR_DESCRIPTIONS[cls][lang]
            conf_str = _conf_str(conf, lang)
            if cls != 'no_tumor':
                descriptions.append(f"{tpl} {conf_str}")
                classes_found.append(cls)
                if cls == 'glioma_tumor' and conf > 0.7:
                    has_severe = True
            else:
                descriptions.append(tpl)

        # Hemorrhage
        elif cls in HEMORRHAGE_DESCRIPTIONS:
            descriptions.append(HEMORRHAGE_DESCRIPTIONS[cls][lang])
            classes_found.append(cls)
            has_severe = True  # all hemorrhages are urgent

        # Dementia
        elif cls in DEMENTIA_DESCRIPTIONS:
            descriptions.append(DEMENTIA_DESCRIPTIONS[cls][lang])
            classes_found.append(cls)

        # BraTS subregions
        elif cls in BRATS_DESCRIPTIONS:
            descriptions.append(BRATS_DESCRIPTIONS[cls][lang])
            classes_found.append(cls)

        # Local triage (panel) — 'abnormal' is a flag, 'normal' is explicitly not a normal read
        elif cls in TRIAGE_DESCRIPTIONS:
            tpl = TRIAGE_DESCRIPTIONS[cls][lang]
            if cls == 'abnormal':
                descriptions.append(f"{tpl} {_conf_str(conf, lang)}")
                classes_found.append('triage_abnormal')
            else:
                descriptions.append(tpl)

    # Default if nothing detected — panel-honest, never a normal certificate
    if not descriptions:
        descriptions.append(NO_FLAG_IMPRESSION[lang])

    description = ' '.join(descriptions)

    # Impression. A single pending detector saying 'no_tumor' does NOT make the
    # study normal — the product never certifies normal.
    if not classes_found or classes_found == ['no_tumor']:
        impression = NO_FLAG_IMPRESSION[lang]
    else:
        impression = {
            'ru': f'МР-признаки: {", ".join(classes_found)}.',
            'uz': f'MR-belgilari: {", ".join(classes_found)}.',
            'en': f'MRI findings of: {", ".join(classes_found)}.',
        }[lang]

    # Recommendation
    if has_severe:
        recommendation = {
            'ru': 'СРОЧНАЯ консультация нейрохирурга / невролога. КТ-ангиография при необходимости. Дальнейшая тактика по клиническим показаниям.',
            'uz': 'SHOSHILINCH neyroxirurg / nevrolog maslahati. Zaruratga ko\'ra KT-angiografiya. Keyingi taktika klinik ko\'rsatmalarga ko\'ra.',
            'en': 'URGENT neurosurgical / neurology consultation. CT angiography as indicated. Further management per clinical correlation.',
        }[lang]
    else:
        recommendation = {
            'ru': 'Динамическое МР-наблюдение через 3–6 месяцев. Консультация невролога.',
            'uz': '3–6 oy ichida dinamik MR-nazorat. Nevrolog maslahati.',
            'en': 'Follow-up MRI in 3–6 months. Neurology consultation.',
        }[lang]

    return {
        'clinical_indication': CLINICAL_INDICATION_BRAIN[lang],
        'technique': TECHNIQUE_BRAIN[lang],
        'description': description,
        'impression': impression,
        'recommendation': recommendation,
    }


def _conf_str(confidence: float, lang: str) -> str:
    pct = round(confidence * 100)
    label = {'ru': 'достоверность', 'uz': 'ishonchlilik', 'en': 'confidence'}[lang]
    return f"({label}: {pct}%)"
