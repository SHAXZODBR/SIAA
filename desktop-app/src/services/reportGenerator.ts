/**
 * Local radiology report generator — Russian / Uzbek / English templates.
 *
 * Used as a fallback when the backend Gemma engine is unavailable.
 * Covers all 18 chest pathologies that TorchXRayVision can detect.
 */

import type { Finding, AIResult } from '../types';

interface ReportTemplate {
  clinicalIndication: string;
  technique: string;
  description: string;
  conclusion: string;
  recommendation: string;
}

type Lang = 'ru' | 'uz' | 'en';

// ---------------------------------------------------------------------------
// Per-pathology descriptions in all 3 languages.
// {location} placeholder is replaced with translated location.
// ---------------------------------------------------------------------------
const FINDING_DESCRIPTIONS: Record<string, Record<Lang, string>> = {
  Pneumonia: {
    ru: 'В проекции {location} определяется участок инфильтрации легочной ткани с нечёткими контурами, средней интенсивности — соответствует пневмонии.',
    uz: '{location} sohasida o\'pka to\'qimasining noaniq konturli infiltratsiya sohasi aniqlanadi — pnevmoniyaga mos keladi.',
    en: 'An area of pulmonary infiltration with indistinct margins is identified in the {location}, consistent with pneumonia.',
  },
  Atelectasis: {
    ru: 'В {location} визуализируется участок снижения пневматизации, соответствующий ателектазу.',
    uz: '{location} sohada atelektazga mos pnevmatizatsiya pasayishi kuzatiladi.',
    en: 'Reduced aeration consistent with atelectasis is seen in the {location}.',
  },
  Cardiomegaly: {
    ru: 'Отмечается увеличение размеров сердечной тени. Кардиоторакальный индекс увеличен.',
    uz: 'Yurak soyasi o\'lchamlari kattalashgan. Kardiotorakal indeks oshgan.',
    en: 'The cardiac silhouette is enlarged. The cardiothoracic ratio is increased.',
  },
  Effusion: {
    ru: 'В {location} плевральной полости определяется наличие свободной жидкости. Реберно-диафрагмальный синус затемнён.',
    uz: '{location} plevral bo\'shliqda erkin suyuqlik mavjudligi aniqlanadi. Qovurg\'a-diafragmal sinus xira ko\'rinmoqda.',
    en: 'Free fluid is identified in the {location} pleural space, with blunting of the costophrenic angle.',
  },
  Infiltration: {
    ru: 'В {location} определяется участок инфильтрации легочной ткани.',
    uz: '{location} sohada o\'pka to\'qimasining infiltratsiya sohasi aniqlanadi.',
    en: 'An area of pulmonary infiltration is identified in the {location}.',
  },
  Mass: {
    ru: 'В {location} визуализируется объёмное образование с чёткими/нечёткими контурами размером более 30 мм.',
    uz: '{location} sohada 30 mm dan kattaroq, aniq/noaniq konturli hajmli hosil aniqlanadi.',
    en: 'A mass lesion measuring more than 30 mm with defined/ill-defined margins is seen in the {location}.',
  },
  Nodule: {
    ru: 'В {location} определяется округлое образование (узел) размером до 30 мм.',
    uz: '{location} sohada 30 mm gacha bo\'lgan dumaloq tugun aniqlanadi.',
    en: 'A nodular opacity up to 30 mm is identified in the {location}.',
  },
  Pneumothorax: {
    ru: 'В {location} определяется наличие воздуха в плевральной полости — признаки пневмоторакса. Лёгкое коллабировано.',
    uz: '{location} sohada plevral bo\'shliqda havo mavjudligi — pnevmotoraks belgilari. O\'pka kollapsga uchragan.',
    en: 'Air is present within the {location} pleural cavity — findings consistent with pneumothorax with associated lung collapse.',
  },
  Consolidation: {
    ru: 'В {location} определяется участок консолидации лёгочной ткани с воздушной бронхограммой.',
    uz: '{location} sohada havoli bronxogramma bilan o\'pka to\'qimasi konsolidatsiya sohasi aniqlanadi.',
    en: 'An area of lung consolidation with air bronchograms is identified in the {location}.',
  },
  Edema: {
    ru: 'Определяются признаки отёка лёгких: усиление лёгочного рисунка, перибронхиальные муфты, нечёткость контуров сосудов.',
    uz: 'O\'pka shishi belgilari aniqlanadi: o\'pka rasmining kuchayishi, peribronxial mufta, qon tomirlari konturlarining xiraligi.',
    en: 'Findings of pulmonary edema are noted: increased interstitial markings, peribronchial cuffing, and indistinct vascular margins.',
  },
  Emphysema: {
    ru: 'Отмечаются признаки эмфиземы лёгких: повышение прозрачности лёгочных полей, обеднение лёгочного рисунка.',
    uz: 'O\'pka emfizemasi belgilari kuzatiladi: o\'pka maydonlarining shaffofligi oshgan, o\'pka rasmi kambag\'allashgan.',
    en: 'Findings of pulmonary emphysema are seen: hyperlucent lung fields and decreased pulmonary vascular markings.',
  },
  Fibrosis: {
    ru: 'В {location} определяются фиброзные изменения лёгочной ткани — линейные тяжи, ретикулярный рисунок.',
    uz: '{location} sohada o\'pka to\'qimasining fibroz o\'zgarishlari — chiziqli tasmalar, retikulyar rasm aniqlanadi.',
    en: 'Fibrotic changes are seen in the {location} — linear strands and reticular pattern.',
  },
  Pleural_Thickening: {
    ru: 'В {location} определяется утолщение плевры.',
    uz: '{location} sohada plevra qalinlashgan.',
    en: 'Pleural thickening is identified in the {location}.',
  },
  Hernia: {
    ru: 'Визуализируются признаки диафрагмальной грыжи.',
    uz: 'Diafragmal churra belgilari aniqlanadi.',
    en: 'Findings consistent with diaphragmatic hernia are identified.',
  },
  Lung_Opacity: {
    ru: 'В {location} определяется участок снижения прозрачности лёгочной ткани (затемнение).',
    uz: '{location} sohada o\'pka to\'qimasi shaffofligi pasaygan soha aniqlanadi.',
    en: 'Reduced lung translucency (opacity) is identified in the {location}.',
  },
  'Lung Opacity': {
    ru: 'В {location} определяется участок снижения прозрачности лёгочной ткани (затемнение).',
    uz: '{location} sohada o\'pka to\'qimasi shaffofligi pasaygan soha aniqlanadi.',
    en: 'Reduced lung translucency (opacity) is identified in the {location}.',
  },
  Lung_Lesion: {
    ru: 'В {location} визуализируется очаговое образование лёгочной ткани.',
    uz: '{location} sohada o\'pka to\'qimasining o\'choqli hosilasi aniqlanadi.',
    en: 'A focal lung lesion is identified in the {location}.',
  },
  'Lung Lesion': {
    ru: 'В {location} визуализируется очаговое образование лёгочной ткани.',
    uz: '{location} sohada o\'pka to\'qimasining o\'choqli hosilasi aniqlanadi.',
    en: 'A focal lung lesion is identified in the {location}.',
  },
  Fracture: {
    ru: 'В {location} определяются признаки перелома костной структуры с нарушением целостности кортикального слоя.',
    uz: '{location} sohada suyak tuzilmasining sinish belgilari, kortikal qatlam yaxlitligi buzilgan.',
    en: 'Fracture findings are identified in the {location} with disruption of the cortical line.',
  },
  Enlarged_Cardiomediastinum: {
    ru: 'Отмечается расширение тени средостения.',
    uz: 'Mediastinum soyasi kengaygan.',
    en: 'Widening of the cardiomediastinal silhouette is noted.',
  },
  'Enlarged Cardiomediastinum': {
    ru: 'Отмечается расширение тени средостения.',
    uz: 'Mediastinum soyasi kengaygan.',
    en: 'Widening of the cardiomediastinal silhouette is noted.',
  },
  Support_Devices: {
    ru: 'Определяются признаки наличия медицинских устройств (трубки, катетеры, электроды).',
    uz: 'Tibbiy qurilmalar (naychalar, kateterlar, elektrodlar) mavjudligi belgilari aniqlanadi.',
    en: 'Indwelling medical support devices (tubes, catheters, electrodes) are identified.',
  },
  'Support Devices': {
    ru: 'Определяются признаки наличия медицинских устройств (трубки, катетеры, электроды).',
    uz: 'Tibbiy qurilmalar (naychalar, kateterlar, elektrodlar) mavjudligi belgilari aniqlanadi.',
    en: 'Indwelling medical support devices (tubes, catheters, electrodes) are identified.',
  },
  Normal: { ru: '', uz: '', en: '' },
  No_Finding: { ru: '', uz: '', en: '' },
};

// ---------------------------------------------------------------------------
// Static text per language
// ---------------------------------------------------------------------------
const NORMAL_BLOCK: Record<Lang, { description: string; conclusion: string; recommendation: string }> = {
  ru: {
    description:
      'На представленных снимках лёгочные поля прозрачны, без очаговых и инфильтративных изменений. Корни лёгких структурны, не расширены. Сердечная тень обычной формы и размеров. Диафрагма расположена обычно. Костные структуры без видимой патологии.',
    conclusion: 'Патологических изменений не выявлено.',
    recommendation: 'Контрольное обследование согласно клиническим показаниям.',
  },
  uz: {
    description:
      "Taqdim etilgan suratlarda o'pka maydonlari tiniq, o'choqli va infiltrativ o'zgarishlarsiz. O'pka ildizlari strukturali, kengaymagan. Yurak soyasi odatiy shakl va o'lchamda. Diafragma odatiy joylashgan. Suyak tuzilmalarida ko'rinarli patologiya yo'q.",
    conclusion: "Patologik o'zgarishlar aniqlanmadi.",
    recommendation: "Klinik ko'rsatmalarga muvofiq nazorat tekshiruvi.",
  },
  en: {
    description:
      'The lung fields are clear, without focal or infiltrative changes. The hila are well-defined and not enlarged. Cardiac silhouette is of normal size and contour. The diaphragm is in normal position. Bony structures appear unremarkable.',
    conclusion: 'No pathological findings.',
    recommendation: 'Follow-up examination per clinical indications.',
  },
};

const CLINICAL_INDICATION: Record<Lang, string> = {
  ru: 'Обследование органов грудной клетки.',
  uz: "Ko'krak qafasi a'zolarini tekshirish.",
  en: 'Chest examination.',
};

const TECHNIQUE: Record<string, Record<Lang, string>> = {
  CR: {
    ru: 'Выполнена рентгенография органов грудной клетки в прямой проекции.',
    uz: "Ko'krak qafasi a'zolarining to'g'ri proyeksiyada rentgenografiyasi bajarildi.",
    en: 'Posterior-anterior chest radiograph was obtained.',
  },
  CT: {
    ru: 'Выполнена компьютерная томография органов грудной клетки.',
    uz: "Ko'krak qafasi a'zolarining kompyuter tomografiyasi bajarildi.",
    en: 'Chest computed tomography was performed.',
  },
  MR: {
    ru: 'Выполнена магнитно-резонансная томография.',
    uz: 'Magnit-rezonans tomografiya bajarildi.',
    en: 'Magnetic resonance imaging was performed.',
  },
  DX: {
    ru: 'Выполнена цифровая рентгенография органов грудной клетки.',
    uz: "Ko'krak qafasining raqamli rentgenografiyasi bajarildi.",
    en: 'Digital chest radiograph was obtained.',
  },
};

const RECOMMENDATIONS: Record<'severe' | 'moderate' | 'normal', Record<Lang, string>> = {
  severe: {
    ru: 'Срочная консультация специалиста. Рекомендуется КТ органов грудной клетки и клиническое обследование.',
    uz: "Shoshilinch mutaxassis maslahati. Ko'krak qafasi KT tekshiruvi va klinik tekshiruv tavsiya etiladi.",
    en: 'Urgent specialist consultation. Chest CT and clinical evaluation are recommended.',
  },
  moderate: {
    ru: 'Динамическое наблюдение. Контрольное обследование через 2–4 недели. Консультация профильного специалиста при необходимости.',
    uz: 'Dinamik kuzatuv. 2–4 hafta ichida nazorat tekshiruvi. Zarur bo\'lganda profil mutaxassis maslahati.',
    en: 'Clinical monitoring. Follow-up examination in 2–4 weeks. Specialist consultation as clinically indicated.',
  },
  normal: {
    ru: 'Контрольное обследование согласно клиническим показаниям.',
    uz: "Klinik ko'rsatmalarga muvofiq nazorat tekshiruvi.",
    en: 'Follow-up per clinical indications.',
  },
};

// ---------------------------------------------------------------------------
// Location translation
// ---------------------------------------------------------------------------
function translateLocation(location: string, language: Lang): string {
  if (language === 'en') return location;
  const map: Record<Lang, Record<string, string>> = {
    ru: {
      'Right upper zone': 'правой верхней доли',
      'Right lower zone': 'правой нижней доли',
      'Left upper zone': 'левой верхней доли',
      'Left lower zone': 'левой нижней доли',
      'Central / mediastinal region': 'центральной/медиастинальной области',
    },
    uz: {
      'Right upper zone': "o'ng yuqori zonada",
      'Right lower zone': "o'ng pastki zonada",
      'Left upper zone': 'chap yuqori zonada',
      'Left lower zone': 'chap pastki zonada',
      'Central / mediastinal region': 'markaziy/mediastinal sohada',
    },
    en: {},
  };
  return map[language]?.[location] || location;
}

// ---------------------------------------------------------------------------
// Confidence note
// ---------------------------------------------------------------------------
function confidenceNote(percent: number, language: Lang): string {
  const label =
    language === 'ru' ? 'достоверность'
    : language === 'uz' ? 'ishonchlilik'
    : 'confidence';
  return `(${label}: ${percent}%)`;
}

// ---------------------------------------------------------------------------
// Build the report template
// ---------------------------------------------------------------------------
export function generateReport(
  aiResult: AIResult,
  language: Lang = 'ru',
  modality: string = 'CR',
  _bodyPart: string = 'CHEST',
): ReportTemplate {
  const clinicalIndication = CLINICAL_INDICATION[language];
  const technique = (TECHNIQUE[modality] || TECHNIQUE['CR'])[language];

  if (aiResult.isNormal || aiResult.findings.length === 0) {
    return {
      clinicalIndication,
      technique,
      description: NORMAL_BLOCK[language].description,
      conclusion: NORMAL_BLOCK[language].conclusion,
      recommendation: NORMAL_BLOCK[language].recommendation,
    };
  }

  // Description — one sentence per finding, with confidence
  const descriptions: string[] = [];
  for (const finding of aiResult.findings) {
    const conf = Math.round(finding.confidence * 100);
    const locTranslated = translateLocation(finding.location, language);
    const tpl = FINDING_DESCRIPTIONS[finding.className];

    if (tpl && tpl[language]) {
      let line = tpl[language].replace('{location}', locTranslated);
      line = line.replace(/\.\s*$/, '');
      line += ` ${confidenceNote(conf, language)}.`;
      descriptions.push(line);
    } else {
      const fallback = {
        ru: `В проекции ${locTranslated} определяется ${finding.className} ${confidenceNote(conf, language)}.`,
        uz: `${locTranslated} ${finding.className} aniqlanadi ${confidenceNote(conf, language)}.`,
        en: `${finding.className} is identified in the ${locTranslated} ${confidenceNote(conf, language)}.`,
      } as const;
      descriptions.push(fallback[language]);
    }
  }

  // Conclusion — list top findings
  const top = aiResult.findings.slice(0, 5).map((f) => f.className);
  const conclusion = (() => {
    if (top.length === 1) {
      return language === 'ru'
        ? `Рентгенологическая картина соответствует: ${top[0]}.`
        : language === 'uz'
          ? `Rentgenologik ko'rinish ${top[0]} ga mos keladi.`
          : `Imaging findings consistent with: ${top[0]}.`;
    }
    return language === 'ru'
      ? `Рентгенологическая картина: ${top.join(', ')}.`
      : language === 'uz'
        ? `Rentgenologik ko'rinish: ${top.join(', ')}.`
        : `Imaging findings: ${top.join(', ')}.`;
  })();

  // Recommendation severity
  const severeFindings = aiResult.findings.some(
    (f) =>
      ['Pneumothorax', 'Mass', 'Nodule', 'Lung_Lesion', 'Lung Lesion'].includes(f.className) &&
      f.confidence > 0.7,
  );
  const moderateFindings = aiResult.findings.some((f) => f.confidence >= 0.6);
  const recommendation =
    severeFindings
      ? RECOMMENDATIONS.severe[language]
      : moderateFindings
        ? RECOMMENDATIONS.moderate[language]
        : RECOMMENDATIONS.normal[language];

  return {
    clinicalIndication,
    technique,
    description: descriptions.join(' '),
    conclusion,
    recommendation,
  };
}

// ---------------------------------------------------------------------------
// Format full report text per language (for the textarea / PDF)
// ---------------------------------------------------------------------------
export function formatFullReport(
  report: ReportTemplate,
  patientId: string,
  studyDate: string,
  doctorName: string,
  language: Lang = 'ru',
): string {
  const dateStr =
    language === 'ru'
      ? new Date().toLocaleDateString('ru-RU')
      : language === 'uz'
        ? new Date().toLocaleDateString('en-GB')
        : new Date().toLocaleDateString('en-US');

  if (language === 'ru') {
    return `ПРОТОКОЛ РЕНТГЕНОЛОГИЧЕСКОГО ИССЛЕДОВАНИЯ

Пациент: ${patientId}
Дата исследования: ${studyDate}

КЛИНИЧЕСКОЕ ПОКАЗАНИЕ:
${report.clinicalIndication}

МЕТОДИКА:
${report.technique}

ОПИСАНИЕ:
${report.description}

ЗАКЛЮЧЕНИЕ:
${report.conclusion}

РЕКОМЕНДАЦИИ:
${report.recommendation}

Врач-рентгенолог: ${doctorName}
Дата: ${dateStr}

---
Отчёт подготовлен при содействии системы ИИ Sentinel Medical AI (siaa.uz).
Результаты ИИ-анализа носят рекомендательный характер и требуют подтверждения врачом.`;
  }

  if (language === 'uz') {
    return `RENTGENOLOGIK TEKSHIRUV BAYONNOMASI

Bemor: ${patientId}
Tekshiruv sanasi: ${studyDate}

KLINIK KO'RSATMA:
${report.clinicalIndication}

METODIKA:
${report.technique}

TAVSIF:
${report.description}

XULOSA:
${report.conclusion}

TAVSIYALAR:
${report.recommendation}

Rentgenolog shifokor: ${doctorName}
Sana: ${dateStr}

---
Hisobot Sentinel Medical AI (siaa.uz) tizimi yordamida tayyorlangan.
AI-tahlil natijalari tavsiya xarakteriga ega va shifokor tasdig'ini talab qiladi.`;
  }

  return `RADIOLOGY REPORT

Patient: ${patientId}
Study Date: ${studyDate}

CLINICAL INDICATION:
${report.clinicalIndication}

TECHNIQUE:
${report.technique}

FINDINGS:
${report.description}

IMPRESSION:
${report.conclusion}

RECOMMENDATIONS:
${report.recommendation}

Radiologist: ${doctorName}
Date: ${dateStr}

---
Report assisted by Sentinel Medical AI (siaa.uz).
AI analysis results are advisory and require physician confirmation.`;
}
