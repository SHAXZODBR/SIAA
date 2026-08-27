/**
 * Translation table for AI finding class names.
 *
 * Source of truth: src/inference/model_registry.py → ModelCard.classes_localized
 * This file mirrors those mappings on the frontend so the right panel can show
 * findings in the doctor's chosen language (RU/UZ/EN).
 *
 * Russian is the default per Sentinel's Uzbek market focus.
 */

export type Lang = 'ru' | 'uz' | 'en';

// Key = exact class_name string from the AI model (English class ID)
// Value = { ru, uz, en, location_ru, location_uz, location_en }
type Tr = { ru: string; uz: string; en: string };

const FINDING_NAMES: Record<string, Tr> = {
  // ─── BRAIN TUMOR CLASSIFIER ─────────────────────────────────────────────
  glioma_tumor:       { ru: 'Глиома',                 uz: 'Glioma',          en: 'Glioma' },
  meningioma_tumor:   { ru: 'Менингиома',             uz: 'Meningioma',      en: 'Meningioma' },
  no_tumor:           { ru: 'Без опухоли',            uz: 'Shish topilmadi', en: 'No tumor' },
  pituitary_tumor:    { ru: 'Аденома гипофиза',       uz: 'Gipofiz shishi',  en: 'Pituitary tumor' },

  // ─── BRAIN 3D BRATS SEGMENTATION ────────────────────────────────────────
  tumor_core:         { ru: 'Ядро опухоли',           uz: 'O\'sma yadrosi',  en: 'Tumor core' },
  whole_tumor:        { ru: 'Вся опухоль',            uz: 'Butun o\'sma',    en: 'Whole tumor' },
  enhancing_tumor:    { ru: 'Контрастируемая часть',  uz: 'Kontrastlanadigan qism', en: 'Enhancing tumor' },
  edema:              { ru: 'Перифокальный отёк',     uz: 'Perifokal shish', en: 'Peritumoral edema' },
  necrotic_core:      { ru: 'Некротическое ядро',     uz: 'Nekrotik yadro',  en: 'Necrotic core' },
  background:         { ru: 'Норма',                  uz: 'Norma',           en: 'Background' },

  // ─── HEAD CT HEMORRHAGE ─────────────────────────────────────────────────
  any:                { ru: 'Кровоизлияние (любое)',     uz: 'Qon quyilishi (har qanday)', en: 'Any hemorrhage' },
  epidural:           { ru: 'Эпидуральное',              uz: 'Epidural',                   en: 'Epidural' },
  intraparenchymal:   { ru: 'Внутримозговое',            uz: 'Miya ichi',                  en: 'Intraparenchymal' },
  intraventricular:   { ru: 'Внутрижелудочковое',        uz: 'Qorinchalararo',             en: 'Intraventricular' },
  subarachnoid:       { ru: 'Субарахноидальное',         uz: 'Subaraxnoidal',              en: 'Subarachnoid' },
  subdural:           { ru: 'Субдуральное',              uz: 'Subdural',                   en: 'Subdural' },

  // ─── DEMENTIA ───────────────────────────────────────────────────────────
  Mild_Demented:        { ru: 'Лёгкая деменция',        uz: 'Yengil demensiya',          en: 'Mild dementia' },
  Moderate_Demented:    { ru: 'Умеренная деменция',     uz: 'O\'rtacha demensiya',       en: 'Moderate dementia' },
  Very_Mild_Demented:   { ru: 'Очень лёгкая деменция',  uz: 'Juda yengil demensiya',     en: 'Very mild dementia' },
  Non_Demented:         { ru: 'Без деменции',           uz: 'Demensiyasiz',              en: 'No dementia' },

  // ─── CHEST X-RAY (TorchXRayVision 18 classes) ───────────────────────────
  Atelectasis:        { ru: 'Ателектаз',              uz: 'Atelektaz',         en: 'Atelectasis' },
  Consolidation:      { ru: 'Консолидация',           uz: 'Konsolidatsiya',    en: 'Consolidation' },
  Infiltration:       { ru: 'Инфильтрация',           uz: 'Infiltratsiya',     en: 'Infiltration' },
  Pneumothorax:       { ru: 'Пневмоторакс',           uz: 'Pnevmotoraks',      en: 'Pneumothorax' },
  Edema:              { ru: 'Отёк лёгких',            uz: 'O\'pka shishi',     en: 'Pulmonary edema' },
  Emphysema:          { ru: 'Эмфизема',               uz: 'Emfizema',          en: 'Emphysema' },
  Fibrosis:           { ru: 'Фиброз',                 uz: 'Fibroz',            en: 'Fibrosis' },
  Effusion:           { ru: 'Плевральный выпот',      uz: 'Plevral suyuqlik',  en: 'Pleural effusion' },
  Pneumonia:          { ru: 'Пневмония',              uz: 'Pnevmoniya',        en: 'Pneumonia' },
  Pleural_Thickening: { ru: 'Утолщение плевры',       uz: 'Plevra qalinlashishi', en: 'Pleural thickening' },
  Cardiomegaly:       { ru: 'Кардиомегалия',          uz: 'Kardiomegaliya',    en: 'Enlarged heart' },
  Nodule:             { ru: 'Узелок',                 uz: 'Tugunchak',         en: 'Lung nodule' },
  Mass:               { ru: 'Образование',            uz: 'Hosil',             en: 'Mass' },
  Hernia:             { ru: 'Диафрагмальная грыжа',   uz: 'Diafragmal churra', en: 'Diaphragmatic hernia' },
  'Lung Lesion':      { ru: 'Очаг в лёгком',          uz: 'O\'pka o\'chog\'i', en: 'Focal lung lesion' },
  Fracture:           { ru: 'Перелом',                uz: 'Sinish',            en: 'Fracture' },
  'Lung Opacity':     { ru: 'Затемнение лёгкого',     uz: 'O\'pka xira',       en: 'Lung opacity' },
  'Enlarged Cardiomediastinum': { ru: 'Расширение средостения', uz: 'Mediastinum kengayishi', en: 'Widened cardiomediastinum' },

  // ─── TB BINARY ─────────────────────────────────────────────────────────
  NORMAL:             { ru: 'Норма',                  uz: 'Norma',             en: 'Normal' },
  TUBERCULOSIS:       { ru: 'Туберкулёз',             uz: 'Tuberkulyoz',       en: 'Tuberculosis' },

  // ─── PNEUMONIA BINARY ──────────────────────────────────────────────────
  PNEUMONIA:          { ru: 'Пневмония',              uz: 'Pnevmoniya',        en: 'Pneumonia' },

  // ─── COVID CT ──────────────────────────────────────────────────────────
  CT_COVID:           { ru: 'COVID-19',               uz: 'COVID-19',          en: 'COVID-19' },
  CT_NonCOVID:        { ru: 'Без COVID-19',           uz: 'COVID-19 yo\'q',    en: 'No COVID-19' },

  // ─── MAMMOGRAPHY ───────────────────────────────────────────────────────
  mass:               { ru: 'Образование (масса)',    uz: 'Hajmli hosila',     en: 'Mass' },
  calcification:      { ru: 'Кальцификаты',           uz: 'Kalsifikatsiya',    en: 'Calcifications' },
  asymmetry:          { ru: 'Асимметрия',             uz: 'Assimetriya',       en: 'Asymmetry' },
  distortion:         { ru: 'Архитектурное искажение',uz: 'Strukturaviy buzilish', en: 'Architectural distortion' },
  normal:             { ru: 'Норма',                  uz: 'Norma',             en: 'Normal' },
  Cavitation:         { ru: 'Полость распада',        uz: 'Yemirilish bo\'shlig\'i', en: 'Cavitation' },

  // ─── MEDSAM ────────────────────────────────────────────────────────────
  mask:               { ru: 'Маска сегментации',      uz: 'Segmentatsiya niqobi', en: 'Segmentation mask' },
};

// Location strings (anatomical regions)
const LOCATION_NAMES: Record<string, Tr> = {
  'Right upper zone':       { ru: 'Правая верхняя зона',          uz: 'O\'ng yuqori zona',          en: 'Right upper zone' },
  'Right lower zone':       { ru: 'Правая нижняя зона',           uz: 'O\'ng pastki zona',          en: 'Right lower zone' },
  'Left upper zone':        { ru: 'Левая верхняя зона',           uz: 'Chap yuqori zona',           en: 'Left upper zone' },
  'Left lower zone':        { ru: 'Левая нижняя зона',            uz: 'Chap pastki zona',           en: 'Left lower zone' },
  'Central / mediastinal region': { ru: 'Центральная/средостение', uz: 'Markaziy/mediastinum',     en: 'Central / mediastinal region' },
  'Central region':         { ru: 'Центральная область',          uz: 'Markaziy soha',              en: 'Central region' },
  'Brain parenchyma':       { ru: 'Паренхима мозга',              uz: 'Miya parenximasi',           en: 'Brain parenchyma' },
  'Left frontal lobe':      { ru: 'Левая лобная доля',            uz: 'Chap peshana ulushi',        en: 'Left frontal lobe' },
  'Right frontal lobe':     { ru: 'Правая лобная доля',           uz: 'O\'ng peshana ulushi',       en: 'Right frontal lobe' },
  'Lateral ventricles':     { ru: 'Боковые желудочки',            uz: 'Yon qorinchalar',            en: 'Lateral ventricles' },
  'Intracranial':           { ru: 'Внутричерепная область',       uz: 'Bosh suyagi ichi',           en: 'Intracranial' },
  'Region of interest':     { ru: 'Область интереса',             uz: 'Qiziqish sohasi',            en: 'Region of interest' },
  'Upper outer quadrant':   { ru: 'Верхне-наружный квадрант',     uz: 'Yuqori-tashqi kvadrant',     en: 'Upper outer quadrant' },
  'Right apex':             { ru: 'Правая верхушка',              uz: 'O\'ng cho\'qqi',             en: 'Right apex' },
};

/** Translate a finding class name (e.g. "Pneumonia" → "Пневмония") */
export function translateFinding(classNameRaw: string, lang: Lang = 'ru'): string {
  const tr = FINDING_NAMES[classNameRaw];
  if (tr) return tr[lang];
  // Fallback: try case-insensitive lookup
  const key = Object.keys(FINDING_NAMES).find(
    (k) => k.toLowerCase() === classNameRaw.toLowerCase(),
  );
  if (key) return FINDING_NAMES[key][lang];
  // Last resort — show the raw class name
  return classNameRaw;
}

/** Translate an anatomical location string */
export function translateLocation(locationRaw: string, lang: Lang = 'ru'): string {
  const tr = LOCATION_NAMES[locationRaw];
  if (tr) return tr[lang];
  return locationRaw;
}

/** Translate a confidence level into a clinical word */
export function severityWord(confidence: number, lang: Lang = 'ru'): string {
  if (confidence >= 0.8) {
    return { ru: 'КРИТИЧНО', uz: 'KRITIK', en: 'CRITICAL' }[lang];
  }
  if (confidence >= 0.6) {
    return { ru: 'СРОЧНО', uz: 'SHOSHILINCH', en: 'URGENT' }[lang];
  }
  if (confidence >= 0.4) {
    return { ru: 'УМЕРЕННО', uz: 'O\'RTACHA', en: 'MODERATE' }[lang];
  }
  return { ru: 'ЛЁГКАЯ', uz: 'YENGIL', en: 'MILD' }[lang];
}

/** UI label translations */
export const UI_LABELS = {
  findings:        { ru: 'Находки',         uz: 'Topilmalar',       en: 'Findings' },
  report:          { ru: 'Отчёт',           uz: 'Hisobot',          en: 'Report' },
  askAI:           { ru: 'Спросить ИИ',     uz: 'AI dan so\'rang',  en: 'Ask AI' },
  details:         { ru: 'Детали',          uz: 'Tafsilotlar',      en: 'Details' },
  compare:         { ru: 'Сравнить',        uz: 'Solishtirish',     en: 'Compare' },
  detectedPath:    { ru: 'Обнаруженные патологии', uz: 'Aniqlangan patologiyalar', en: 'Detected Pathologies' },
  sortConfidence:  { ru: 'Сорт.: Достоверность ↓', uz: 'Saralash: Ishonchlilik ↓',  en: 'Sort: Confidence ↓' },
  conf:            { ru: 'дост.',           uz: 'ishonch',          en: 'conf.' },
  sensitivity:     { ru: 'Чувствительность',uz: 'Sezgirlik',        en: 'Sensitivity' },
  specificity:     { ru: 'Специфичность',   uz: 'Spetsiflik',       en: 'Specificity' },
  area:            { ru: 'Площадь',         uz: 'Maydon',           en: 'Area' },
  severity:        { ru: 'Серьёзность',     uz: 'Og\'irlik',        en: 'Severity' },
  rerunAI:         { ru: 'Запустить ИИ заново', uz: 'AI tahlilini qayta ishga tushirish', en: 'Re-run AI Analysis' },
  comparePrior:    { ru: 'Сравнить с предыдущим', uz: 'Oldingilari bilan solishtirish', en: 'Compare Prior' },
  submitFeedback:  { ru: 'Отправить отзыв', uz: 'Fikr bildirish',   en: 'Submit Feedback' },
  noPathology:     { ru: 'Патология не обнаружена', uz: 'Patologiya topilmadi', en: 'No Pathology Detected' },
  analyzing:       { ru: 'Анализ…',         uz: 'Tahlil…',          en: 'Analyzing…' },
  failed:          { ru: 'Анализ не выполнен', uz: 'Tahlil bajarilmadi', en: 'Analysis Failed' },
  notSupported:    { ru: 'Пока не поддерживается', uz: 'Hozircha qo\'llab-quvvatlanmaydi', en: 'Not Yet Supported' },
  noAnalysisYet:   { ru: 'Анализ ещё не выполнен', uz: 'Tahlil hali bajarilmagan', en: 'No analysis yet' },
  selectStudy:     { ru: 'Выберите исследование в списке слева', uz: 'Chap tomondagi ro\'yxatdan tekshiruvni tanlang', en: 'Select a study from the worklist' },
};

export function L(key: keyof typeof UI_LABELS, lang: Lang = 'ru'): string {
  return UI_LABELS[key][lang];
}
