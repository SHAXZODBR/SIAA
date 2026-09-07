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

export type FindingUrgency = 'review' | 'unvalidated' | 'not_flagged';

/**
 * Urgency is derived from the server's `positive` flag and the detector's
 * validation status — never from the raw confidence value.
 */
export function findingUrgency(f: { positive: boolean; status: string }): FindingUrgency {
  if (f.positive && f.status === 'validated') return 'review';
  if (f.positive) return 'unvalidated';
  return 'not_flagged';
}

export function urgencyWord(urgency: FindingUrgency, lang: Lang = 'ru'): string {
  const words: Record<FindingUrgency, Tr> = {
    review:      { ru: 'ТРЕБУЕТ ПРОСМОТРА', uz: 'KO\'RIB CHIQISH KERAK', en: 'REVIEW' },
    unvalidated: { ru: 'НЕВАЛИДИРОВАННЫЙ ФЛАГ', uz: 'TASDIQLANMAGAN BELGI', en: 'UNVALIDATED FLAG' },
    not_flagged: { ru: 'НЕ ОТМЕЧЕНО', uz: 'BELGILANMAGAN', en: 'NOT FLAGGED' },
  };
  return words[urgency][lang];
}

export function statusWord(status: string, lang: Lang = 'ru'): string {
  const words: Record<string, Tr> = {
    validated:    { ru: 'Валидировано', uz: 'Tasdiqlangan', en: 'Validated' },
    pending:      { ru: 'Валидация не завершена', uz: 'Tasdiqlash kutilmoqda', en: 'Pending validation' },
    experimental: { ru: 'Экспериментально', uz: 'Eksperimental', en: 'Experimental' },
  };
  return (words[status] || words.experimental)[lang];
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

  // ─── Pilot hardening: status, review, health ─────────────────────────
  intendedUse:     { ru: 'Поддержка принятия решений — рентгенолог читает и подписывает каждое исследование',
                     uz: 'Qaror qabul qilishga yordam — har bir tekshiruvni rentgenolog o\'qiydi va imzolaydi',
                     en: 'Decision support — the radiologist reads and signs every study' },
  notAnalyzed:     { ru: 'Не проанализировано — требуется полный просмотр рентгенологом', uz: 'Tahlil qilinmadi — rentgenolog to\'liq ko\'rib chiqishi kerak', en: 'Not analyzed — requires full radiologist review' },
  noFindingFlagged:{ ru: 'ИИ не отметил находок — это НЕ заключение о норме', uz: 'AI hech qanday topilma belgilamadi — bu norma xulosasi EMAS', en: 'No finding flagged by AI — NOT a normal read' },
  abnormalFlagged: { ru: 'ИИ отметил находку — требуется просмотр', uz: 'AI topilma belgiladi — ko\'rib chiqish kerak', en: 'AI flagged a finding — requires review' },
  overallAssessment:{ ru: 'Общая оценка', uz: 'Umumiy baho', en: 'Overall assessment' },
  models:          { ru: 'Модели', uz: 'Modellar', en: 'Models' },
  threshold:       { ru: 'Порог', uz: 'Chegara', en: 'Threshold' },
  detector:        { ru: 'Детектор', uz: 'Detektor', en: 'Detector' },
  sequence:        { ru: 'Последовательность', uz: 'Ketma-ketlik', en: 'Sequence' },
  status:          { ru: 'Статус', uz: 'Holat', en: 'Status' },
  reason:          { ru: 'Причина', uz: 'Sabab', en: 'Reason' },
  aiServerDown:    { ru: 'Сервер ИИ не запущен — обратитесь в ИТ-службу', uz: 'AI serveri ishlamayapti — IT xizmatiga murojaat qiling', en: 'AI server is not running — contact IT' },
  aiServerDegraded:{ ru: 'Сервер ИИ работает частично: валидированная модель не загружена — обратитесь в ИТ-службу', uz: 'AI serveri qisman ishlamoqda: tasdiqlangan model yuklanmadi — IT xizmatiga murojaat qiling', en: 'AI server is degraded: the validated model failed to load — contact IT' },
  aiServerError:   { ru: 'Сервер ИИ сообщает об ошибке — обратитесь в ИТ-службу', uz: 'AI serveri xato haqida xabar bermoqda — IT xizmatiga murojaat qiling', en: 'AI server reports an error — contact IT' },
  llmUnavailable:  { ru: 'Шаблонный отчёт — ИИ-ассистент недоступен', uz: 'Shablon hisobot — AI yordamchisi mavjud emas', en: 'Template report — AI assistant unavailable' },
  analysisFailed:  { ru: 'Анализ не выполнен — обратитесь в ИТ-службу', uz: 'Tahlil bajarilmadi — IT xizmatiga murojaat qiling', en: 'Analysis failed — contact IT' },
  uploadStudy:     { ru: 'Загрузить исследование', uz: 'Tekshiruvni yuklash', en: 'Upload study' },
  uploadFolder:    { ru: 'Папка', uz: 'Papka', en: 'Folder' },
  uploadFiles:     { ru: 'Файлы', uz: 'Fayllar', en: 'Files' },
  uploading:       { ru: 'Загрузка', uz: 'Yuklanmoqda', en: 'Uploading' },
  analyzingFiles:  { ru: 'Анализ', uz: 'Tahlil', en: 'Analyzing' },
  files:           { ru: 'файлов', uz: 'fayl', en: 'files' },
  dropHint:        { ru: 'Перетащите папку или файлы DICOM сюда', uz: 'DICOM papkasi yoki fayllarini shu yerga tashlang', en: 'Drop a DICOM folder or files here' },
  noDicomFiles:    { ru: 'В выбранных файлах нет DICOM (.dcm)', uz: 'Tanlangan fayllar orasida DICOM (.dcm) yo\'q', en: 'No DICOM (.dcm) files in the selection' },
  noStudies:       { ru: 'Исследований пока нет', uz: 'Hozircha tekshiruvlar yo\'q', en: 'No studies yet' },
  noStudiesHint:   { ru: 'Загрузите папку DICOM, чтобы начать', uz: 'Boshlash uchun DICOM papkasini yuklang', en: 'Upload a DICOM folder to begin' },
  needsReview:     { ru: 'Нужен просмотр', uz: 'Ko\'rib chiqish kerak', en: 'Needs review' },
  signReport:      { ru: 'Подписать отчёт', uz: 'Hisobotni imzolash', en: 'Sign report' },
  signing:         { ru: 'Подписание…', uz: 'Imzolanmoqda…', en: 'Signing…' },
  signedLocked:    { ru: 'Подписано и заблокировано', uz: 'Imzolangan va qulflangan', en: 'Signed & locked' },
  exportSignedPdf: { ru: 'Экспорт подписанного PDF', uz: 'Imzolangan PDF eksporti', en: 'Export signed PDF' },
  exportDraftPdf:  { ru: 'Экспорт черновика PDF', uz: 'Qoralama PDF eksporti', en: 'Export draft PDF' },
  aiDraftWatermark:{ ru: 'ЧЕРНОВИК ИИ — НЕ ПОДПИСАНО', uz: 'AI QORALAMASI — IMZOLANMAGAN', en: 'AI DRAFT — NOT SIGNED' },
  autoTranslated:  { ru: 'Автоматический перевод ИИ — требует проверки', uz: 'AI avtomatik tarjimasi — tekshirish kerak', en: 'Auto-translated by AI — requires review' },
  signFailed:      { ru: 'Не удалось подписать отчёт', uz: 'Hisobotni imzolab bo\'lmadi', en: 'Could not sign the report' },
  alreadySigned:   { ru: 'Отчёт на этом языке уже подписан', uz: 'Bu tildagi hisobot allaqachon imzolangan', en: 'A report in this language is already signed' },
  noPermissionSign:{ ru: 'Ваша роль не позволяет подписывать отчёты', uz: 'Sizning rolingiz hisobot imzolashga ruxsat bermaydi', en: 'Your role cannot sign reports' },
  exportOpened:    { ru: 'Открытые языки', uz: 'Ochilgan tillar', en: 'Opened languages' },
  currentLanguage: { ru: 'Текущий язык', uz: 'Joriy til', en: 'Current language' },
  patient:         { ru: 'Пациент', uz: 'Bemor', en: 'Patient' },
  study:           { ru: 'Исследование', uz: 'Tekshiruv', en: 'Study' },
  equipment:       { ru: 'Оборудование', uz: 'Uskuna', en: 'Equipment' },
  aiPipeline:      { ru: 'ИИ-анализ', uz: 'AI tahlili', en: 'AI analysis' },
  noImagePreview:  { ru: 'Изображение недоступно', uz: 'Tasvir mavjud emas', en: 'No image preview available' },
  restoredHint:    { ru: 'Исследование восстановлено из базы — предпросмотр среза не сохраняется', uz: 'Tekshiruv bazadan tiklandi — kesim ko\'rinishi saqlanmaydi', en: 'Restored from the database — slice preview is not stored' },
  loginTitle:      { ru: 'Вход в систему', uz: 'Tizimga kirish', en: 'Sign in' },
  loginSubtitle:   { ru: 'Введите учётные данные, выданные администратором', uz: 'Administrator bergan hisob ma\'lumotlarini kiriting', en: 'Enter the credentials issued by your administrator' },
  username:        { ru: 'Имя пользователя', uz: 'Foydalanuvchi nomi', en: 'Username' },
  password:        { ru: 'Пароль', uz: 'Parol', en: 'Password' },
  signIn:          { ru: 'Войти', uz: 'Kirish', en: 'Sign in' },
  authenticating:  { ru: 'Проверка…', uz: 'Tekshirilmoqda…', en: 'Authenticating…' },
  badCredentials:  { ru: 'Неверное имя пользователя или пароль', uz: 'Foydalanuvchi nomi yoki parol noto\'g\'ri', en: 'Invalid username or password' },
  enterBoth:       { ru: 'Введите имя пользователя и пароль', uz: 'Foydalanuvchi nomi va parolni kiriting', en: 'Please enter username and password' },
  changePassword:  { ru: 'Смена пароля', uz: 'Parolni o\'zgartirish', en: 'Change password' },
  changePasswordHint:{ ru: 'Перед началом работы необходимо сменить временный пароль', uz: 'Ishni boshlashdan oldin vaqtinchalik parolni o\'zgartirish kerak', en: 'You must replace the temporary password before continuing' },
  currentPassword: { ru: 'Текущий пароль', uz: 'Joriy parol', en: 'Current password' },
  newPassword:     { ru: 'Новый пароль', uz: 'Yangi parol', en: 'New password' },
  confirmPassword: { ru: 'Повторите новый пароль', uz: 'Yangi parolni takrorlang', en: 'Confirm new password' },
  passwordsDiffer: { ru: 'Пароли не совпадают', uz: 'Parollar mos kelmadi', en: 'Passwords do not match' },
  passwordTooShort:{ ru: 'Пароль должен содержать не менее 8 символов', uz: 'Parol kamida 8 ta belgidan iborat bo\'lishi kerak', en: 'Password must be at least 8 characters' },
  changeFailed:    { ru: 'Не удалось сменить пароль — проверьте текущий пароль', uz: 'Parolni o\'zgartirib bo\'lmadi — joriy parolni tekshiring', en: 'Could not change password — check the current password' },
  save:            { ru: 'Сохранить', uz: 'Saqlash', en: 'Save' },
  support:         { ru: 'Поддержка', uz: 'Qo\'llab-quvvatlash', en: 'Support' },
  supportPlaceholder:{ ru: 'Контакт поддержки — укажите в настройках', uz: 'Qo\'llab-quvvatlash kontakti — sozlamalarda ko\'rsating', en: 'Support contact — set in Settings' },
  clinicPlaceholder:{ ru: 'Название клиники — укажите в настройках', uz: 'Klinika nomi — sozlamalarda ko\'rsating', en: 'Clinic name — set in Settings' },
  aiServer:        { ru: 'Сервер ИИ', uz: 'AI serveri', en: 'AI server' },
  online:          { ru: 'в сети', uz: 'ishlamoqda', en: 'online' },
  offline:         { ru: 'не в сети', uz: 'ishlamayapti', en: 'offline' },
  degraded:        { ru: 'частично', uz: 'qisman', en: 'degraded' },
  version:         { ru: 'Версия', uz: 'Versiya', en: 'Version' },
  loadingResult:   { ru: 'Загрузка результата…', uz: 'Natija yuklanmoqda…', en: 'Loading result…' },
  studiesRestored: { ru: 'Список исследований восстановлен', uz: 'Tekshiruvlar ro\'yxati tiklandi', en: 'Worklist restored' },
  sessionExpired:  { ru: 'Сессия завершена — войдите снова', uz: 'Sessiya tugadi — qayta kiring', en: 'Session expired — please sign in again' },
  // ─── Report / sign / export ──────────────────────────────────────────
  signedBy:        { ru: 'Подписал', uz: 'Imzolagan', en: 'Signed by' },
  exportPdf:       { ru: 'Экспорт PDF', uz: 'PDF eksporti', en: 'Export PDF' },
  exporting:       { ru: 'Экспорт…', uz: 'Eksport…', en: 'Exporting…' },
  exportFailed:    { ru: 'Не удалось экспортировать PDF', uz: 'PDF eksport qilib bo\'lmadi', en: 'Could not export the PDF' },
  pdfSaved:        { ru: 'PDF сохранён', uz: 'PDF saqlandi', en: 'PDF saved' },
  noReportYet:     { ru: 'Отчёта пока нет', uz: 'Hisobot hali yo\'q', en: 'No report yet' },
  selectStudyForReport: { ru: 'Выберите исследование, чтобы подготовить отчёт', uz: 'Hisobot tayyorlash uchun tekshiruvni tanlang', en: 'Select a study to prepare a report' },
  generatingReport:{ ru: 'Генерация отчёта', uz: 'Hisobot yaratilmoqda', en: 'Generating report' },
  editReport:      { ru: 'Редактировать', uz: 'Tahrirlash', en: 'Edit' },
  draftEdited:     { ru: 'Черновик изменён врачом', uz: 'Qoralama shifokor tomonidan o\'zgartirilgan', en: 'Draft edited by the doctor' },
  draft:           { ru: 'Черновик', uz: 'Qoralama', en: 'Draft' },
  flagged:         { ru: 'Отмечено', uz: 'Belgilangan', en: 'Flagged' },
  time:            { ru: 'Время', uz: 'Vaqt', en: 'Time' },
  model:           { ru: 'Модель', uz: 'Model', en: 'Model' },

  // ─── Details tab (real DICOM fields) ─────────────────────────────────
  noStudySelected: { ru: 'Исследование не выбрано', uz: 'Tekshiruv tanlanmagan', en: 'No study selected' },
  selectStudyForDetails: { ru: 'Выберите исследование, чтобы посмотреть детали', uz: 'Tafsilotlarni ko\'rish uchun tekshiruvni tanlang', en: 'Select a study to view details' },
  compareUnavailable: { ru: 'Сравнение с предыдущими исследованиями недоступно: в этой версии нет подключения к архиву PACS', uz: 'Oldingi tekshiruvlar bilan solishtirish mavjud emas: bu versiyada PACS arxiviga ulanish yo\'q', en: 'Comparison with prior studies is unavailable: this build has no PACS archive connection' },
  patientId:       { ru: 'ID пациента', uz: 'Bemor ID', en: 'Patient ID' },
  patientName:     { ru: 'Имя', uz: 'Ismi', en: 'Name' },
  sex:             { ru: 'Пол', uz: 'Jinsi', en: 'Sex' },
  age:             { ru: 'Возраст', uz: 'Yoshi', en: 'Age' },
  birthDate:       { ru: 'Дата рождения', uz: 'Tug\'ilgan sana', en: 'Date of birth' },
  studyUid:        { ru: 'Study UID', uz: 'Study UID', en: 'Study UID' },
  accession:       { ru: 'Номер направления', uz: 'Yo\'llanma raqami', en: 'Accession' },
  studyDate:       { ru: 'Дата исследования', uz: 'Tekshiruv sanasi', en: 'Study date' },
  modality:        { ru: 'Модальность', uz: 'Modallik', en: 'Modality' },
  bodyPart:        { ru: 'Область', uz: 'Soha', en: 'Body part' },
  description:     { ru: 'Описание', uz: 'Tavsif', en: 'Description' },
  manufacturer:    { ru: 'Производитель', uz: 'Ishlab chiqaruvchi', en: 'Manufacturer' },
  scannerModel:    { ru: 'Модель аппарата', uz: 'Apparat modeli', en: 'Scanner model' },
  filesCount:      { ru: 'Файлов', uz: 'Fayllar soni', en: 'Files' },
  series:          { ru: 'Серии', uz: 'Seriyalar', en: 'Series' },
  inferenceTime:   { ru: 'Время анализа', uz: 'Tahlil vaqti', en: 'Inference time' },
  analyzedAt:      { ru: 'Проанализировано', uz: 'Tahlil qilingan', en: 'Analyzed at' },
  appVersion:      { ru: 'Версия приложения', uz: 'Ilova versiyasi', en: 'App version' },
  device:          { ru: 'Устройство', uz: 'Qurilma', en: 'Device' },
};

export function L(key: keyof typeof UI_LABELS, lang: Lang = 'ru'): string {
  return UI_LABELS[key][lang];
}
