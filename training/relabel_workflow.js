// =============================================================================
//  SENTINEL — relabel_workflow.js  (Claude Code Workflow script, stage 2 "label")
// =============================================================================
//  WHAT
//    LLM-labels brain MRI/CT report conclusions (Russian ЗАКЛЮЧЕНИЕ) with the
//    fixed 9-class + abnormal yes/no policy of docs/dev/release_pipeline_contract.md.
//    This is the script that labelled batch 1 (the 894-patient reference in
//    data/matched_clean.csv), adapted to read the JSONL that
//    scripts/label_reports.py --export-for-llm writes and to return labels in
//    the exact shape --import-llm-labels consumes.
//
//  HOW TO RUN (Claude Code Workflow tool, NOT node):
//    1. python scripts/label_reports.py --reports data/<batch>_reports.csv \
//         --out data/<batch>_labels.csv --export-for-llm data/<batch>_review.jsonl   [--all]
//    2. In Claude Code:  Workflow({ scriptPath: 'training/relabel_workflow.js',
//                                  args: { file: 'data/<batch>_review.jsonl' } })
//       The script reads the JSONL through a reader agent (scripts have no
//       filesystem access), labels it in batches of 75 lines and returns
//       { labels: [{patient_id, study_date, modality, section_index, label, abnormal}],
//         n_in, n_out, missing }.
//       modality + section_index are ECHOED UNCHANGED from the export: they are
//       part of the import key, so a CT brain and an MR brain of one patient on
//       one day (or two report sections) never overwrite each other. The script
//       takes the key fields from the rows it read, not from the labelling agent,
//       and de-duplicates on the full 4-part key. Lines of an older export that
//       lack them are carried through as '' (import then falls back to the
//       unique (patient_id, study_date) match).
//    3. Save the returned `labels` array as JSONL (one object per line — a single
//       line holding the whole {labels:[…]} object is accepted too) and merge:
//       python scripts/label_reports.py --reports data/<batch>_reports.csv \
//         --out data/<batch>_labels.csv --import-llm-labels data/<batch>_llm_labels.jsonl
//
//  PHI: the JSONL carries only patient_id / study_date / modality / section_index /
//       conclusion — no names.
//       Agents must never print anything else from data/.
//  Plain JS (no TypeScript), no Date.now()/Math.random(), meta is a pure literal.
// =============================================================================

export const meta = {
  name: 'relabel-brain-reports',
  description: 'LLM-label brain report conclusions (9 findings + abnormal) from an --export-for-llm JSONL',
  phases: [
    { title: 'Read', detail: 'load the JSONL exported by label_reports.py' },
    { title: 'Label', detail: 'batches of 75 conclusions, one agent each' },
  ],
}

// Rubric — VERBATIM copy of RUBRIC in scripts/label_reports.py (and the Ollama prompt).
const RUBRIC =
  'You are a neuroradiologist labelling brain MRI/CT report conclusions (Russian ЗАКЛЮЧЕНИЕ). ' +
  'For EACH report choose the SINGLE most significant BRAIN finding: normal = no significant brain ' +
  'pathology; tumor = опухоль / новообразование / глиома / менингиома / аденома / mts / объёмное ' +
  'образование; stroke = инсульт / инфаркт / ишемия / очаг ишемии / лакунарный; atrophy = атрофия / ' +
  'церебральная микроангиопатия / Fazekas / расширение борозд; hemorrhage = внутримозговое ' +
  'кровоизлияние / гематома (NOT scalp/подкожная); ms_wml = демиелинизация / рассеянный склероз / ' +
  'очаги белого вещества; hydrocephalus = гидроцефалия / выраженное расширение желудочков; cyst = ' +
  'киста ГОЛОВНОГО МОЗГА / арахноидальная (NOT киста пазухи); other = other genuine brain finding. ' +
  'CRITICAL: many conclusions describe ONLY sinus/nose (гайморит, синусит, пазух, носовой) or scalp ' +
  'findings while the BRAIN is normal — label those normal, abnormal=false. abnormal=false ONLY when ' +
  'the brain itself is normal.'

const LABELS = ['normal', 'tumor', 'stroke', 'atrophy', 'hemorrhage', 'ms_wml', 'hydrocephalus', 'cyst', 'other']
const BATCH = 75

const READ_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['rows'],
  properties: {
    rows: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['patient_id', 'study_date', 'modality', 'section_index', 'conclusion'],
        properties: {
          patient_id: { type: 'string' },
          study_date: { type: 'string' },
          modality: { type: 'string' },
          section_index: { type: 'string' },
          conclusion: { type: 'string' },
        },
      },
    },
  },
}

const LABEL_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['labels'],
  properties: {
    labels: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['row', 'patient_id', 'study_date', 'modality', 'section_index', 'label', 'abnormal'],
        properties: {
          row: { type: 'integer' },
          patient_id: { type: 'string' },
          study_date: { type: 'string' },
          modality: { type: 'string' },
          section_index: { type: 'string' },
          label: { type: 'string', enum: LABELS },
          abnormal: { type: 'boolean' },
        },
      },
    },
  },
}

const file = args && args.file
if (!file) throw new Error('args.file is required: the JSONL written by label_reports.py --export-for-llm')

// ---------------------------------------------------------------- Phase: Read
phase('Read')
const readResult = await agent(
  `Read the JSONL file ${file} (repo root /Users/shakhzodbtr/Desktop/Siaa_ai/sentinel; use the Read tool or ` +
  `\`cat\`; if it is large, read it in chunks with sed -n). Every line is a JSON object with the keys ` +
  `patient_id, study_date, modality, section_index, conclusion. Return ALL lines as rows, unchanged and in ` +
  `file order (patient_id, study_date, modality and section_index as strings — use "" when a line lacks ` +
  `modality or section_index — and conclusion verbatim). Do not summarise, skip or de-duplicate ` +
  `anything, and do not print the file contents anywhere except in your structured result ` +
  `(it is hospital data — numeric ids only, no names).`,
  { label: 'read-jsonl', phase: 'Read', schema: READ_SCHEMA },
)
const rows = (readResult && readResult.rows) || []
log(`read ${rows.length} conclusions from ${file}`)
if (!rows.length) return { labels: [], n_in: 0, n_out: 0, missing: [] }

// --------------------------------------------------------------- Phase: Label
phase('Label')
const batches = []
for (let i = 0; i < rows.length; i += BATCH) batches.push(rows.slice(i, i + BATCH))
log(`${batches.length} batch(es) of up to ${BATCH}`)

// The 4-part key --import-llm-labels matches on (label_reports.py import_llm_labels).
const keyOf = r => [r.patient_id, r.study_date, r.modality, r.section_index].map(v => String(v == null ? '' : v)).join('|')
const norm = r => ({
  patient_id: String(r.patient_id),
  study_date: String(r.study_date || ''),
  modality: String(r.modality || ''),
  section_index: String(r.section_index == null ? '' : r.section_index),
})

const labelBatch = (batch, idx) => agent(
  RUBRIC +
  '\n\nLabel EVERY report below. Return exactly one entry per report: `row` is the report\'s row number ' +
  'and patient_id, study_date, modality and section_index are copied verbatim (all strings, "" stays ""). ' +
  'The same patient may appear more than once (e.g. CT and MR on one day) — label each row on its own. ' +
  'label must be one of ' + LABELS.join(' | ') + '; abnormal is a boolean and is ' +
  'false ONLY when label is normal. Negations matter: a keyword inside "не выявлено / без патологии / ' +
  'не изменены / отсутствуют" is NOT a finding; hedges ("нельзя исключить", "не исключается", "под вопросом", ' +
  '"(?)") are NOT negations — keep the hedged finding. Do not print or echo the reports anywhere except in the ' +
  'structured result.\n\nREPORTS (JSON lines):\n' +
  batch.map((r, k) => JSON.stringify({ row: idx * BATCH + k, ...norm(r), conclusion: r.conclusion })).join('\n'),
  { label: `label-batch-${idx + 1}`, phase: 'Label', schema: LABEL_SCHEMA },
)

const results = await pipeline(batches, (batch, _item, idx) => labelBatch(batch, idx))

const labels = []
const seen = new Set()
for (const r of results) {
  if (!r || !Array.isArray(r.labels)) continue
  for (const l of r.labels) {
    // The key fields come from the row that was READ (echoed unchanged), never
    // from the labelling agent's copy; a row number that does not point back to
    // the same patient is dropped (it stays in `missing`, i.e. needs_review=yes).
    const src = Number.isInteger(l.row) ? rows[l.row] : undefined
    if (!src || String(src.patient_id) !== String(l.patient_id)) continue
    const key = norm(src)
    const k = keyOf(key)
    if (seen.has(k)) continue
    seen.add(k)
    labels.push({
      ...key,
      label: l.label,
      abnormal: l.label === 'normal' ? false : Boolean(l.abnormal),
    })
  }
}
const missing = rows.filter(r => !seen.has(keyOf(norm(r)))).map(norm)
if (missing.length) log(`${missing.length} row(s) came back without a label — re-run for them or leave needs_review=yes`)
log(`labelled ${labels.length}/${rows.length}`)

return { labels, n_in: rows.length, n_out: labels.length, missing }
