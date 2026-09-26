#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — PARSE HOSPITAL RADIOLOGY REPORTS INTO ONE ROW PER STUDY SECTION
================================================================================
  Release-pipeline stage 1 ("parse-reports", docs/dev/release_pipeline_contract.md).

  WHAT
    Reads hospital radiology reports and writes a CSV keyed by the numeric
    PatientID so the rows auto-join to the DICOM studies (same PatientID in the
    headers) in stage 3.  Accepted inputs (a folder, or a single file):
      * PDF  — Shox International Hospital layout, named <patientID>_<Name>.pdf.
               ONE PDF may hold SEVERAL studies.  Each study section is a lone
               modality line ("МРТ" / "МСКТ") followed by a header such as
               "МСКТ головного мозга (2026-07-21)", the protocol text, a
               "ЗАКЛЮЧЕНИЕ:" line and boilerplate ("Заключение не является
               диагнозом…", "Результат подготовил(а): <name>", "Врач: ___",
               "Уважаемые пациенты!…").  One output row per section.
      * XLSX / CSV — the request-sheet layout (Russian headers
               "ID пациента (PatientID)", "Дата исследования", "Модальность",
               "Область", "Описание исследования", "Заключение врача …");
               every sheet whose header row has the ID + conclusion columns.
      * DOCX — paragraphs are parsed like the PDF text.

  WHY (bugs of the previous keyword parser this replaces)
    1. modality was "МСКТ" for every report because the letterhead says
       "Результаты МРТ / МСКТ" — now modality comes from each SECTION header.
    2. multi-study PDFs produced one row with the first date and a conclusion
       that ran into the next study — now one row per section.
    3. the conclusion carried boilerplate and the doctor's name and was cut at
       600 chars — now it is the ЗАКЛЮЧЕНИЕ text only, whitespace-collapsed,
       never truncated, with every boilerplate/signature line removed.

  OUTPUT CSV columns (backward compatible prefix, then the contract's extras):
    patient_id, modality (MR|CT|?), body_part (brain|spine|chest|…|other),
    study_date (YYYY-MM-DD or ''), label (''; stage 2 fills it), conclusion,
    description (protocol text), source_file (basename, name part -> sha256[:16]),
    section_index (1-based within the source file), section_header

  PHI: patient NAMES never reach the output — the file name's name part is
  replaced by a hash, signature lines are stripped, only the numeric ID is kept.
  Reports WITHOUT a dated section header (DOCX exports, other templates) take a
  fallback path that first cuts everything before the first protocol / study
  line, and every description + conclusion + section_header (all input kinds)
  then passes a PHI scrubber (Ф.И.О., дата рождения / выдачи, ID:, Пациент /
  Больной / Фамилия / Имя / Отчество labels in ANY case, "Фамилия И.О." names,
  "1980 г.р.", signer fragments). section_header is further cut to the study
  title (text before the first comma / patient label).
  Error messages and the summary never echo an original file name (hashed).
  Keep the source PDFs off git/cloud.  Nothing but the --out CSV is written.

  Usage:
    python scripts/parse_hospital_reports.py --src "<folder-or-file>" \
        --out data/<batch>_reports.csv [--limit N] [--workers 4]
  Exit codes: 0 ok · 1 no input / no rows parsed.
================================================================================
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

G, Y, B, R, NC = '\033[0;32m', '\033[1;33m', '\033[0;34m', '\033[0;31m', '\033[0m'

PDFTOTEXT_CANDIDATES = ('/opt/homebrew/bin/pdftotext', 'pdftotext')
SUPPORTED_SUFFIXES = ('.pdf', '.xlsx', '.csv', '.docx')

OUTPUT_COLUMNS = ['patient_id', 'modality', 'body_part', 'study_date', 'label', 'conclusion',
                  'description', 'source_file', 'section_index', 'section_header']

# ------------------------------------------------------------------ vocab
# lone-line / first-token modality words -> canonical modality
MODALITY_TOKENS = {
    'мрт': 'MR', 'mri': 'MR', 'mr': 'MR', 'магнитно-резонансная': 'MR',
    'мскт': 'CT', 'кт': 'CT', 'скт': 'CT', 'ct': 'CT', 'компьютерная': 'CT',
}

# body-part keywords -> canonical.  ORDER MATTERS: the first group with a hit
# wins, so specific regions (brain, sinus, orbit) come before the generic ones,
# chest ("грудной клетки") before spine ("грудного отдела позвоночника"), and
# hip ("тазобедренный сустав") before joint/pelvis.  Cyrillic keys match as
# substrings, Latin keys (DICOM BodyPartExamined words) as whole words.
BODY_PARTS: list[tuple[str, list[str]]] = [
    ('brain',   ['головного мозга', 'головной мозг', 'гипофиз', 'головы', 'голова',
                 'внутричерепн', 'brain', 'head', 'skull', 'cerebr']),
    ('sinus',   ['придаточных пазух', 'пазух носа', 'ппн', 'околоносов', 'sinus', 'paranasal']),
    ('orbit',   ['орбит', 'глазниц', 'orbit']),
    ('chest',   ['грудной клетки', 'органов грудн', 'огк', 'легк', 'средостен',
                 'chest', 'lung', 'thorax']),
    ('spine',   ['позвоноч', 'пояснич', 'крестц', 'копчик', 'шейного отдела', 'грудного отдела',
                 'spine', 'lspine', 'cspine', 'tspine', 'lumbar', 'cervical']),
    ('abdomen', ['брюшн', 'печен', 'почек', 'почки', 'забрюшин', 'надпочечн',
                 'abdomen', 'liver', 'kidney']),
    ('pelvis',  ['малого таза', 'таза', 'pelvis']),
    ('neck',    ['шеи', 'гортан', 'щитовидн', 'neck', 'thyroid']),
    ('knee',    ['коленн', 'knee']),
    ('hip',     ['тазобедрен', 'hip']),
    ('joint',   ['сустав', 'плечев', 'локтев', 'голеностоп', 'стоп', 'кисти', 'лучезапяст',
                 'joint', 'shoulder', 'elbow', 'ankle', 'wrist', 'foot']),
    ('vessel',  ['ангиограф', 'коронарограф', 'флебограф', 'сосуд', 'артери', 'аорт',
                 'angio', 'vessel', 'aorta']),
    ('bone',    ['кост', 'лицевого скелета', 'челюст', 'bone', 'skelet', 'jaw']),
]

# a section header: "<title> (YYYY-MM-DD)" or "<title> (DD.MM.YYYY)" at line end
HEADER_RE = re.compile(r'^\s*(?P<title>\S.{0,180}?)\s*\((?P<date>\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4})\)\s*$')
# a lone modality line ("МРТ", "МСКТ", possibly indented)
LONE_MODALITY_RE = re.compile(r'^\s*([A-Za-zА-Яа-яЁё\-]{2,24})\s*$')
# the conclusion marker at a line start; the boilerplate sentence
# "Заключение не является диагнозом…" is NOT a marker
CONCLUSION_RE = re.compile(r'^\s*(?:заключение|выводы?)\s*[:.]?\s*(?P<rest>.*)$', re.IGNORECASE)
CONCLUSION_PREFIX_RE = re.compile(r'^\s*(МСКТ|МРТ|КТ|СКТ|MRI|MR|CT)\b', re.IGNORECASE)
# any of these ends the conclusion / description and never reaches the CSV
BOILERPLATE_PATTERNS = [
    r'заключение\s+не\s+является',
    r'результат\s+подготовил',
    # "Врач:", "Врач-рентгенолог:", "Врач рентгенолог:", "Врач-радиолог:" — the signer line
    r'(?:^|\s)врач(?:[\s\-/]+[а-яё\-]{2,30})?\s*:',
    r'уважаемые\s+пациенты',
    r'результаты\s+исследований\s+и\s+снимки',
    r'проверка\s+подлинности',
    r'_{4,}',
]
BOILERPLATE_RE = re.compile('|'.join(BOILERPLATE_PATTERNS), re.IGNORECASE)
LETTERHEAD_ID_RE = re.compile(r'\bID\s*[:№]?\s*(\d{3,})')
FILE_ID_RE = re.compile(r'^(\d+)[_\-\s]+(.*)$')

# fallback path (no dated section header): the description starts at the first
# line that opens a protocol / study — a modality word, "ПРОТОКОЛ", "На серии",
# "МР-…", "КТ-…" — never at the letterhead
STUDY_START_RE = re.compile(
    r'^\s*(?:протокол|на\s+серии|мр[\s\-]|кт[\s\-]|мскт|мрт|скт|mri\b|ct\b|'
    r'магнитно-резонансн|компьютерн(?:ая|ой)\s+томограф)', re.IGNORECASE)

# ---- PHI scrubber (belt and braces on top of the section split) -----------
# Letterhead / signature material that must never reach description,
# conclusion or section_header.  Letterhead LABELS match in any case
# ("ПАЦИЕНТ:", "ДАТА РОЖДЕНИЯ:" — uppercase DOCX / template tables, review
# round 2); the name words after a label must start with a capital, so clinical
# prose ("у пациента выявлено", "врача-невролога") survives.
_PHI_DATE = r'(?:\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}|\d{4}-\d{2}-\d{2})'
_NAME_WORD = r"(?:[А-ЯЁA-Z][А-ЯЁа-яёA-Za-z\-']+\.?|[А-ЯЁA-Z]\.)"
_NAME_SUFFIX = r'(?:\s+(?:оглы|угли|ўғли|кызы|қизи|кизи))?'
_NAMES_OPT = rf'(?:\s*{_NAME_WORD}{_NAME_SUFFIX}){{0,5}}'
_NAMES_REQ = rf'(?:\s*{_NAME_WORD}{_NAME_SUFFIX}){{1,5}}'
_FIO = r'(?i:ф\.\s?и\.\s?о\.?|\bфио\b)'
_PROPER_WORD = r"(?:[А-ЯЁ][а-яё][а-яё\-']*\.?|[А-ЯЁ]\.)"
# letterhead labels whose value is a name ("Фамилия: …", "Больной: …"); only with a colon/dash,
# because "больной" / "имя" can occur in prose
_NAME_LABEL = r'(?i:пациент(?:ка)?|больн(?:ой|ая)|фамилия|имя|отчество)'
# "Иванов И.И." / "И.И. Иванов" / "ИВАНОВ И.И." — a surname next to initials is a name wherever it is
_INITIALS_NAME = (r'\b[А-ЯЁ][А-ЯЁа-яё]+(?:-[А-ЯЁ][А-ЯЁа-яё]+)?\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.'
                  r'|\b[А-ЯЁ]\.\s?[А-ЯЁ]\.\s?[А-ЯЁ][а-яё]+')
_SIGNER = (r'(?:[Рр]езультат\s+подготовил\w*(?:\s*\([а-яё]+\))?'
           r'|[Вв]рач(?:[\s\-/]+[а-яё\-]{2,30})?'
           r'|(?:[Вв]рач[\s\-/]+)?(?:[Рр]ентгенолог|[Рр]адиолог)\w*)')
PHI_FRAGMENT_PATTERNS = [
    # ORDER MATTERS: date / id fragments first, so that a name pattern cannot
    # swallow the "Дата" of "Дата рождения" before its own rule runs
    rf'(?i:дата\s+рожд\w*)\s*[:\-–]?\s*{_PHI_DATE}?(?:\s*г\.?\s*р?\.?)?',
    rf'(?i:дата\s+выдачи)\s*[:\-–]?\s*{_PHI_DATE}?',
    rf'(?i:рожд\w*)\W{{0,30}}?{_PHI_DATE}',                # "рождения: 01.01.2031"
    rf'{_PHI_DATE}\W{{0,30}}?(?:г\.?\s*р\.|(?i:рожд\w*))',  # "01.01.2031 г.р."
    # "1980 г.р." / "1980 года рождения" (year only) — never "2026г. ранее"
    r'\b(?:19|20)\d{2}\s*(?:г\.?\s*р(?:\.|\b)|(?i:года?\s+рожд\w*))',
    r'(?i:\bID)\s*[:№#]?\s*\d+',
    rf'{_FIO}\s*[:\-–]?{_NAMES_OPT}',
    rf'{_NAME_LABEL}\s*[:\-–]\s*{_NAMES_OPT}',
    rf'[Пп]ациент(?:ка)?\s*{_NAMES_REQ}',
    # inflected forms ("пациента Иванова Ивана") only before PROPER names (Capital + lower-case, or
    # an initial) — "у пациентки СПО" (an abbreviation) stays
    rf'(?i:пациент(?:а|у|ом|ки|ке|кой))(?:\s+{_PROPER_WORD}){{1,3}}',
    _INITIALS_NAME,
    rf'{_SIGNER}\s*(?:[:_\-–]+\s*{_NAMES_OPT}|{_NAMES_REQ})',
]
PHI_FRAGMENT_RE = re.compile('|'.join(f'(?:{p})' for p in PHI_FRAGMENT_PATTERNS))
# a whole line is dropped when it carries a letterhead-only field (these words
# never occur in clinical prose).  Mentions that CAN sit inside a clinical
# sentence ("Пациентка Х., 01.01.2031 г.р.") are removed as fragments by
# scrub_phi() instead, so the rest of that protocol / conclusion line survives.
PHI_LINE_RE = re.compile(
    rf'{_FIO}|(?i:дата\s+рожд|дата\s+выдачи|\bID\s*[:№#]?\s*\d{{3,}})'
    rf'|^\s*{_NAME_LABEL}\s*[:\-–]')
# a letterhead LABEL alone on its line (DOCX tables give one paragraph per cell:
# "Ф.И.О." / "<name>" / "Дата рождения" / "<date>") — its value line goes too
PHI_LABEL_ONLY_RE = re.compile(
    rf'^\s*(?:{_FIO}|(?i:дата\s+рожд\w*|дата\s+выдачи|id)|{_NAME_LABEL})\s*[:\-–№#]?\s*$')
PHI_VALUE_LINE_RE = re.compile(
    rf'^\s*(?:{_NAMES_REQ}|{_PHI_DATE}(?:\s*г\.?\s*р?\.?)?|\d{{3,}})\s*$')

# request-sheet header words (lower-case substrings)
SHEET_COLUMNS = {
    'patient_id':  ('patientid', 'id пациента', 'patient_id', 'patient id'),
    'study_date':  ('дата исследования', 'study_date', 'study date', 'дата'),
    'modality':    ('модальность', 'modality'),
    'body_part':   ('область', 'body_part', 'bodypart', 'body part'),
    'description': ('описание', 'description', 'study_description'),
    'conclusion':  ('заключение', 'conclusion'),
}


# ------------------------------------------------------------- text readers
def pdf_text(path: Path) -> str:
    """Text of a PDF via pdftotext -layout ('' when the tool or the text is missing)."""
    for exe in PDFTOTEXT_CANDIDATES:
        try:
            r = subprocess.run([exe, '-layout', str(path), '-'], capture_output=True, timeout=60)
        except FileNotFoundError:
            continue
        except Exception:
            return ''
        return r.stdout.decode('utf-8', 'ignore') if r.returncode == 0 else ''
    return ''


def docx_text(path: Path) -> str:
    """Paragraph text of a .docx (word/document.xml), one paragraph per line."""
    W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read('word/document.xml'))
    lines = []
    for p in root.iter(W + 'p'):
        parts = []
        for node in p.iter():
            if node.tag == W + 't':
                parts.append(node.text or '')
            elif node.tag in (W + 'br', W + 'cr'):
                parts.append('\n')
            elif node.tag == W + 'tab':
                parts.append('\t')
        lines.append(''.join(parts))
    return '\n'.join(lines)


def _col_index(ref: str) -> int:
    """'A1' -> 0, 'AB7' -> 27."""
    n = 0
    for ch in ref:
        if ch.isalpha():
            n = n * 26 + (ord(ch.upper()) - 64)
        else:
            break
    return max(n - 1, 0)


def xlsx_sheets(path: Path) -> list[tuple[str, list[list[str]]]]:
    """All worksheets of an .xlsx as (sheet_name, rows) using zipfile + ElementTree only.
    Handles shared strings, inline strings, numbers and booleans."""
    M = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    RID = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        shared: list[str] = []
        if 'xl/sharedStrings.xml' in names:
            for si in ET.fromstring(z.read('xl/sharedStrings.xml')).iter(M + 'si'):
                shared.append(''.join(t.text or '' for t in si.iter(M + 't')))
        # sheet name -> xml path (workbook.xml + its rels); fall back to sheetN.xml order
        sheets: list[tuple[str, str]] = []
        try:
            rels = {}
            for rel in ET.fromstring(z.read('xl/_rels/workbook.xml.rels')):
                target = rel.get('Target', '')
                target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
                rels[rel.get('Id')] = target
            for s in ET.fromstring(z.read('xl/workbook.xml')).iter(M + 'sheet'):
                target = rels.get(s.get(RID), '')
                if target in names:
                    sheets.append((s.get('name', target), target))
        except Exception:
            sheets = []
        if not sheets:
            paths = sorted((n for n in names if re.match(r'xl/worksheets/sheet\d+\.xml$', n)),
                           key=lambda n: int(re.search(r'(\d+)\.xml$', n).group(1)))
            sheets = [(Path(p).stem, p) for p in paths]
        out = []
        for sheet_name, target in sheets:
            root = ET.fromstring(z.read(target))
            rows: list[list[str]] = []
            for row in root.iter(M + 'row'):
                cells: dict[int, str] = {}
                for c in row.findall(M + 'c'):
                    idx = _col_index(c.get('r', '')) if c.get('r') else len(cells)
                    t = c.get('t')
                    if t == 'inlineStr':
                        val = ''.join(x.text or '' for x in c.iter(M + 't'))
                    else:
                        v = c.find(M + 'v')
                        val = (v.text or '') if v is not None else ''
                        if t == 's' and val:
                            try:
                                val = shared[int(val)]
                            except (ValueError, IndexError):
                                pass
                        elif t == 'b':
                            val = 'TRUE' if val == '1' else 'FALSE'
                    cells[idx] = val
                width = (max(cells) + 1) if cells else 0
                rows.append([cells.get(i, '') for i in range(width)])
            out.append((sheet_name, rows))
    return out


def csv_rows(path: Path) -> list[list[str]]:
    with open(path, encoding='utf-8-sig', newline='') as fh:
        sample = fh.read(4096)
        fh.seek(0)
        delimiter = ';' if sample.count(';') > sample.count(',') else ','
        return [list(r) for r in csv.reader(fh, delimiter=delimiter)]


# ---------------------------------------------------------- canonicalisers
def normalize_date(value: str | float | int | None) -> str:
    """'21.07.2026' / '2026-07-21' / '20260721' / Excel serial -> 'YYYY-MM-DD', else ''."""
    if value is None:
        return ''
    s = str(value).strip()
    if not s:
        return ''
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', s)
    if m:
        y, mo, d = m.groups()
    else:
        m = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', s)
        if m:
            d, mo, y = m.groups()
        else:
            m = re.fullmatch(r'(\d{4})(\d{2})(\d{2})', s)
            if m:
                y, mo, d = m.groups()
            elif re.fullmatch(r'\d{4,6}(\.\d+)?', s):  # Excel serial number
                try:
                    dt = datetime(1899, 12, 30) + timedelta(days=float(s))
                    return dt.strftime('%Y-%m-%d')
                except (ValueError, OverflowError):
                    return ''
            else:
                return ''
    try:
        return date(int(y), int(mo), int(d)).isoformat()
    except ValueError:
        return ''


def canon_modality(value: str) -> str:
    """Modality word ('МРТ', 'МСКТ', 'MR', 'CT', a header's first token) -> MR|CT|?."""
    s = (value or '').strip().lower()
    if not s:
        return '?'
    if s in MODALITY_TOKENS:
        return MODALITY_TOKENS[s]
    first = re.match(r'[a-zа-яё\-]+', s)
    if first and first.group(0) in MODALITY_TOKENS:
        return MODALITY_TOKENS[first.group(0)]
    if 'магнитно' in s or 'mri' in s:
        return 'MR'
    if 'компьютерн' in s or 'мскт' in s:
        return 'CT'
    return '?'


def _keyword_hit(kw: str, text_lc: str) -> bool:
    if kw.isascii():
        return re.search(r'(?<![a-z])' + re.escape(kw) + r'(?![a-z])', text_lc) is not None
    return kw in text_lc


def canon_body_part(value: str) -> str:
    """Header / DICOM body-part text -> canonical body part (first BODY_PARTS group with a hit)."""
    s = (value or '').lower()
    if not s.strip():
        return 'other'
    for name, kws in BODY_PARTS:
        if any(_keyword_hit(k, s) for k in kws):
            return name
    return 'other'


def collapse_ws(text: str) -> str:
    return re.sub(r'\s+', ' ', text.replace('\f', ' ')).strip()


def strip_boilerplate(text: str) -> str:
    """Cut the text at the first boilerplate / signature marker and collapse whitespace."""
    t = collapse_ws(text)
    m = BOILERPLATE_RE.search(t)
    if m:
        t = t[:m.start()].rstrip()
    return t.lstrip(' .-–—:;,').rstrip(' -–—:;,')


def scrub_phi(text: str) -> str:
    """Remove letterhead / signature fragments (Ф.И.О. + name, дата рождения,
    дата выдачи, ID: <n>, Пациент: <name>, Врач / рентгенолог / Результат
    подготовил + name) from an already joined description or conclusion."""
    t = collapse_ws(text)
    if not t:
        return ''
    t = PHI_FRAGMENT_RE.sub(' ', t)
    t = re.sub(r'\(\s*\)', ' ', t)
    t = collapse_ws(t)
    return t.lstrip(' .-–—:;,').rstrip(' -–—:;,')


def scrub_header(text: str) -> str:
    """section_header: the study TITLE only ("МРТ головного мозга"). PHI fragments are
    scrubbed, then everything from the first comma / semicolon OUTSIDE parentheses, or from a
    patient label, is cut (a title line that names the patient: "МРТ головного мозга,
    пациент …"); "(почки, мочеточники)" inside a title stays."""
    t = scrub_phi(text)
    t = re.split(r'(?i:\bпациент|\bбольн(?:ой|ая)\b)', t, maxsplit=1)[0]
    depth = 0
    for i, ch in enumerate(t):
        depth += ch == '('
        depth -= ch == ')' and depth > 0
        if ch in ',;' and depth == 0:
            t = t[:i]
            break
    return collapse_ws(t).rstrip(' -–—:;,(')[:120]


def drop_phi_lines(lines: Iterable[str]) -> list[str]:
    """Lines without the ones that carry a letterhead field (name, birth date, …).
    A label alone on its line also takes the next non-empty line when that looks
    like its value (a name, a date or a number)."""
    out: list[str] = []
    drop_value = False
    for line in lines:
        if drop_value and line.strip():
            drop_value = False
            if PHI_VALUE_LINE_RE.match(line):
                continue
        if PHI_LABEL_ONLY_RE.match(line):
            drop_value = True
            continue
        if PHI_LINE_RE.search(line):
            continue
        out.append(line)
    return out


def cut_letterhead(lines: list[str]) -> list[str]:
    """Fallback path: the lines from the first protocol / study line on (a modality
    line, ПРОТОКОЛ, На серии, МР-/КТ-/МСКТ/МРТ …); when there is none, from the
    conclusion marker on; [] when neither exists — the letterhead is never kept."""
    for i, line in enumerate(lines):
        if STUDY_START_RE.match(line) or _is_lone_modality(line):
            return lines[i:]
    for i, line in enumerate(lines):
        m = CONCLUSION_RE.match(line)
        if m and not re.match(r'не\s+является', m.group('rest').strip(), re.IGNORECASE):
            return lines[i:]
    return []


def hashed_source_name(filename: str) -> str:
    """'<pid>_<Name Surname>.pdf' -> '<pid>_<sha256(name)[:16]>.pdf'; a file without the
    numeric prefix keeps only a hash of its stem — a person's name never reaches the CSV."""
    p = Path(filename)
    stem, suffix = p.stem, p.suffix.lower()
    m = FILE_ID_RE.match(stem)
    if m:
        pid, rest = m.group(1), m.group(2).strip()
        digest = hashlib.sha256(rest.encode('utf-8')).hexdigest()[:16]
        return f'{pid}_{digest}{suffix}'
    if stem.isdigit():
        return f'{stem}{suffix}'
    return f'{hashlib.sha256(stem.encode("utf-8")).hexdigest()[:16]}{suffix}'


def patient_id_from_name(filename: str) -> str:
    m = re.match(r'^(\d+)', Path(filename).stem)
    return m.group(1) if m else ''


# ------------------------------------------------------------ PDF/DOCX text
def _is_lone_modality(line: str) -> bool:
    m = LONE_MODALITY_RE.match(line)
    return bool(m) and m.group(1).lower() in MODALITY_TOKENS


def split_sections(text: str) -> list[dict]:
    """Find the study sections of a report text.
    A section starts at a header line "<title> (<date>)" that is preceded by a lone
    modality line (or whose first word is itself a modality word).  Returns
    [{'header', 'modality', 'date', 'lines'}] in document order; [] when none."""
    lines = text.split('\n')
    starts: list[tuple[int, int, str, str, str]] = []  # (start_idx, header_idx, header, modality, date)
    for i, line in enumerate(lines):
        m = HEADER_RE.match(line.replace('\f', ''))
        if not m:
            continue
        title = m.group('title').strip()
        j = i - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        lone = j >= 0 and _is_lone_modality(lines[j])
        modality = canon_modality(lines[j]) if lone else canon_modality(title)
        if not lone and modality == '?':
            continue  # a dated line inside the protocol text, not a header
        starts.append((j if lone else i, i, title, modality, normalize_date(m.group('date'))))
    sections = []
    for k, (start, hdr, title, modality, sdate) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        sections.append({'header': title, 'modality': modality, 'date': sdate,
                         'lines': lines[hdr + 1:end]})
    return sections


def extract_conclusion(lines: Iterable[str]) -> tuple[str, str]:
    """(description, conclusion) of one section's lines.  The conclusion starts at the
    first 'ЗАКЛЮЧЕНИЕ' marker (any case, colon optional, text may continue on the next
    lines) and ends at the first boilerplate / signature line."""
    lines = [l.replace('\f', '') for l in lines]
    marker = -1
    for i, line in enumerate(lines):
        m = CONCLUSION_RE.match(line)
        if m and not re.match(r'не\s+является', m.group('rest').strip(), re.IGNORECASE):
            marker = i
            break
    if marker < 0:
        return strip_boilerplate(' '.join(lines)), ''
    description = strip_boilerplate(' '.join(lines[:marker]))
    first = CONCLUSION_RE.match(lines[marker]).group('rest')
    body = [first]
    for line in lines[marker + 1:]:
        if BOILERPLATE_RE.search(line):
            break
        body.append(line)
    return description, strip_boilerplate(' '.join(body))


def parse_report_text(text: str, patient_id: str, source_file: str) -> list[dict]:
    """Rows (one per study section) of a PDF/DOCX report text."""
    if not text.strip():
        return []
    if not patient_id:
        m = LETTERHEAD_ID_RE.search(text)
        patient_id = m.group(1) if m else ''
    sections = split_sections(text)
    rows = []
    if sections:
        for k, sec in enumerate(sections, start=1):
            description, conclusion = extract_conclusion(drop_phi_lines(sec['lines']))
            description, conclusion = scrub_phi(description), scrub_phi(conclusion)
            body_part = canon_body_part(sec['header'])
            if body_part == 'other':
                body_part = canon_body_part(description[:200])
            rows.append({
                'patient_id': patient_id, 'modality': sec['modality'], 'body_part': body_part,
                'study_date': sec['date'], 'label': '', 'conclusion': conclusion,
                'description': description, 'source_file': source_file, 'section_index': k,
                'section_header': scrub_header(sec['header']),
            })
        return rows
    # fallback — no dated section header (DOCX export, another clinic's template):
    # one row.  The letterhead (Ф.И.О., дата рождения, ID) is cut BEFORE anything
    # is joined, then the PHI scrubber runs over the joined fields as well.
    lines = drop_phi_lines(cut_letterhead([l.replace('\f', '') for l in text.split('\n')]))
    description, conclusion = extract_conclusion(lines)
    description, conclusion = scrub_phi(description), scrub_phi(conclusion)
    if not conclusion and not description:
        return []
    header = lines[0].strip() if lines else ''
    if len(header) > 120 or canon_modality(header) == '?':
        header = ''
    pm = CONCLUSION_PREFIX_RE.match(conclusion)
    modality = canon_modality(pm.group(1)) if pm else canon_modality(header)
    body_part = canon_body_part(header)
    if body_part == 'other':
        body_part = canon_body_part(conclusion)
    if body_part == 'other':
        body_part = canon_body_part(description[:300])
    joined = '\n'.join(lines)
    m = (re.search(r'\((\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4})\)', joined)
         or re.search(r'^\s*(\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4})\s*$', joined, re.MULTILINE))
    rows.append({
        'patient_id': patient_id, 'modality': modality, 'body_part': body_part,
        'study_date': normalize_date(m.group(1)) if m else '', 'label': '',
        'conclusion': conclusion, 'description': description, 'source_file': source_file,
        'section_index': 1, 'section_header': scrub_header(header),
    })
    return rows


# ------------------------------------------------------------ sheet layout
def _find_header(rows: list[list[str]]) -> tuple[int, dict[str, int]] | None:
    """Index of the header row + column map, or None when the sheet has no ID+conclusion.
    Every key takes a DISTINCT cell and at least three keys must match, so a summary
    sheet whose legend cell mentions both "ID пациента" and "заключение" is not a table."""
    for i, row in enumerate(rows[:50]):
        cells = [str(c or '').strip().lower() for c in row]
        cols: dict[str, int] = {}
        for key, words in SHEET_COLUMNS.items():
            for j, cell in enumerate(cells):
                if j in cols.values() or not cell or not any(w in cell for w in words):
                    continue
                if key == 'study_date' and 'рожден' in cell:  # 'дата' must not take 'дата рождения'
                    continue
                cols[key] = j
                break
        if 'patient_id' in cols and 'conclusion' in cols and len(cols) >= 3:
            return i, cols
    return None


def parse_sheet_rows(rows: list[list[str]], source_file: str, sheet_name: str = '',
                     stats: Counter | None = None) -> list[dict]:
    """Rows of one request-sheet table (xlsx sheet or csv).  Rows without a conclusion
    are skipped (counted in stats['sheet_rows_without_conclusion'])."""
    found = _find_header(rows)
    if not found:
        return []
    hdr, cols = found

    def cell(row: list[str], key: str) -> str:
        j = cols.get(key)
        return str(row[j]).strip() if j is not None and j < len(row) and row[j] is not None else ''

    out = []
    for n, row in enumerate(rows[hdr + 1:], start=1):
        pid = re.sub(r'\.0$', '', cell(row, 'patient_id'))
        conclusion = strip_boilerplate(cell(row, 'conclusion'))
        if not pid.strip():
            continue
        if not pid.isdigit():  # the hospital PatientID is numeric; anything else is a note row
            if stats is not None:
                stats['sheet_rows_bad_patient_id'] += 1
            continue
        if not conclusion:
            if stats is not None:
                stats['sheet_rows_without_conclusion'] += 1
            continue
        conclusion = scrub_phi(conclusion)
        area, desc = cell(row, 'body_part'), scrub_phi(cell(row, 'description'))
        body_part = canon_body_part(area)
        if body_part == 'other':
            body_part = canon_body_part(desc) if desc else 'other'
        if body_part == 'other':
            body_part = canon_body_part(conclusion[:120])
        modality = canon_modality(cell(row, 'modality'))
        if modality == '?':
            pm = CONCLUSION_PREFIX_RE.match(conclusion)
            modality = canon_modality(pm.group(1)) if pm else '?'
        out.append({
            'patient_id': pid, 'modality': modality, 'body_part': body_part,
            'study_date': normalize_date(cell(row, 'study_date')), 'label': '',
            'conclusion': conclusion, 'description': desc, 'source_file': source_file,
            'section_index': n, 'section_header': scrub_phi(f'{sheet_name} {area}'.strip()),
        })
    return out


def _dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = (r['patient_id'], r['study_date'], r['modality'], r['body_part'], r['conclusion'])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# --------------------------------------------------------------- per file
def sanitize_error(exc: BaseException, path: Path, source: str) -> str:
    """'<ExcType>: <message>' with every form of the original path / file name
    (OSError embeds it) replaced by the hashed source name, and the tokens of the
    name part removed — the message goes to stdout and to the release logs."""
    msg = f'{type(exc).__name__}: {exc}'
    forms = {str(path), path.name, path.stem}
    for fn in (path.resolve, path.absolute):
        try:
            forms.add(str(fn()))
        except Exception:
            pass
    for form in sorted((f for f in forms if f), key=len, reverse=True):
        msg = msg.replace(form, source)
    m = FILE_ID_RE.match(path.stem)
    name_part = m.group(2).strip() if m else ('' if path.stem.isdigit() else path.stem)
    if name_part:
        msg = msg.replace(name_part, source)
        for tok in re.split(r'[\s_\-\.]+', name_part):
            if len(tok) >= 3 and re.search(r'[^\W\d_]', tok) and tok in msg:
                msg = msg.replace(tok, '…')
    return msg


def display_path(p: Path) -> str:
    """A path for messages: the file name part hashed when it is a report file."""
    if p.suffix.lower() in SUPPORTED_SUFFIXES:
        return str(p.parent / hashed_source_name(p.name))
    return str(p)


def parse_file(path: Path, stats: Counter | None = None) -> tuple[list[dict], str]:
    """(rows, error) for one input file.  error != '' when the file yielded nothing."""
    source = hashed_source_name(path.name)
    suffix = path.suffix.lower()
    try:
        if suffix == '.pdf':
            rows = parse_report_text(pdf_text(path), patient_id_from_name(path.name), source)
            return rows, ('' if rows else 'no text / no sections')
        if suffix == '.docx':
            rows = parse_report_text(docx_text(path), patient_id_from_name(path.name), source)
            return rows, ('' if rows else 'no text / no sections')
        if suffix == '.xlsx':
            rows = []
            for sheet_name, sheet_rows in xlsx_sheets(path):
                rows.extend(parse_sheet_rows(sheet_rows, source, sheet_name, stats))
            rows = _dedupe(rows)
            for k, r in enumerate(rows, start=1):
                r['section_index'] = k
            return rows, ('' if rows else 'no sheet with ID + conclusion columns (or all conclusions empty)')
        if suffix == '.csv':
            rows = _dedupe(parse_sheet_rows(csv_rows(path), source, '', stats))
            return rows, ('' if rows else 'no ID + conclusion columns (or all conclusions empty)')
    except Exception as e:  # unreadable zip / xml / encoding / ACL — never abort the batch
        return [], sanitize_error(e, path, source)
    return [], f'unsupported suffix {suffix}'


def collect_inputs(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    files = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
             and not p.name.startswith('.')]
    return sorted(files, key=lambda p: p.name)


# -------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Parse hospital radiology reports (PDF folder / xlsx / csv / docx) '
                                             'into one CSV row per study section.')
    ap.add_argument('--src', required=True, help='folder of reports or a single report file')
    ap.add_argument('--out', default='data/hospital_reports_labels.csv')
    ap.add_argument('--limit', type=int, default=0, help='parse only the first N files (0 = all)')
    ap.add_argument('--workers', type=int, default=4, help='parallel pdftotext processes')
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        print(f'{R}✗ --src not found: {display_path(src)}{NC}')
        return 1
    files = collect_inputs(src)
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f'{R}✗ no {"/".join(SUPPORTED_SUFFIXES)} files under {display_path(src)}{NC}')
        return 1
    print(f'Parsing {len(files)} report file(s) from {display_path(src)} …')

    stats: Counter = Counter()
    results: list[tuple[list[dict], str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for i, res in enumerate(ex.map(lambda p: parse_file(p, stats), files), start=1):
            results.append(res)
            if i % 500 == 0:
                print(f'  …{i}/{len(files)}')

    rows: list[dict] = []
    unreadable: list[tuple[str, str]] = []
    sections_hist: Counter = Counter()
    for f, (frows, err) in zip(files, results):
        if err:
            unreadable.append((hashed_source_name(f.name), err))
            continue
        if f.suffix.lower() in ('.pdf', '.docx'):
            sections_hist[len(frows)] += 1
        rows.extend(frows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    mod_counter = Counter(r['modality'] for r in rows)
    bp_counter = Counter(r['body_part'] for r in rows)
    no_concl = sum(1 for r in rows if not r['conclusion'])
    no_date = sum(1 for r in rows if not r['study_date'])
    no_pid = sum(1 for r in rows if not r['patient_id'])

    print('\n' + '=' * 70)
    print(f'  {B}PARSED {len(rows)} STUDY SECTIONS from {len(files) - len(unreadable)} files  '
          f'({len(unreadable)} unreadable){NC}')
    print('=' * 70)
    print(f'  Modality: ' + ', '.join(f'{k}={c}' for k, c in sorted(mod_counter.items())))
    print(f'\n  {B}Body part:{NC}')
    for k, c in bp_counter.most_common():
        mark = G if k == 'brain' else ''
        print(f'    {mark}{c:>5}  {k}{NC}')
    brain = bp_counter.get('brain', 0)
    share = (100.0 * brain / len(rows)) if rows else 0.0
    print(f'\n  {G}{B}BRAIN sections (our focus): {brain}  ({share:.1f}% of rows){NC}')
    if sections_hist:
        print(f'\n  {B}Sections per PDF/DOCX:{NC}')
        for n, c in sorted(sections_hist.items()):
            print(f'    {c:>5} files × {n} section(s)')
    if stats.get('sheet_rows_without_conclusion') or stats.get('sheet_rows_bad_patient_id'):
        print(f'\n  {Y}Sheet rows skipped: {stats["sheet_rows_without_conclusion"]} with an empty conclusion, '
              f'{stats["sheet_rows_bad_patient_id"]} with a non-numeric patient id{NC}')
    if no_concl or no_date or no_pid:
        print(f'\n  {Y}Rows with empty conclusion: {no_concl} · empty study_date: {no_date} · '
              f'empty patient_id: {no_pid}{NC}')
    if unreadable:
        print(f'\n  {Y}Unreadable files ({len(unreadable)}):{NC}')
        for name, err in unreadable[:25]:
            print(f'    {name}: {err}')
        if len(unreadable) > 25:
            print(f'    … and {len(unreadable) - 25} more')
    print(f'\n  ✓ Reports CSV (keyed by patient_id for scan matching): {out}')
    print(f'  ⚠ Names dropped (file name part hashed, signatures stripped); keep source files off git/cloud (PHI).')
    if not rows:
        print(f'{R}✗ nothing parsed{NC}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
