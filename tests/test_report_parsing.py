"""Release-pipeline stage 1 — scripts/parse_hospital_reports.py.

Synthetic fixtures only (patient ids >= 9000000, every date in 2031, invented
names; tests/test_phi_guard.py fails the suite if a real id ever lands here):
the multi-section Russian report text mimics the Shox
International Hospital PDF layout (letterhead that says "Результаты МРТ / МСКТ",
a lone modality line + "<title> (<date>)" header per study, ЗАКЛЮЧЕНИЕ, then the
boilerplate and signature lines).  pdftotext is monkeypatched away, the xlsx and
docx inputs are built with zipfile in tmp_path.  The one test over the real PDF
folder skips when the folder (or pdftotext) is absent and writes only to tmp_path.
"""

import csv
import importlib.util
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "parse_hospital_reports.py"
REAL_PDF_DIR = Path("/Users/shakhzodbtr/Desktop/Siaa_ai/mrt_mskt_hospital_2026-04-29_2026-07-27 2")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("parse_hospital_reports", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


FAKE_PATIENT = "Тестов Тест Тестович"
FAKE_DOCTOR = "Докторов Доктор Докторович"
FAKE_SIGNER = "Лаборантов Лаборант"

MULTI_SECTION_TEXT = f"""
                                                            Республика Узбекистан
                                              Проверка подлинности
                                                  результатов
        Ф.И.О.                           {FAKE_PATIENT}                                Результаты МРТ
  Дата рождения                                             01.01.2031                                     / МСКТ
   Дата выдачи                       27.07.2031(SYNTHETIC CLINIC)                             ID: 9000001


                                                    МСКТ
                                  МСКТ грудной клетки (2031-07-13)

                                                 МСКТ-ИССЛЕДОВАНИЯ
                                    ПРОТОКОЛ ОПИСАНИЯ МСКТ ОРГАНОВ ГРУДНОЙ КЛЕТКИ
Легочные поля прозрачные. Очаговых и инфильтративных изменений не выявлено.

ЗАКЛЮЧЕНИЕ: МСКТ - признаки очаговых и инфильтративных изменений обеих легких не выявлено.


Заключение не является диагнозом и требует дальнейшей консультации с врачом, направившим на обследование.

Результат подготовил(а): {FAKE_SIGNER}

Врач: ___________________ {FAKE_DOCTOR}
\f
                                                       МРТ
                                           МРТ головного мозга (03.08.2031)

                                            ПРОТОКОЛ МРТ ИССЛЕДОВАНИЯ ГОЛОВНОГО МОЗГА
Срединные структуры не смещены. Желудочки мозга не расширены.

Заключение:
МРТ-признаки церебральной микроангиопатии (Fazekas Grade II).
Умеренная атрофия лобно-височных областей с обеих сторон.
Рекомендовано: консультация невролога.

Заключение не является диагнозом и требует дальнейшей консультации с врачом, направившим на обследование.

Результат подготовил(а): {FAKE_SIGNER}
Врач: ___________________ {FAKE_DOCTOR}

                                                        МСКТ
                                     МСКТ ППН (придаточных пазух носа) (2031-07-21)

ПРОТОКОЛ МСКТ ИССЛЕДОВАНИЯ ПРИДАТОЧНЫХ ПАЗУХ НОСА
Носовая перегородка искривлена.

ЗАКЛЮЧЕНИЕ
КТ признаки левостороннего гайморита. Искривление носовой перегородки.
Результат подготовил(а): {FAKE_SIGNER}
Врач: {FAKE_DOCTOR}

  Уважаемые пациенты!
  Результаты исследований и снимки хранятся в архиве клиники в течение 1 месяца, после чего утилизируются.
"""

SINGLE_MR_TEXT = f"""
        Ф.И.О.                           {FAKE_PATIENT}                                Результаты МРТ
  Дата рождения                                             01.01.2031                                     / МСКТ
   Дата выдачи                       27.07.2031(SYNTHETIC CLINIC)                             ID: 9000003

                                                    МРТ
                                  МРТ гипофиза (2031-06-30)

ПРОТОКОЛ МРТ ИССЛЕДОВАНИЯ ГИПОФИЗА
Гипофиз обычной формы.

ЗАКЛЮЧЕНИЕ: МРТ-признаков патологии гипофиза не выявлено. Результат подготовил(а): {FAKE_SIGNER}
Врач: {FAKE_DOCTOR}
"""


def _patch_pdf(monkeypatch, mod, text: str):
    monkeypatch.setattr(mod, "pdf_text", lambda path: text)


def _no_names(rows) -> None:
    blob = "\n".join("|".join(str(v) for v in r.values()) for r in rows)
    for tok in (FAKE_PATIENT, FAKE_DOCTOR, FAKE_SIGNER, "Тестов", "Докторов", "Лаборантов"):
        assert tok not in blob, tok
    for boiler in ("Результат подготовил", "Врач", "Уважаемые", "не является диагнозом", "____"):
        assert boiler not in blob, boiler


# ------------------------------------------------------------- PDF text

def test_multi_section_pdf_gives_one_row_per_section(monkeypatch, mod):
    _patch_pdf(monkeypatch, mod, MULTI_SECTION_TEXT)
    rows, err = mod.parse_file(Path(f"9000001_{FAKE_PATIENT}.pdf"))
    assert err == ""
    assert [r["section_index"] for r in rows] == [1, 2, 3]
    assert [r["modality"] for r in rows] == ["CT", "MR", "CT"]
    assert [r["body_part"] for r in rows] == ["chest", "brain", "sinus"]
    assert [r["study_date"] for r in rows] == ["2031-07-13", "2031-08-03", "2031-07-21"]
    assert all(r["patient_id"] == "9000001" for r in rows)
    assert all(r["label"] == "" for r in rows)
    assert rows[0]["conclusion"] == "МСКТ - признаки очаговых и инфильтративных изменений обеих легких не выявлено."
    assert rows[1]["conclusion"] == ("МРТ-признаки церебральной микроангиопатии (Fazekas Grade II). "
                                     "Умеренная атрофия лобно-височных областей с обеих сторон. "
                                     "Рекомендовано: консультация невролога.")
    assert rows[2]["conclusion"] == "КТ признаки левостороннего гайморита. Искривление носовой перегородки."
    assert "ПРОТОКОЛ ОПИСАНИЯ МСКТ ОРГАНОВ ГРУДНОЙ КЛЕТКИ" in rows[0]["description"]
    assert "ЗАКЛЮЧЕНИЕ" not in rows[0]["description"]
    assert rows[1]["section_header"] == "МРТ головного мозга"
    assert set(mod.OUTPUT_COLUMNS) == set(rows[0].keys())
    _no_names(rows)


def test_modality_comes_from_section_not_letterhead(monkeypatch, mod):
    # letterhead says "Результаты МРТ / МСКТ" — the old parser called everything МСКТ
    _patch_pdf(monkeypatch, mod, SINGLE_MR_TEXT)
    rows, _ = mod.parse_file(Path(f"9000003_{FAKE_PATIENT}.pdf"))
    assert len(rows) == 1
    assert rows[0]["modality"] == "MR"
    assert rows[0]["body_part"] == "brain"
    assert rows[0]["study_date"] == "2031-06-30"
    # inline signature on the conclusion line is cut off
    assert rows[0]["conclusion"] == "МРТ-признаков патологии гипофиза не выявлено."
    _no_names(rows)


def test_source_file_hashes_the_name_part(mod):
    src = mod.hashed_source_name(f"9000001_{FAKE_PATIENT}.pdf")
    assert re.fullmatch(r"9000001_[0-9a-f]{16}\.pdf", src)
    assert FAKE_PATIENT not in src
    assert mod.hashed_source_name("9000001.pdf") == "9000001.pdf"
    assert re.fullmatch(r"[0-9a-f]{16}\.xlsx", mod.hashed_source_name("Иванов report.xlsx"))
    assert mod.hashed_source_name("requests_2031.xlsx") != "requests_2031.xlsx"
    assert mod.patient_id_from_name(f"9000001_{FAKE_PATIENT}.pdf") == "9000001"
    assert mod.patient_id_from_name("nodigits.pdf") == ""


def test_patient_id_falls_back_to_letterhead_id(monkeypatch, mod):
    _patch_pdf(monkeypatch, mod, SINGLE_MR_TEXT)
    rows, _ = mod.parse_file(Path("scan.pdf"))
    assert rows[0]["patient_id"] == "9000003"


def test_unreadable_pdf_is_reported_not_raised(monkeypatch, mod):
    _patch_pdf(monkeypatch, mod, "")
    rows, err = mod.parse_file(Path("1_x.pdf"))
    assert rows == [] and err


def test_date_normalisation(mod):
    assert mod.normalize_date("21.07.2031") == "2031-07-21"
    assert mod.normalize_date("2031-07-21") == "2031-07-21"
    assert mod.normalize_date("20310721") == "2031-07-21"
    assert mod.normalize_date(" 03.08.2031 ") == "2031-08-03"
    assert mod.normalize_date("48050") == "2031-07-21"  # Excel serial
    assert mod.normalize_date("") == ""
    assert mod.normalize_date(None) == ""
    assert mod.normalize_date("не указана") == ""
    assert mod.normalize_date("31.02.2031") == ""


def test_modality_canonicalisation(mod):
    assert mod.canon_modality("МРТ") == "MR"
    assert mod.canon_modality("  МСКТ ") == "CT"
    assert mod.canon_modality("КТ") == "CT"
    assert mod.canon_modality("MR") == "MR"
    assert mod.canon_modality("CT") == "CT"
    assert mod.canon_modality("МСКТ головного мозга") == "CT"
    assert mod.canon_modality("Ангиография нижних конечностей") == "?"
    assert mod.canon_modality("") == "?"


def test_body_part_canonicalisation(mod):
    for s in ("МРТ головного мозга", "МСКТ головы", "МРТ гипофиза", "головной мозг",
              "МРТ головного мозга и сосудов головного мозга (ангиорежим)", "BRAIN", "HEAD HEAD"):
        assert mod.canon_body_part(s) == "brain", s
    assert mod.canon_body_part("МСКТ грудной клетки") == "chest"          # not spine
    assert mod.canon_body_part("МРТ грудного отдела позвоночника") == "spine"
    assert mod.canon_body_part("МРТ пояснично-крестцового отдела позвоночника") == "spine"
    assert mod.canon_body_part("МСКТ ППН (придаточных пазух носа)") == "sinus"
    assert mod.canon_body_part("МСКТ орбит") == "orbit"
    assert mod.canon_body_part("МРТ тазобедренных суставов") == "hip"
    assert mod.canon_body_part("МРТ коленного сустава") == "knee"
    assert mod.canon_body_part("МСКТ органов брюшной полости") == "abdomen"
    assert mod.canon_body_part("CHEST") == "chest"
    assert mod.canon_body_part("LSPINE") == "spine"
    assert mod.canon_body_part("") == "other"
    assert mod.canon_body_part("МРТ контраст 20мг") == "other"
    assert mod.canon_body_part("ship") == "other"  # Latin keys match whole words only


def test_boilerplate_stripping(mod):
    t = ("КТ-признаков патологии не выявлено.   Рекомендовано:\n консультация невролога.  "
         f"Результат подготовил(а): {FAKE_SIGNER} Врач: ____ {FAKE_DOCTOR}")
    assert mod.strip_boilerplate(t) == "КТ-признаков патологии не выявлено. Рекомендовано: консультация невролога."
    assert mod.strip_boilerplate("Норма. Заключение не является диагнозом и требует") == "Норма."
    assert mod.strip_boilerplate(": . Норма. ____") == "Норма."
    assert mod.strip_boilerplate("Уважаемые пациенты! ...") == ""


def test_conclusion_marker_variants(mod):
    for marker in ("ЗАКЛЮЧЕНИЕ:", "ЗАКЛЮЧЕНИЕ", "Заключение :", "Заключение:", "ЗАКЛЮЧЕНИЕ.", "ВЫВОД:", "Выводы:"):
        lines = ["ПРОТОКОЛ", "Описание.", marker, "Первая строка.", "Вторая строка.",
                 "Заключение не является диагнозом", f"Врач: {FAKE_DOCTOR}"]
        desc, concl = mod.extract_conclusion(lines)
        assert concl == "Первая строка. Вторая строка.", marker
        assert desc == "ПРОТОКОЛ Описание."
    # no marker at all -> everything is description, conclusion empty
    desc, concl = mod.extract_conclusion(["ПРОТОКОЛ", "Описание.", f"Врач: {FAKE_DOCTOR}"])
    assert (desc, concl) == ("ПРОТОКОЛ Описание.", "")


def test_no_headers_falls_back_to_single_row(monkeypatch, mod):
    text = ("Описание исследования без шапки.\n"
            "ЗАКЛЮЧЕНИЕ: МРТ-признаки атрофии головного мозга.\n"
            f"Результат подготовил(а): {FAKE_SIGNER}\n")
    _patch_pdf(monkeypatch, mod, text)
    rows, err = mod.parse_file(Path("9000005_someone.pdf"))
    assert err == "" and len(rows) == 1
    assert rows[0]["modality"] == "MR" and rows[0]["body_part"] == "brain"
    assert rows[0]["conclusion"] == "МРТ-признаки атрофии головного мозга."
    assert rows[0]["patient_id"] == "9000005" and rows[0]["study_date"] == ""


# ------------------------------------------------------------- xlsx / csv

XLSX_HEADER = ["ID пациента (PatientID)", "Дата исследования", "Время", "Модальность", "Область",
               "Описание исследования", "Accession №", "Пол", "Возраст", "Приоритет",
               "Заключение врача (вставить сюда)"]


def _xlsx_cell(ref: str, value: str, shared: list[str] | None) -> str:
    if value == "":
        return f'<c r="{ref}"/>'
    if shared is not None:  # shared-string cell
        if value not in shared:
            shared.append(value)
        return f'<c r="{ref}" t="s"><v>{shared.index(value)}</v></c>'
    esc = value.replace("&", "&amp;").replace("<", "&lt;")
    return f'<c r="{ref}" t="inlineStr"><is><t>{esc}</t></is></c>'


def _sheet_xml(rows: list[list[str]], shared: list[str] | None) -> str:
    out = ['<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>']
    for r, row in enumerate(rows, start=1):
        cells = "".join(_xlsx_cell(f"{chr(65 + c)}{r}", v, shared) for c, v in enumerate(row))
        out.append(f'<row r="{r}">{cells}</row>')
    out.append("</sheetData></worksheet>")
    return "".join(out)


def build_xlsx(path: Path, sheets: dict[str, list[list[str]]], use_shared: bool) -> None:
    shared: list[str] | None = [] if use_shared else None
    names = list(sheets)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml",
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        z.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>')
        z.writestr("xl/workbook.xml",
                   '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                   'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
                   + "".join(f'<sheet name="{n}" sheetId="{i + 1}" r:id="rId{i + 1}"/>' for i, n in enumerate(names))
                   + "</sheets></workbook>")
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   + "".join(f'<Relationship Id="rId{i + 1}" Type="x/worksheet" Target="worksheets/sheet{i + 1}.xml"/>'
                             for i in range(len(names)))
                   + "</Relationships>")
        sheet_xmls = [_sheet_xml(rows, shared) for rows in sheets.values()]
        for i, xml in enumerate(sheet_xmls):
            z.writestr(f"xl/worksheets/sheet{i + 1}.xml", xml)
        if shared is not None:
            z.writestr("xl/sharedStrings.xml",
                       '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                       + "".join(f"<si><t>{s}</t></si>" for s in shared) + "</sst>")


SHEET_ROWS = [
    XLSX_HEADER,
    ["9000101", "06.07.2031", "00:00:01", "MR", "BRAIN", "HEAD HEAD", "", "F", "099Y",
     "1 — МРТ головного мозга", "МРТ-признаки церебральной микроангиопатии."],
    ["9000102", "18.07.2031", "00:00:02", "CT", "HEAD", "HEAD", "", "M", "098Y", "", ""],  # unfilled
    ["9000103", "29.06.2031", "00:00:03", "CT", "CHEST", "CHEST", "", "M", "097Y", "",
     "МСКТ - признаки очаговых изменений не выявлено."],
    ["", "", "", "", "", "", "", "", "", "", ""],
]


@pytest.mark.parametrize("use_shared", [False, True], ids=["inlineStr", "sharedStrings"])
def test_xlsx_request_sheet(tmp_path, mod, use_shared):
    xlsx = tmp_path / "report_request.xlsx"
    summary = [["Сводка"], ["МРТ мозг", "2"], ["КТ голова", "1"]]
    build_xlsx(xlsx, {"0_Сводка": summary, "1_МРТ_мозг": SHEET_ROWS[:3],
                      "3_Все": SHEET_ROWS}, use_shared)
    out = tmp_path / "reports.csv"
    assert mod.main(["--src", str(xlsx), "--out", str(out)]) == 0
    with open(out, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == mod.OUTPUT_COLUMNS
    # brain row appears in two sheets -> deduplicated; the unfilled row is skipped
    assert [(r["patient_id"], r["modality"], r["body_part"], r["study_date"]) for r in rows] == [
        ("9000101", "MR", "brain", "2031-07-06"),
        ("9000103", "CT", "chest", "2031-06-29"),
    ]
    assert rows[0]["conclusion"] == "МРТ-признаки церебральной микроангиопатии."
    assert rows[0]["description"] == "HEAD HEAD"
    assert rows[0]["label"] == ""
    assert [r["section_index"] for r in rows] == ["1", "2"]
    assert all(r["source_file"].endswith(".xlsx") for r in rows)


def test_csv_request_sheet(tmp_path, mod):
    src = tmp_path / "request.csv"
    with open(src, "w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows(SHEET_ROWS)
    rows, err = mod.parse_file(src)
    assert err == ""
    assert [(r["patient_id"], r["modality"], r["body_part"]) for r in rows] == [
        ("9000101", "MR", "brain"), ("9000103", "CT", "chest")]
    assert rows[0]["study_date"] == "2031-07-06"


def test_xlsx_without_id_and_conclusion_columns_is_unreadable(tmp_path, mod):
    xlsx = tmp_path / "other.xlsx"
    # a summary sheet whose legend cell mentions both column names is NOT a data table
    legend = [["Запрос заключений"], ["Что просим:", "Заключение врача в колонку ID пациента (PatientID)"],
              ["Дата", "Все исследования"], ["27.06.2031", "1"], ["29.06.2031", "2"]]
    build_xlsx(xlsx, {"Sheet1": [["a", "b"], ["1", "2"]], "0_Сводка": legend}, use_shared=False)
    rows, err = mod.parse_file(xlsx)
    assert rows == [] and err


def test_sheet_rows_with_non_numeric_patient_id_are_skipped(mod):
    from collections import Counter
    stats = Counter()
    rows = mod.parse_sheet_rows([XLSX_HEADER,
                                 ["итого", "", "", "", "", "", "", "", "", "", "12 заключений"],
                                 ["9000101.0", "06.07.2031", "", "MR", "BRAIN", "", "", "", "", "", "Норма."]],
                                "req.xlsx", "s", stats)
    assert [r["patient_id"] for r in rows] == ["9000101"]
    assert stats["sheet_rows_bad_patient_id"] == 1


# ------------------------------------------------------------------- docx

def build_docx(path: Path, paragraphs: list[str]) -> None:
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(
        f"<w:p><w:r><w:t xml:space=\"preserve\">"
        f"{p.replace(chr(12), '').replace('&', '&amp;').replace('<', '&lt;')}</w:t></w:r></w:p>"
        for p in paragraphs)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml",
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        z.writestr("word/document.xml", f'<w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>')


def test_docx_report(tmp_path, mod):
    docx = tmp_path / f"9000004_{FAKE_PATIENT}.docx"
    build_docx(docx, MULTI_SECTION_TEXT.split("\n"))
    rows, err = mod.parse_file(docx)
    assert err == ""
    assert [r["modality"] for r in rows] == ["CT", "MR", "CT"]
    assert [r["body_part"] for r in rows] == ["chest", "brain", "sinus"]
    assert rows[1]["conclusion"].startswith("МРТ-признаки церебральной микроангиопатии")
    assert all(r["patient_id"] == "9000004" for r in rows)
    assert re.fullmatch(r"9000004_[0-9a-f]{16}\.docx", rows[0]["source_file"])
    _no_names(rows)


# ------------------------------------------------------------ docx tables

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _esc(s: str) -> str:
    return s.replace(chr(12), "").replace("&", "&amp;").replace("<", "&lt;")


def _p(text: str) -> str:
    return f'<w:p><w:r><w:t xml:space="preserve">{_esc(text)}</w:t></w:r></w:p>'


def _tc(*paragraphs: str, raw: str = "") -> str:
    return "<w:tc>" + "".join(_p(x) for x in paragraphs) + raw + "</w:tc>"


def _tbl(*rows: list[str]) -> str:
    return "<w:tbl><w:tblPr/>" + "".join("<w:tr>" + "".join(r) + "</w:tr>" for r in rows) + "</w:tbl>"


def build_docx_xml(path: Path, body: str) -> None:
    """A minimal .docx whose word/document.xml body is `body` (raw WordprocessingML)."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml",
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        z.writestr("word/document.xml",
                   f'<w:document xmlns:w="{W_NS}"><w:body>{body}<w:sectPr/></w:body></w:document>')


def test_docx_text_reads_tables_in_reading_order(tmp_path, mod):
    nested = _tbl([_tc("вложенная"), _tc("таблица")])
    body = (_p("до таблицы")
            + _tbl([_tc("Область"), _tc("Дата")],
                   [_tc("головной мозг"), _tc(), _tc("03.08.2031", "повтор")],      # empty (vMerge) cell drops out
                   [_tc("одна ячейка", "вторая строка ячейки")],                    # layout row: one line per paragraph
                   [_tc("внешняя", raw=nested), _tc("справа")])
            + "<w:sdt><w:sdtContent>" + _p("в элементе управления") + "</w:sdtContent></w:sdt>"
            + _p("после таблицы"))
    f = tmp_path / "9000301_x.docx"
    build_docx_xml(f, body)
    assert mod.docx_text(f).split("\n") == [
        "до таблицы",
        "Область | Дата",
        "головной мозг | 03.08.2031 повтор",
        "одна ячейка",
        "вторая строка ячейки",
        "внешняя вложенная | таблица | справа",
        "в элементе управления",
        "после таблицы",
    ]


def test_docx_report_in_a_two_column_table(tmp_path, mod):
    body = (_p("Результаты МРТ / МСКТ")
            + _tbl([_tc("МРТ")],
                   [_tc("МРТ головного мозга (03.08.2031)")],
                   [_tc("Протокол"), _tc("Срединные структуры не смещены.", "Желудочки не расширены.")],
                   [_tc("Заключение:"), _tc("МРТ-признаки церебральной микроангиопатии.", "Рекомендовано: невролог.")],
                   [_tc(f"Результат подготовил(а): {FAKE_SIGNER}")]))
    f = tmp_path / f"9000302_{FAKE_PATIENT}.docx"
    build_docx_xml(f, body)
    rows, err = mod.parse_file(f)
    assert err == "" and len(rows) == 1
    r = rows[0]
    assert (r["patient_id"], r["modality"], r["body_part"], r["study_date"]) == ("9000302", "MR", "brain", "2031-08-03")
    assert r["conclusion"].startswith("МРТ-признаки церебральной микроангиопатии")
    assert "|" not in r["conclusion"]
    _no_names(rows)


def test_docx_report_inside_a_layout_table_matches_the_paragraph_version(tmp_path, mod):
    plain = tmp_path / f"9000004_{FAKE_PATIENT}.docx"
    build_docx(plain, MULTI_SECTION_TEXT.split("\n"))
    boxed = tmp_path / f"9000004_{FAKE_PATIENT}_t.docx"
    build_docx_xml(boxed, _tbl([_tc(*[x for x in MULTI_SECTION_TEXT.split("\n") if x.strip()])]))
    want, _ = mod.parse_file(plain)
    got, err = mod.parse_file(boxed)
    assert err == ""
    keys = ("modality", "body_part", "study_date", "conclusion")
    assert [[r[k] for k in keys] for r in got] == [[r[k] for k in keys] for r in want]
    _no_names(got)


def test_conclusion_marker_accepts_a_table_cell_separator(mod):
    assert mod.CONCLUSION_RE.match("Заключение: | Без патологии.").group("rest") == "Без патологии."
    assert mod.CONCLUSION_RE.match("ЗАКЛЮЧЕНИЕ | норма").group("rest") == "норма"
    assert mod.CONCLUSION_RE.match("ЗАКЛЮЧЕНИЕ: норма").group("rest") == "норма"


# --------------------------------------------------------------- CLI / folder

def test_cli_over_folder_writes_csv_and_summary(tmp_path, monkeypatch, mod, capsys):
    _patch_pdf(monkeypatch, mod, MULTI_SECTION_TEXT)
    folder = tmp_path / "reports"
    folder.mkdir()
    (folder / f"9000001_{FAKE_PATIENT}.pdf").write_bytes(b"%PDF-1.4 fake")
    (folder / f"9000002_{FAKE_PATIENT}.pdf").write_bytes(b"%PDF-1.4 fake")
    (folder / "notes.txt").write_text("ignored")
    out = tmp_path / "out" / "reports.csv"
    assert mod.main(["--src", str(folder), "--out", str(out), "--limit", "1", "--workers", "2"]) == 0
    with open(out, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3 and rows[0]["patient_id"] == "9000001"
    text = capsys.readouterr().out
    assert "PARSED 3 STUDY SECTIONS" in text and "1 files × 3 section(s)" in text
    assert FAKE_PATIENT not in text
    _no_names(rows)


def test_cli_fails_on_missing_src_or_no_rows(tmp_path, monkeypatch, mod):
    assert mod.main(["--src", str(tmp_path / "nope"), "--out", str(tmp_path / "o.csv")]) == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    assert mod.main(["--src", str(empty), "--out", str(tmp_path / "o.csv")]) == 1
    _patch_pdf(monkeypatch, mod, "")
    (empty / "1_x.pdf").write_bytes(b"")
    assert mod.main(["--src", str(empty), "--out", str(tmp_path / "o.csv")]) == 1


# ------------------------------------------------------------ PHI: fallback path

SYN_NAME = "Синтетов Синтет Синтетович"
SYN_SURNAME = "Синтетов"
SYN_BIRTH = "02.03.2031"


def _assert_no_phi(row) -> None:
    for field in ("description", "conclusion", "section_header"):
        v = row[field]
        for tok in (SYN_NAME, SYN_SURNAME, "Синтет ", SYN_BIRTH, "Ф.И.О", "ФИО", "Дата рождения",
                    "Дата выдачи", "ID:", "рентгенолог", "Результат подготовил"):
            assert tok not in v, (field, tok, v)


def test_fallback_docx_without_dated_header_drops_letterhead(tmp_path, mod):
    # reviewer scenario: DOCX export, study header without "(date)", the date on
    # its own line, a letterhead with Ф.И.О. + birth date above it
    paragraphs = [
        'ООО "Синтетика Клиник"', "Результаты МРТ / МСКТ",
        f"Ф.И.О. {SYN_NAME}", f"Дата рождения {SYN_BIRTH}", "Дата выдачи 20.05.2031", "ID: 9000201",
        "", "МРТ головного мозга", "14.05.2031",
        "На серии МР-томограмм, взвешенных по Т1 и Т2, срединные структуры не смещены.",
        "Желудочки мозга не расширены.",
        "ЗАКЛЮЧЕНИЕ: МРТ-признаки очагового поражения белого вещества.",
        "Рекомендовано: консультация невролога.",
        "Заключение не является диагнозом.",
        f"Результат подготовил(а): {FAKE_SIGNER}",
        f"Врач-рентгенолог: {FAKE_DOCTOR}",
    ]
    docx = tmp_path / f"9000201_{SYN_NAME}.docx"
    build_docx(docx, paragraphs)
    rows, err = mod.parse_file(docx)
    assert err == "" and len(rows) == 1
    r = rows[0]
    assert (r["patient_id"], r["modality"], r["body_part"], r["study_date"]) == (
        "9000201", "MR", "brain", "2031-05-14")
    assert r["description"].startswith("МРТ головного мозга")
    assert "срединные структуры не смещены" in r["description"]
    assert r["conclusion"] == ("МРТ-признаки очагового поражения белого вещества. "
                               "Рекомендовано: консультация невролога.")
    assert "Клиник" not in r["description"]
    _assert_no_phi(r)
    _no_names(rows)


def test_fallback_scrubber_works_without_the_letterhead_cut(mod):
    # the letterhead sits AFTER the first study line (so cut_letterhead keeps it)
    # and a DOCX table puts each label and its value on separate lines — the
    # PHI scrubber must still remove every name / birth-date fragment
    text = "\n".join([
        "МРТ головного мозга",
        "Ф.И.О.", SYN_NAME, "Дата рождения:", SYN_BIRTH, "ID", "9000202",
        f"Пациент: {SYN_NAME}",
        f"На серии МР-томограмм срединные структуры не смещены. Пациент {SYN_SURNAME} С. обследован.",
        f"ЗАКЛЮЧЕНИЕ: МРТ-признаки атрофии. Пациентка {SYN_SURNAME}а С. С., {SYN_BIRTH} г.р.",
        f"Рентгенолог {SYN_SURNAME} С.",
    ])
    rows = mod.parse_report_text(text, "9000202", "9000202_x.docx")
    assert len(rows) == 1
    r = rows[0]
    _assert_no_phi(r)
    # clinical content on the same lines survives the fragment scrub
    assert "срединные структуры не смещены" in r["description"] and "обследован" in r["description"]
    assert r["conclusion"] == "МРТ-признаки атрофии."


def test_scrub_phi_fragments(mod):
    s = mod.scrub_phi
    # a capitalised word right after a name is taken as part of it (fails safe)
    assert s(f"Ф.И.О.: {SYN_NAME} Норма.") == ""
    assert s(f"Ф.И.О.: {SYN_NAME}, без патологии.") == "без патологии."
    assert s(f"Норма. Дата рождения: {SYN_BIRTH} г.р.") == "Норма."
    assert s(f"Норма. рожд. {SYN_BIRTH}") == "Норма."
    assert s("Норма. ID: 9000203") == "Норма."
    assert s(f"Норма. Врач-рентгенолог {SYN_SURNAME} С.С.") == "Норма."
    assert s(f"Норма. Результат подготовил(а): {SYN_SURNAME} С.") == "Норма."
    # clinical words that merely resemble the patterns are kept
    assert s("Со слов пациента, головные боли. Консультация врача-невролога.") == \
        "Со слов пациента, головные боли. Консультация врача-невролога."


def test_sheet_conclusion_and_description_are_scrubbed(mod):
    rows = mod.parse_sheet_rows(
        [XLSX_HEADER,
         ["9000204", "06.07.2031", "", "MR", "BRAIN", f"Пациент: {SYN_NAME} HEAD",
          "", "", "", "", f"МРТ-признаки нормы. Врач-рентгенолог {SYN_SURNAME} С."]],
        "req.xlsx", "s")
    assert len(rows) == 1
    _assert_no_phi(rows[0])
    assert rows[0]["conclusion"] == "МРТ-признаки нормы."


# ------------------------------------------------------ PHI: error messages

def _chmod_blocks_reads(p: Path) -> bool:
    return not os.access(p, os.R_OK)


@pytest.mark.parametrize("suffix", [".docx", ".xlsx", ".csv"])
def test_unreadable_file_error_uses_hashed_name(tmp_path, mod, suffix):
    f = tmp_path / f"9000205_{SYN_NAME}{suffix}"
    f.write_bytes(b"x")
    f.chmod(0)
    try:
        if not _chmod_blocks_reads(f):
            pytest.skip("running with privileges that ignore file modes")
        rows, err = mod.parse_file(f)
    finally:
        f.chmod(0o600)
    assert rows == [] and err.startswith("PermissionError")
    assert mod.hashed_source_name(f.name) in err
    for tok in (SYN_NAME, SYN_SURNAME, "Синтет", "Синтетович"):
        assert tok not in err, err


def test_corrupt_xlsx_missing_file_and_directory_errors_carry_no_name(tmp_path, mod):
    corrupt = tmp_path / f"9000206_{SYN_NAME}.xlsx"
    corrupt.write_bytes(b"PK\x03\x04 not really a zip " + SYN_SURNAME.encode())
    missing = tmp_path / f"9000207_{SYN_NAME}.docx"
    folder = tmp_path / f"9000208_{SYN_NAME}.csv"
    folder.mkdir()
    bad_member = tmp_path / f"9000209_{SYN_NAME}.docx"
    with zipfile.ZipFile(bad_member, "w") as z:
        z.writestr(f"word/{SYN_NAME}.xml", "<x/>")   # no word/document.xml -> KeyError
    for f in (corrupt, missing, folder, bad_member):
        rows, err = mod.parse_file(f)
        assert rows == [] and err, f.suffix
        for tok in (SYN_NAME, SYN_SURNAME, "Синтет"):
            assert tok not in err, err


def test_cli_unreadable_listing_never_prints_the_name(tmp_path, monkeypatch, mod, capsys):
    _patch_pdf(monkeypatch, mod, MULTI_SECTION_TEXT)
    folder = tmp_path / "reports"
    folder.mkdir()
    (folder / f"9000001_{FAKE_PATIENT}.pdf").write_bytes(b"%PDF-1.4 fake")
    locked = folder / f"9000210_{SYN_NAME}.docx"
    locked.write_bytes(b"x")
    locked.chmod(0)
    try:
        if not _chmod_blocks_reads(locked):
            pytest.skip("running with privileges that ignore file modes")
        assert mod.main(["--src", str(folder), "--out", str(tmp_path / "o.csv")]) == 0
    finally:
        locked.chmod(0o600)
    out = capsys.readouterr().out
    assert "Unreadable files (1)" in out and mod.hashed_source_name(locked.name) in out
    for tok in (SYN_NAME, SYN_SURNAME, "Синтет", FAKE_PATIENT):
        assert tok not in out


# ------------------------------------------------------------- real data

@pytest.mark.skipif(not REAL_PDF_DIR.is_dir(), reason="real hospital PDF folder not on this machine")
def test_real_pdf_folder_sample(tmp_path, mod):
    if not any(shutil.which(p) or Path(p).exists() for p in mod.PDFTOTEXT_CANDIDATES):
        pytest.skip("pdftotext not installed")
    files = sorted(REAL_PDF_DIR.glob("*.pdf"))[:120]
    if not files:
        pytest.skip("no PDFs in the real folder")
    # tokens of the name part of every file name — none may appear in the CSV
    name_tokens: set[str] = set()
    for p in files:
        m = re.match(r"^\d+[_\-\s]+(.*)$", p.stem)
        if m:
            name_tokens.update(t for t in re.split(r"[\s_\-\.]+", m.group(1)) if len(t) >= 3)
    out = tmp_path / "real_sample.csv"
    rc = mod.main(["--src", str(REAL_PDF_DIR), "--out", str(out), "--limit", str(len(files))])
    assert rc == 0
    csv_text = out.read_text(encoding="utf-8")
    with open(out, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= len(files) * 0.9
    modalities = {r["modality"] for r in rows}
    assert {"MR", "CT"} <= modalities
    assert any(r["body_part"] == "brain" for r in rows)
    assert sum(1 for r in rows if not r["conclusion"]) <= 0.05 * len(rows)
    for r in rows:
        assert "Результат подготовил" not in r["conclusion"]
        assert "Врач:" not in r["conclusion"] and "Врач:" not in r["description"]
        assert "подготовил" not in r["description"]
        assert "не является диагнозом" not in r["conclusion"]
        assert re.fullmatch(r"\d+", r["patient_id"])
        assert re.fullmatch(r"\d+_[0-9a-f]{16}\.pdf", r["source_file"])
        assert r["study_date"] == "" or re.fullmatch(r"\d{4}-\d{2}-\d{2}", r["study_date"])
    # whole-word match: 3-letter fragments of a name may sit inside Latin terms ("Gallenii", "Stoller")
    leaked = [t for t in name_tokens
              if re.search(r"(?<![A-Za-zА-Яа-яЁё])" + re.escape(t) + r"(?![A-Za-zА-Яа-яЁё])", csv_text)]
    assert not leaked, f"{len(leaked)} file-name tokens leaked into the CSV"
    assert not re.search(r"Ф\.И\.О|SHOX INTERNATIONAL|Дата рождения", csv_text)


# ------------------------------------------------------------ review round 2

@pytest.mark.parametrize("text", [
    # fallback path: a title line that names the patient
    "МРТ головного мозга, пациент Тестов Тест Тестович, 01.01.2031 г.р.\n"
    "На серии МР-томограмм очагов нет.\nЗАКЛЮЧЕНИЕ: Патологии не выявлено.",
    "МРТ головного мозга пациента Тестова Т.Т., 2031 г.р.\n"
    "На серии МР-томограмм очагов нет.\nЗАКЛЮЧЕНИЕ: Патологии не выявлено.",
    # sectioned path: the dated header carries the name
    "МРТ\nМРТ головного мозга Тестова Т.Т. (2031-05-14)\nНа серии очагов нет.\nЗАКЛЮЧЕНИЕ: Норма.",
])
def test_section_header_is_scrubbed(mod, text):
    """section_header is a CSV column: it must be name- and birth-date-free (round 2, line 555)."""
    rows = mod.parse_report_text(text, "9000001", "9000001_x.pdf")
    assert rows
    for r in rows:
        assert r["section_header"].startswith("МРТ головного мозга"), r["section_header"]
        for field in ("section_header", "description", "conclusion"):
            for leak in ("Тестов", "Т.Т.", "01.01.2031", "2031 г.р", "г.р."):
                assert leak not in r[field], (field, leak, r[field])


def test_section_header_keeps_parenthesised_titles(mod):
    assert mod.scrub_header("МСКТ урологический комплекс (почки, мочеточники , мочевой пузырь)") == \
        "МСКТ урологический комплекс (почки, мочеточники , мочевой пузырь)"


@pytest.mark.parametrize("line", [
    "ДАТА РОЖДЕНИЯ: 01.01.2031", "ПАЦИЕНТ: ТЕСТОВ ТЕСТ ТЕСТОВИЧ", "Больной: Тестов Тест Тестович",
    "Больная: Тестова Тест", "Фамилия: Тестова", "Имя: Тест", "Отчество: Тестовна", "Ф.И.О.: Тестов Т.Т.",
    "ФИО ТЕСТОВ ТЕСТ",
])
def test_uppercase_and_other_letterhead_labels_are_dropped(mod, line):
    """Letterhead labels match in ANY case, plus Больной / Фамилия / Имя / Отчество (round 2, line 173)."""
    assert mod.drop_phi_lines([line]) == []
    assert "Тест" not in mod.scrub_phi(line) and "ТЕСТ" not in mod.scrub_phi(line)


def test_uppercase_letterhead_after_the_title_never_reaches_the_csv(mod):
    text = ("МРТ головного мозга\nДАТА РОЖДЕНИЯ:\n01.01.2031\nПАЦИЕНТ: ТЕСТОВ ТЕСТ ТЕСТОВИЧ\n"
            "На серии МР-томограмм очагов нет.\nЗАКЛЮЧЕНИЕ: Патологии не выявлено.\nФамилия: Тестова")
    rows = mod.parse_report_text(text, "9000001", "9000001_x.pdf")
    assert len(rows) == 1
    r = rows[0]
    blob = " ".join(str(v) for v in r.values())
    for leak in ("ТЕСТОВ", "Тестова", "01.01.2031"):
        assert leak not in blob, leak
    assert r["study_date"] == "", "a birth date must never become the study date"
    assert r["conclusion"] == "Патологии не выявлено."


def test_scrubber_keeps_clinical_prose(mod):
    for s in ("у пациентки СПО, после операции", "с МРТ от 17.03.2031г. ранее выявленный очаг",
              "у пациента выявлено очаговое поражение", "больной жалуется на головную боль"):
        assert mod.scrub_phi(s) == s, s
    assert mod.scrub_phi("Пациент Тестов Т.Т., 2031 г.р. очаг") == "очаг"
