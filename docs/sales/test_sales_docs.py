"""Drift guard for docs/sales/ — the documents the vendor hands to a clinic.

Run explicitly:  python -m pytest -q docs/sales/test_sales_docs.py
(pytest.ini sets testpaths = tests, so the default `pytest -q` does not collect this file,
exactly like docs/test_docs_consistency.py.)

What it enforces
  * Every performance figure in the sales documents is the validated triage figure exposed by
    src/inference/validated_metrics.py, formatted the way the module formats it (dec / ci,
    decimal comma in RU/UZ) — and no other two-decimal figure appears anywhere ("every number
    traceable to validated_metrics"). The model identity is deliberately NOT pinned in these
    documents (it changes with every release; the clinic reads it from the vendor README).
  * The anchors that scripts/bump_model_release.py needs to rewrite these numbers after a model
    release (BUMP_ENTRIES below — to be appended to its TABLE) match exactly once, render the
    current text byte for byte, and actually move the numbers for a different release.
  * Language versions have the same structure (sections, table rows, checkboxes).
  * Claims stay inside the intended use statement: required limitations present, hype absent.
  * The pilot proposal carries the legal-review header, the protocol §7/§9 success criteria with
    the release-gate floors, the support commitment and the pricing placeholders.
  * The onboarding checklist covers the contract's delivery path (fingerprint → licence with
    model key → encrypted bundle import → verification) and a 60-minute training agenda.
  * docs/README.md links every sales document and the links resolve.

Pure text checks: validated_metrics and bump_model_release are loaded by path (both torch-free);
no server, no model, no network. No PHI: the documents are templates with placeholders only.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

SALES: Path = Path(__file__).resolve().parent
DOCS: Path = SALES.parent
REPO: Path = DOCS.parent

ONE_PAGER_LANGS = ("ru", "uz", "en")
TWO_LANGS = ("ru", "en")
FILES: dict[str, tuple[str, ...]] = {
    "one_pager": ONE_PAGER_LANGS,
    "pilot_proposal": TWO_LANGS,
    "clinic_onboarding_checklist": TWO_LANGS,
}
ALL_DOCS: list[str] = [f"{stem}_{lang}.md" for stem, langs in FILES.items() for lang in langs]

DRAFT_HEADER_EN = "**Draft for legal review — not a legal document until reviewed by counsel**"
DRAFT_HEADER_RU = "**Проект для юридической проверки — не является юридическим документом до проверки юристом**"


# --------------------------------------------------------------------------- helpers

def _load_by_path(name: str, rel: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    assert spec and spec.loader, rel
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses resolve `from __future__` annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def vm() -> ModuleType:
    """src/inference/validated_metrics.py — the single source of truth for the numbers."""
    return _load_by_path("_sales_validated_metrics", "src/inference/validated_metrics.py")


@pytest.fixture(scope="module")
def bump() -> ModuleType:
    """scripts/bump_model_release.py (imports only validated_metrics + src.utils.offline)."""
    return _load_by_path("_sales_bump_model_release", "scripts/bump_model_release.py")


def _text(name: str) -> str:
    return (SALES / name).read_text(encoding="utf-8")


def _lang_of(name: str) -> str:
    return name.rsplit("_", 1)[1].removesuffix(".md")


# --------------------------------------------------------------------------- bump anchors
# (file relative to repo root, regex that must match exactly once, str.format template using
#  bump_model_release.build_context keys). Same wildcard fragments as the bump script.
D = r"\d[.,]\d{2}"
I = r"\d+"

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

BUMP_ENTRIES: list[tuple[str, str, str]] = [
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


def _rewrite(text: str, entries: list[tuple[str, str]], ctx: dict[str, str]) -> str:
    for pattern, template in entries:
        matches = list(re.finditer(pattern, text, re.M))
        assert len(matches) == 1, f"pattern matched {len(matches)}× (need exactly 1): {pattern}"
        m = matches[0]
        text = text[:m.start()] + template.format(**ctx) + text[m.end():]
    return text


def _entries_by_file() -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {}
    for rel, pat, tpl in BUMP_ENTRIES:
        out.setdefault(rel, []).append((pat, tpl))
    return out


# --------------------------------------------------------------------------- files / index

@pytest.mark.parametrize("name", ALL_DOCS)
def test_sales_document_exists_and_is_substantial(name):
    p = SALES / name
    assert p.is_file(), p
    assert p.stat().st_size > 2500, f"{name} is suspiciously short"


def test_readme_links_every_sales_document():
    readme = (DOCS / "README.md").read_text(encoding="utf-8")
    assert "## Sales & onboarding" in readme
    links = re.findall(r"\[[^\]]+\]\((sales/[^)]+\.md)\)", readme)
    for name in ALL_DOCS:
        assert f"sales/{name}" in links, f"docs/README.md does not link sales/{name}"
    missing = [t for t in links if not (DOCS / t).is_file()]
    assert not missing, missing


def _referenced_paths() -> list[str]:
    """Repo-relative paths of every markdown file quoted in backticks: `docs/…md` as written,
    a bare `name_xx.md` relative to docs/sales/ (the sibling sales documents)."""
    refs: set[str] = set()
    for name in ALL_DOCS:
        for q in re.findall(r"`([\w./-]+\.md)`", _text(name)):
            refs.add(q if q.startswith("docs/") else f"docs/sales/{q}")
    return sorted(refs)


_REFERENCED = _referenced_paths()
# Created in the same release round by another work package (finish-line contract, "Docs").
_PENDING_IN_THIS_ROUND = {"docs/vendor_provisioning_en.md", "docs/vendor_provisioning_ru.md"}


@pytest.mark.parametrize("rel", _REFERENCED)
def test_documents_referenced_by_path_exist(rel):
    if rel in _PENDING_IN_THIS_ROUND and not (REPO / rel).exists():
        pytest.skip(f"{rel} is written by the ci-docs work package of this round")
    assert (REPO / rel).is_file(), rel


# --------------------------------------------------------------------------- structure parity

@pytest.mark.parametrize("stem", sorted(FILES))
def test_structure_matches_across_languages(stem):
    shapes = {}
    for lang in FILES[stem]:
        t = _text(f"{stem}_{lang}.md")
        shapes[lang] = (
            len(re.findall(r"^## ", t, re.M)),
            len(re.findall(r"^### ", t, re.M)),
            len(re.findall(r"^\|", t, re.M)),
            len(re.findall(r"^- \[ \]", t, re.M)),
            len(re.findall(r"^\d+\. ", t, re.M)),
        )
    assert len(set(shapes.values())) == 1, f"{stem}: structure differs between languages: {shapes}"


@pytest.mark.parametrize("lang", ONE_PAGER_LANGS)
def test_one_pager_fits_one_printed_page(lang):
    t = _text(f"one_pager_{lang}.md")
    words = len(re.findall(r"\S+", t))
    assert words <= 650, f"one_pager_{lang}.md has {words} words — no longer one printed page"
    assert len(re.findall(r"^## ", t, re.M)) == 6


# --------------------------------------------------------------------------- numbers

@pytest.mark.parametrize("name", [n for n in ALL_DOCS if not n.startswith("clinic_onboarding")])
def test_validated_figures_quoted_exactly(vm, name):
    lang = _lang_of(name)
    m = vm.METRICS
    t = _text(name)
    for tok in (vm.dec(m.sens, lang), vm.dec(m.spec, lang), vm.ci(m.spec_ci, lang),
                str(m.n_studies), str(m.tp), str(m.n_abnormal), str(m.tn), str(m.n_normal)):
        assert tok in t, (name, tok)
    if name.startswith("pilot_proposal"):
        for tok in (vm.dec(m.min_sens, lang), vm.dec(m.min_spec, lang)):
            assert tok in t, (name, tok)


@pytest.mark.parametrize("name", ALL_DOCS)
def test_every_two_decimal_figure_is_traceable_to_validated_metrics(vm, name):
    """No figure that validated_metrics does not expose (no hand-typed or rounded-up number)."""
    lang = _lang_of(name)
    m = vm.METRICS
    allowed = {vm.dec(x, lang) for x in (m.sens, m.spec, m.acc, m.min_sens, m.min_spec, m.threshold,
                                         *m.sens_ci, *m.spec_ci)}
    found = set(re.findall(r"(?<![\d.,])\d[.,]\d{2}(?![\d])", _text(name)))
    assert found <= allowed, f"{name}: figures not from validated_metrics: {sorted(found - allowed)}"


@pytest.mark.parametrize("name", ALL_DOCS)
def test_model_identity_is_not_pinned(vm, name):
    """The identity changes with every release; the clinic compares against the vendor README."""
    t = _text(name)
    assert vm.IDENTITY not in t
    assert not re.search(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{12}\b", t), name


@pytest.mark.parametrize("lang", ONE_PAGER_LANGS)
def test_plain_words_follow_from_the_numbers(vm, bump, lang):
    ctx = bump.build_context(vm.METRICS)
    t = _text(f"one_pager_{lang}.md")
    expect = {
        "en": [f"about 1 in {ctx['miss']} abnormal studies is not flagged", ctx["normals_flagged_en"]],
        "ru": [f"примерно 1 из {ctx['miss']} {ctx['miss_abn_ru']} не отмечается", ctx["normals_flagged_ru"]],
        "uz": [f"har {ctx['miss']} ta patologik tekshiruvdan taxminan 1 tasi belgilanmaydi", ctx["normals_flagged_uz"]],
    }[lang]
    for tok in expect:
        assert tok in t, (lang, tok)


# --------------------------------------------------------------------------- bump anchors

def test_bump_entries_cover_every_sales_document_with_figures():
    covered = {rel for rel, _, _ in BUMP_ENTRIES}
    for name in ALL_DOCS:
        if not name.startswith("clinic_onboarding"):
            assert f"docs/sales/{name}" in covered, name


@pytest.mark.parametrize("rel", sorted(_entries_by_file()))
def test_bump_anchors_match_once_and_are_in_sync(vm, bump, rel):
    """Rendering the anchors with the current release reproduces the file byte for byte."""
    ctx = bump.build_context(vm.METRICS)
    old = (REPO / rel).read_text(encoding="utf-8")
    assert _rewrite(old, _entries_by_file()[rel], ctx) == old, f"{rel} is out of sync with validated_metrics"


@pytest.mark.parametrize("rel", sorted(_entries_by_file()))
def test_bump_anchors_move_the_numbers_for_another_release(vm, bump, rel):
    """A hypothetical next release (more normals, higher specificity) rewrites every figure."""
    m = vm.METRICS
    new = replace(m, tp=410, fn=40, tn=338, fp=142, n_studies=930, sens=410 / 450, spec=338 / 480,
                  sens_ci=vm.wilson_ci(410, 450), spec_ci=vm.wilson_ci(338, 480),
                  min_sens=0.86, min_spec=0.45)
    ctx = bump.build_context(new)
    old = (REPO / rel).read_text(encoding="utf-8")
    out = _rewrite(old, _entries_by_file()[rel], ctx)
    lang = _lang_of(rel)
    fmt = "ru" if lang in ("ru", "uz") else "en"
    assert vm.dec(new.spec, fmt) in out and str(new.n_normal) in out and str(new.n_studies) in out, rel
    assert vm.ci(new.spec_ci, fmt) in out, rel
    assert vm.dec(m.spec, fmt) not in out, f"{rel}: old specificity survived the rewrite"
    if "pilot_proposal" in rel:
        assert vm.dec(new.min_spec, fmt) in out and vm.dec(m.min_spec, fmt) not in out, rel
    # idempotent: a second rewrite with the same release changes nothing
    assert _rewrite(out, _entries_by_file()[rel], ctx) == out


# --------------------------------------------------------------------------- claims

REQUIRED_ONE_PAGER = {
    "en": ["reads and signs every study", "not been certified as a medical device", "never certifies a study as normal",
           "switched off by default", "not analysed by AI", "does not make a diagnosis", "without internet",
           "PACS", "SHA-256", "60-minute", "One afternoon"],
    "ru": ["описывает и подписывает врач-рентгенолог", "не сертифицирована как медицинское изделие",
           "Никогда не признаёт исследование нормой", "по умолчанию выключен", "не проанализировано ИИ",
           "Не ставит диагноз", "без подключения к интернету", "PACS", "SHA-256", "60-минутное", "Полдня"],
    "uz": ["rentgenolog shifokor o‘qiydi va imzolaydi", "tibbiy buyum sifatida sertifikatlanmagan",
           "norma deb tasdiqlamaydi", "sukut bo‘yicha o‘chirilgan", "AI tomonidan tahlil qilinmagan",
           "Tashxis qo‘ymaydi", "internetsiz", "PACS", "SHA-256", "60 daqiqalik", "yarim kuni"],
}

FORBIDDEN = [r"guarantee", r"100\s?%", r"state[- ]of[- ]the[- ]art", r"revolution", r"replaces? the radiologist",
             r"detects tumou?rs", r"гарантир", r"революц", r"заменяет врача", r"kafolat", r"inqilob",
             r"best[- ]in", r"лучш", r"\bAI diagnos"]


@pytest.mark.parametrize("lang", ONE_PAGER_LANGS)
def test_one_pager_states_the_limits(lang):
    t = _text(f"one_pager_{lang}.md")
    for tok in REQUIRED_ONE_PAGER[lang]:
        assert tok in t, (lang, tok)


@pytest.mark.parametrize("name", ALL_DOCS)
def test_no_hype_and_no_claim_beyond_intended_use(name):
    t = _text(name)
    hits = [p for p in FORBIDDEN if re.search(p, t, re.I)]
    assert not hits, f"{name}: {hits}"


@pytest.mark.parametrize("name", ALL_DOCS)
def test_no_phi_like_content(name):
    """Templates only: no long numeric identifiers, no concrete dates, no e-mail addresses."""
    t = _text(name)
    assert not re.search(r"(?<![\w-])\d{7,}(?![\w-])", t), name
    assert not re.search(r"\b(19|20)\d\d-\d\d-\d\d\b", t), name
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", t), name


# --------------------------------------------------------------------------- pilot proposal

def test_pilot_proposal_en_header_is_the_legal_review_line():
    first = _text("pilot_proposal_en.md").splitlines()[0]
    assert first == DRAFT_HEADER_EN


def test_pilot_proposal_ru_header_is_the_legal_review_line_in_both_languages():
    lines = _text("pilot_proposal_ru.md").splitlines()
    assert lines[0] == DRAFT_HEADER_RU
    assert DRAFT_HEADER_EN.strip("*") in lines[1]


PROPOSAL_REQUIRED = {
    "en": ["§7", "§9", "Wilson score method", "**lower bound**", "At least 100 adjudicated studies",
           "at least 40 adjudicated as normal", "No safety event attributable to AI output", "at least 95%",
           "**within 1 business day**", "release gate", "consecutively", "[N] weeks",
           "Pilot fee", "Per-workstation licence", "Support (after the pilot)",
           "stay on the Clinic's premises", "only with the Clinic's written consent", "ZRU-547",
           "**localisation requirement**", "belongs to the Vendor", "keeps all rights to its data",
           "Intellectual property and confidentiality", "Duration and termination", "Signatures",
           "does not change clinical care", "Switched off during the pilot"],
    "ru": ["§ 7", "§ 9", "метод Уилсона", "**нижняя граница**", "Не менее 100 размеченных исследований",
           "не менее 40 отнесённых к норме", "Ни одного события безопасности", "не менее 95 %",
           "**в течение 1 рабочего дня**", "порога выпуска", "**подряд**", "[N] недель",
           "Плата за пилот (pilot fee)", "per-workstation licence", "Поддержка (support",
           "остаются на территории Клиники", "только с письменного согласия Клиники", "ЗРУ-547",
           "**требование о локализации**", "принадлежит Поставщику", "сохраняются за Клиникой",
           "Интеллектуальная собственность и конфиденциальность", "Срок действия и прекращение", "Подписи",
           "не меняет медицинскую помощь", "Выключено на время пилота"],
}


@pytest.mark.parametrize("lang", TWO_LANGS)
def test_pilot_proposal_contains_contracted_terms(lang):
    t = _text(f"pilot_proposal_{lang}.md")
    for tok in PROPOSAL_REQUIRED[lang]:
        assert tok in t, (lang, tok)


@pytest.mark.parametrize("lang", TWO_LANGS)
def test_pilot_proposal_success_criteria_are_the_six_of_protocol_section_9(lang):
    t = _text(f"pilot_proposal_{lang}.md")
    sec = t[t.index("## 4."):t.index("## 5.")]
    assert re.findall(r"^(\d)\. ", sec, re.M) == ["1", "2", "3", "4", "5", "6"]


def test_pilot_proposal_floors_equal_the_protocol_floors(vm):
    """The proposal's criterion 2 must be the protocol's criterion 2 (same floors, same wording)."""
    proto = (DOCS / "pilot_evaluation_protocol_en.md").read_text(encoding="utf-8")
    prop = _text("pilot_proposal_en.md")
    phrase = (f"Triage sensitivity point estimate ≥ {vm.dec(vm.MIN_SENS)} **and** specificity ≥ "
              f"{vm.dec(vm.MIN_SPEC)}")
    assert phrase in proto and phrase in prop


# --------------------------------------------------------------------------- onboarding checklist

CHECKLIST_SECTIONS = {
    "en": ["Signed agreement", "IT prerequisites", "Installation", "Machine fingerprint",
           "Licence and encrypted model delivery", "Verification", "Radiologist training session (60 minutes)",
           "Pilot start", "Weekly check-ins", "Pilot report"],
    "ru": ["Подписанное соглашение", "ИТ-требования", "Установка", "Отпечаток машины",
           "Лицензия и поставка зашифрованной модели", "Проверка", "Обучение рентгенологов (60 минут)",
           "Начало пилота", "Еженедельные встречи", "Отчёт о пилоте"],
}


@pytest.mark.parametrize("lang", TWO_LANGS)
def test_checklist_follows_the_onboarding_path_in_order(lang):
    heads = re.findall(r"^## \d+\. (.+)$", _text(f"clinic_onboarding_checklist_{lang}.md"), re.M)
    assert heads == CHECKLIST_SECTIONS[lang]


@pytest.mark.parametrize("lang", TWO_LANGS)
def test_checklist_names_the_contracted_delivery_mechanics(lang):
    t = _text(f"clinic_onboarding_checklist_{lang}.md")
    for tok in ("-NO-VALIDATED-MODEL", "sentinel-backend.exe --print-fingerprint", "provision_clinic.py",
                "license.dat", "brain_triage_finetuned_<identity>_<fp8>.zip", "README_admin_ru.txt",
                "README_admin_en.txt", "license_model_key_missing", "models.brain_triage.identity",
                "bundle_not_for_this_machine", "/health", "ai_settings.show_pending_detectors", "license.mode",
                "/audit/verify", "AI DRAFT — NOT SIGNED", "1.1.0", f"vendor_provisioning_{lang}.md"):
        assert tok in t, (lang, tok)


@pytest.mark.parametrize("lang", TWO_LANGS)
def test_checklist_ui_labels_exist_in_the_app_dictionary(lang):
    """Labels the checklist tells IT to click must be the app's own words (desktop-app/src/i18n)."""
    i18n = REPO / "desktop-app" / "src" / "i18n" / f"{lang}.ts"
    if not i18n.is_file():
        pytest.skip("desktop-app sources not in this checkout")
    d = i18n.read_text(encoding="utf-8")
    t = _text(f"clinic_onboarding_checklist_{lang}.md")
    for key in ("settings.importBundle", "about.exportDiagnostics", "license.import", "settings.ai",
                "settings.license", "settings.about"):
        m = re.search(r"'" + re.escape(key) + r"':\s*'([^']+)'", d)
        assert m, f"{key} missing from {i18n.name}"
        assert m.group(1) in t, (lang, key, m.group(1))


@pytest.mark.parametrize("lang", TWO_LANGS)
def test_training_agenda_covers_exactly_sixty_minutes(lang):
    t = _text(f"clinic_onboarding_checklist_{lang}.md")
    sec = t[t.index("## 7."):t.index("## 8.")]
    spans = [(int(a), int(b)) for a, b in re.findall(r"^\| (\d+)–(\d+) \|", sec, re.M)]
    assert spans and spans[0][0] == 0 and spans[-1][1] == 60, spans
    assert all(prev[1] == nxt[0] for prev, nxt in zip(spans, spans[1:])), spans


def test_pc_specification_matches_the_installation_guide():
    guide = (DOCS / "admin_install_guide_en.md").read_text(encoding="utf-8")
    spec = guide[guide.index("## 2."):guide.index("## 3.")]
    for tok in ("Windows 10/11 64-bit", "8 GB", "16 GB", "20 GB", "50 GB", "1600×1000", "8 cores"):
        assert tok in spec, tok
    for name in ("one_pager_en.md", "clinic_onboarding_checklist_en.md"):
        t = _text(name)
        for tok in ("Windows 10/11 64-bit", "8 GB", "16 GB", "20 GB", "50 GB", "1600×1000"):
            assert tok in t, (name, tok)


def test_bump_script_carries_the_sales_entries(bump):
    """A model release (scripts/bump_model_release.py) rewrites the sales figures too:
    every anchor above is in its TABLE, verbatim."""
    missing = [e for e in BUMP_ENTRIES if e not in bump.TABLE]
    assert not missing, f"{len(missing)} sales anchors missing from bump_model_release.TABLE"
    assert {rel for rel, _, _ in BUMP_ENTRIES} <= set(bump.FILES)
