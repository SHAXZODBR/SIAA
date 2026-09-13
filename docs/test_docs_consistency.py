"""Drift guard for docs/ — keeps the hospital-facing documents honest against the code.

Run explicitly:  python -m pytest -q docs/test_docs_consistency.py
(pytest.ini sets testpaths = tests, so the default `pytest -q` does not collect this file.)

Pure text checks — nothing under src/ is imported (no torch, no server start-up).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent
LANGS = ("ru", "uz", "en")
STEMS = ("user_manual", "admin_install_guide", "intended_use", "pilot_evaluation_protocol")


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _doc(stem: str, lang: str) -> str:
    return (DOCS / f"{stem}_{lang}.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- index / files

def test_readme_links_resolve():
    readme = (DOCS / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^\]]+\]\(([^)]+\.md)\)", readme)
    assert targets, "README has no markdown links"
    missing = [t for t in targets if not (DOCS / t).exists()]
    assert not missing, f"README links to missing files: {missing}"


@pytest.mark.parametrize("stem", STEMS)
def test_readme_links_all_three_languages(stem):
    readme = (DOCS / "README.md").read_text(encoding="utf-8")
    for lang in LANGS:
        assert f"{stem}_{lang}.md" in readme, f"README does not link {stem}_{lang}.md"


@pytest.mark.parametrize("stem", STEMS)
@pytest.mark.parametrize("lang", LANGS)
def test_every_language_version_exists_and_is_substantial(stem, lang):
    p = DOCS / f"{stem}_{lang}.md"
    assert p.exists(), p
    assert p.stat().st_size > 3000, f"{p.name} is suspiciously short"


@pytest.mark.parametrize("stem", STEMS)
@pytest.mark.parametrize("lang", ("ru", "uz"))
def test_reviewer_note_on_translated_versions(stem, lang):
    first = _doc(stem, lang).splitlines()[0]
    assert first.startswith("> **"), f"{stem}_{lang}.md must open with the radiologist-review note"
    assert ("рентгенолог" in first) or ("rentgenolog" in first)


@pytest.mark.parametrize("stem", STEMS)
def test_section_structure_matches_across_languages(stem):
    counts = {}
    for lang in LANGS:
        text = _doc(stem, lang)
        counts[lang] = (
            len(re.findall(r"^## ", text, re.M)),
            len(re.findall(r"^### ", text, re.M)),
            len(re.findall(r"^\|", text, re.M)),  # table rows
        )
    assert len(set(counts.values())) == 1, f"{stem}: section/table structure differs: {counts}"


# ---------------------------------------------------------------- routes

ROUTES = ["/auth/login", "/analyze/study", "/report/sign", "/studies", "/health", "/audit/verify",
          "/audit/log", "/license/status", "/auth/register", "/auth/users", "/auth/me",
          "/report/save_correction", "/study/{id}"]


@pytest.mark.parametrize("route", ROUTES)
def test_routes_quoted_in_docs_exist_in_server(route):
    server = _read("src/inference/server.py")
    auth = _read("src/inference/auth_routes.py")
    assert 'APIRouter(prefix="/auth"' in auth
    if route.startswith("/auth/"):
        assert f'"{route[len("/auth"):]}"' in auth, route
    elif route == "/study/{id}":
        assert '@app.get("/study/{study_id}")' in server
    else:
        assert re.search(r'@app\.(get|post)\("' + re.escape(route) + r'"', server), route
    for lang in LANGS:
        assert route in _doc("admin_install_guide", lang), f"{route} not mentioned in admin guide ({lang})"


# ---------------------------------------------------------------- env vars

def _code_blob() -> str:
    parts = []
    for folder in ("src", "scripts", "desktop-app/electron"):
        for p in (REPO / folder).rglob("*"):
            if p.suffix in (".py", ".js", ".sh") and "node_modules" not in p.parts:
                parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def test_env_vars_in_admin_guides_exist_in_code():
    code = _code_blob()
    for lang in LANGS:
        names = set(re.findall(r"`(SENTINEL_[A-Z_]+|DEV_BYPASS_LICENSE|ORTHANC_URL)", _doc("admin_install_guide", lang)))
        for name in names:
            if name.endswith("_"):  # SENTINEL_MODEL_DIR_OVERRIDE_<KEY>, SENTINEL_TLS_* style prefixes
                assert name in code, (lang, name)
            else:
                assert re.search(r"\b" + re.escape(name) + r"\b", code), (lang, name)


@pytest.mark.parametrize("name", ["SENTINEL_DEV_INSECURE", "SENTINEL_ADMIN_PASSWORD", "SENTINEL_DATA_DIR",
                                  "SENTINEL_HOSPITAL_BUILD", "SENTINEL_OFFLINE", "SENTINEL_MODEL_KEY_HEX"])
def test_required_env_vars_documented_in_every_language(name):
    for lang in LANGS:
        assert f"`{name}`" in _doc("admin_install_guide", lang), (lang, name)


def test_hospital_build_semantics_are_default_not_forced():
    offline = _read("src/utils/offline.py")
    assert '_DEFAULT = "1" if (HOSPITAL_BUILD or not _DEV_INSECURE) else "0"' in offline
    assert 'OFFLINE: bool = os.environ.get("SENTINEL_OFFLINE", _DEFAULT) == "1"' in offline
    for lang in LANGS:
        row = next(l for l in _doc("admin_install_guide", lang).splitlines() if l.startswith("| `SENTINEL_HOSPITAL_BUILD`"))
        assert "SENTINEL_OFFLINE=0" in row, f"{lang}: row must say an explicit SENTINEL_OFFLINE=0 still overrides"
        assert "forced on" not in row and "принудительно" not in row


def test_training_corpus_lives_in_data_dir():
    paths = _read("src/utils/paths.py")
    assert 'TRAINING_CORPUS_DIR: Path = DATA_DIR / "training_corpus"' in paths
    assert "state.data_collector = TrainingDataCollector()" in _read("src/inference/server.py")
    for lang in LANGS:
        text = _doc("admin_install_guide", lang)
        assert "`training_corpus\\`" in text, lang
        assert "data\\training_corpus" not in text, f"{lang}: stale backend-relative corpus path"


# ---------------------------------------------------------------- CLI / scripts

def test_model_crypto_cli_commands_documented():
    src = _read("src/utils/model_crypto.py")
    for cmd in ("encrypt", "decrypt", "keyinfo"):
        assert f'add_parser("{cmd}"' in src
        for lang in LANGS:
            assert f"python -m src.utils.model_crypto {cmd}" in _doc("admin_install_guide", lang), (lang, cmd)
    assert "SENTINEL_MODEL_KEY_HEX" in src


def test_offline_bundle_step_documented():
    script = REPO / "scripts/download_all_models.py"
    assert script.exists()
    assert "'--only'" in script.read_text(encoding="utf-8")
    hint = _read("src/utils/offline.py")
    assert "run scripts/download_all_models.py --only {key}" in hint
    for lang in LANGS:
        text = _doc("admin_install_guide", lang)
        assert "scripts/download_all_models.py --only <key>" in text, lang
        assert "python scripts/download_all_models.py" in text, lang


def test_license_cli_documented():
    lic = _read("src/utils/license.py")
    assert "sub.add_parser('fingerprint'" in lic and "sub.add_parser('verify'" in lic
    assert "'--license-file'" in lic
    for lang in LANGS:
        text = _doc("admin_install_guide", lang)
        assert "-m src.utils.license fingerprint" in text, lang
        assert "-m src.utils.license verify --license-file" in text, lang


# ---------------------------------------------------------------- detectors / validation numbers

def test_detector_statuses_match_brain_analysis():
    ba = _read("src/inference/brain_analysis.py")
    assert "Detector('triage', 'Study triage — normal vs abnormal (local)', 'validated'" in ba
    assert "Detector('tumor_class', 'Brain tumor', 'pending'" in ba
    assert "Detector('stroke', 'Ischemic stroke', 'experimental'" in ba
    assert "Detector('atrophy', 'Cerebral atrophy / dementia', 'experimental'" in ba
    # whole-study route never runs experimental detectors
    server = _read("src/inference/server.py")
    sync = server[server.index("def _analyze_study_sync"):server.index("@app.post(\"/analyze/study\")")]
    assert "include_experimental=False" in sync


def test_validated_numbers_match_eval_json_and_manifest():
    ev = json.loads(_read("models/brain_triage_finetuned/study_level_eval.json"))
    gate = ev["gate"]
    assert round(gate["sens"], 2) == 0.90 and round(gate["spec"], 2) == 0.47
    assert (gate["tp"], gate["fn"], gate["tn"], gate["fp"]) == (118, 13, 22, 25)
    assert (gate["min_sens"], gate["min_spec"], gate["passed"]) == (0.85, 0.4, True)
    assert ev["n_studies"] == 178 and ev["n_slices"] == 890
    assert ev["evaluated_at"].startswith("2026-09-07")
    man = json.loads(_read("models/brain_triage_finetuned/MANIFEST.json"))
    assert man["files"]["model.safetensors"].startswith("0d559766ce58")
    gate_src = _read("scripts/eval_study_level.py")
    assert "default=0.85" in gate_src and "default=0.40" in gate_src

    for lang in LANGS:
        iu = _doc("intended_use", lang)
        assert "178" in iu and "118" in iu and "131" in iu and "22" in iu and "890" in iu, lang
        assert "0d559766ce58" in iu, lang
        assert "2026-09-07" in iu, lang
        for stem in ("user_manual", "intended_use", "pilot_evaluation_protocol"):
            text = _doc(stem, lang)
            assert re.search(r"0[.,]90", text) and re.search(r"0[.,]47", text) and "178" in text, (stem, lang)
        for stem in ("intended_use", "pilot_evaluation_protocol", "admin_install_guide"):
            text = _doc(stem, lang)
            assert re.search(r"0[.,]85", text) and re.search(r"0[.,]40", text), (stem, lang)
    readme = (DOCS / "README.md").read_text(encoding="utf-8")
    assert "0.90" in readme and "0.47" in readme and "178" in readme and "0d559766ce58" in readme


def test_ui_strings_quoted_in_manual_exist_in_app_dictionaries():
    """The manuals quote on-screen messages verbatim; keep them in sync with desktop-app/src/i18n."""
    expected = {
        "en": ["No finding flagged by AI — NOT a normal read", "Template report — AI assistant unavailable",
               "A report in this language is already signed", "AI DRAFT — NOT SIGNED",
               "Session expired — please sign in again", "No DICOM (.dcm) files in the selection"],
        "uz": ["AI hech qanday topilma belgilamadi — bu norma xulosasi EMAS",
               "Shablon hisobot — AI yordamchisi mavjud emas",
               "Bu tildagi hisobot allaqachon imzolangan", "AI QORALAMASI — IMZOLANMAGAN",
               "Sessiya tugadi — qayta kiring", "Tanlangan fayllar orasida DICOM (.dcm) yo‘q"],
    }
    for lang, strings in expected.items():
        dictionary = _read(f"desktop-app/src/i18n/{lang}.ts")
        manual = _doc("user_manual", lang)
        for s in strings:
            assert s in dictionary, (lang, s)
            assert s in manual, (lang, s)


def test_health_json_example_lists_offline_field():
    assert "'offline': OFFLINE," in _read("src/inference/server.py")
    for lang in LANGS:
        assert '"offline": true' in _doc("admin_install_guide", lang), lang


def test_audit_chain_documented():
    db = _read("src/utils/database.py")
    assert "def verify_audit_chain" in db and "prev_hash" in db and "row_hash" in db
    for lang in LANGS:
        text = _doc("admin_install_guide", lang)
        assert "/audit/verify" in text and "prev_hash" in text and "row_hash" in text, lang
