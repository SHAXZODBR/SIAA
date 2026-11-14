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
          "/report/save_correction", "/study/{id}",
          # v1.1.0 (finish-line contract): AI settings, diagnostics export, per-clinic model import
          "/settings/ai", "/diagnostics", "/models/import"]


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
                                  "SENTINEL_HOSPITAL_BUILD", "SENTINEL_OFFLINE", "SENTINEL_MODEL_KEY_HEX",
                                  "SENTINEL_SHOW_PENDING_DETECTORS", "SENTINEL_ENCRYPT_MODELS_AT_REST"])
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
    """Single source of truth: models/brain_triage_finetuned/{study_level_eval,MANIFEST}.json,
    read by src/inference/validated_metrics.py (loaded by path — torch-free, no src package
    import). No literal numbers here: the expectations are derived from the JSON (or from the
    module's FALLBACK block when the model is not in this checkout, e.g. CI), and
    scripts/bump_model_release.py --check proves every pinned copy in the docs matches."""
    import importlib.util
    import subprocess
    import sys

    spec = importlib.util.spec_from_file_location("_docs_validated_metrics", REPO / "src/inference/validated_metrics.py")
    vm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = vm     # dataclasses resolve `from __future__` annotations via sys.modules
    spec.loader.exec_module(vm)
    m = vm.METRICS

    ev_path = REPO / "models/brain_triage_finetuned/study_level_eval.json"
    man_path = REPO / "models/brain_triage_finetuned/MANIFEST.json"
    if ev_path.exists() and man_path.exists():
        assert m.source == "json", m
        ev = json.loads(ev_path.read_text(encoding="utf-8"))
        gate = ev["gate"]
        assert gate["passed"] is True
        assert (m.tp, m.fn, m.tn, m.fp) == (gate["tp"], gate["fn"], gate["tn"], gate["fp"])
        assert (m.sens, m.spec, m.min_sens, m.min_spec) == (gate["sens"], gate["spec"], gate["min_sens"], gate["min_spec"])
        assert (m.n_studies, m.n_slices) == (ev["n_studies"], ev["n_slices"])
        assert ev["evaluated_at"].startswith(m.evaluated_on)
        man = json.loads(man_path.read_text(encoding="utf-8"))
        assert man["files"]["model.safetensors"].startswith(m.identity)
    else:
        assert m.source == "fallback", m
    # internal consistency + the gate the release was shipped under
    assert m.n_studies == m.n_abnormal + m.n_normal == m.tp + m.fn + m.tn + m.fp
    assert m.sens >= m.min_sens and m.spec >= m.min_spec
    assert len(m.identity) == 12 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", m.evaluated_on)
    gate_src = _read("scripts/eval_study_level.py")
    assert f"default={m.min_sens:.2f}" in gate_src and f"default={m.min_spec:.2f}" in gate_src

    # the documents quote exactly these values (decimal comma in RU/UZ)
    for lang in LANGS:
        iu = _doc("intended_use", lang)
        for tok in (str(m.n_studies), str(m.tp), str(m.n_abnormal), str(m.tn), str(m.n_normal),
                    str(m.n_slices), m.identity, m.evaluated_on, vm.ci(m.spec_ci, lang)):
            assert tok in iu, (lang, tok)
        for stem in ("user_manual", "intended_use", "pilot_evaluation_protocol"):
            text = _doc(stem, lang)
            assert vm.dec(m.sens, lang) in text and vm.dec(m.spec, lang) in text and str(m.n_studies) in text, (stem, lang)
        for stem in ("intended_use", "pilot_evaluation_protocol", "admin_install_guide"):
            text = _doc(stem, lang)
            assert vm.dec(m.min_sens, lang) in text and vm.dec(m.min_spec, lang) in text, (stem, lang)
        assert m.identity in _doc("admin_install_guide", lang), lang
    readme = (DOCS / "README.md").read_text(encoding="utf-8")
    assert vm.dec(m.sens) in readme and vm.dec(m.spec) in readme and str(m.n_studies) in readme and m.identity in readme

    # the runtime strings (report templates / registry notes) are generated from the same module
    for lang in LANGS:
        assert vm.pct(m.sens) in vm.triage_note(lang) and vm.pct(m.spec) in vm.triage_note(lang), lang
    assert vm.dec(m.sens) in vm.registry_notes() and str(m.n_studies) in vm.registry_notes()

    # every pinned copy (12 docs, READMEs, laptop checklists, the FALLBACK block) is in sync
    r = subprocess.run([sys.executable, str(REPO / "scripts/bump_model_release.py"), "--check"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, "stale pinned numbers — run scripts/bump_model_release.py\n" + r.stdout + r.stderr


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


# ---------------------------------------------------------------- v1.1.0 "clinic-ready" (docs/dev/finish_line_contract.md)

VENDOR_LANGS = ("en", "ru")
DIAGNOSTICS_MEMBERS = ("health.json", "license.json", "ai_settings.json", "versions.json", "models.json",
                       "audit_tail.jsonl", "settings_redacted.json")


def _i18n_value(lang: str, key: str) -> str:
    m = re.search(r"'" + re.escape(key) + r"':\s*'([^']*)'", _read(f"desktop-app/src/i18n/{lang}.ts"))
    assert m, (lang, key)
    return m.group(1)


def test_app_version_quoted_in_docs_matches_the_code():
    """Intended use identifies the software by version: the documents, the desktop app and the
    server must quote the same one (bump all three together — desktop-app/package.json,
    server.APP_VERSION, the version lines of the hospital documents)."""
    pkg = json.loads(_read("desktop-app/package.json"))["version"]
    m = re.search(r'^APP_VERSION = "([^"]+)"', _read("src/inference/server.py"), re.M)
    assert m and m.group(1) == pkg, f"server APP_VERSION {m and m.group(1)} != desktop-app/package.json {pkg}"
    word = {"en": "Version", "ru": "Версия", "uz": "Versiya"}
    for lang in LANGS:
        for stem in ("user_manual", "admin_install_guide", "intended_use", "pilot_evaluation_protocol"):
            text = _doc(stem, lang)
            assert f"{word[lang]} {pkg} " in text, (stem, lang, pkg)
            stale = {v for v in re.findall(r"\b1\.\d+\.\d+\b", text) if v != pkg}
            # the only other x.y.z tokens allowed are tag examples/torch pins, never another app version
            assert not {v for v in stale if f"Setup {v}.exe" in text or f'"version": "{v}"' in text
                        or f"{word[lang]} {v} " in text}, (stem, lang, stale)
        assert f'"version": "{pkg}"' in _doc("admin_install_guide", lang), lang
        assert f"Setup {pkg}.exe" in _doc("admin_install_guide", lang), lang


def test_founder_checklists_and_kit_quote_the_current_version_and_d1_behaviour():
    """The founder's Windows acceptance checklist (en/ru), the kit README and attributions.md
    follow desktop-app/package.json, and the checklist expects the D1 default (pending detectors
    hidden → 1 triage finding), so a correct build is never reported as broken."""
    pkg = json.loads(_read("desktop-app/package.json"))["version"]
    for lang in ("en", "ru"):
        text = _read(f"docs/windows_laptop_test_{lang}.md")
        assert f"Setup {pkg}.exe" in text, lang
        assert not {v for v in re.findall(r"Setup (\d+\.\d+\.\d+)\.exe", text) if v != pkg}, lang
        assert {"en": "exactly 1 finding", "ru": "ровно 1 находка"}[lang] in text, lang
        assert {"en": "2 findings →", "ru": "2 находки →"}[lang] not in text, lang
    kit = _read("packaging/make_windows_kit.sh")
    assert "Setup ${APP_VERSION}.exe" in kit and not re.search(r"Setup \d+\.\d+\.\d+\.exe", kit)
    assert f"Version {pkg} " in _read("docs/attributions.md")


@pytest.mark.parametrize("lang", VENDOR_LANGS)
def test_vendor_provisioning_doc_exists_and_names_the_flow(lang):
    p = DOCS / f"vendor_provisioning_{lang}.md"
    assert p.exists(), p
    text = p.read_text(encoding="utf-8")
    assert p.stat().st_size > 3000, f"{p.name} is suspiciously short"
    for token in ("packaging/provision_clinic.py", "/models/import", "data/licenses/ledger.csv",
                  "--with-model-key", "bundle_not_for_this_machine", "model_key_secret",
                  "include_validated_model", "NO-VALIDATED-MODEL", "README_admin_ru.txt", "/health"):
        assert token in text, (lang, token)


def test_vendor_provisioning_structure_matches_across_languages():
    counts = {}
    for lang in VENDOR_LANGS:
        text = (DOCS / f"vendor_provisioning_{lang}.md").read_text(encoding="utf-8")
        counts[lang] = (len(re.findall(r"^## ", text, re.M)), len(re.findall(r"^### ", text, re.M)),
                        len(re.findall(r"^\|", text, re.M)))
    assert len(set(counts.values())) == 1, f"vendor_provisioning: section/table structure differs: {counts}"


def test_vendor_provisioning_cli_flags_exist_in_the_tool():
    """Every --flag of the provision_clinic.py command quoted in the vendor docs is a real option."""
    tool = REPO / "packaging" / "provision_clinic.py"
    assert tool.exists(), "packaging/provision_clinic.py (finish-line contract, vendor-tools) is missing"
    src = tool.read_text(encoding="utf-8")
    for lang in VENDOR_LANGS:
        text = (DOCS / f"vendor_provisioning_{lang}.md").read_text(encoding="utf-8")
        block = text[text.index("python packaging/provision_clinic.py"):]
        block = block[:block.index("```")]
        flags = set(re.findall(r"(--[a-z][a-z-]+)", block))
        assert {"--fingerprint", "--customer", "--expires", "--out"} <= flags, (lang, flags)
        missing = [f for f in flags if f"'{f}'" not in src and f'"{f}"' not in src]
        assert not missing, (lang, missing)


def test_license_issue_with_model_key_is_a_real_option():
    lic = _read("src/utils/license.py")
    assert "--with-model-key" in lic, "src/utils/license.py issue has no --with-model-key (finish-line contract)"
    for lang in LANGS:
        assert "-m src.utils.license issue … --with-model-key" in _doc("admin_install_guide", lang), lang


def test_admin_guides_cover_clinic_flavour_import_and_diagnostics():
    for lang in LANGS:
        g = _doc("admin_install_guide", lang)
        for token in ("sentinel-windows-x64-full", "sentinel-windows-x64-NO-VALIDATED-MODEL",
                      "bundle_not_for_this_machine", "model_inbox\\", "brain_triage_finetuned_prev_",
                      "`SENTINEL_ENCRYPT_MODELS_AT_REST=1`", "`SENTINEL_SHOW_PENDING_DETECTORS`",
                      "settings.json", "sentinel-diagnostics-", "include_validated_model", "--print-fingerprint"):
            assert token in g, (lang, token)
        for member in DIAGNOSTICS_MEMBERS:
            assert f"`{member}`" in g, (lang, member)
        # the UI path names are the app's own labels (Preferences → AI models / About)
        for key in ("settings.ai", "settings.about", "settings.pendingTitle", "settings.importBundle",
                    "about.exportDiagnostics"):
            assert _i18n_value(lang, key) in g, (lang, key)
        # §5.4 and §5.5 exist in every language (section parity is checked separately)
        assert "\n### 5.4 " in g and "\n### 5.5 " in g, lang


def test_diagnostics_zip_members_quoted_in_docs_exist_in_server():
    server = _read("src/inference/server.py")
    for member in DIAGNOSTICS_MEMBERS:
        assert member in server, f"admin guides list {member} in the diagnostics zip but server.py never writes it"


def test_run_server_supports_the_documented_fingerprint_flag():
    assert "'--print-fingerprint' in sys.argv" in _read("run_server.py")


def test_user_manual_explains_pending_detectors_hidden_by_default():
    marker = {"en": "Pending detectors are hidden by default", "ru": "Pending-детекторы по умолчанию скрыты",
              "uz": "Pending detektorlar standart holatda yashirilgan"}
    for lang in LANGS:
        text = _doc("user_manual", lang)
        assert marker[lang] in text, lang
        assert _i18n_value(lang, "settings.ai") in text, lang          # Preferences → AI models
        assert _i18n_value(lang, "settings.pendingTitle") in text, lang  # the admin's switch
        assert _i18n_value(lang, "findings.noValidatedDetector") in text, lang  # the CT-head note, verbatim
        assert _i18n_value(lang, "findings.notAnalyzed") in text, lang  # what a CT head shows
        assert _i18n_value(lang, "worklist.needsReview") in text, lang


def test_intended_use_states_unvalidated_detectors_disabled_by_default():
    sentence = {"en": "Unvalidated (pending and experimental) detectors are disabled by default; only an administrator",
                "ru": "Невалидированные детекторы (pending и experimental) по умолчанию отключены; включить pending-детекторы может только администратор",
                "uz": "Tasdiqlanmagan detektorlar (pending va experimental) standart holatda o‘chirilgan; pending detektorlarni faqat administrator"}
    for lang in LANGS:
        assert sentence[lang] in _doc("intended_use", lang), lang




def test_model_import_reason_codes_quoted_in_admin_guides_exist_in_server():
    """The §5.4 table quotes the server's reason codes verbatim (the notification shows them)."""
    server = _read("src/inference/server.py")
    for lang in LANGS:
        g = _doc("admin_install_guide", lang)
        sec = g[g.index("\n### 5.4 "):g.index("\n### 5.5 ")]
        rows = [l for l in sec.splitlines() if l.startswith("| `")]
        codes = {c for l in rows for c in re.findall(r"`([a-z_]+)`", l.split("|")[1])}
        assert "bundle_not_for_this_machine" in codes and len(codes) >= 8, (lang, codes)
        missing = [c for c in codes if f"'{c}'" not in server]
        assert not missing, (lang, missing)


def test_provisioning_refusal_codes_quoted_in_vendor_docs_exist_in_the_tool():
    src = (REPO / "packaging" / "provision_clinic.py").read_text(encoding="utf-8")
    for lang in VENDOR_LANGS:
        text = (DOCS / f"vendor_provisioning_{lang}.md").read_text(encoding="utf-8")
        for code in ("bad_fingerprint", "bad_expiry", "model_missing", "manifest_missing", "release_check",
                     "forced_promotion", "private_key", "output_exists"):
            assert f"`{code}`" in text, (lang, code)
            assert code in src, code
        # every ledger column the docs name is one the tool writes
        fields = re.search(r"LEDGER_FIELDS = \(([^)]*)\)", src, re.S).group(1)
        for col in ("customer", "fp8", "identity", "issued", "expires", "tier", "features",
                    "key_fingerprint", "bundle_sha256"):
            assert f"`{col}`" in text and f"'{col}'" in fields, (lang, col)


# ---------------------------------------------------------------- release runbooks: publish rules

RUNBOOK_LANGS = ("en", "ru")
# Phrases of the pre-2026-11 publish section. Each one told the operator to do something that now leaks
# the model or is no longer needed: publish the tag's draft release (it holds the full installer, and the
# code repository is public), paste the MODEL_BUNDLE_URL secret for every release (CI resolves the bundle
# by the identity the bump pins), or download the full installer from the workflow artifacts (it is never
# one). Matched case-insensitively, with runs of whitespace collapsed.
RUNBOOK_STALE_PUBLISH = {
    "en": ("publish the draft release** with",
           "draft release published with numbers",
           "copy the *manifest asset api url*",
           "→ secret `model_bundle_url`",
           "`model_bundle_url` secret updated",
           "update the secret **before** tagging",
           "re-paste the secret",
           "restores the model from `model_bundle_url`",
           "to the environment secret `model_bundle_url`",
           "from the actions run artifacts"),
    "ru": ("опубликуйте черновик релиза",
           "черновик релиза опубликован с числами",
           "скопируйте *api-url ассета манифеста*",
           "→ секрет `model_bundle_url`",
           "секрет `model_bundle_url` обновлён",
           "обновите секрет **до** тега",
           "вставьте секрет снова",
           "восстанавливает модель из `model_bundle_url`",
           "в секрет окружения `model_bundle_url`",
           "из артефактов запуска actions"),
}
RUNBOOK_DO_NOT_PUBLISH = {
    "en": ("**do not publish the draft release** while it holds the full installer",
           "draft release not published (it holds the full installer)"),
    "ru": ("**не публикуйте черновик релиза**, пока в нём лежит полный установщик",
           "черновик релиза не опубликован (в нём полный установщик)"),
}


def runbook_publish_problems(text: str, lang: str) -> list[str]:
    """What is wrong with a release runbook's publish rules (empty = fine)."""
    flat = re.sub(r"\s+", " ", text).lower()
    problems = [f"stale instruction: {p!r}" for p in RUNBOOK_STALE_PUBLISH[lang] if p in flat]
    problems += [f"missing rule: {p!r}" for p in RUNBOOK_DO_NOT_PUBLISH[lang] if p not in flat]
    return problems


@pytest.mark.parametrize("lang", RUNBOOK_LANGS)
def test_release_runbook_never_publishes_the_draft_or_pastes_the_secret(lang):
    """The runbook must not tell the operator to publish the draft release (it holds the installer with
    the model, in a public repository), to paste the MODEL_BUNDLE_URL secret per release, or to download
    the full installer from the workflow artifacts — and it must keep the "do not publish" rule."""
    text = (DOCS / f"release_runbook_{lang}.md").read_text(encoding="utf-8")
    assert not runbook_publish_problems(text, lang), (lang, runbook_publish_problems(text, lang))


# ---------------------------------------------------------------- the age limitation (measured 2026-11-14)

AGE_LIMITATION = {
    "en": ("**The flag follows patient age.**", "AUC 0.78 for both", "29 of 30 normal studies flagged",
           "42 were flagged, but so were 396 of the other 472"),
    "ru": ("**Флаг следует за возрастом пациента.**", "AUC 0,78 в обоих случаях", "отмечено 29 из 30 нормальных исследований",
           "отмечено 42, но отмечены и 396 из остальных 472"),
    "uz": ("**Belgi bemor yoshiga bog‘liq.**", "ikkalasida AUC 0,78", "30 ta normal tekshiruvdan 29 tasi belgilangan",
           "42 tasi belgilangan, biroq qolgan 472 tadan 396 tasi ham belgilangan"),
}
PILOT_AGE_AND_PRIORITY = {
    "en": ("age_band (under 40 / 40-59 / 60+) | priority_finding (yes/no)", "**in each age band**",
           "flag every patient aged 40 or more", "By age band (under 40 | 40-59 | 60+)"),
    "ru": ("возрастная_группа (до 40 / 40–59 / 60+) | приоритетная_находка (да/нет)", "**в каждой возрастной группе**",
           "отмечать каждого пациента 40 лет и старше", "По возрастным группам (до 40 | 40–59 | 60+)"),
    "uz": ("age_band (under 40 / 40-59 / 60+) | priority_finding (yes/no)", "**har bir yosh guruhida**",
           "40 va undan katta yoshdagi har bir bemorni belgilash", "Yosh guruhlari bo‘yicha (40 gacha | 40–59 | 60+)"),
}
SALES_AGE_SENTENCE = {
    "sales/one_pager_en.md": "the flag largely followed patient age: from the age of 40 almost every study was flagged",
    "sales/one_pager_ru.md": "флаг во многом следовал за возрастом пациента: с 40 лет отмечалось почти каждое исследование",
    "sales/one_pager_uz.md": "belgi asosan bemor yoshiga bog‘liq bo‘ldi: 40 yoshdan boshlab",
    "sales/pilot_proposal_en.md": "the flag largely followed patient age (from the age of 40 almost every study was flagged)",
    "sales/pilot_proposal_ru.md": "флаг во многом следовал за возрастом пациента (с 40 лет отмечалось почти каждое исследование)",
}


@pytest.mark.parametrize("lang", LANGS)
def test_intended_use_states_that_the_flag_follows_patient_age(lang):
    """November 2026: patient age alone predicts the report as well as the model does, and from the age of
    40 almost every study is flagged. A clinic must be able to read this next to the validated figures
    (docs/dev/model_findings_2026-11.md, section 6)."""
    text = _doc("intended_use", lang)
    for phrase in AGE_LIMITATION[lang]:
        assert phrase in text, (lang, phrase)


@pytest.mark.parametrize("lang", LANGS)
def test_pilot_protocol_reports_by_age_band_and_priority_finding(lang):
    """The pilot must show whether the flag beats a plain age rule and whether it singles out priority
    findings: the adjudication log has the two columns and the report has the two lines."""
    text = _doc("pilot_evaluation_protocol", lang)
    for phrase in PILOT_AGE_AND_PRIORITY[lang]:
        assert phrase in text, (lang, phrase)


@pytest.mark.parametrize("rel", sorted(SALES_AGE_SENTENCE))
def test_sales_documents_say_that_the_flag_follows_age(rel):
    assert SALES_AGE_SENTENCE[rel] in (DOCS / rel).read_text(encoding="utf-8"), rel


# ------------------------------------------ release runbooks: age baseline (2026-11-14) --

RELEASE_RUNBOOK_AGE_PRIORITY = {
    "en": ("**A model that does not beat the age baseline is not evidence of detection.**",
           "**The priority rules are provisional** (`rules-v2`)",
           "`--index data/<batch>_index.csv` (patient ages)",
           "`data/<batch>_priority.csv`",
           "age alone AUC a (model m; model − age [lo, hi]), age-matched p · priority findings flagged k/K, "
           "others o/O",
           "**The index table is required.**",
           "with `--modality any` it reads MR\n  reports only",
           "**A prospective result from before the age lines**"),
    "ru": ("**Модель, которая не лучше одного только возраста, не доказывает обнаружение.**",
           "**Правила приоритетных находок предварительные** (`rules-v2`)",
           "`--index data/<batch>_index.csv` (возраст пациентов)",
           "`data/<batch>_priority.csv`",
           "age alone AUC a (model m; model − age [lo, hi]), age-matched p · priority findings flagged k/K, "
           "others o/O",
           "**Таблица индекса обязательна.**",
           "при `--modality any` — только заключения MR",
           "Если AUC модели явно\n  не превышает AUC одного только возраста",
           "отмечает порог\n    поставляемой модели",
           "**Результат prospective, записанный до строк про возраст**"),
}
# Sentences the review of 2026-11-14 found false (a missing index only "warns" and the stage "still
# runs"; in fact eval_prospective.py refuses a missing --index file) or unidiomatic: they must not return.
RELEASE_RUNBOOK_RETIRED = {
    "en": ("guard before `prospective` warns about it, the stage still runs",
           "is only a warning (the age lines are then empty)"),
    "ru": ("предупреждает об этом, этап всё равно выполняется",
           "только предупреждение (строки\nпро возраст тогда пусты)",
           "не\n  выше явно AUC", "отмечает порог\n    поставки;"),
}


@pytest.mark.parametrize("lang", sorted(RELEASE_RUNBOOK_RETIRED))
def test_release_runbook_no_longer_says_a_missing_index_only_warns(lang):
    text = _read(f"docs/release_runbook_{lang}.md")
    for phrase in RELEASE_RUNBOOK_RETIRED[lang]:
        assert phrase not in text, (lang, phrase)


def test_release_contract_says_a_missing_index_is_an_error_and_schema_2():
    text = _read("docs/dev/release_pipeline_contract.md")
    for phrase in ("A missing index is an ERROR of\n  `guard_pre_prospective`", "`schema_version: 2`",
                   "`age_new`, `priority_new`", "argparse rejects the unknown arguments (exit 2) and the stage FAILS"):
        assert phrase in text, phrase
    for phrase in ("A missing index is a WARNING", "`schema_version: 1`", "is done\n  with a warning, not failed"):
        assert phrase not in text, phrase


@pytest.mark.parametrize("lang", sorted(RELEASE_RUNBOOK_AGE_PRIORITY))
def test_release_runbook_explains_the_age_baseline_and_priority_findings(lang):
    """The prospective stage prints the age-alone AUC and the priority-finding counts for every batch
    (on 2026-11-14 the shipped model was matched by patient age alone). The runbook that tells the
    founder how to read the stage must say what the lines mean, in both languages."""
    text = _read(f"docs/release_runbook_{lang}.md")
    for phrase in RELEASE_RUNBOOK_AGE_PRIORITY[lang]:
        assert phrase in text, (lang, phrase)


def test_release_contract_names_the_priority_command_and_the_legacy_rule():
    text = _read("docs/dev/release_pipeline_contract.md")
    for phrase in ("scripts/report_priority.py --matched data/<batch>_matched.csv --labels data/<batch>_labels.csv",
                   "--index data/<batch>_index.csv --priority data/<batch>_priority.csv",
                   "`PREDATES_AGE_PRIORITY`", "A model that does not beat the age baseline is not evidence of detection",
                   "priority rules are provisional"):
        assert phrase in text, phrase
