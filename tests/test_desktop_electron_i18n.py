"""Electron main-process dictionary parity (desktop-app/electron/main.js).

Native dialogs (file pickers, backend crash/restart boxes) are shown by the
Electron main process, outside React, so main.js carries its own tiny
MESSAGES = {en, ru, uz} table and a 'set-language' IPC fed by the renderer.
These checks mirror tests/test_desktop_i18n.py for that table: every language
has the same keys, no value is empty, {placeholder} sets match English, and no
dialog string is still hard-coded in English.

main.js requires 'electron' at import time, so the table is parsed with a
regex — no Node/Electron toolchain is needed.
"""

import re
from pathlib import Path

import pytest

DESKTOP = Path(__file__).resolve().parents[1] / "desktop-app"
MAIN_JS = DESKTOP / "electron" / "main.js"
PRELOAD_JS = DESKTOP / "electron" / "preload.js"
LANGS = ("en", "ru", "uz")

# Values that are intentionally identical across languages (product/format names).
IDENTICAL_OK = {"pdfFiles", "ok", "signal"}

_LANG_BLOCK = re.compile(r"^\s{2}(en|ru|uz):\s*\{\n(.*?)^\s{2}\},?$", re.M | re.S)
_ENTRY = re.compile(r"^\s*(\w+):\s*'((?:[^'\\]|\\.)*)',?\s*$", re.M)


def _load() -> dict[str, dict[str, str]]:
    if not MAIN_JS.exists():
        pytest.skip(f"{MAIN_JS} not present in this checkout")
    src = MAIN_JS.read_text(encoding="utf-8")
    start = src.find("const MESSAGES = {")
    assert start >= 0, "main.js: MESSAGES dictionary not found"
    end = src.find("\n};", start)
    body = src[start:end]
    out: dict[str, dict[str, str]] = {}
    for lang, block in _LANG_BLOCK.findall(body):
        entries = _ENTRY.findall(block)
        assert entries, f"main.js: no entries parsed for '{lang}'"
        keys = [k for k, _ in entries]
        assert len(keys) == len(set(keys)), f"main.js '{lang}': duplicate keys"
        out[lang] = dict(entries)
    return out


@pytest.fixture(scope="module")
def dicts() -> dict[str, dict[str, str]]:
    return _load()


def test_all_three_languages_present(dicts):
    assert set(dicts) == set(LANGS), sorted(dicts)


def test_key_sets_are_identical(dicts):
    en = set(dicts["en"])
    for lang in ("ru", "uz"):
        other = set(dicts[lang])
        assert not (en - other), f"main.js '{lang}' is missing keys: {sorted(en - other)}"
        assert not (other - en), f"main.js '{lang}' has keys absent from en: {sorted(other - en)}"


def test_no_empty_values(dicts):
    for lang, d in dicts.items():
        empty = [k for k, v in d.items() if not v.strip()]
        assert not empty, f"main.js '{lang}' empty values: {empty}"


def test_placeholders_match_english(dicts):
    ph = re.compile(r"\{(\w+)\}")
    en = dicts["en"]
    for lang in ("ru", "uz"):
        bad = {}
        for k, v in dicts[lang].items():
            want, got = set(ph.findall(en[k])), set(ph.findall(v))
            if want != got:
                bad[k] = (sorted(want), sorted(got))
        assert not bad, f"main.js '{lang}' placeholder mismatch: {bad}"


def test_translations_are_actually_translated(dicts):
    en = dicts["en"]
    for lang in ("ru", "uz"):
        same = [k for k, v in dicts[lang].items() if v == en[k] and k not in IDENTICAL_OK]
        assert not same, f"main.js '{lang}' identical to English: {same}"


def test_uzbek_uses_typographic_apostrophe(dicts):
    """Uzbek Latin o‘ / g‘ must use U+2018 (project convention, see uz.ts)."""
    bad = [k for k, v in dicts["uz"].items() if re.search(r"[og]'", v)]
    assert not bad, f"main.js 'uz' uses ASCII apostrophe in: {bad}"


def test_default_language_is_russian():
    src = MAIN_JS.read_text(encoding="utf-8")
    assert re.search(r"^let uiLang = 'ru';", src, re.M), "main.js must default dialogs to Russian"


def test_set_language_ipc_is_wired():
    """main.js listens, preload exposes setLanguage, the renderer pushes settings.language."""
    main = MAIN_JS.read_text(encoding="utf-8")
    preload = PRELOAD_JS.read_text(encoding="utf-8")
    app_tsx = (DESKTOP / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "ipcMain.on('set-language'" in main
    assert "ipcRenderer.send('set-language'" in preload
    assert "setLanguage" in preload
    assert "setLanguage?.(settings.language)" in app_tsx


def test_dialogs_reference_only_known_keys(dicts):
    """Every msg('...') call in main.js must exist in the English table."""
    src = MAIN_JS.read_text(encoding="utf-8")
    used = set(re.findall(r"\bmsg\(\s*'(\w+)'", src))
    assert used, "main.js: no msg() calls found"
    unknown = used - set(dicts["en"])
    assert not unknown, f"main.js msg() keys missing from MESSAGES.en: {sorted(unknown)}"


def test_no_hardcoded_english_dialog_strings():
    """The strings the localisation replaced must not come back as literals
    outside the MESSAGES table."""
    src = MAIN_JS.read_text(encoding="utf-8")
    start = src.find("const MESSAGES = {")
    end = src.find("\n};", start)
    outside = src[:start] + src[end:]
    for literal in (
        "'Open DICOM File'", "'Open DICOM Folder'", "'Save Report PDF'",
        "'Backend not found'", "'Backend failed to start'",
        "'AI server stopped'", "'AI server restarting'", "buttons: ['OK']",
    ):
        assert literal not in outside, f"main.js still hard-codes {literal}"
