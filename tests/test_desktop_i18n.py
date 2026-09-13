"""Desktop-app dictionary parity (desktop-app/src/i18n/{en,ru,uz}.ts).

en.ts is the source of truth for the key set; ru.ts / uz.ts must carry every
key (and nothing else), every value must be non-empty, and the {placeholder}
set of each translated string must match the English one so t(key, params)
never leaves a stale "{count}" on screen in one language only.

The files are flat objects with quoted keys, so a regex is enough — no Node
toolchain is required to run this check.
"""

import re
from pathlib import Path

import pytest

I18N_DIR = Path(__file__).resolve().parents[1] / "desktop-app" / "src" / "i18n"
LANGS = ("en", "ru", "uz")

# Keys whose value is intentionally identical in every language (language
# names, badges, product abbreviations).
IDENTICAL_OK = {
    "lang.ru", "lang.uz", "lang.en",
    "lang.short.ru", "lang.short.uz", "lang.short.en",
    "tool.zoomShort", "tool.widthShort", "tool.levelShort",
    "report.sha", "viewer.acc", "status.srv", "status.app",
    "urgency.review", "settings.pacs",
}

_ENTRY = re.compile(r"^\s*'([^']+)':\s*'((?:[^'\\]|\\.)*)',?\s*$", re.M)


def _load(lang: str) -> dict[str, str]:
    path = I18N_DIR / f"{lang}.ts"
    if not path.exists():
        pytest.skip(f"{path} not present in this checkout")
    src = path.read_text(encoding="utf-8")
    entries = _ENTRY.findall(src)
    assert entries, f"{path.name}: no entries parsed"
    keys = [k for k, _ in entries]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f"{path.name}: duplicate keys {dupes}"
    return dict(entries)


@pytest.fixture(scope="module")
def dicts() -> dict[str, dict[str, str]]:
    return {lang: _load(lang) for lang in LANGS}


def test_en_exports_key_type():
    src = (I18N_DIR / "en.ts").read_text(encoding="utf-8")
    assert "export type I18nKey = keyof typeof en" in src


def test_key_sets_are_identical(dicts):
    en = set(dicts["en"])
    for lang in ("ru", "uz"):
        other = set(dicts[lang])
        assert not (en - other), f"{lang}.ts is missing keys: {sorted(en - other)}"
        assert not (other - en), f"{lang}.ts has keys absent from en.ts: {sorted(other - en)}"


def test_key_counts_are_equal(dicts):
    counts = {lang: len(d) for lang, d in dicts.items()}
    assert len(set(counts.values())) == 1, counts


def test_no_empty_values(dicts):
    for lang, d in dicts.items():
        empty = [k for k, v in d.items() if not v.strip()]
        assert not empty, f"{lang}.ts empty values: {empty}"


def test_placeholders_match_english(dicts):
    ph = re.compile(r"\{(\w+)\}")
    en = dicts["en"]
    for lang in ("ru", "uz"):
        bad = {}
        for k, v in dicts[lang].items():
            want = set(ph.findall(en[k]))
            got = set(ph.findall(v))
            if want != got:
                bad[k] = (sorted(want), sorted(got))
        assert not bad, f"{lang}.ts placeholder mismatch: {bad}"


def test_translations_are_actually_translated(dicts):
    """A translated dictionary that merely copies English is not a translation.
    Allow the small set of deliberately identical strings and at most a few
    percent of accidental matches (proper nouns, units, abbreviations)."""
    en = dicts["en"]
    for lang in ("ru", "uz"):
        same = [k for k, v in dicts[lang].items() if v == en[k] and k not in IDENTICAL_OK]
        assert len(same) <= max(5, len(en) // 25), f"{lang}.ts identical to English: {same}"


def test_uzbek_uses_typographic_apostrophe(dicts):
    """Uzbek Latin o‘ / g‘ must use U+2018 (project convention, see uz.ts)."""
    bad = [k for k, v in dicts["uz"].items() if re.search(r"[og]'", v)]
    assert not bad, f"uz.ts uses ASCII apostrophe in: {bad}"


def test_components_reference_only_known_keys(dicts):
    """Every t('...') / translate(lang, '...') literal in the renderer must exist in en.ts."""
    src_dir = I18N_DIR.parent
    known = set(dicts["en"])
    call = re.compile(r"\b(?:t|translate)\(\s*(?:[a-zA-Z_]+,\s*)?'([a-zA-Z0-9_.]+)'")
    missing = {}
    for path in src_dir.rglob("*.ts*"):
        if path.is_relative_to(I18N_DIR):
            continue
        for key in call.findall(path.read_text(encoding="utf-8")):
            if key not in known:
                missing.setdefault(path.name, set()).add(key)
    assert not missing, f"unknown i18n keys referenced: {missing}"
