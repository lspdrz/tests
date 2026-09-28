"""Guard: every locale catalog is offered in the picker and parses.

The frontend picks a language from src/lib/i18n/locales/languages.json and then loads
locales/<code>/translation.json. A contributor who adds a locale directory and forgets
languages.json leaves a translation nobody can select, and a trailing comma or an unescaped quote
in any translation.json breaks that catalog. A built page cannot see a catalog the manifest does
not name, and a broken one fails before any page loads, so both stay a data lint over the JSON
files. That every offered language has a name, is listed once and loads its catalog, and
that English shows no raw keys, is covered in the browser by
e2e/frontend/test_locale_catalog_integrity.py.

Discriminates: passes on ef67cc3fa; a locale directory missing from languages.json fails the
picker test, and a catalog with a trailing comma fails the JSON test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

LOCALES_DIR = Path("src") / "lib" / "i18n" / "locales"


@pytest.fixture(scope="module")
def locales_dir(open_webui_backend: Path) -> Path:
    path = open_webui_backend.parent / LOCALES_DIR
    assert path.is_dir(), f"no locales directory at {path}; retarget LOCALES_DIR"
    return path


@pytest.fixture(scope="module")
def locale_dirs(locales_dir: Path) -> list[str]:
    dirs = sorted(entry.name for entry in locales_dir.iterdir() if entry.is_dir())
    assert dirs, f"no locale directories under {locales_dir}"
    return dirs


@pytest.fixture(scope="module")
def declared_languages(locales_dir: Path) -> list[dict]:
    manifest = locales_dir / "languages.json"
    assert manifest.is_file(), f"no languages.json at {manifest}; retarget the language manifest"
    return json.loads(manifest.read_text(encoding="utf-8"))


def test_every_catalog_is_offered_in_the_picker(
    declared_languages: list[dict], locale_dirs: list[str]
) -> None:
    """A translated language nobody can select is a wasted contribution."""
    declared = {entry["code"] for entry in declared_languages}
    unlisted = [code for code in locale_dirs if code not in declared]
    assert not unlisted, f"locale directories missing from languages.json: {unlisted}"


def test_every_catalog_is_valid_json(locales_dir: Path, locale_dirs: list[str]) -> None:
    broken = []
    for code in locale_dirs:
        catalog = locales_dir / code / "translation.json"
        if not catalog.is_file():
            broken.append((code, "no translation.json"))
            continue
        try:
            parsed = json.loads(catalog.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            broken.append((code, str(error)))
            continue
        if not isinstance(parsed, dict):
            broken.append((code, f"top level is {type(parsed).__name__}, expected an object"))
    assert not broken, f"unusable translation catalogs: {broken}"
