"""Guard: the "What's New" notes are the five newest releases of CHANGELOG.md, one entry each.

`open_webui.env` turns every `## [version] - date` heading of CHANGELOG.md into an entry at
import, taking the bracketed part as the version and the rest as the date, and /api/changelog
serves the newest five to the dialog, signed in or not. A repeated version merges two releases
and pulls an older one in, anything around the brackets ends up in the version the dialog shows
and a heading without its date stops the instance from booting.

Twin of unit/config/test_changelog_boot_parse.py; its sweep over every heading of the file, most
of which the endpoint never serves, stays there.

Discriminates: passes on dev ef67cc3fa; in a backend copy whose CHANGELOG.md repeats the second
heading, or writes the newest one as `## [0.11.4] (hotfix) - 2026-09-21`, the first test fails
(the served versions differ from the file's newest headings), and with the date dropped from the
newest heading the instance exits during boot.
"""

from __future__ import annotations

import re

import httpx
import pytest
from packaging.version import Version

from harness.instance import resolve_backend

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

RELEASE_HEADING = re.compile(r"^## (?P<heading>.+)$", re.MULTILINE)
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SERVED_RELEASES = 5


def newest_release_headings(count: int = SERVED_RELEASES) -> list[tuple[str, str]]:
    """The (version, date) pairs of the file's newest headings, read the way a person reads them."""
    backend = resolve_backend()
    candidates = [backend.parent / "CHANGELOG.md", backend / "open_webui" / "CHANGELOG.md"]
    source = next(path for path in candidates if path.is_file()).read_text(encoding="utf-8")
    releases = []
    for match in list(RELEASE_HEADING.finditer(source))[:count]:
        bracketed, _, date = match["heading"].partition(" - ")
        releases.append((bracketed.strip().removeprefix("[").removesuffix("]"), date.strip()))
    return releases


@pytest.fixture(scope="module")
def served(instance) -> dict:
    with httpx.Client(base_url=instance.base_url, timeout=60.0) as signed_out:
        changelog = signed_out.get("/api/changelog")
    assert changelog.status_code == 200, changelog.text
    return changelog.json()


def test_the_dialog_is_served_the_five_newest_releases(served):
    releases = newest_release_headings()

    assert [(version, entry["date"]) for version, entry in served.items()] == releases


def test_every_served_release_is_a_dated_version_number(served):
    for version, entry in served.items():
        assert str(Version(version)) == version, f"{version!r} is not a plain version number"
        assert DATE.match(entry["date"]), f"{version} is dated {entry['date']!r}"


def test_every_served_release_carries_its_notes(served):
    empty = [
        version
        for version, entry in served.items()
        if not any(isinstance(items, list) and items for items in entry.values())
    ]
    assert not empty, f"releases served without any notes: {empty}"
