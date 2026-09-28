"""Guard: the "What's New" dialog lists the five newest releases of CHANGELOG.md, each dated.

`open_webui.env` turns every `## [version] - date` heading of CHANGELOG.md into an entry at
import and the dialog an admin opens from Settings > General shows the newest five as `v<version>`
with the date spelled out. A repeated version merges two releases and pulls an older one in, and
anything around the brackets ends up in the version shown. Closing the dialog records the version
the admin has seen.

Twin of unit/config/test_changelog_boot_parse.py.

Discriminates: passes on dev ef67cc3fa; in a backend copy whose CHANGELOG.md repeats the second
heading the listing test fails (0.11.3 shows once and 0.11.0 is listed as well).
"""

from __future__ import annotations

from datetime import date

import pytest
from playwright.sync_api import Locator, Page, expect

from integration.config.test_changelog_boot_parse import newest_release_headings

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def _spelled_out(iso_date: str) -> str:
    day = date.fromisoformat(iso_date)
    return f"{day:%b} {day.day}, {day.year}"


def _version_heading(dialog: Locator, version: str) -> Locator:
    return dialog.page.get_by_role("heading", name=f"v{version}", exact=True)


def _open_whats_new(page: Page) -> Locator:
    page.goto("/admin/settings/general")
    page.get_by_role("button", name="See what's new").click()
    dialog = page.get_by_role("dialog").filter(has_text="What's New in")
    expect(dialog).to_be_visible()
    return dialog


def test_the_dialog_lists_the_five_newest_releases_with_their_dates(page_for, make_user):
    releases = newest_release_headings()
    dialog = _open_whats_new(page_for(make_user(role="admin")))

    listed = dialog.get_by_role("heading", level=3)
    expect(listed).to_have_text([f"v{version}" for version, _ in releases])
    for version, released in releases:
        section = dialog.locator("section").filter(has=_version_heading(dialog, version))
        expect(section).to_contain_text(_spelled_out(released))


def test_closing_the_dialog_records_the_version_seen(page_for, make_user):
    admin = make_user(role="admin")
    page = page_for(admin)
    dialog = _open_whats_new(page)

    with page.expect_response(lambda response: "/users/user/settings/update" in response.url):
        dialog.get_by_role("button", name="Okay, Let's Go!").click()

    expect(dialog).to_be_hidden()
    with admin.client() as client:
        version = client.get("/api/config").json()["version"]
        seen = client.get("/api/v1/users/user/settings").json()["ui"]["version"]
    assert seen == version
