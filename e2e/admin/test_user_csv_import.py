"""A user CSV import reported the blank line after its last row as invalid, #31373.

Fix commit `db697c7e6` (open-webui/open-webui#31374). The admin's CSV import splits the file on
line breaks and checks every row after the header for four columns. A file ending in a line
break, as most editors and spreadsheet exports write it, has an empty last line, which failed that
check and showed "Row N: invalid format." next to the success message. Empty lines are now
skipped.

Discriminates: passes on dev efe63bd34; with db697c7e6 reverted the file ending in a line break
shows "Row 4: invalid format."; the malformed row is reported on both.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness.actors import Actor

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

HEADER = "Name,Email,Password,Role"


def _import_csv(page: Page, csv: str) -> None:
    page.goto("/admin/users")
    page.get_by_role("main").get_by_role("button", name="Add User").click()
    form = page.get_by_role("dialog").filter(has_text="Add User")
    form.get_by_role("button", name="CSV Import").click()
    with page.expect_file_chooser() as chooser:
        form.get_by_role("button", name="Click here to select a csv file.").click()
    chooser.value.set_files(
        files=[{"name": "users.csv", "mimeType": "text/csv", "buffer": csv.encode()}]
    )
    form.get_by_role("button", name="Save").click()


def _emails(suffix: str, count: int) -> list[str]:
    return [f"csv-{index}-{suffix}@example.com" for index in range(count)]


def _registered(admin: Actor, email: str) -> bool:
    with admin.client() as client:
        found = client.get("/api/v1/users/", params={"query": email}).json()
    return any(user["email"] == email for user in found["users"])


def test_a_file_ending_in_a_line_break_imports_without_an_error(page_for, make_user, admin):
    suffix = uuid.uuid4().hex[:8]
    emails = _emails(suffix, 2)
    rows = [f"Csv {index},{email},csvpassword123,user" for index, email in enumerate(emails)]
    page = page_for(make_user(role="admin"))

    _import_csv(page, "\n".join([HEADER, *rows]) + "\n")

    expect(page.get_by_text("Successfully imported 2 users.")).to_be_visible()
    # a retrying expect would pass once the error toast times out
    page.wait_for_timeout(500)
    row_errors = page.get_by_text(re.compile(r"invalid format")).all_inner_texts()
    assert row_errors == [], "the blank last line was reported as a row (#31373)"
    assert all(_registered(admin, email) for email in emails)


# ---------------------------------------------------------------- nearby


def test_a_malformed_row_is_still_reported(page_for, make_user, admin):
    suffix = uuid.uuid4().hex[:8]
    (email,) = _emails(suffix, 1)
    page = page_for(make_user(role="admin"))

    _import_csv(page, f"{HEADER}\nCsv 0,{email},csvpassword123,user\nonly,three,columns\n")

    expect(page.get_by_text("Row 3: invalid format.")).to_be_visible()
    expect(page.get_by_text("Successfully imported 1 users.")).to_be_visible()
    assert _registered(admin, email)
