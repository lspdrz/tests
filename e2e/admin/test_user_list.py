"""Journey: an admin edits, deletes, searches and sorts accounts in Admin Panel > Users.

The edit dialog changes an account's name, email, role and password at once: the list shows the
new name and role, the old credentials stop working, the new ones sign in through the auth page
and the promoted account opens the admin panel. Deleting an account from its row takes it off the
list and its sign-in is refused. The search box narrows the list by name or email, and the Name
and Email column headers sort what is left, flipping direction on a second click. Each test works
as a fresh admin on accounts of its own.

Discriminates: passes on dev 176d31d1d; in a frontend copy, the edit dialog sending the stored
role and no password turns the edit test red (the role stays user), the delete confirmation doing
nothing turns the delete test red (the row stays) and the Email header sorting by name turns the
sort test red; in a backend copy, the user search matching emails alone turns the search test red
(the name finds nobody).

The header test is red on dev 176d31d1d (open-webui/open-webui#31581): the sorted column's
`aria-sort`, added by #27501 to tell screen readers the sort, keeps the value it had on load,
since in legacy mode Svelte does not re-run a template call to a helper when the state that
helper reads changes. Writing the comparison into each header's attribute turns it green.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

INVALID_CREDENTIALS = "The email or password provided is incorrect."


@pytest.fixture
def admin_page(make_user, page_for) -> Page:
    """A fresh admin's page, so nothing here touches the shared admin's settings."""
    return page_for(make_user(role="admin"))


def _user_list(page: Page) -> Locator:
    page.goto("/admin/users")
    users = page.get_by_role("main")
    expect(users.get_by_role("button", name="Add User")).to_be_visible()
    return users


def _search(users: Locator, query: str) -> None:
    users.get_by_role("textbox", name="Search").fill(query)


def _row(users: Locator, email: str) -> Locator:
    return users.get_by_role("row").filter(has_text=email)


def _sign_in(page: Page, email: str, password: str) -> None:
    page.goto("/auth")
    page.get_by_label("Email").fill(email)
    page.get_by_label("Password", exact=True).fill(password)
    page.get_by_role("button", name="Sign in", exact=True).click()


def _dismiss_changelog(account: Actor) -> None:
    # an account promoted to admin meets the changelog on its first page load
    with account.client() as client:
        dismissed = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"showChangelog": False}}
        )
    dismissed.raise_for_status()


def test_an_edited_account_signs_in_with_its_new_details_and_opens_the_admin_panel(
    admin_page, make_user, page
):
    account = make_user()
    _dismiss_changelog(account)
    suffix = uuid.uuid4().hex[:8]
    new_name, new_email, new_password = (
        f"Edited {suffix}",
        f"edited-{suffix}@example.com",
        "edited-pw-1",
    )

    users = _user_list(admin_page)
    _search(users, account.email)
    _row(users, account.email).get_by_role("button", name="Edit User").click()
    editing = admin_page.get_by_role("dialog").filter(has_text="Edit User")
    editing.get_by_role("combobox", name="Role").select_option(label="Admin")
    editing.get_by_role("textbox", name="Name").fill(new_name)
    editing.get_by_role("textbox", name="Email").fill(new_email)
    editing.get_by_label("New Password").fill(new_password)
    editing.get_by_role("button", name="Save").click()
    expect(editing).to_be_hidden()

    _search(users, new_email)
    edited = _row(users, new_email)
    expect(edited).to_contain_text(new_name)
    expect(edited.get_by_role("button", name="Change User Role")).to_have_text("admin")

    _sign_in(page, account.email, account.password)
    expect(page.get_by_text(INVALID_CREDENTIALS)).to_be_visible()
    _sign_in(page, new_email, new_password)
    expect(chat_input(page)).to_be_visible()
    promoted_users = _user_list(page)
    _search(promoted_users, new_email)
    expect(_row(promoted_users, new_email)).to_be_visible()


def test_a_deleted_account_leaves_the_list_and_cannot_sign_in(admin_page, make_user, page):
    account = make_user()
    users = _user_list(admin_page)
    _search(users, account.email)

    _row(users, account.email).get_by_role("button", name="Delete User").click()
    admin_page.get_by_role("button", name="Confirm").click()

    expect(_row(users, account.email)).to_have_count(0)
    admin_page.reload()
    _search(users, account.email)
    expect(users.get_by_role("row")).to_have_count(1)  # the header alone
    _sign_in(page, account.email, account.password)
    expect(page.get_by_text(INVALID_CREDENTIALS)).to_be_visible()
    expect(chat_input(page)).to_be_hidden()


def test_the_search_narrows_the_list_by_name_or_email(admin_page, make_user):
    suffix = uuid.uuid4().hex[:8]
    wanted = make_user(name=f"Wanted {suffix}", email=f"wanted-{suffix}@example.com")
    other = make_user(name=f"Other {suffix}", email=f"other-{suffix}@example.com")
    users = _user_list(admin_page)

    _search(users, suffix)
    expect(_row(users, wanted.email)).to_be_visible()
    expect(_row(users, other.email)).to_be_visible()

    _search(users, f"Wanted {suffix}")
    expect(_row(users, other.email)).to_have_count(0)
    expect(_row(users, wanted.email)).to_be_visible()

    _search(users, other.email)
    expect(_row(users, wanted.email)).to_have_count(0)
    expect(_row(users, other.email)).to_be_visible()


def test_the_column_headers_sort_the_list_both_ways(admin_page, make_user):
    suffix = uuid.uuid4().hex[:8]
    # created, name and email orders all differ, so each sort shows a list of its own
    for name, email_prefix in (("Abe", "b"), ("Zed", "c"), ("Mia", "a")):
        make_user(name=f"{name} {suffix}", email=f"{email_prefix}-sort-{suffix}@example.com")
    users = _user_list(admin_page)
    _search(users, f"sort-{suffix}")
    rows = users.get_by_role("row").filter(has_text=suffix)
    expect(rows).to_have_count(3)

    def expect_order(*names: str) -> None:
        expect(rows).to_have_text([re.compile(f"{name} {suffix}") for name in names])

    name_header = users.get_by_role("columnheader", name="Name")
    name_header.get_by_role("button").click()
    expect_order("Abe", "Mia", "Zed")
    name_header.get_by_role("button").click()
    expect_order("Zed", "Mia", "Abe")

    email_header = users.get_by_role("columnheader", name="Email")
    email_header.get_by_role("button").click()
    expect_order("Mia", "Abe", "Zed")
    email_header.get_by_role("button").click()
    expect_order("Zed", "Abe", "Mia")


def test_the_sorted_column_header_tells_assistive_technology_its_direction(admin_page):
    users = _user_list(admin_page)
    name_header = users.get_by_role("columnheader", name="Name")
    created_header = users.get_by_role("columnheader", name="Created at")
    expect(created_header).to_have_attribute("aria-sort", "ascending")

    with admin_page.expect_response(lambda response: "order_by=name" in response.url):
        name_header.get_by_role("button").click()

    stale = "aria-sort keeps its load value after a header click (open-webui/open-webui#31581)"
    expect(name_header, stale).to_have_attribute("aria-sort", "ascending")
    expect(created_header, stale).to_have_attribute("aria-sort", "none")
