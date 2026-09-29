"""Regression: a fresh install died on a circular import before it served a page (issue #29280,
fix 8c0c7b3b6, shipped in v0.11.3).

Importing the config module runs the migrations, and Alembic's `env.py` imports the calendar
model, whose import chain reached back into the still-loading config module. The server never
started, so the first visitor got no page at all. Here the server starts on an empty data
directory with the built frontend, and the first visitor walks the onboarding: the welcome
screen, the admin account form and the chat it lands on, as the admin.

Twin of unit/imports/test_config_imports_cleanly.py; the API side is
integration/imports/test_fresh_install.py.
Discriminates: passes on dev ef67cc3fa; a module-scope `from open_webui.config import
ENABLE_SIGNUP` in a copy's `models/calendar.py` (the #29280 cycle) turns it red (no server
starts).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from utils.chat_ui import chat_input

pytestmark = [
    pytest.mark.regression,
    pytest.mark.slow,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
]

ADMIN_EMAIL = "owner@example.com"
ADMIN_PASSWORD = "owner-password-123"


def test_the_first_visitor_creates_the_admin_account_and_lands_in_chat(browser, fresh_install):
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080}, base_url=fresh_install.base_url
    )
    try:
        page = context.new_page()
        page.goto("/")

        page.get_by_role("button", name="Get started").click()
        expect(page.get_by_text("Get started with Open WebUI")).to_be_visible()
        page.get_by_label("Name").fill("Owner")
        page.get_by_label("Email").fill(ADMIN_EMAIL)
        page.get_by_label("Password", exact=True).fill(ADMIN_PASSWORD)
        page.get_by_role("button", name="Create Admin Account").click()

        expect(chat_input(page)).to_be_visible(timeout=30_000)
        token = page.evaluate("localStorage.token")
    finally:
        context.close()

    with fresh_install.client(token) as client:
        session = client.get("/api/v1/auths/")
    assert session.status_code == 200, f"HTTP {session.status_code}: {session.text}"
    assert session.json()["email"] == ADMIN_EMAIL
    assert session.json()["role"] == "admin", "the onboarding account is not the admin"
