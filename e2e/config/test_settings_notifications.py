"""Journey: Settings > Notifications, the browser notification for a reply finished elsewhere.

A user who allows notifications switches Browser Notifications on, starts a chat and moves to a
new chat before the reply is done; the finished reply then shows as a browser notification as
well as the in-page toast. The switch is still on after a reload, and a second account that never
switched it on gets only the toast.

A browser that denies the permission gets the error toast and nothing is saved, but the switch
still shows On until the dialog is reopened: it keeps its own flipped state because the tab hands
it the setting one way. That test stays red until the switch follows the setting.

Discriminates: passes on dev 176d31d1d apart from the denied-permission test, which fails there
(the switch shows On); in a frontend copy, showing the finished-reply notification without
reading `notificationEnabled` turns the second account's check red.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ANSWER = "The tide turns at noon."
DENIED = "Response notifications cannot be activated as the website permissions have been denied."

# keeps what the page shows, so a test can read the notifications that reached the browser
RECORD_NOTIFICATIONS = """
window.shownNotifications = [];
window.Notification = new Proxy(window.Notification, {
    construct(target, args) {
        window.shownNotifications.push({ title: args[0], body: args[1]?.body });
        return Reflect.construct(target, args);
    }
});
"""


def recording_page(page_for, account, allow: bool) -> Page:
    page = page_for(account)
    if allow:
        page.context.grant_permissions(["notifications"])
    page.add_init_script(RECORD_NOTIFICATIONS)
    page.goto("/")
    return page


def notifications_switch(page: Page) -> Locator:
    page.goto("/?settings=notifications")
    switch = page.get_by_role("switch", name="Browser Notifications")
    expect(switch).to_be_visible()
    return switch


def finish_a_reply_elsewhere(page: Page, upstream, question: str) -> list[dict]:
    upstream.queue(
        reply.text(
            ["The tide ", "turns at noon."], chunk_delay=1.0, match=reply.answering(question)
        )
    )
    page.goto("/")
    send(page, question)
    expect(page).to_have_url(re.compile(r"/c/"))
    page.get_by_role("link", name="New Chat").click()
    expect(page.get_by_text(ANSWER)).to_be_visible()
    return page.evaluate("window.shownNotifications")


def test_a_reply_finished_in_another_chat_shows_a_browser_notification(
    page_for, make_user, upstream
):
    page = recording_page(page_for, make_user(), allow=True)
    switch = notifications_switch(page)
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        switch.click()
    expect(switch).to_have_attribute("aria-checked", "true")

    shown = finish_a_reply_elsewhere(page, upstream, "When does the tide turn?")
    assert [notification["body"] for notification in shown] == [ANSWER]

    page.reload()
    expect(notifications_switch(page)).to_have_attribute("aria-checked", "true")

    other = recording_page(page_for, make_user(), allow=True)
    expect(notifications_switch(other)).to_have_attribute("aria-checked", "false")
    assert finish_a_reply_elsewhere(other, upstream, "When does the tide turn today?") == []


def test_a_denied_permission_leaves_the_switch_off(page_for, make_user):
    account = make_user()
    page = recording_page(page_for, account, allow=False)
    switch = notifications_switch(page)

    switch.click()

    expect(page.get_by_text(DENIED)).to_be_visible()
    with account.client() as client:
        saved = client.get("/api/v1/users/user/settings").json()["ui"]
    assert saved.get("notificationEnabled") is not True
    expect(switch, "the refused switch still shows On").to_have_attribute("aria-checked", "false")
