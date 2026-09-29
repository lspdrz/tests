"""Journey: a user's own direct connection, from Settings to a chat answered by their browser.

When the admin allows direct connections, a user adds an OpenAI-shaped endpoint of their own in
Settings > Connections. The page itself lists that endpoint's models in the model selector and
sends the chat to it: the instance's own provider is never called. With the admin switch off
the Connections tab is gone and the saved endpoint's models drop out of the selector.

Discriminates: in a frontend build, skipping the direct connections in the model list fetch turns
the chat test and the switch-off test red, dropping the tab's forwarding of the provider's stream
turns the chat test red alone, and showing the tab and the connections whatever the admin switch
says (and listing a connection whatever its own switch says) turns the three switch tests red.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.browser_provider import serve
from utils.chat_ui import chat_input, conversation, expect_reply, send
from utils.personal_connections import add_connection_form, open_personal_connections

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SELECTOR_BUTTON = re.compile("Select a model|Selected model")
CONNECTIONS_CONFIG = ("/api/v1/configs/connections", "/api/v1/configs/connections")


def set_direct_connections(admin, preserve, enabled: bool) -> None:
    preserve(CONNECTIONS_CONFIG)
    with admin.client() as client:
        current = client.get(CONNECTIONS_CONFIG[0]).json()
        changed = {**current, "ENABLE_DIRECT_CONNECTIONS": enabled}
        client.post(CONNECTIONS_CONFIG[1], json=changed).raise_for_status()


def save_connections(account, url: str, key: str = "", enabled: bool = True) -> None:
    connections = {
        "OPENAI_API_BASE_URLS": [url],
        "OPENAI_API_KEYS": [key],
        "OPENAI_API_CONFIGS": {"0": {"enable": enabled}},
    }
    with account.client() as client:
        client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"directConnections": connections}}
        ).raise_for_status()


def model_options(page: Page, name: str) -> Locator:
    search = page.get_by_role("textbox", name="Search In Models")
    if not search.is_visible():
        page.get_by_role("button", name=SELECTOR_BUTTON).click()
    search.fill(name)
    return page.get_by_role("listbox", name="Available models").get_by_role(
        "option", name=f"Select {name} model"
    )


def select_model(page: Page, name: str) -> None:
    model_options(page, name).click()


@pytest.fixture
def direct_model_id() -> str:
    return f"my-own-{uuid.uuid4().hex[:6]}"


def test_a_user_adds_a_connection_and_chats_with_its_model_straight_from_the_browser(
    page_for, make_user, admin, preserve, listener, upstream, direct_model_id
):
    set_direct_connections(admin, preserve, True)
    provider = serve(listener, direct_model_id)
    provider.reply_with("answered by the local fake")
    page = page_for(make_user())
    tab = open_personal_connections(page)

    tab.get_by_role("button", name="Add Connection").click()
    form = add_connection_form(page)
    form.get_by_role("combobox", name="URL").fill(provider.base_url)
    form.get_by_role("textbox", name="API Key").fill("sk-own-key")
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        form.get_by_role("button", name="Save").click()
    expect(form).to_be_hidden()
    page.get_by_role("dialog").get_by_role("button", name="Back").click()

    select_model(page, direct_model_id)
    prompt = f"Say something, {uuid.uuid4().hex[:6]}."
    upstream.queue(
        reply.text("the instance provider must stay silent", match=reply.answering(prompt))
    )
    send(page, prompt)
    expect_reply(page, "answered by the local fake")

    [request] = provider.chat_requests()
    assert request.headers["Authorization"] == "Bearer sk-own-key"
    assert request.headers["Origin"] == page.url.split("/c/")[0].rstrip("/")
    assert request.json()["model"] == direct_model_id
    assert prompt in request.body.decode()
    assert upstream.chat_requests() == [], "the instance's own provider answered a direct chat"
    expect(conversation(page)).to_contain_text(prompt)


def test_the_connections_tab_and_saved_models_are_gone_while_the_admin_switch_is_off(
    page_for, make_user, admin, preserve, listener, direct_model_id
):
    set_direct_connections(admin, preserve, False)
    provider = serve(listener, direct_model_id)
    account = make_user()
    save_connections(account, provider.base_url)
    page = page_for(account)
    expect(chat_input(page)).to_be_visible()

    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    page.get_by_role("button", name="Settings").click()
    expect(page.get_by_role("tab", name="General")).to_be_visible()
    expect(page.get_by_role("tab", name="Connections")).to_have_count(0)
    page.get_by_role("dialog").get_by_role("button", name="Back").click()
    expect(model_options(page, "mock-model")).to_have_count(1)
    expect(model_options(page, direct_model_id)).to_have_count(0)
    assert provider.models_requests() == [], "the browser asked a switched-off connection"


def test_switching_direct_connections_off_hides_them_from_a_user_who_has_one(
    page_for, make_user, admin, preserve, listener, direct_model_id
):
    set_direct_connections(admin, preserve, True)
    provider = serve(listener, direct_model_id)
    account = make_user()
    save_connections(account, provider.base_url)
    user_page = page_for(account)
    open_personal_connections(user_page)
    user_page.get_by_role("dialog").get_by_role("button", name="Back").click()
    expect(model_options(user_page, direct_model_id)).to_have_count(1)

    admin_page = page_for(make_user(role="admin"))
    admin_page.goto("/admin/settings/connections")
    switch = admin_page.get_by_role("dialog").get_by_role("switch", name="Direct Connections")
    expect(switch).to_be_checked()
    with admin_page.expect_response(lambda response: "/configs/connections" in response.url):
        switch.click()
    expect(switch).not_to_be_checked()

    user_page.reload()
    expect(chat_input(user_page)).to_be_visible()
    expect(model_options(user_page, "mock-model")).to_have_count(1)
    expect(model_options(user_page, direct_model_id)).to_have_count(0)


def test_a_connection_switched_off_in_its_own_settings_lists_no_models(
    page_for, make_user, admin, preserve, listener, direct_model_id
):
    set_direct_connections(admin, preserve, True)
    provider = serve(listener, direct_model_id)
    account = make_user()
    save_connections(account, provider.base_url)
    page = page_for(account)
    tab = open_personal_connections(page)

    tab.get_by_role("button", name="Open modal to configure connection").click()
    editing = page.get_by_role("dialog").filter(
        has=page.get_by_role("heading", name="Edit Connection")
    )
    editing.get_by_role("switch", name="Toggle whether current connection is active.").click()
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        editing.get_by_role("button", name="Save").click()
    expect(editing).to_be_hidden()

    page.reload()
    expect(chat_input(page)).to_be_visible()
    # the instance's own model shows the list has loaded
    expect(model_options(page, "mock-model")).to_have_count(1)
    expect(model_options(page, direct_model_id)).to_have_count(0)
