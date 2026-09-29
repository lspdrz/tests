"""Journey: an admin connects a pgvector table in the Integrations settings and a chat uses it.

The admin picks pgvector as the provider of a knowledge connection in Admin Settings >
Integrations, which swaps the API key for the table and collection fields, fills in the layout
Open WebUI writes itself and can only be saved after its test query returned rows. A source whose
collection holds nothing cannot be created. A chat that attaches the resulting knowledge base with
`#` is sent the rows nearest to its question. The database is a real Postgres with the vector
extension, so the tests read what the settings showed and what the model was sent.

Discriminates: red on dev 176d31d1d for the tests that search the table, since every search fails
there (open-webui/open-webui#26663, fix in #31112); they pass once the query embedding is sent as
a vector. In a frontend copy, with the provider switch leaving the API key field in place the
fields test fails, with the table name changed in the source the add test fails (the search reads
a table that is not there) and with the Create button enabled before a test passed the no-rows
test fails. In a backend copy with that fixed, searching the other collections turns the chat
test red (another collection's row is sent).
"""

from __future__ import annotations

import json
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.external_knowledge import CONNECTIONS, external_connection, external_knowledge_base
from harness.pgvector_source import COLLECTION, SOURCE_CONFIG, chunk, serving_pgvector
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHUNKS = [
    chunk("far", "Moorings are inspected every May.", [0.3, 0.2, 0.1], source="moorings.md"),
    chunk(
        "near", "The tide tables hang in the harbour office.", [0.1, 0.2, 0.3], source="tides.md"
    ),
    chunk("elsewhere", "The canteen closes at four.", [0.1, 0.2, 0.3], collection="canteen_docs"),
]
QUESTION = "where are the tide tables?"


@pytest.fixture
def curator(make_user):
    """A fresh admin, whose external knowledge is deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for knowledge in client.get(
            "/api/v1/knowledge/search", params={"source": "external"}
        ).json()["items"]:
            if knowledge["user_id"] == account.id:
                client.delete(f"/api/v1/knowledge/{knowledge['id']}/delete")
        for connection in client.get(CONNECTIONS).json()["items"]:
            if connection["created_by"] == account.id:
                client.delete(f"{CONNECTIONS}/{connection['id']}")


@pytest.fixture
def source_form():
    with serving_pgvector(CHUNKS) as form:
        yield form


def _open_integrations(page: Page) -> Locator:
    """The Integrations tab of the admin settings, freshly loaded from the server."""
    page.goto("/admin/settings/integrations")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_text("External Knowledge Sources", exact=True)).to_be_visible()
    return settings


def _add_button(settings: Locator) -> Locator:
    # the plus button has no accessible name, so it is found next to the section title
    title = settings.get_by_text("External Knowledge Sources", exact=True)
    return title.locator("xpath=ancestor::div[.//button][1]").get_by_role("button")


def _open_new_pgvector_form(page: Page) -> Locator:
    _add_button(_open_integrations(page)).click()
    form = page.get_by_role("dialog").filter(
        has=page.get_by_role("heading", name="Add Knowledge Connection")
    )
    form.get_by_label("Provider").select_option("pgvector")
    return form


def _fill_new_source(form: Locator, name: str, endpoint: str, collection: str) -> None:
    form.get_by_label("Name", exact=True).fill(name)
    form.get_by_label("Endpoint").fill(endpoint)
    form.get_by_label("Collection", exact=True).fill(collection)
    form.get_by_label("Test Query").fill(QUESTION)


def test_choosing_pgvector_swaps_the_api_key_for_the_table_fields(page_for, curator):
    page = page_for(curator)
    _add_button(_open_integrations(page)).click()
    form = page.get_by_role("dialog").filter(
        has=page.get_by_role("heading", name="Add Knowledge Connection")
    )
    expect(form.get_by_label("API Key / Token")).to_be_visible()
    expect(form.get_by_label("Table")).to_have_count(0)

    form.get_by_label("Provider").select_option("pgvector")

    expect(form.get_by_label("API Key / Token")).to_have_count(0)
    expect(form.get_by_label("Table")).to_have_value(SOURCE_CONFIG["table_name"])
    expect(form.get_by_label("Collection Field")).to_have_value(SOURCE_CONFIG["collection_field"])
    expect(form.get_by_label("Content Field")).to_have_value(SOURCE_CONFIG["content_field"])
    expect(form.get_by_label("Vector Field")).to_have_value(SOURCE_CONFIG["vector_field"])
    expect(form.get_by_label("Metadata Field")).to_have_value(SOURCE_CONFIG["metadata_field"])
    expect(form.get_by_role("button", name="Create")).to_be_disabled()


def test_a_source_is_added_once_its_test_query_finds_rows_and_survives_a_reload(
    page_for, curator, source_form
):
    name = f"Harbour {uuid.uuid4().hex[:6]}"
    page = page_for(curator)
    form = _open_new_pgvector_form(page)
    _fill_new_source(form, name, source_form["endpoint"], COLLECTION)
    expect(form.get_by_role("button", name="Create")).to_be_disabled()

    form.get_by_role("button", name="Verify Connection").click()
    expect(page.get_by_text("Test succeeded.").first).to_be_visible()
    form.get_by_role("button", name="Create").click()
    expect(page.get_by_text("Knowledge source created.").first).to_be_visible()

    settings = _open_integrations(page)
    expect(settings.get_by_text(name)).to_be_visible()
    expect(settings.get_by_text(f"pgvector · {COLLECTION}")).to_be_visible()
    with curator.client() as client:
        [saved] = [
            connection
            for connection in client.get(CONNECTIONS).json()["items"]
            if connection["endpoint"] == source_form["endpoint"]
        ]
    assert saved["provider"] == "pgvector" and saved["enabled"] is True


def test_a_source_whose_collection_holds_no_rows_cannot_be_created(page_for, curator, source_form):
    page = page_for(curator)
    form = _open_new_pgvector_form(page)
    _fill_new_source(
        form, f"Empty {uuid.uuid4().hex[:6]}", source_form["endpoint"], "no_such_collection"
    )

    form.get_by_role("button", name="Verify Connection").click()

    expect(page.get_by_text("Test returned no results.").first).to_be_visible()
    expect(form.get_by_role("button", name="Create")).to_be_disabled()


def test_a_chat_with_the_source_attached_is_sent_the_nearest_rows_of_its_collection(
    page_for, curator, source_form, upstream
):
    name = f"Harbour {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("In the harbour office.", match=reply.answering(QUESTION)))
    page = page_for(curator)
    with (
        curator.client() as client,
        external_connection(client, source_form) as connection_id,
        external_knowledge_base(client, connection_id, COLLECTION, name, config=SOURCE_CONFIG),
    ):
        page.goto("/")
        chat_input(page).click()
        page.keyboard.type("#Harbour")
        page.get_by_role("tooltip").get_by_role("button", name=name).click()
        send(page, QUESTION)
        expect_reply(page, "In the harbour office.")

    [chat_request] = [body for body in upstream.chat_requests() if reply.answering(QUESTION)(body)]
    sent = json.dumps(chat_request["messages"])
    assert "The tide tables hang in the harbour office." in sent
    assert "Moorings are inspected every May." in sent
    assert "canteen" not in sent, "a row of another collection was sent"
