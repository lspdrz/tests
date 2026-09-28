"""Dependency smoke: Azure OpenAI signed in with Microsoft Entra ID, and Azure AI Search.

An Azure OpenAI connection whose auth type is Microsoft Entra ID sends no key: every request gets
a bearer token from azure-identity's `DefaultAzureCredential` through `get_bearer_token_provider`
for the Cognitive Services scope. Here the credential finds an App Service managed identity
(`IDENTITY_ENDPOINT` and `IDENTITY_HEADER`, only read at boot), played by a local service that
also answers as the Azure OpenAI deployment, so a test reads which token reached the provider.
The Azure AI Search web search engine queries an index with azure-search-documents'
`SearchClient` and an `AzureKeyCredential`, and turns each hit's `url`, `title` and `content`
into a search result. The admin panel offers no fields for it, so it is set up through the
environment as the documentation describes, on the same instance and the same local service.

Discriminates: passes on dev ef67cc3fa; in a backend copy whose Entra token lookup returns None
both Entra tests fail (the provider gets no bearer token), and one whose Azure search sends
`search_text="*"` in place of the query fails the search test.
"""

from __future__ import annotations

import json

import pytest

from harness.actors import admin_of
from harness.chat import ask
from harness.listener import ReceivedRequest, json_answer, listening
from harness.second_provider import OPENAI_CONFIG, sse

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

ENTRA_TOKEN = "entra-token-for-the-harbour"
IDENTITY_HEADER = "managed-identity-secret"
COGNITIVE_SERVICES = "https://cognitiveservices.azure.com"
DEPLOYMENT = "harbour-gpt"
API_VERSION = "2024-10-21"
SEARCH_KEY = "azure-search-query-key"
SEARCH_INDEX = "harbour-pages"
SEARCH_PATH = f"/indexes('{SEARCH_INDEX}')/docs/search.post.search"


def _managed_identity_token(request: ReceivedRequest):
    return json_answer(
        {
            "access_token": ENTRA_TOKEN,
            "expires_on": "4102444800",
            "resource": COGNITIVE_SERVICES,
            "token_type": "Bearer",
        }
    )


def _search_answer(request: ReceivedRequest):
    hits = [
        {
            "@search.score": 2.5,
            "url": "https://harbour.example/tides",
            "title": "Tide tables",
            "content": "High water at the harbour mouth is at noon.",
        },
        {"@search.score": 1.5, "title": "A hit with no link", "content": "dropped"},
    ]
    return json_answer({"value": hits})


@pytest.fixture(scope="module")
def azure():
    """The managed identity endpoint, the Azure OpenAI deployment and the search index, as one."""
    with listening() as service:
        service.route("GET", "/msi/token", _managed_identity_token)
        service.route("GET", "/openai/models", json_answer({"data": [{"id": DEPLOYMENT}]}))
        service.route("POST", SEARCH_PATH, _search_answer)
        service.route(
            "POST",
            f"/openai/deployments/{DEPLOYMENT}/chat/completions",
            sse({"role": "assistant", "content": ""}, {"content": "signed in with Entra"}),
        )
        yield service


@pytest.fixture
def managed_identity(instance_with, azure):
    return instance_with(
        {
            "IDENTITY_ENDPOINT": f"{azure.base_url}/msi/token",
            "IDENTITY_HEADER": IDENTITY_HEADER,
            "ENABLE_WEB_SEARCH": "true",
            "WEB_SEARCH_ENGINE": "azure",
            "WEB_SEARCH_RESULT_COUNT": "4",
            "BYPASS_WEB_SEARCH_WEB_LOADER": "true",
            "BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL": "true",
            "AZURE_AI_SEARCH_API_KEY": SEARCH_KEY,
            "AZURE_AI_SEARCH_ENDPOINT": azure.base_url,
            "AZURE_AI_SEARCH_INDEX_NAME": SEARCH_INDEX,
        }
    )


def _entra_connection() -> dict:
    return {
        "enable": True,
        "azure": True,
        "api_version": API_VERSION,
        "auth_type": "microsoft_entra_id",
        "model_ids": [DEPLOYMENT],
    }


@pytest.fixture
def entra_admin(managed_identity, azure, preserve):
    """The admin of the managed-identity instance, with the Entra connection added."""
    preserve(OPENAI_CONFIG, on=managed_identity)
    admin = admin_of(managed_identity)
    with admin.client() as client:
        current = client.get(OPENAI_CONFIG[0]).json()
        index = str(len(current["OPENAI_API_BASE_URLS"]))
        updated = {
            **current,
            "OPENAI_API_BASE_URLS": [*current["OPENAI_API_BASE_URLS"], azure.base_url],
            "OPENAI_API_KEYS": [*current["OPENAI_API_KEYS"], ""],
            "OPENAI_API_CONFIGS": {**current["OPENAI_API_CONFIGS"], index: _entra_connection()},
        }
        client.post(OPENAI_CONFIG[1], json=updated).raise_for_status()
        client.get("/api/models").raise_for_status()
    return admin


def _token_requests(azure) -> list[ReceivedRequest]:
    return azure.requests_to("/msi/token")


def test_a_chat_on_an_entra_connection_carries_the_managed_identity_token(entra_admin, azure):
    with entra_admin.client() as client:
        _, message = ask(client, "hello?", model=DEPLOYMENT)

    [completion] = azure.requests_to(f"/openai/deployments/{DEPLOYMENT}/chat/completions")
    assert completion.headers.get("Authorization") == f"Bearer {ENTRA_TOKEN}"
    assert "api-key" not in {name.lower() for name in completion.headers}
    assert f"api-version={API_VERSION}" in completion.path
    assert message["content"] == "signed in with Entra"
    asked = _token_requests(azure)[-1]
    assert asked.headers.get("X-IDENTITY-HEADER") == IDENTITY_HEADER
    assert f"resource={COGNITIVE_SERVICES}" in asked.path


def test_verifying_an_entra_connection_signs_in_with_the_token(managed_identity, azure):
    with admin_of(managed_identity).client() as client:
        verified = client.post(
            "/openai/verify",
            json={"url": azure.base_url, "key": "", "config": _entra_connection()},
        )

    assert verified.status_code == 200, verified.text
    assert verified.json()["data"] == [{"id": DEPLOYMENT}]
    [listing] = azure.requests_to("/openai/models")
    assert listing.headers.get("Authorization") == f"Bearer {ENTRA_TOKEN}"


def test_an_azure_ai_search_web_search_returns_the_index_hits(managed_identity, azure):
    with admin_of(managed_identity).client() as client:
        searched = client.post("/api/v1/retrieval/process/web/search", json={"queries": ["tides"]})

    assert searched.status_code == 200, searched.text
    [query] = azure.requests_to(SEARCH_PATH)
    assert query.headers.get("api-key") == SEARCH_KEY
    assert json.loads(query.body) == {"search": "tides", "top": 4}
    [document] = searched.json()["docs"]
    assert document["content"] == "High water at the harbour mouth is at noon."
    assert document["metadata"]["source"] == "https://harbour.example/tides"
    assert document["metadata"]["title"] == "Tide tables"
