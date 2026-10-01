"""Journey: `ENABLE_ORJSON` on and off, where the switch is live and where the two values differ.

The switch moves request parsing, JSON responses, every stored JSON column, socket payloads and
provider traffic from stdlib `json` to orjson (`72fdf238a`, #27583; native JSON columns since
`8d6a7c830`, #28396), and has been on by default since #31616 (`f98ca224c`). Two instances share
one database, one per value set explicitly, so every test writes through either and reads through
either, as a deployment flipping the switch or rolling it out worker by worker does.

* The stored text of a JSON column shows which codec wrote it: stdlib spaces its separators and
  orjson writes compact. Both write non-ASCII raw since #31615 (`321a24dfe`); stdlib escaped it
  before. So the "on" path really runs, and it is the one the shared instance runs unset.
* JSON response bodies are byte for byte the same on both values, and a request body is read
  the same whether the client wrote it compact, escaped or pretty-printed; a broken one is
  refused with the same 422.
* The size limit on user and chat variables counts the characters of their JSON text, which
  stdlib wrote six per non-ASCII character until #31615. Long non-ASCII variables were refused
  (user variables, HTTP 400) or silently left out of the system prompt (chat variables) with the
  switch off, and accepted with it on.
* Searching automation prompts, and on Postgres filtering models by a tag, match the stored JSON
  text case-insensitively. Folding case turns a raw `Ü` into `ü` but leaves its escape `\\u00dc`
  alone, so while stdlib escaped (until #31615) a lower-case query for capitalised non-ASCII
  prompt text, or a capitalised non-ASCII tag on Postgres, found rows written with the switch on
  and missed rows written with it off. SQLite matches a non-ASCII tag exact-case, so the tag
  filter behaved the same there.
* The Anthropic-compatible stream carries the same text and tool arguments on both values, with
  every frame parsing when split the way the Anthropic SDK splits lines.

The variable and search tests pin the differences #31615 removed. The CJK tag and automation
search (#28399, #31422), the line separators and the codec options are pinned by
integration/models/test_automations_and_calendar.py and integration/config/test_json_codec.py.

Discriminates: on dev a5bc78300 every test passes. On dev 176d31d1d the
variable and search tests fail too (the tag test only on Postgres), and so does the stdlib side of
the stored-spelling test, which finds the old escapes, and the default test, the switch being off
there. In backend copies of dev 176d31d1d, the orjson codec decoding its output as Latin-1 turns
the orjson side of the stored-spelling and Anthropic stream tests red; orjson request parsing that
mangles non-ASCII (with indented responses) turns the byte-identity and the compact and pretty
request tests red; the orjson request parser answering an unparseable body with an empty object
turns the broken-body test red; the stdlib codec writing mojibake turns the stdlib sides red; and
the stdlib codec writing compact raw UTF-8 like orjson turns all four difference tests green.
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest

from harness import backends
from harness.actors import admin_of, create_user
from harness.chat import send_message, wait_for_reply
from harness.json_codecs import CODECS, MIXED_TEXT, codec_pair, nested, stored_text
from harness.upstream import MOCK_MODEL_ID
from harness.upstream import text as reply_text
from harness.upstream import tool_call as reply_tool_call

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

# a contract or style guide a user pastes into a variable, well under the per-value limit
LONG_RUSSIAN = (MIXED_TEXT["russian"] + ". ") * 330
CAPITALISED_TAG = "Überblick"
CAPITALISED_PROMPT = "Отчёт о продажах за квартал"


@pytest.fixture(scope="module")
def pair(instance_with):
    return codec_pair(instance_with)


@pytest.fixture(scope="module")
def admin_client(pair):
    """The admin's client on each instance, by codec."""
    clients = {codec: admin_of(pair[codec]).client() for codec in CODECS}
    yield clients
    for client in clients.values():
        client.close()


# ---------------------------------------------------------------- the switch is live


def _stored_rows(client: httpx.Client) -> dict[tuple[str, str], str]:
    """A note, a chat and a model preset carrying `nested()` as `extra`, by (table, column)."""
    note = client.post(
        "/api/v1/notes/create",
        json={"title": MIXED_TEXT["german"], "data": {}, "meta": {"extra": nested()}},
    )
    chat = client.post(
        "/api/v1/chats/new",
        json={"chat": {"title": MIXED_TEXT["german"], "models": [], "extra": nested()}},
    )
    model_id = f"codec-spelling-{uuid.uuid4().hex[:8]}"
    model = client.post(
        "/api/v1/models/create",
        json={
            "id": model_id,
            "name": MIXED_TEXT["german"],
            "base_model_id": MOCK_MODEL_ID,
            "meta": {"extra": nested()},
            "params": {},
        },
    )
    for created in (note, chat, model):
        assert created.status_code == 200, created.text
    return {
        ("note", "meta"): note.json()["id"],
        ("chat", "chat"): chat.json()["id"],
        ("model", "meta"): model_id,
    }


@pytest.mark.parametrize("codec", CODECS)
def test_the_stored_json_shows_which_codec_wrote_it(pair, admin_client, codec):
    rows = _stored_rows(admin_client[codec])

    for (table, column), row_id in rows.items():
        stored = stored_text(pair[codec], table, column, row_id)
        if codec == "orjson":
            assert "Grüße" in stored and '": ' not in stored, (
                f"with ENABLE_ORJSON=true {table}.{column} was not written by orjson: "
                f"{stored[:200]}"
            )
        else:
            assert "Grüße" in stored and '": ' in stored, (
                f"with ENABLE_ORJSON=false {table}.{column} was not written by stdlib json: "
                f"{stored[:200]}"
            )
        assert json.loads(stored)["extra"] == nested()


def test_the_shared_instance_writes_with_orjson_by_default(instance, admin):
    """#31616: with `ENABLE_ORJSON` unset every test on the shared instance runs orjson."""
    with admin.client() as client:
        note = client.post(
            "/api/v1/notes/create",
            json={"title": MIXED_TEXT["german"], "data": {}, "meta": {"extra": nested()}},
        )
    assert note.status_code == 200, note.text

    stored = stored_text(instance, "note", "meta", note.json()["id"])
    assert "Grüße" in stored and '": ' not in stored, (
        f"with ENABLE_ORJSON unset note.meta was not written by orjson: {stored[:200]}"
    )


# ---------------------------------------------------------------- the same bytes and bodies


def test_json_response_bodies_are_byte_identical(admin_client):
    model_id = f"codec-bytes-{uuid.uuid4().hex[:8]}"
    created = admin_client["stdlib"].post(
        "/api/v1/models/create",
        json={
            "id": model_id,
            "name": MIXED_TEXT["emoji"],
            "base_model_id": MOCK_MODEL_ID,
            "meta": {"description": MIXED_TEXT["arabic"], "tags": [{"name": "plain"}]},
            "params": {"temperature": 0.7},
        },
    )
    assert created.status_code == 200, created.text

    bodies = {}
    for codec, client in admin_client.items():
        listed = client.get("/api/models")
        assert listed.status_code == 200, listed.text
        missing = client.get(f"/api/v1/chats/{uuid.uuid4()}")
        bodies[codec] = (listed.content, missing.status_code, missing.content)

    assert model_id in bodies["stdlib"][0].decode()
    assert bodies["stdlib"] == bodies["orjson"], (
        "the same data served different JSON bytes with ENABLE_ORJSON on and off"
    )


REQUEST_SPELLINGS = {
    "compact-raw": lambda body: json.dumps(body, ensure_ascii=False, separators=(",", ":")),
    "escaped-spaced": lambda body: json.dumps(body),
    "pretty-crlf": lambda body: json.dumps(body, indent=2, ensure_ascii=False).replace(
        "\n", "\r\n"
    ),
}


@pytest.mark.parametrize("spelling", sorted(REQUEST_SPELLINGS))
@pytest.mark.parametrize("codec", CODECS)
def test_a_request_body_reads_the_same_in_every_spelling(admin_client, codec, spelling):
    body = {"title": MIXED_TEXT["hebrew"], "data": {"content": {"md": MIXED_TEXT["pdf_paste"]}}}
    raw = REQUEST_SPELLINGS[spelling](body).encode("utf-8")

    created = admin_client[codec].post(
        "/api/v1/notes/create", content=raw, headers={"Content-Type": "application/json"}
    )

    assert created.status_code == 200, created.text
    stored = created.json()
    assert stored["title"] == body["title"]
    assert stored["data"]["content"]["md"] == body["data"]["content"]["md"]


@pytest.mark.parametrize("codec", CODECS)
def test_a_broken_request_body_is_refused_the_same_way(admin_client, codec):
    refused = admin_client[codec].post(
        "/api/v1/notes/create",
        content='{"title": "Grüße", "data": '.encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"][0]["type"] == "json_invalid"


# ---------------------------------------------------------------- differences: variable limits


def test_long_non_ascii_user_variables_are_saved_on_both(pair):
    variables = {"style_guide": LONG_RUSSIAN, "contract": LONG_RUSSIAN}
    statuses = {}
    for codec in CODECS:
        account = create_user(pair[codec])
        with account.client() as client:
            saved = client.post(
                "/api/v1/users/user/variables/update", json={"variables": variables}
            )
        statuses[codec] = (saved.status_code, saved.json().get("detail"))

    assert statuses["stdlib"] == statuses["orjson"], (
        f"two user variables of {len(LONG_RUSSIAN)} Cyrillic characters: {statuses}. The limit "
        "counts JSON characters, which stdlib json writes as six-character escapes"
    )


def _model_with_chat_variable(client: httpx.Client) -> str:
    model_id = f"codec-vars-{uuid.uuid4().hex[:8]}"
    created = client.post(
        "/api/v1/models/create",
        json={
            "id": model_id,
            "name": "Contract reviewer",
            "base_model_id": MOCK_MODEL_ID,
            "meta": {},
            "params": {
                "system": "Review:\n{{chat.variables.document}}\nNotes:\n{{chat.variables.notes}}"
            },
            "access_grants": [
                {"principal_type": "user", "principal_id": "*", "permission": "read"}
            ],
        },
    )
    assert created.status_code == 200, created.text
    return model_id


def _system_prompt_sent(instance, client: httpx.Client, model_id: str) -> str:
    instance.upstream.reset()
    instance.upstream.queue(reply_text("reviewed"))
    turn = send_message(
        client,
        "Please review",
        model=model_id,
        chat_variables={"document": LONG_RUSSIAN, "notes": LONG_RUSSIAN},
    )
    wait_for_reply(client, turn)
    sent = instance.upstream.chat_requests()[-1]["messages"]
    return next(message["content"] for message in sent if message["role"] == "system")


def test_long_non_ascii_chat_variables_reach_the_model_on_both(pair, admin_client):
    model_id = _model_with_chat_variable(admin_client["stdlib"])
    has_document = {}
    for codec in CODECS:
        admin_client[codec].get("/api/models").raise_for_status()  # as the model selector loads
        system = _system_prompt_sent(pair[codec], admin_client[codec], model_id)
        has_document[codec] = LONG_RUSSIAN.strip() in system

    assert has_document == {"stdlib": True, "orjson": True}, (
        f"two {len(LONG_RUSSIAN)}-character Cyrillic chat variables reached the model: "
        f"{has_document}. "
        "Over the JSON-character limit every chat variable renders empty, and stdlib json "
        "counts six characters per Cyrillic letter"
    )


# ---------------------------------------------------------------- differences: Postgres search


def _tagged_model(client: httpx.Client, tag: str) -> str:
    model_id = f"codec-tag-{uuid.uuid4().hex[:8]}"
    created = client.post(
        "/api/v1/models/create",
        json={
            "id": model_id,
            "name": f"Tagged {model_id}",
            "base_model_id": MOCK_MODEL_ID,
            "meta": {"tags": [{"name": tag}]},
            "params": {},
        },
    )
    assert created.status_code == 200, created.text
    return model_id


def test_a_capitalised_non_ascii_tag_finds_models_whichever_codec_wrote_them(admin_client):
    written = {writer: _tagged_model(admin_client[writer], CAPITALISED_TAG) for writer in CODECS}

    found = {}
    for reader in CODECS:
        listed = admin_client[reader].get("/api/v1/models/list", params={"tag": CAPITALISED_TAG})
        assert listed.status_code == 200, listed.text
        ids = {item["id"] for item in listed.json()["items"]}
        found[reader] = {writer: written[writer] in ids for writer in CODECS}

    everywhere = {writer: True for writer in CODECS}
    assert found == {reader: everywhere for reader in CODECS}, (
        f"filtering by the tag {CAPITALISED_TAG!r} found (reader: {{writer: found}}) {found}: "
        "LOWER() folds the raw tag orjson stores, not the \\u00dc escape stdlib json stores"
    )


def _automation(client: httpx.Client) -> str:
    created = client.post(
        "/api/v1/automations/create",
        json={
            "name": f"weekly {uuid.uuid4().hex[:6]}",
            "data": {
                "prompt": CAPITALISED_PROMPT,
                "model_id": MOCK_MODEL_ID,
                "rrule": "RRULE:FREQ=WEEKLY",
            },
            "is_active": False,
        },
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def test_a_lower_case_query_finds_capitalised_non_ascii_prompts_the_same_way(pair, admin_client):
    written = {writer: _automation(admin_client[writer]) for writer in CODECS}
    query = CAPITALISED_PROMPT.split()[0].lower()

    found = {}
    try:
        for reader in CODECS:
            listed = admin_client[reader].get("/api/v1/automations/list", params={"query": query})
            assert listed.status_code == 200, listed.text
            ids = {item["id"] for item in listed.json()["items"]}
            found[reader] = {writer: written[writer] in ids for writer in CODECS}
    finally:
        for automation_id in written.values():
            admin_client["stdlib"].delete(f"/api/v1/automations/{automation_id}/delete")

    by_writer = {found[reader][writer] for reader in CODECS for writer in CODECS}
    database = "Postgres" if backends.on_postgres(pair["stdlib"]) else "SQLite"
    assert len(by_writer) == 1, (
        f"on {database}, searching {query!r} found (reader: {{writer: found}}) {found}: a "
        "case-insensitive match folds the raw text orjson stores, not the escapes stdlib stores"
    )


# ---------------------------------------------------------------- the Anthropic-compatible API

REPLY_PIECES = [f"{text} " for text in MIXED_TEXT.values()]
CITY_ARGUMENTS = {"city": "Zürich", "units": "°C", "days": 3, "detailed": True}


def _anthropic_stream(instance, client: httpx.Client) -> list[dict]:
    instance.upstream.reset()
    instance.upstream.queue(
        reply_text(REPLY_PIECES),
        reply_tool_call("forecast", CITY_ARGUMENTS),
    )
    events = []
    for turn in ("write it", "look it up"):
        response = client.post(
            "/api/v1/messages",
            json={
                "model": MOCK_MODEL_ID,
                "max_tokens": 256,
                "stream": True,
                "messages": [{"role": "user", "content": turn}],
            },
        )
        assert response.status_code == 200, response.text
        # the Anthropic SDK reads the stream with httpx's line decoder, which splits like this
        events += [
            json.loads(line.removeprefix("data:"))
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
    return events


@pytest.mark.parametrize("codec", CODECS)
def test_the_anthropic_messages_stream_carries_the_same_text_and_arguments(
    pair, admin_client, codec
):
    events = _anthropic_stream(pair[codec], admin_client[codec])

    deltas = [event["delta"] for event in events if event["type"] == "content_block_delta"]
    text = "".join(delta.get("text", "") for delta in deltas)
    arguments = "".join(delta.get("partial_json", "") for delta in deltas)
    assert text == "".join(REPLY_PIECES)
    assert json.loads(arguments) == CITY_ARGUMENTS
