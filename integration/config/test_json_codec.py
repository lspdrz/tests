"""Regression: streamed replies and stored JSON under the orjson codec, and oversized stream lines.

Three open-webui v0.11.1 fixes:

* PR #27819 (commit 1d6735ff0): stdlib json escapes U+2028, U+2029 and U+0085, orjson writes them
  raw. Python reads all three as line breaks, so under `ENABLE_ORJSON` a chunk the server
  re-serialized (every chunk, once a stream filter is installed) split into lines that no longer
  parse for a consumer reading the stream with `splitlines()`. `dumps` now escapes them.
* Commit 78ed5a0235: the orjson codec swallowed `dumps` and `loads` options. A note stored with
  structured markdown came back as one compact line instead of the indented JSON block the note
  sanitizer asks for, and a tag search stopped matching the ASCII-escaped spelling stdlib json
  had stored, so switching `ENABLE_ORJSON` on hid every model tagged with a non-ASCII name.
* Commit a33fa05adc: when `CHAT_STREAM_RESPONSE_CHUNK_MAX_BUFFER_SIZE` meant "no limit" (unset, 0
  or negative) the provider stream was read through aiohttp's own line reader, which aborted any
  reply that arrived as one line over its limit (512 KiB on aiohttp 3.14). A configured limit
  drops only the line over it.

The codec options no route passes (`sort_keys`, other `separators`, `object_hook`) stay in
unit/config/test_json_codec.py.

Twin of unit/config/test_json_codec.py.

Discriminates: passes on bbfa876af; dropping the separator escaping from `ORJSONCodec.dumps` fails
the separator test, dropping its option fallback fails the note and tag search tests, and handing
back the raw reader for a disabled limit fails the oversized line tests. The plain-text, configured
limit and empty stream tests pass on both.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing

import pytest

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ask
from harness.instance import ADMIN_EMAIL, ADMIN_PASSWORD
from harness.plugins import installed_function
from harness.prepared_data import serving
from harness.raw_provider import connect
from harness.second_provider import OPENAI_CONFIG
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

ORJSON = {"ENABLE_ORJSON": "true"}
BUFFER_SIZE = "CHAT_STREAM_RESPONSE_CHUNK_MAX_BUFFER_SIZE"
OVERSIZED_LINE = 1_000_000  # characters, past any aiohttp line limit
LINE_SEPARATORS = ("\u2028", "\u2029", "\x85")

# Makes the server parse and re-serialize every streamed chunk.
PASS_THROUGH_STREAM_FILTER = """
class Filter:
    def stream(self, event):
        return event
"""


@pytest.fixture
def orjson_instance(instance_with):
    return instance_with(ORJSON)


def _streamed_frames(instance, pieces: list[str]) -> list[str]:
    """The data lines of a filtered streamed reply, split the way `str.splitlines()` splits."""
    instance.upstream.queue(reply.text(pieces))
    admin = admin_of(instance)
    request = {
        "model": MOCK_MODEL_ID,
        "messages": [{"role": "user", "content": "say it"}],
        "stream": True,
    }
    with installed_function(admin, PASS_THROUGH_STREAM_FILTER, is_global=True):
        with admin.client() as client:
            streamed = client.post("/api/chat/completions", json=request)
    assert streamed.status_code == 200, streamed.text
    return [
        line.removeprefix("data:").strip()
        for line in streamed.text.splitlines()
        if line.strip() and line.strip() != "data: [DONE]"
    ]


def _content(frames: list[str]) -> str:
    chunks = [json.loads(frame) for frame in frames]
    return "".join(
        choice["delta"].get("content") or "" for chunk in chunks for choice in chunk["choices"]
    )


def test_line_separators_in_a_reply_keep_every_frame_parseable(orjson_instance):
    pieces = [f"before{separator}after " for separator in LINE_SEPARATORS]

    frames = _streamed_frames(orjson_instance, pieces)

    unparseable = []
    for frame in frames:
        try:
            json.loads(frame)
        except ValueError:
            unparseable.append(frame)
    assert unparseable == [], (
        "a line separator in the reply was written raw, so the frame split into lines that do "
        f"not parse (#27819): {unparseable}"
    )
    assert _content(frames) == "".join(pieces)


def test_plain_text_streams_through_the_codec_unchanged(orjson_instance):
    pieces = ["héllo ", "wörld\n", "done"]

    assert _content(_streamed_frames(orjson_instance, pieces)) == "".join(pieces)


def test_a_note_with_structured_markdown_is_stored_as_indented_json(orjson_instance):
    structured = {"steps": ["mix", "bake"], "serves": 4}

    with admin_of(orjson_instance).client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": "Recipe", "data": {"content": {"md": structured}}},
        )
    assert created.status_code == 200, created.text

    stored = created.json()["data"]["content"]["md"]
    body = stored.removeprefix("```json\n").removesuffix("\n```")
    assert json.loads(body) == structured
    assert len(body.splitlines()) > 1, (
        "the note's JSON was stored on one line: the orjson codec dropped the indent option "
        f"the sanitizer passed: {stored!r}"
    )


def test_a_reply_arriving_as_one_oversized_line_is_stored_whole(user, upstream):
    oversized = "x" * OVERSIZED_LINE
    upstream.queue(reply.text(oversized))

    with user.client() as client:
        _, message = ask(client, "one very long line please")

    assert message["content"] == oversized, (
        "a reply that arrived as one line over aiohttp's limit was cut off on default settings, "
        f"because the stream was read through the raw reader: {len(message['content'])} chars"
    )


@pytest.mark.parametrize("disabled", ["0", "-1"])
def test_a_reply_arriving_as_one_oversized_line_survives_a_disabled_limit(instance_with, disabled):
    unlimited = instance_with({BUFFER_SIZE: disabled})
    oversized = "y" * OVERSIZED_LINE
    unlimited.upstream.queue(reply.text(oversized))

    with admin_of(unlimited).client() as client:
        _, message = ask(client, "one very long line please")

    assert message["content"] == oversized, (
        f"{BUFFER_SIZE}={disabled} means no limit, yet the reply was cut off at aiohttp's own "
        f"line limit: {len(message['content'])} chars"
    )


def test_a_configured_limit_drops_only_the_oversized_line(instance_with):
    limited = instance_with({BUFFER_SIZE: "100000"})
    limited.upstream.queue(reply.text(["before ", "z" * 200_000, "after"]))

    with admin_of(limited).client() as client:
        _, message = ask(client, "three lines, one too long")

    assert message["content"] == "before after"


def test_an_empty_provider_stream_ends_the_reply_quietly(admin, listener, preserve):
    preserve(OPENAI_CONFIG)
    provider = connect(admin, listener)
    provider.stream(b"")

    with admin.client() as client:
        _, message = ask(client, "anything?", model=provider.model_id)

    assert message["content"] == ""
    assert not message.get("error"), message


def _stored_meta(data_dir, model_id: str) -> str:
    with closing(sqlite3.connect(data_dir / "webui.db")) as database:
        row = database.execute("SELECT meta FROM model WHERE id = ?", (model_id,)).fetchone()
    return row[0]


def test_a_non_ascii_tag_stored_by_stdlib_json_is_still_found_under_orjson(tmp_path):
    tag = f"Café {uuid.uuid4().hex[:6]}"
    model_id = f"tagged-{uuid.uuid4().hex[:8]}"
    account = {"name": "Admin", "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}

    with serving(tmp_path) as stdlib_backend, stdlib_backend.client() as client:
        signed_up = client.post("/api/v1/auths/signup", json=account)
        assert signed_up.status_code == 200, signed_up.text
        client.headers["Authorization"] = f"Bearer {signed_up.json()['token']}"
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "name": "Tagged preset",
                "base_model_id": MOCK_MODEL_ID,
                "meta": {"tags": [{"name": tag}]},
                "params": {},
            },
        )
        assert created.status_code == 200, created.text
    assert "\\u00e9" in _stored_meta(tmp_path, model_id), "stdlib json no longer escapes the tag"

    with serving(tmp_path, ORJSON) as orjson_backend, orjson_backend.client() as client:
        signed_in = client.post(
            "/api/v1/auths/signin", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
        )
        assert signed_in.status_code == 200, signed_in.text
        client.headers["Authorization"] = f"Bearer {signed_in.json()['token']}"
        found = client.get("/api/v1/models/list", params={"tag": tag})

    assert found.status_code == 200, found.text
    assert [item["id"] for item in found.json()["items"]] == [model_id], (
        "with ENABLE_ORJSON on, the tag search only looked for the raw spelling: the codec "
        "ignored ensure_ascii=True, so a tag stored escaped by stdlib json was never found"
    )
