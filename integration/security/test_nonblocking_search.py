"""Regression: the built-in text searches ran on the event loop and froze the whole worker.

open-webui 0.11.4 fix `82f11b14c` (#29621) plus the knowledge_fs half in `d9c8de9c`:
`grep_chat_files` and `grep_knowledge_files` matched every line of every file directly on the
event loop, and `kb_exec`'s grep built its matcher and scanned inline, so while a model searched
a large file no other request on the worker was answered. All three now hand the matching to a
worker thread.

The model searches a large knowledge file on an instance of its own while the test keeps asking
the server for `/health` and the chat, one request after the other, and records the slowest
answer. A search on the loop holds every one of those answers back until it is done.

Twin of unit/security/test_nonblocking_search.py.

Discriminates: passes on dev ef67cc3fa, fails with the thread hand-off in each tool replaced by a
direct call (a request waits for the whole matching pass).
"""

from __future__ import annotations

import time

import httpx
import pytest

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import send_message
from harness.knowledge_bases import KB_EXEC, add_text_file
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

# `|$` matches every line, and working out where costs RE2 over half a millisecond per line
PATTERN = "(.{0,50}a){8}q|$"
FILLER_LINE = ("abcd efgh ijkl mnop 1234 5678 " * 33)[:990]
FILLER_LINES = 2500
BUDGET_SPENT = "Search exceeded"
# a few large chunks keep the upload of the big file to a few embedding requests
LARGE_CHUNKS = {"CHUNK_SIZE": "200000", "RAG_EMBEDDING_BATCH_SIZE": "64"}


def archive(launched) -> dict:
    """A large knowledge file, and the models that search it.

    One model has the knowledge base attached, which offers `kb_exec`; the other reads a chat's
    files through tools, and without knowledge attached is offered `grep_knowledge_files` too.
    Nothing is removed afterwards: the instance is this module's own and stops with it.
    """
    content = "\n".join([FILLER_LINE] * FILLER_LINES)
    with admin_of(launched).client() as client:
        created = client.post(
            "/api/v1/knowledge/create",
            json={"name": "Archive", "description": "", "access_grants": []},
        )
        assert created.status_code == 200, created.text
        kb_id = created.json()["id"]
        file_id = add_text_file(client, kb_id, "archive.txt", content)
        presets = {
            "kb_exec": {"knowledge": [{"type": "collection", "id": kb_id, "name": "Archive"}]},
            "reader": {"capabilities": {"file_upload": True, "file_context": False}},
        }
        models = {}
        for role, meta in presets.items():
            model_id = f"{role}-{kb_id[:8]}".replace("_", "-")
            form = {
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": model_id,
                "meta": meta,
                "params": {},
                "access_grants": [EVERYONE_READS],
            }
            added = client.post("/api/v1/models/create", json=form)
            assert added.status_code == 200, added.text
            models[role] = model_id
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    return {"instance": launched, "file_id": file_id, "models": models}


@pytest.fixture(scope="module")
def archives(instance_with):
    """The archive on an instance with `kb_exec` switched on, and on one without it."""
    return {
        "shell": archive(instance_with({**KB_EXEC, **LARGE_CHUNKS})),
        "plain": archive(instance_with(LARGE_CHUNKS)),
    }


def timed_search(large_file, tool: str, arguments: dict) -> tuple[str, float, float]:
    """Run the tool; returns its result, the slowest answer the server gave meanwhile and the
    whole turn's time."""
    launched = large_file["instance"]
    launched.upstream.reset()
    launched.upstream.queue(reply.tool_call(tool, arguments), reply.text("done"))
    attached = [{"type": "file", "id": large_file["file_id"], "name": "archive.txt"}]
    chat_files = attached if tool == "grep_chat_files" else None
    model = large_file["models"]["kb_exec" if tool == "kb_exec" else "reader"]
    waits: list[float] = []

    def timed(send) -> httpx.Response:
        asked = time.perf_counter()
        answer = send()
        waits.append(time.perf_counter() - asked)
        return answer

    with admin_of(launched).client() as client, httpx.Client(base_url=launched.base_url) as probe:
        started = time.perf_counter()
        turn = send_message(client, f"use {tool}", model=model, chat_files=chat_files)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            for _ in range(10):
                timed(lambda: probe.get("/health")).raise_for_status()
            stored = timed(lambda: client.get(f"/api/v1/chats/{turn.chat_id}"))
            messages = stored.json()["chat"]["history"]["messages"]
            if messages.get(turn.assistant_message_id, {}).get("done"):
                break
        else:
            raise AssertionError(f"{tool} never finished")
        elapsed = time.perf_counter() - started
    sent_back = launched.upstream.chat_requests()[-1]["messages"]
    result = [entry["content"] for entry in sent_back if entry["role"] == "tool"][-1]
    return result, max(waits), elapsed


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("grep_chat_files", {"pattern": PATTERN}),
        ("grep_knowledge_files", {"pattern": PATTERN}),
        ("kb_exec", {"command": f'grep -E "{PATTERN}" archive.txt'}),
    ],
    ids=["grep_chat_files", "grep_knowledge_files", "kb_exec"],
)
def test_a_long_search_leaves_the_worker_answering(archives, tool, arguments):
    large_file = archives["shell" if tool == "kb_exec" else "plain"]
    result, stall, elapsed = timed_search(large_file, tool, arguments)

    assert FILLER_LINE[:40] in result or BUDGET_SPENT in result, result[:300]
    assert stall < elapsed / 3, (
        f"a request waited {stall * 1000:.0f} ms during a {elapsed * 1000:.0f} ms search by "
        f"{tool}, so every other request on the worker waited for the matching pass (#29621)"
    )
