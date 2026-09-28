"""Regression: one knowledge-search pattern must not be able to stall the worker.

open-webui 0.11.0 fix `3ab202626` (PR #27471): the knowledge search tools compiled the pattern
the model wrote with Python's backtracking `re` and ran it over every line of every reachable
file, with no timeout. A catastrophic pattern such as `(a|aa)+$` against a long non-matching line
costs exponential time, and one such search froze the whole instance for minutes.

dev matches with RE2, which is linear-time, and gives each `kb_exec` command one matching budget
of two seconds, charged only for the time spent inside searches: every grep of a pipeline draws
on it, and the next command starts with a full one. The model calls the tools on an instance of
its own, so a regression wedges nothing the rest of the suite uses, and the narrow tests poll
with short timeouts, so a search that never returns fails the test instead of hanging it.

Twin of unit/security/test_knowledge_search_match_budget.py, which keeps the test that time spent
between two searches of one command is not charged: no request can stretch that gap.

Discriminates: passes on dev ef67cc3fa; with RE2 swapped for `re` the catastrophic searches never
answer, with a fresh budget per matcher the long pipeline runs to its end, and with a budget that
is never reset the next command's search is refused.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import send_message
from harness.knowledge_bases import KB_EXEC, add_text_file
from harness.python_tools import EVERYONE_READS
from harness.tool_calls import run_tool
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

CATASTROPHIC_PATTERN = "(a|aa)+$"
NON_MATCHING_LINE = "a" * 48 + "!"
# a backtracking engine takes minutes here, RE2 microseconds
ANSWER_CEILING_SECONDS = 20.0

# RE2 spends about half a millisecond per line on this pattern, and `|$` matches every line,
# so each grep of a pipeline passes the whole file on to the next one.
SLOW_PATTERN = "(.{0,50}a){8}q|$"
SLOW_LINE = ("abcd efgh ijkl mnop 1234 5678 " * 33)[:990]
SLOW_LINES = 600
PIPELINE_GREPS = 16

LINES = ["alpha beta", "gamma delta", "ALPHA omega", "epsilon"]
BUDGET_SPENT = "Search exceeded"
# a few large chunks keep the upload of the slow file to a few embedding requests
LARGE_CHUNKS = {"CHUNK_SIZE": "200000", "RAG_EMBEDDING_BATCH_SIZE": "64"}


def library_on(launched) -> dict:
    """A knowledge base holding the catastrophic line, a slow file and a short one.

    Nothing is removed afterwards: the instance is this module's own and stops with it, and after
    a regression it would answer no request anyway.
    """
    with admin_of(launched).client() as client:
        created = client.post(
            "/api/v1/knowledge/create",
            json={"name": "Search budget", "description": "", "access_grants": []},
        )
        assert created.status_code == 200, created.text
        kb_id = created.json()["id"]
        trap_id = add_text_file(client, kb_id, "trap.txt", NON_MATCHING_LINE)
        add_text_file(client, kb_id, "slow.txt", "\n".join([SLOW_LINE] * SLOW_LINES))
        add_text_file(client, kb_id, "lines.txt", "\n".join(LINES))
        model_id = f"search-budget-{kb_id[:8]}"
        form = {
            "id": model_id,
            "base_model_id": MOCK_MODEL_ID,
            "name": model_id,
            "meta": {"knowledge": [{"type": "collection", "id": kb_id, "name": "Search budget"}]},
            "params": {},
            "access_grants": [EVERYONE_READS],
        }
        added = client.post("/api/v1/models/create", json=form)
        assert added.status_code == 200, added.text
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    return {"instance": launched, "model": model_id, "trap": trap_id}


@pytest.fixture(scope="module")
def libraries(instance_with):
    """`kb_exec` is only offered on an instance booted with it, `grep_knowledge_files` without."""
    return {
        "kb_exec": library_on(instance_with({**KB_EXEC, **LARGE_CHUNKS})),
        "grep_knowledge_files": library_on(instance_with(LARGE_CHUNKS)),
    }


@pytest.fixture
def run(libraries):
    """`run(tool, arguments)` has the model call the tool and returns what the tool returned."""
    clients: dict[str, httpx.Client] = {}

    def call(tool: str, arguments: dict) -> str:
        library = libraries[tool]
        launched = library["instance"]
        if tool not in clients:
            clients[tool] = admin_of(launched).client()
        launched.upstream.reset()
        return run_tool(
            clients[tool], launched.upstream, tool, arguments, model=model_offering(library, tool)
        )

    yield call
    for client in clients.values():
        client.close()


def model_offering(library, tool: str) -> str:
    # `kb_exec` reads the attached knowledge; the account-wide tools need a model without any
    return library["model"] if tool == "kb_exec" else MOCK_MODEL_ID


def kb_exec(run, command: str) -> str:
    return run("kb_exec", {"command": command})


def answered_within(library, tool: str, arguments: dict, ceiling: float) -> str | None:
    """The tool's result if the chat finished within `ceiling` seconds, else None."""
    launched = library["instance"]
    launched.upstream.reset()
    launched.upstream.queue(reply.tool_call(tool, arguments), reply.text("done"))
    token = admin_of(launched).token
    headers = {"Authorization": f"Bearer {token}"}
    # short timeouts: a worker stuck in a regex answers nothing at all
    with httpx.Client(base_url=launched.base_url, headers=headers, timeout=5.0) as client:
        turn = send_message(client, f"use {tool}", model=model_offering(library, tool))
        deadline = time.monotonic() + ceiling
        while time.monotonic() < deadline:
            try:
                stored = client.get(f"/api/v1/chats/{turn.chat_id}")
            except httpx.TimeoutException:
                continue
            messages = stored.json()["chat"]["history"]["messages"]
            if messages.get(turn.assistant_message_id, {}).get("done"):
                break
            time.sleep(0.2)
        else:
            return None
    sent_back = launched.upstream.chat_requests()[-1]["messages"]
    return [entry["content"] for entry in sent_back if entry["role"] == "tool"][-1]


# ── broad: one budget per command, shared by its searches, full again next time ──


def slow_pipeline() -> str:
    first = f'grep -E "{SLOW_PATTERN}" slow.txt'
    rest = [f'grep -E "{SLOW_PATTERN}"'] * (PIPELINE_GREPS - 1)
    return " | ".join([first, *rest, "wc"])


def test_the_searches_of_one_command_share_a_single_budget(run):
    """Each grep alone fits the budget; together they must not get one each."""
    alone = kb_exec(run, f'grep -c -E "{SLOW_PATTERN}" slow.txt')
    assert alone.endswith(f": {SLOW_LINES}"), f"one slow grep did not fit the budget: {alone}"

    piped = kb_exec(run, slow_pipeline())

    assert BUDGET_SPENT in piped, (
        f"{PIPELINE_GREPS} slow greps in one command all ran to the end, so a pipeline multiplies "
        f"the budget by its length and one command can hold the worker that long (#27471): "
        f"{piped[:200]}"
    )


def test_the_next_command_starts_with_a_full_budget(run):
    """A spent budget must not leak into the next tool call and refuse everything."""
    spent = kb_exec(run, slow_pipeline())
    assert BUDGET_SPENT in spent, spent[:200]

    after = kb_exec(run, 'grep -E "alpha|epsilon" lines.txt')

    assert after == "1: alpha beta\n4: epsilon", (
        f"a later command inherited the previous command's spent budget (#27471): {after}"
    )


# ── nearby: ordinary searching still behaves ─────────────────────────────


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('grep -E "alpha|gamma" lines.txt', ["1: alpha beta", "2: gamma delta"]),
        ('grep -i -E "alpha" lines.txt', ["1: alpha beta", "3: ALPHA omega"]),
        ('grep "beta" lines.txt', ["1: alpha beta"]),
        ('grep -i "OMEGA" lines.txt', ["3: ALPHA omega"]),
        (r'grep "alpha\|epsilon" lines.txt', ["1: alpha beta", "4: epsilon"]),
    ],
    ids=["alternation", "regex-ignoring-case", "literal", "literal-ignoring-case", "escaped-pipe"],
)
def test_ordinary_searches_return_the_right_lines(run, command, expected):
    assert kb_exec(run, command).splitlines() == expected


@pytest.mark.parametrize("pattern", ["(unclosed", "*nothing-to-repeat", "[a-"])
def test_an_invalid_regex_comes_back_as_a_readable_error(run, pattern):
    """The model gets an error it can act on, not a failed tool call."""
    from_shell = kb_exec(run, f'grep -E "{pattern}" lines.txt')
    from_tool = json.loads(run("grep_knowledge_files", {"pattern": f"{pattern}|zzz"}))

    assert from_shell.startswith("Invalid"), from_shell
    assert from_tool.get("error", "").startswith("Invalid"), from_tool


# ── narrow: the catastrophic pattern answers instead of running forever ──
# Last in the module: after a regression the instance answers nothing more.


@pytest.mark.parametrize("tool", ["kb_exec", "grep_knowledge_files"])
def test_a_catastrophic_pattern_answers_without_stalling(libraries, tool):
    library = libraries[tool]
    if tool == "kb_exec":
        arguments = {"command": f'grep -E "{CATASTROPHIC_PATTERN}" trap.txt'}
    else:
        arguments = {"pattern": CATASTROPHIC_PATTERN, "file_id": library["trap"]}

    result = answered_within(library, tool, arguments, ANSWER_CEILING_SECONDS)

    assert result is not None, (
        f"matching {CATASTROPHIC_PATTERN!r} against a {len(NON_MATCHING_LINE)}-character line "
        f"was still running after {ANSWER_CEILING_SECONDS:g}s, so a single model-issued "
        "knowledge search holds the worker and every other user waits (#27471)"
    )
    assert "No matches" in result, result
