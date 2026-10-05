"""Journey: a stored chat opens on the newest message of the branch it was left on.

A chat whose current message has later descendants (an import, a crash that left `currentId`
behind, an older client) is repaired on read to the end of that branch: at each step the last
child that is really there, really a reply to its parent and has a role. Links that are stale
(an id with no message), malformed (not a string, no role, a parent that is someone else) or
already visited are skipped, and a branch whose links loop ends instead of spinning. A current
message at the end of its branch, or on an older branch, stays where it is.

Discriminates: in a backend copy that repairs only chats with a context summary (the change
undone), seven of the nine tests go red (every stranded chat, stale link and loop). With only the
child filter back to "the id exists in the history", the role-less child, the reply to someone
else and the looping branch go red.
"""

from __future__ import annotations

import httpx
import pytest

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def message(message_id: str, parent_id: str | None, children: list, role: str = "user", **extra):
    return {
        "id": message_id,
        "parentId": parent_id,
        "childrenIds": children,
        "role": role,
        "content": f"{role} {message_id}",
        "timestamp": 1_700_000_000,
        **extra,
    }


def stored_current_id(client: httpx.Client, current_id: str, *messages: dict) -> str:
    history = {"currentId": current_id, "messages": {entry["id"]: entry for entry in messages}}
    created = client.post(
        "/api/v1/chats/new", json={"chat": {"title": "Left behind", "history": history}}
    )
    assert created.status_code == 200, created.text
    read = client.get(f"/api/v1/chats/{created.json()['id']}")
    assert read.status_code == 200, read.text
    return read.json()["chat"]["history"]["currentId"]


@pytest.fixture
def client(make_user):
    with make_user().client() as owner_client:
        yield owner_client


def test_a_chat_left_on_an_early_message_opens_on_the_newest_one(client):
    opened_on = stored_current_id(
        client,
        "q1",
        message("q1", None, ["a1"]),
        message("a1", "q1", ["q2"], "assistant"),
        message("q2", "a1", ["a2"]),
        message("a2", "q2", [], "assistant"),
    )
    assert opened_on == "a2", "a current message with later replies was not moved to the newest"


def test_the_last_child_wins_at_each_branch_point(client):
    opened_on = stored_current_id(
        client,
        "q1",
        message("q1", None, ["a1", "a1b"]),
        message("a1", "q1", ["q2"], "assistant"),
        message("q2", "a1", [], "assistant"),
        message("a1b", "q1", ["q3"], "assistant"),
        message("q3", "a1b", [], "user"),
    )
    assert opened_on == "q3"


def test_a_message_on_an_older_branch_stays_on_that_branch(client):
    opened_on = stored_current_id(
        client,
        "a1",
        message("q1", None, ["a1", "a1b"]),
        message("a1", "q1", ["q2"], "assistant"),
        message("q2", "a1", [], "user"),
        message("a1b", "q1", ["q3"], "assistant"),
        message("q3", "a1b", [], "user"),
    )
    assert opened_on == "q2"


def test_a_message_at_the_end_of_its_branch_stays_put(client):
    opened_on = stored_current_id(
        client,
        "a1",
        message("q1", None, ["a1"]),
        message("a1", "q1", [], "assistant"),
    )
    assert opened_on == "a1"


def test_a_link_to_a_message_that_is_not_there_is_skipped(client):
    opened_on = stored_current_id(
        client,
        "q1",
        message("q1", None, ["a1", "ghost"]),
        message("a1", "q1", [], "assistant"),
    )
    assert opened_on == "a1", "a stale child link stranded the reader on the parent"


def test_a_child_link_that_is_not_an_id_is_skipped(client):
    opened_on = stored_current_id(
        client,
        "q1",
        message("q1", None, ["a1", 7, None]),
        message("a1", "q1", [], "assistant"),
    )
    assert opened_on == "a1"


def test_a_child_without_a_role_is_skipped(client):
    opened_on = stored_current_id(
        client,
        "q1",
        message("q1", None, ["a1", "blank"]),
        message("a1", "q1", [], "assistant"),
        {"id": "blank", "parentId": "q1", "childrenIds": []},
    )
    assert opened_on == "a1"


def test_a_child_that_is_a_reply_to_someone_else_is_skipped(client):
    opened_on = stored_current_id(
        client,
        "q1",
        message("q1", None, ["a1", "elsewhere"]),
        message("a1", "q1", [], "assistant"),
        message("elsewhere", "other", [], "assistant"),
        message("other", None, ["elsewhere"]),
    )
    assert opened_on == "a1", "a message that is no reply to the current one was opened"


def test_a_branch_whose_links_loop_ends_on_the_last_new_message(client):
    opened_on = stored_current_id(
        client,
        "q1",
        message("q1", None, ["a1"]),
        message("a1", "q1", ["q1"], "assistant"),
    )
    assert opened_on == "a1"
