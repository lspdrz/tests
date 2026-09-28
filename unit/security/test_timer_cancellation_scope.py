"""Regression: marking a chat as read cancelled every user's timers on that chat.

open-webui 0.11.0 fix `e140d8f3c` (#27472): `cancel_timers_for_chat` selected pending timers on
the parent chat without filtering on their owner, so a read or a message by someone else
cancelled the owner's timers. The fix makes `user_id` a required parameter and filters on
`Chat.user_id`.

The behaviour is pinned over HTTP by integration/security/test_timer_cancellation_scope.py. This
keeps an `ast` audit that every caller in the backend passes the acting user, which also guards
a call site upstream adds later, and the query's refusal to cancel a timer that is already
running, which stays unit because no request can land a read inside a timer's run reliably.

Discriminates: passes on dev `ef67cc3fa`; a call to `cancel_timers_for_chat` with only the chat
and the event fails the audit, and dropping both the pending-status and the due-time filters
from the query cancels the running timer.
"""

from __future__ import annotations

import ast
import inspect
import time
from uuid import uuid4

import pytest

pytestmark = pytest.mark.regression

CANCEL_TARGET = "cancel_timers_for_chat"


def test_every_caller_passes_the_acting_user(open_webui_backend):
    """Guards the next unscoped sweep, wherever in the backend it is called from."""
    backend = open_webui_backend / "open_webui"
    call_sites, unscoped = [], []
    for path in sorted(backend.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            if CANCEL_TARGET not in (
                getattr(node.func, "id", None),
                getattr(node.func, "attr", None),
            ):
                continue
            site = f"{path.relative_to(backend)}:{node.lineno}"
            call_sites.append(site)
            if len(node.args) < 3 and not any(kw.arg == "user_id" for kw in node.keywords):
                unscoped.append(site)

    assert call_sites, f"nothing calls {CANCEL_TARGET} any more: retarget this audit"
    assert unscoped == [], (
        f"{CANCEL_TARGET} is called without the acting user at {unscoped}: that call cancels "
        "the pending timers of every user on the chat (#27472)"
    )


@pytest.fixture(scope="module")
def timers(owui_module):
    owui_module("open_webui.config")  # runs the migrations, so the `chat` table exists
    try:
        return owui_module("open_webui.utils.timers")
    except pytest.skip.Exception:
        pytest.fail("open_webui.utils.timers is gone: retarget these tests at the timer module")


@pytest.fixture
def ids():
    """Unique id prefix so tests sharing the scratch database cannot collide."""
    return uuid4().hex[:12]


def _timer(timers, timer_id, owner_id, parent_chat_id, cancel_on=("chat.read",), status="pending"):
    now = int(time.time())
    due_at = time.time_ns()
    row = timers.Chat(
        id=timer_id,
        user_id=owner_id,
        title=f"Timer: {timer_id}",
        chat={},
        meta={
            "internal": True,
            "type": "timer",
            "parent_chat_id": parent_chat_id,
            "status": status,
            "cancel_on": list(cancel_on),
            "timer_at": due_at,
        },
        created_at=now,
        updated_at=now,
    )
    # 0.11.1 moved the due time onto a column the queries key off; claiming clears it
    if "timer_at" in timers.Chat.__table__.columns:
        row.timer_at = due_at if status == "pending" else None
    return row


async def _seed(timers, rows):
    async with timers.get_async_db() as db:
        for row in rows:
            db.add(row)
        await db.commit()


async def _statuses(timers, timer_ids):
    async with timers.get_async_db() as db:
        return {
            timer_id: (await db.get(timers.Chat, timer_id)).meta.get("status")
            for timer_id in timer_ids
        }


async def _cancel_as(timers, parent_chat_id, event, user_id):
    """The pre-fix signature has no `user_id`; calling it without keeps the failure behavioural."""
    arguments = {"parent_chat_id": parent_chat_id, "event": event}
    if "user_id" in inspect.signature(timers.cancel_timers_for_chat).parameters:
        arguments["user_id"] = user_id
    await timers.cancel_timers_for_chat(**arguments)


@pytest.mark.asyncio
async def test_a_running_timer_is_not_cancelled(timers, ids):
    parent_chat_id = f"{ids}-read-chat"
    running = _timer(timers, f"{ids}-running", "alice", parent_chat_id, status="running")
    await _seed(timers, [running])

    await _cancel_as(timers, parent_chat_id, "chat.read", "alice")

    assert await _statuses(timers, [running.id]) == {running.id: "running"}
