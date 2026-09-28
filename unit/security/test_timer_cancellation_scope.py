"""Regression: marking a chat as read cancelled every user's timers on that chat.

open-webui 0.11.0 fix `e140d8f3c` (#27472): `cancel_timers_for_chat` selected pending timers on
the parent chat without filtering on their owner, so a read or a message by someone else
cancelled the owner's timers. The fix makes `user_id` a required parameter and filters on
`Chat.user_id`.

The behaviour is pinned over HTTP by integration/security/test_timer_cancellation_scope.py. This
keeps an `ast` audit that every caller in the backend passes the acting user, which also guards
a call site upstream adds later.

Discriminates: passes on dev `ef67cc3fa`; a call to `cancel_timers_for_chat` with only the chat
and the event fails it.
"""

from __future__ import annotations

import ast

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
