"""Socket/runtime guards from v0.11.0 to v0.11.1 that no request can observe.

The rest of these fixes are pinned from outside: the cleanup locks in
integration/chat/test_socket_cleanup_locks.py, the task command listener in
integration/chat/test_redis_task_commands.py, the event call timeout in
integration/chat/test_socket_runtime.py and the shared model pool signature in
integration/models/test_shared_model_pool_cache.py. Two stay here:

- 108+109 (211906d79, PR #28053): the lifespan created its background coroutines without keeping
  a reference, so the event loop could collect them mid-run. Garbage collection of a pending task
  is event-loop timing no request can stage, so the stored and cancelled handles are audited.
- 152 (bf3a58dbcd, #28909): while Redis stays down, the task command listener backs off between
  reconnects. The attempts never reach a Redis that is down, so only a patched clock can count
  the waits.

Discriminates: passes on bbfa876af; fails with a lifespan task handle dropped or never cancelled,
and with the reconnect wait not doubling.
"""

from __future__ import annotations

import ast
import asyncio
from unittest.mock import create_autospec, patch

import pytest
import redis
import redis.asyncio
import redis.asyncio.client
from fastapi import FastAPI

pytestmark = pytest.mark.regression

LOOP_DRIVE_TIMEOUT = 5


class _LoopExit(BaseException):
    """Stops a production `while True` loop after a fixed number of calls."""


def _sleep_until(limit: int, sleeps: list[float]):
    async def sleep(delay=0, *args, **kwargs):
        sleeps.append(delay)
        if len(sleeps) >= limit:
            raise _LoopExit

    return sleep


async def _drive_until_exit(coroutine) -> None:
    with pytest.raises(_LoopExit):
        await asyncio.wait_for(coroutine, timeout=LOOP_DRIVE_TIMEOUT)


@pytest.fixture(scope="session")
def tasks_module(owui_module):
    return owui_module("open_webui.tasks")


# --- 108 + 109: the lifespan keeps and cancels its background task handles ----------------


def _stored_and_cancelled_tasks(main_tree: ast.Module) -> tuple[dict[str, str], set[str]]:
    """`{coroutine: app.state attribute}` for stored `asyncio.create_task` handles, and the
    handles `.cancel()` is called on."""
    stored: dict[str, str] = {}
    cancelled: set[str] = set()
    for node in ast.walk(main_tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if ast.unparse(call.func) == "asyncio.create_task" and call.args:
                targets = [ast.unparse(target) for target in node.targets]
                handles = [target for target in targets if target.startswith("app.state.")]
                if handles and isinstance(call.args[0], ast.Call):
                    stored[ast.unparse(call.args[0].func)] = handles[0]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "cancel":
                cancelled.add(ast.unparse(node.func.value))
    return stored, cancelled


@pytest.mark.parametrize(
    "coroutine",
    [
        "periodic_usage_pool_cleanup",
        "periodic_session_pool_cleanup",
        "scheduler_worker_loop",
        "redis_task_command_listener",
    ],
)
def test_the_lifespan_keeps_and_cancels_each_background_task(open_webui_backend, coroutine):
    main_py = open_webui_backend / "open_webui" / "main.py"
    stored, cancelled = _stored_and_cancelled_tasks(ast.parse(main_py.read_text("utf-8")))

    assert coroutine in stored, (
        f"the {coroutine} task handle is not kept on app.state, so the loop may collect it"
    )
    assert stored[coroutine] in cancelled, f"{stored[coroutine]} is never cancelled on shutdown"


# --- 152: the task command listener backs off while Redis is down ------------------------


async def _stream_that_ends():
    yield {"type": "subscribe", "data": 1}


def _redis_whose_pubsubs(fail_subscribe: bool, limit: int):
    """A specced async client handing out `limit` pubsubs, then stopping the loop."""
    pubsubs: list = []

    def open_pubsub():
        if len(pubsubs) >= limit:
            raise _LoopExit
        pubsub = create_autospec(redis.asyncio.client.PubSub, instance=True)
        pubsub.listen.side_effect = _stream_that_ends
        if fail_subscribe:
            pubsub.subscribe.side_effect = redis.exceptions.ConnectionError("cache is down")
        pubsubs.append(pubsub)
        return pubsub

    client = create_autospec(redis.asyncio.Redis, instance=True)
    client.pubsub.side_effect = open_pubsub
    return client, pubsubs


def _listener_app(client) -> FastAPI:
    app = FastAPI()
    app.state.redis = client
    return app


@pytest.mark.asyncio
async def test_reconnects_back_off_while_the_cache_stays_down(tasks_module):
    client, _ = _redis_whose_pubsubs(fail_subscribe=True, limit=3)
    sleeps: list[float] = []
    with patch.object(asyncio, "sleep", _sleep_until(20, sleeps)):
        await _drive_until_exit(tasks_module.redis_task_command_listener(_listener_app(client)))

    first = tasks_module.REDIS_PUBSUB_RECONNECT_INTERVAL
    assert sleeps == [first, first * 2, first * 4]
    assert max(sleeps) <= tasks_module.REDIS_PUBSUB_MAX_RECONNECT_INTERVAL
