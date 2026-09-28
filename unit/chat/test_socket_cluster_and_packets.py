"""The task command listener initializes a Redis Cluster client before it subscribes.

a5ea8b0b8 (PR #29165, issue #19840): `redis_task_command_listener` called `redis.pubsub()`
straight away. A `RedisCluster` client of the redis-py releases the fix was made against cannot
route a subscribe until `initialize()` has filled its slot cache, so the listener never subscribed
and a stop sent from another instance was dropped while the reply ran to the end. The pinned
redis-py 8.0.1 subscribes on an uninitialized cluster client as well, so a real cluster shows no
difference: integration/chat/test_redis_task_commands.py stops a reply on a one-node cluster as
journey coverage, and this guard stays with a cluster stand-in that refuses to route first.

The packet and session reaper fixes of the same release are pinned from outside in
integration/notes/test_binary_note_updates.py and integration/chat/test_socket_cleanup_locks.py.

The cluster client is a `create_autospec` stand-in, and every loop is bounded by construction: a
patched `asyncio.sleep` or the stand-in raises `_LoopExit` (a `BaseException` the production
`except Exception` cannot swallow) after a fixed number of calls, inside `asyncio.wait_for`.

Discriminates: passes on bbfa876af; fails with the `initialize()` call removed from the listener.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import create_autospec, patch

import pytest
import redis
import redis.asyncio.client
import redis.asyncio.cluster
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


# --- a5ea8b0b8: stop-generation across instances on a Redis Cluster ----------------------


def _cluster_carrying(messages: list[dict], subscriptions: int):
    """A specced RedisCluster that cannot route a subscribe before `initialize()`."""
    cluster = create_autospec(redis.asyncio.cluster.RedisCluster, instance=True)
    state = {"initialized": False, "pubsubs": []}

    async def initialize():
        state["initialized"] = True

    async def stream():
        for message in messages:
            yield message

    def open_pubsub():
        if len(state["pubsubs"]) >= subscriptions:
            raise _LoopExit
        if not state["initialized"]:
            raise redis.exceptions.RedisClusterException("no slot cache yet")
        pubsub = create_autospec(redis.asyncio.client.PubSub, instance=True)
        pubsub.listen.side_effect = stream
        state["pubsubs"].append(pubsub)
        return pubsub

    cluster.initialize.side_effect = initialize
    cluster.pubsub.side_effect = open_pubsub
    return cluster, state["pubsubs"]


async def _listen_on(tasks_module, cluster) -> None:
    app = FastAPI()
    app.state.redis = cluster
    with patch.object(asyncio, "sleep", _sleep_until(20, [])):
        await _drive_until_exit(tasks_module.redis_task_command_listener(app))


def _command(action: str, task_id: str) -> dict:
    return {"type": "message", "data": json.dumps({"action": action, "task_id": task_id})}


@pytest.mark.asyncio
async def test_a_stop_sent_through_the_cluster_cancels_the_local_task(tasks_module):
    running = asyncio.get_running_loop().create_task(asyncio.Event().wait())
    cluster, pubsubs = _cluster_carrying([_command("stop", "task-1")], subscriptions=1)

    with patch.dict(tasks_module.tasks, {"task-1": running}):
        await _listen_on(tasks_module, cluster)
        await asyncio.wait([running], timeout=0.5)

    assert running.cancelled(), "the listener never subscribed on the cluster client (#19840)"
    pubsubs[0].subscribe.assert_awaited_once_with(tasks_module.REDIS_PUBSUB_CHANNEL)


@pytest.mark.asyncio
async def test_every_reconnect_initializes_the_cluster_again(tasks_module):
    cluster, pubsubs = _cluster_carrying([], subscriptions=3)

    await _listen_on(tasks_module, cluster)

    assert len(pubsubs) == 3
    assert cluster.initialize.await_count == cluster.pubsub.call_count, (
        "a reconnect subscribed without refilling the cluster's slot cache first"
    )
