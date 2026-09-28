"""Socket and Redis runtime guards from v0.11.0 that no request can observe.

The rest of these fixes are pinned from outside: the task id and the temporary-chat tool gating
in integration/chat/test_socket_and_redis_runtime.py, the cleanup locks in
integration/chat/test_socket_cleanup_locks.py, the socket timeout default in
integration/chat/test_redis_task_commands.py and the Sentinel retries in
integration/resilience/test_redis_sentinel_failover.py. What stays here:

- 🧹 846ba80: `RedisLock` released with GET then DEL, so a takeover landing between the two
  deleted another instance's lock. The window lies inside one release call, which no outside
  client can hit on cue; the renew half is pinned from outside.
- 🧊 fc4906c: the connection cache key omitted `redis_cluster`, so a cluster and a single-server
  request for the same address shared one client. Only a deployment naming one address as both
  reaches it, and no request can tell which client answered.
- 🫥 d484a2a (issue #27432): the status emitter persisted status updates of a `temporary:` chat.
  The write looks up a chats row that cannot exist, so nothing a request or the database shows
  changes.

Discriminates: passes on bbfa876af; fails with `RedisLock` releasing with GET then DEL,
`redis_cluster` dropped from the cache key, and the status emitter keyed on `local:` only.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, create_autospec, patch

import pytest
import redis
import redis.cluster

pytestmark = pytest.mark.regression

SAVED_CHAT_ID = "8e2b5f0c-2c3f-4d1e-9a77-0a1b2c3d4e5f"


@pytest.fixture(scope="session")
def socket_main(owui_module):
    return owui_module("open_webui.socket.main")


@pytest.fixture(scope="session")
def socket_utils(owui_module):
    return owui_module("open_webui.socket.utils")


@pytest.fixture(scope="session")
def redis_utils(owui_module):
    return owui_module("open_webui.utils.redis")


# --- 🧹 846ba80: RedisLock only releases a lock it holds ------------------------------------


def _redis_holding(lock_name: str, holder: str):
    """A specced sync Redis client holding one key, with EVAL applied atomically."""
    store = {lock_name: holder}
    client = create_autospec(redis.Redis, instance=True)
    client.get.side_effect = store.get
    client.set.side_effect = lambda name, value, nx=False, xx=False, ex=None, **kw: (
        None
        if (nx and name in store) or (xx and name not in store)
        else store.update({name: value}) or True
    )
    client.delete.side_effect = lambda *names: sum(
        store.pop(name, None) is not None for name in names
    )

    def run_script(script, numkeys, *keys_and_args):
        key, owner = keys_and_args[0], keys_and_args[1]
        if store.get(key) != owner:
            return 0
        return 1 if "'expire'" in script else int(store.pop(key, None) is not None)

    client.eval.side_effect = run_script
    return client, store


def _lock_as(socket_utils, client, owner: str):
    with patch.object(socket_utils, "get_redis_connection", return_value=client):
        lock = socket_utils.RedisLock(
            redis_url="redis://127.0.0.1:1/0", lock_name="cleanup-lock", timeout_secs=30
        )
    lock.lock_id = owner
    return lock


def test_release_does_not_delete_a_lock_another_instance_holds(socket_utils):
    client, store = _redis_holding("cleanup-lock", "other-instance")
    client.get.side_effect = lambda name: "this-instance"  # read before a takeover lands

    _lock_as(socket_utils, client, "this-instance").release_lock()

    assert store == {"cleanup-lock": "other-instance"}


# --- 🧊 fc4906c: the connection factory ----------------------------------------

REDIS_URL = "redis://127.0.0.1:1/0"


@pytest.fixture
def empty_connection_cache(redis_utils):
    with patch.object(redis_utils, "_CONNECTION_POOL", {}):
        yield


def test_cluster_and_single_server_connections_are_not_shared(redis_utils, empty_connection_cache):
    cluster_client = create_autospec(redis.cluster.RedisCluster, instance=True)
    # a real RedisCluster fills its slot cache on construction, so that call is the boundary
    with patch.object(redis.cluster.RedisCluster, "from_url", return_value=cluster_client):
        cluster = redis_utils.get_redis_connection(REDIS_URL, redis_cluster=True)
    single = redis_utils.get_redis_connection(REDIS_URL, redis_cluster=False)

    assert cluster is cluster_client
    assert isinstance(single, redis.Redis), "the single-server caller got the cluster client"
    assert redis_utils.get_redis_connection(REDIS_URL) is single


# --- 🫥 d484a2a / issue #27432: a temporary chat's status is not persisted ---------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chat_id", "writes"),
    [("temporary:abc", 0), ("local:abc", 0), (SAVED_CHAT_ID, 1)],
    ids=["temporary", "legacy-local", "saved"],
)
async def test_status_updates_are_persisted_for_saved_chats_only(socket_main, chat_id, writes):
    chats = create_autospec(type(socket_main.Chats), instance=True)
    with (
        patch.object(socket_main, "Chats", chats),
        patch.object(socket_main.sio, "emit", AsyncMock()),
    ):
        emitter = await socket_main.get_event_emitter(
            {"user_id": "u-1", "chat_id": chat_id, "message_id": "m-1"}
        )
        await emitter({"type": "status", "data": {"description": "working"}})

    assert chats.add_message_status_to_chat_by_id_and_message_id.await_count == writes
