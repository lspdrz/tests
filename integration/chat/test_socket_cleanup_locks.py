"""The socket cleanup tasks share their work across instances through Redis locks, and keep at it.

With `WEBSOCKET_MANAGER=redis` every instance runs two cleanup loops, each behind a Redis lock: the
session cleanup reaps socket sessions whose tab stopped sending heartbeats, the usage cleanup
drops models no tab reports using any more. Fixes from 0.11.0 to 0.11.2 made them hold up:

* `bf35f64` and 5586964bb (PR #28834): a loop returned for good when it lost the lock race, a
  renew failed or a sweep raised, so one bad moment stopped the cleanup cluster-wide.
* 846ba80: the lock renewed with `SET XX` and released with GET then DEL, taking over or
  deleting a lock another instance held.
* 939bcdb79e (#27762): the session cleanup slept its whole cycle without renewing, so its lock
  lapsed mid-cycle and another instance started a second sweep.
* d7674c517 (PR #28835): the sweep read the whole pool with one HKEYS and did a GET and a DEL per
  session. It now walks HSCAN batches and deletes each batch in one HDEL. The members of a
  channel room are this worker's own sockets, read without touching the pool.

One instance runs on a real Redis with a 4 second lock timeout; `harness.redis_monitor` records
every command it sends. Before it boots, the test holds both locks as "another instance" would
and fills the session pool with stale and live sessions and the usage pool with an idle and a
busy model; later tests take a lock away, break the pool and look at what the instance does.

Twin of unit/chat/test_socket_runtime.py, unit/chat/test_socket_and_redis_runtime.py and
unit/chat/test_socket_cluster_and_packets.py for these fixes.

Discriminates: passes on dev ef67cc3fa. It fails with a cleanup loop that returns once it lost
the lock race twice (the held-lock tests), that returns on a failed renew (the stolen-lock
tests) or on a failed sweep (the broken-pool test), with the renew done as `SET XX` (the stolen
lock is overwritten), with the session cleanup sleeping its whole cycle between renews (the lock
lapses), with the sweep deleting session by session or reading the pool with HKEYS, and with
room members looked up in the session pool.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Callable, Iterator

import pytest
import redis

from harness import backends
from harness.actors import create_user
from harness.channel_quotes import enable_channels, group_channel, post_message
from harness.redis_monitor import RedisRecorder, recording
from harness.socket_client import connected

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

LOCK_TIMEOUT = 4
USAGE_LOCK = "open-webui:usage_cleanup_lock"
SESSION_LOCK = "open-webui:session_cleanup_lock"
SESSION_POOL = "open-webui:session_pool"
USAGE_POOL = "open-webui:usage_pool"
SOMEONE_ELSE = "another-instance"
STALE_SESSIONS = 450
LIVE_SESSION = "live-session"
POOL_READS = ("HGET", "HGETALL", "HKEYS", "HVALS", "HSCAN", "HEXISTS")


class HeldLock:
    """A cleanup lock held by another instance, renewed until `release()`."""

    def __init__(self, client: redis.Redis, key: str):
        self.client, self.key = client, key
        self.stopped = threading.Event()
        self.client.set(key, SOMEONE_ELSE, ex=LOCK_TIMEOUT)
        self.renewer = threading.Thread(target=self._renew, daemon=True)
        self.renewer.start()

    def _renew(self) -> None:
        while not self.stopped.wait(1):
            self.client.set(self.key, SOMEONE_ELSE, ex=LOCK_TIMEOUT)

    def release(self) -> None:
        if not self.stopped.is_set():
            self.stopped.set()
            self.renewer.join()
            if self.client.get(self.key) == SOMEONE_ELSE:
                self.client.delete(self.key)


def _wait_until(condition: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.2)
    return condition()


def _session(last_seen_at: int) -> str:
    return json.dumps({"id": "user-gone", "name": "Gone", "last_seen_at": last_seen_at})


@pytest.fixture(scope="module")
def redis_url() -> Iterator[str]:
    with backends.redis_server() as url:
        yield url


@pytest.fixture(scope="module")
def store(redis_url) -> Iterator[redis.Redis]:
    client = redis.Redis.from_url(redis_url, decode_responses=True)
    yield client
    client.close()


@pytest.fixture(scope="module")
def monitor(redis_url) -> Iterator[RedisRecorder]:
    with recording(redis_url) as recorder:
        yield recorder


@pytest.fixture(scope="module")
def held_locks(store) -> Iterator[dict[str, HeldLock]]:
    locks = {key: HeldLock(store, key) for key in (USAGE_LOCK, SESSION_LOCK)}
    yield locks
    for lock in locks.values():
        lock.release()


@pytest.fixture(scope="module")
def filled_pools(store) -> None:
    long_ago = int(time.time()) - 3600
    sessions = {f"stale-{index}": _session(long_ago) for index in range(STALE_SESSIONS)}
    store.hset(SESSION_POOL, mapping={**sessions, LIVE_SESSION: _session(int(time.time()) + 3600)})
    busy = {"sid-busy": {"updated_at": int(time.time()) + 3600}}
    idle = {"sid-idle": {"updated_at": long_ago}}
    store.hset(USAGE_POOL, mapping={"busy-model": json.dumps(busy), "idle-model": json.dumps(idle)})


@pytest.fixture(scope="module")
def cleanup_instance(instance_with, redis_url, monitor, held_locks, filled_pools):
    return instance_with(
        {
            "REDIS_URL": redis_url,
            "WEBSOCKET_MANAGER": "redis",
            "WEBSOCKET_REDIS_LOCK_TIMEOUT": str(LOCK_TIMEOUT),
        }
    )


def _takes(monitor: RedisRecorder, key: str) -> int:
    return sum("NX" in (arg.upper() for arg in args) for args in monitor.sent("SET", key))


def _held_by_the_instance(store: redis.Redis, key: str) -> bool:
    return store.get(key) not in (None, SOMEONE_ELSE)


@pytest.mark.parametrize("key", [USAGE_LOCK, SESSION_LOCK], ids=["usage", "session"])
def test_a_lock_held_elsewhere_is_asked_for_until_it_frees(
    cleanup_instance, store, monitor, held_locks, key
):
    assert _wait_until(lambda: _takes(monitor, key) >= 3, timeout=20), (
        f"the instance stopped asking for {key} after losing the race {_takes(monitor, key)} times"
    )
    assert store.get(key) == SOMEONE_ELSE, "the instance took over a lock another instance held"

    held_locks[key].release()

    assert _wait_until(lambda: _held_by_the_instance(store, key), timeout=15), (
        f"the instance never took {key} once it was free"
    )


def test_stale_sessions_are_reaped_in_batches(cleanup_instance, store, monitor, held_locks):
    held_locks[SESSION_LOCK].release()

    assert _wait_until(lambda: store.hlen(SESSION_POOL) == 1, timeout=20), (
        f"{store.hlen(SESSION_POOL) - 1} stale sessions were never reaped"
    )
    scans = len(monitor.sent("HSCAN", SESSION_POOL))
    deletes = monitor.sent("HDEL", SESSION_POOL)
    assert scans, "the sweep never walked the pool with HSCAN"
    assert len(deletes) <= scans, (
        f"{STALE_SESSIONS} stale sessions took {len(deletes)} HDEL calls, not one per batch"
    )
    whole_reads = monitor.sent("HKEYS", SESSION_POOL) + monitor.sent("HGETALL", SESSION_POOL)
    assert not whole_reads, "the sweep read the whole session pool in one call"
    assert not monitor.sent("HGET", SESSION_POOL), "the sweep read the pool session by session"
    assert store.hkeys(SESSION_POOL) == [LIVE_SESSION], "a session still heartbeating was reaped"


def test_the_session_cleanup_keeps_its_lock_between_sweeps(cleanup_instance, store, held_locks):
    held_locks[SESSION_LOCK].release()
    assert _wait_until(lambda: _held_by_the_instance(store, SESSION_LOCK), timeout=15)
    holder = store.get(SESSION_LOCK)

    samples = []
    deadline = time.monotonic() + LOCK_TIMEOUT * 2.5
    while time.monotonic() < deadline:
        samples.append(store.get(SESSION_LOCK))
        time.sleep(0.25)

    assert set(samples) == {holder}, (
        "the session cleanup lock lapsed while its holder waited for the next sweep (#27762)"
    )


@pytest.mark.parametrize("key", [USAGE_LOCK, SESSION_LOCK], ids=["usage", "session"])
def test_a_lock_taken_over_elsewhere_is_left_alone_and_asked_for_again(
    cleanup_instance, store, held_locks, key
):
    held_locks[key].release()
    assert _wait_until(lambda: _held_by_the_instance(store, key), timeout=15)

    store.set(key, SOMEONE_ELSE, ex=LOCK_TIMEOUT + 1)  # another instance took it after a lapse
    kept = []
    deadline = time.monotonic() + LOCK_TIMEOUT
    while time.monotonic() < deadline:
        kept.append(store.get(key))
        time.sleep(0.25)

    assert set(kept) <= {SOMEONE_ELSE, None}, (
        f"the failed renew wrote the instance's id over the lock another instance now holds: {kept}"
    )
    assert _wait_until(lambda: _held_by_the_instance(store, key), timeout=15), (
        f"after a failed renew the instance never asked for {key} again"
    )


def test_a_model_no_tab_reports_any_more_drops_out_of_the_usage_pool(
    cleanup_instance, store, held_locks
):
    held_locks[USAGE_LOCK].release()

    assert _wait_until(lambda: not store.hexists(USAGE_POOL, "idle-model"), timeout=15), (
        "the usage cleanup never dropped a model idle for an hour"
    )
    assert store.hexists(USAGE_POOL, "busy-model"), "a model still in use was dropped"


def test_a_sweep_that_fails_is_retried(cleanup_instance, store, held_locks):
    held_locks[SESSION_LOCK].release()
    assert _wait_until(lambda: _held_by_the_instance(store, SESSION_LOCK), timeout=15)
    offset = cleanup_instance.log_size()
    store.delete(SESSION_POOL)
    store.set(SESSION_POOL, "not a hash")  # every sweep now fails with WRONGTYPE
    store.set(SESSION_LOCK, SOMEONE_ELSE, ex=1)  # so the instance takes it again and sweeps

    failed = _wait_until(
        lambda: "Session pool cleanup failed" in cleanup_instance.log_since(offset), timeout=20
    )
    assert failed, "the broken pool never failed a sweep; the test set it up wrongly"
    store.delete(SESSION_POOL)
    store.hset(SESSION_POOL, mapping={"stale-after-failure": _session(0)})

    assert _wait_until(lambda: store.hlen(SESSION_POOL) == 0, timeout=20), (
        "the session cleanup stopped for good after one failed sweep"
    )


def test_a_channel_post_reads_its_room_from_this_workers_sockets(
    cleanup_instance, preserve, monitor
):
    preserve("admin_config", on=cleanup_instance)
    owner = create_user(cleanup_instance, role="admin")
    member = create_user(cleanup_instance)
    enable_channels(owner)
    channel_id = group_channel(owner, member)
    with connected(owner) as socket:
        socket.call("join-channels", {"auth": {"token": owner.token}})
        monitor.clear()

        post_message(owner, channel_id, "hello room")
        reads = {command: len(monitor.sent(command, SESSION_POOL)) for command in POOL_READS}

    assert not any(reads.values()), (
        f"posting to a channel read the session pool for its room members: {reads}"
    )
