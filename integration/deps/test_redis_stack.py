"""Dependency smoke: what two instances share through Redis, driven through redis-py.

With `REDIS_URL` and `WEBSOCKET_MANAGER=redis` Open WebUI keeps its cross-instance state in Redis
through redis-py's asyncio client: a signed-out token is stored as revoked with `SET ... EX` for
the rest of its lifetime (worked out on pytz's UTC clock), the socket session pool is a Redis hash
every instance reads, running replies are listed in a hash and a set per chat written through
pipelines, and a note's live edits are a Redis list (`RPUSH`, `LRANGE`) any instance can rebuild
the document from. python-socketio's `AsyncRedisManager` carries emits and disconnects between the
instances over pub/sub. `/ready` pings Redis. A `rediss://` URL speaks TLS, with the authority
named in the URL (`harness.tls_authority`).

Two instances here share one database and one `redis-server`; the other Redis paths are driven
by integration/chat/test_redis_task_commands.py (a stop over pub/sub, a restart, a Redis
Cluster), integration/chat/test_socket_cleanup_locks.py (locks and the session pool sweep),
integration/resilience/test_redis_sentinel_failover.py (Sentinel) and
integration/chat/test_socket_delivery_across_instances.py (streams across instances).

Discriminates: passes on dev ef67cc3fa. In a backend copy, a revoked token stored without its
expiry fails the sign-out test, `disconnect` answered by python-socketio's pub/sub manager
without acting (patched in at import) fails the cross-instance sign-out, a task saved without
its chat's set fails the task listing, a note update not pushed to Redis fails the note test,
a readiness check that skips the ping fails the readiness test and a Redis client that drops
the URL's CA file fails the TLS test.
"""

from __future__ import annotations

import contextlib
import dataclasses
import shutil
import subprocess
import time
import uuid
from typing import Callable, Iterator

import httpx
import jwt
import pytest
import redis

from harness import backends
from harness.actors import Actor, create_user
from harness.chat import wait_for_reply
from harness.inflight import start_slow_reply
from harness.instance import LaunchedInstance, free_port
from harness.socket_client import connected
from harness.tls_authority import issue_certificate

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

KEY_PREFIX = "open-webui"
WAIT = 15.0


@pytest.fixture(scope="module")
def redis_url() -> Iterator[str]:
    with backends.redis_server() as url:
        yield url


@pytest.fixture(scope="module")
def pair(redis_url, instance_with) -> tuple[LaunchedInstance, LaunchedInstance]:
    """Two instances on one database and one Redis."""
    shared = {"REDIS_URL": redis_url, "WEBSOCKET_MANAGER": "redis"}
    first = instance_with({**shared, "WEBUI_NAME": "first"})
    second = instance_with({**shared, "WEBUI_NAME": "second", "DATABASE_URL": first.database_url})
    return first, second


def _on(instance: LaunchedInstance, account: Actor) -> Actor:
    """The same account, talking to another instance."""
    return dataclasses.replace(account, base_url=instance.base_url)


def _eventually(condition: Callable[[], bool], timeout: float = WAIT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.1)
    return condition()


def test_a_signed_out_token_stays_revoked_in_redis_for_the_rest_of_its_life(pair, redis_url):
    first, second = pair
    account = create_user(first)
    claims = jwt.decode(account.token, options={"verify_signature": False})

    with account.client() as client:
        signed_out = client.post("/api/v1/auths/signout")
    with _on(second, account).client() as client:
        refused = client.get("/api/v1/auths/")
    remaining = redis.Redis.from_url(redis_url).ttl(
        f"{KEY_PREFIX}:auth:token:{claims['jti']}:revoked"
    )

    assert signed_out.status_code == 200, signed_out.text
    assert refused.status_code == 401, "the other instance still accepts the signed-out token"
    assert abs(remaining - (claims["exp"] - time.time())) < 60, (
        f"the revocation lives {remaining}s, the token {claims['exp'] - time.time():.0f}s"
    )


def test_signing_out_on_one_instance_disconnects_the_tab_on_the_other(pair):
    first, second = pair
    account = create_user(first)

    with connected(_on(second, account)) as tab:
        with account.client() as client:
            signed_out = client.post("/api/v1/auths/signout")

        assert signed_out.status_code == 200, signed_out.text
        assert _eventually(lambda: not tab.client.connected), (
            "the tab on the other instance kept its socket after the sign-out"
        )


def test_a_reply_running_on_one_instance_is_listed_by_the_other(pair):
    first, second = pair
    account = create_user(first)

    with account.client() as client, _on(second, account).client() as elsewhere:
        turn = start_slow_reply(client, first.upstream, chunk_delay=0.2)
        listed = elsewhere.get(f"/api/tasks/chat/{turn.chat_id}")
        wait_for_reply(client, turn, timeout=30)
        after = elsewhere.get(f"/api/tasks/chat/{turn.chat_id}")

    assert listed.status_code == 200, listed.text
    assert len(listed.json()["task_ids"]) == 1, listed.json()
    assert after.json()["task_ids"] == [], "the finished reply is still listed"


def test_a_note_edited_through_one_instance_opens_edited_through_the_other(pair):
    first, second = pair
    account = create_user(first)
    with account.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": f"shared {uuid.uuid4().hex[:8]}", "data": {"content": {"md": ""}}},
        )
    assert created.status_code == 200, created.text
    note_id = created.json()["id"]

    with connected(account) as writer, connected(_on(second, account)) as watcher:
        writer.join_note(note_id)
        watcher.join_note(note_id)
        writer.edit_note(note_id, "tides at noon")
        passed_on = watcher.note_update(note_id)
        with connected(_on(second, account)) as later:
            later.join_note(note_id)
            opened = later.note_state(note_id)

    assert passed_on["document_id"] == f"note:{note_id}"
    assert opened == "<paragraph>tides at noon</paragraph>"


@pytest.fixture
def short_lived_redis(instance_with) -> Iterator[tuple[LaunchedInstance, Callable[[], None]]]:
    """An instance on a Redis of its own, and a way to take that Redis away."""
    with contextlib.ExitStack() as stack:
        url = stack.enter_context(backends.redis_server())
        instance = instance_with({"REDIS_URL": url, "WEBSOCKET_MANAGER": "redis"})

        def stop() -> None:
            redis.Redis.from_url(url).shutdown(nosave=True)

        yield instance, stop


def test_ready_answers_while_redis_answers_and_refuses_without_it(short_lived_redis):
    instance, stop_redis = short_lived_redis
    ready = httpx.get(f"{instance.base_url}/ready", timeout=30)

    stop_redis()
    refused = httpx.get(f"{instance.base_url}/ready", timeout=30)

    assert ready.status_code == 200, ready.text
    assert refused.status_code == 503, refused.text
    assert refused.json()["detail"] == "Redis not ready"


@pytest.fixture(scope="module")
def tls_redis_url(tmp_path_factory) -> Iterator[str]:
    """A redis-server that speaks only TLS, and the URL that trusts its authority."""
    binary = shutil.which("redis-server")
    if binary is None:
        pytest.skip("needs a redis-server binary on PATH")
    issued = issue_certificate(tmp_path_factory.mktemp("redis-tls"), "Redis Test CA", ["localhost"])
    port = free_port()
    command = [
        binary,
        *("--port", "0", "--tls-port", str(port), "--bind", "127.0.0.1", "--save", ""),
        *("--tls-cert-file", str(issued.certificate), "--tls-key-file", str(issued.key)),
        *("--tls-ca-cert-file", str(issued.authority), "--tls-auth-clients", "no"),
    ]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    url = f"rediss://localhost:{port}/0?ssl_ca_certs={issued.authority}"
    try:
        client = redis.Redis.from_url(url)
        if not _eventually(lambda: _answers(client), timeout=10):
            pytest.fail("the TLS redis-server did not start")
        yield url
    finally:
        process.terminate()
        process.wait(timeout=10)


def _answers(client: redis.Redis) -> bool:
    try:
        return bool(client.ping())
    except redis.exceptions.RedisError:
        return False


def test_an_instance_on_a_tls_redis_keeps_its_revocations_there(instance_with, tls_redis_url):
    instance = instance_with({"REDIS_URL": tls_redis_url, "WEBSOCKET_MANAGER": "redis"})
    account = create_user(instance)
    claims = jwt.decode(account.token, options={"verify_signature": False})

    with account.client() as client:
        signed_out = client.post("/api/v1/auths/signout")
        refused = client.get("/api/v1/auths/")
    ready = httpx.get(f"{instance.base_url}/ready", timeout=30)
    stored = redis.Redis.from_url(tls_redis_url).exists(
        f"{KEY_PREFIX}:auth:token:{claims['jti']}:revoked"
    )

    assert signed_out.status_code == 200, signed_out.text
    assert refused.status_code == 401, refused.text
    assert ready.status_code == 200, ready.text
    assert stored == 1, "the revocation never reached the TLS Redis"
