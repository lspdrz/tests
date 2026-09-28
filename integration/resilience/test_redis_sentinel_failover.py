"""On Redis Sentinel, a stalled master is waited out and the master is looked up only when needed.

0.11.0 75a8a00 (issue #27210): Open WebUI's Sentinel proxy retried a Redis call on a connection
error but not on a timeout, which is how a master mid-failover (or stalled) shows, and it asked
Sentinel for the master on every call. It now retries timeouts too, looking the master up again
before each retry, and keeps the master it found while it answers.

The instance runs on a real master behind a real Sentinel, with a one second socket timeout.
Every signed-in request checks Redis for a revoked token, and that check lets the request
through when Redis fails, so a timeout the proxy gives up on shows as a signed-out token
accepted again. The master is stalled with `DEBUG SLEEP` for a few socket timeouts. A master
looked up afresh through Sentinel comes with a connection of its own, so the master's count of
accepted connections shows how often that happened.

Twin of unit/chat/test_socket_and_redis_runtime.py for the Sentinel proxy.

Discriminates: passes on dev ef67cc3fa; with `TimeoutError` removed from the retryable errors the
signed-out token is accepted during the stall, and with the master looked up on every call every
request opens a new connection to it.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

import pytest
import redis

from harness.actors import create_user
from harness.instance import free_port

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

SERVICE = "mymaster"
# several one-second socket timeouts long, and well inside the proxy's eight tries
STALL_SECONDS = 4
REQUESTS = 20


def _wait_until(condition: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.25)
    return condition()


def _answers(port: int) -> bool:
    client = redis.Redis(port=port, socket_connect_timeout=1)
    try:
        return bool(client.ping())
    except redis.exceptions.RedisError:
        return False
    finally:
        client.close()


def _start(command: list[str], port: int) -> subprocess.Popen:
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if not _wait_until(lambda: _answers(port), timeout=10):
        process.terminate()
        pytest.fail(f"{command[0]} did not start on port {port}")
    return process


@pytest.fixture(scope="module")
def sentinel_setup() -> Iterator[tuple[int, int]]:
    """A master and a Sentinel watching it; yields their ports."""
    binary = shutil.which("redis-server")
    if binary is None:
        pytest.skip("needs a redis-server binary on PATH")
    workdir = Path(tempfile.mkdtemp(prefix="owui-sentinel-"))
    master_port, sentinel_port = free_port(), free_port()
    config = workdir / "sentinel.conf"
    config.write_text(
        f"port {sentinel_port}\nbind 127.0.0.1\ndir {workdir}\n"
        f"sentinel monitor {SERVICE} 127.0.0.1 {master_port} 1\n"
        f"sentinel down-after-milliseconds {SERVICE} 600000\n"
    )
    master = _start(
        [
            binary,
            "--port",
            str(master_port),
            "--bind",
            "127.0.0.1",
            "--save",
            "",
            "--dir",
            str(workdir),
            "--enable-debug-command",
            "local",
        ],
        master_port,
    )
    sentinel = _start([binary, str(config), "--sentinel"], sentinel_port)
    try:
        yield master_port, sentinel_port
    finally:
        for process in (sentinel, master):
            process.terminate()
            process.wait(timeout=10)
        shutil.rmtree(workdir, ignore_errors=True)


@pytest.fixture(scope="module")
def sentinel_instance(instance_with, sentinel_setup):
    _, sentinel_port = sentinel_setup
    return instance_with(
        {
            "REDIS_URL": f"redis://{SERVICE}:6379/0",
            "REDIS_SENTINEL_HOSTS": "127.0.0.1",
            "REDIS_SENTINEL_PORT": str(sentinel_port),
            "REDIS_SOCKET_TIMEOUT": "1",
            "REDIS_RECONNECT_DELAY": "500",
            "REDIS_SENTINEL_MAX_RETRY_COUNT": "8",
        }
    )


def _master_connections(master_port: int) -> int:
    """How many connections the master has accepted so far."""
    client = redis.Redis(port=master_port)
    try:
        return client.info("stats")["total_connections_received"]
    finally:
        client.close()


def _signed_out_token(instance) -> str:
    account = create_user(instance)
    with account.client() as client:
        client.post("/api/v1/auths/signout").raise_for_status()
        assert client.get("/api/v1/auths/").status_code == 401, "signing out did not revoke"
    return account.token


def test_the_master_is_looked_up_once_while_it_answers(sentinel_instance, sentinel_setup):
    master_port, _ = sentinel_setup
    account = create_user(sentinel_instance)
    with account.client() as client:
        client.get("/api/v1/auths/").raise_for_status()
        before = _master_connections(master_port)
        for _ in range(REQUESTS):
            assert client.get("/api/v1/auths/").status_code == 200

    # one of the new connections is this test's own counter
    opened = _master_connections(master_port) - before - 1
    assert opened < REQUESTS / 2, (
        f"{REQUESTS} requests opened {opened} new connections to the master, one per fresh "
        "lookup of it through Sentinel"
    )


def test_a_signed_out_token_stays_refused_while_the_master_stalls(
    sentinel_instance, sentinel_setup
):
    master_port, _ = sentinel_setup
    token = _signed_out_token(sentinel_instance)

    stall = threading.Thread(
        target=lambda: redis.Redis(port=master_port).execute_command(
            "DEBUG", "SLEEP", STALL_SECONDS
        )
    )
    stall.start()
    time.sleep(0.5)
    try:
        with sentinel_instance.client(token) as client:
            during_stall = client.get("/api/v1/auths/")
    finally:
        stall.join()

    assert during_stall.status_code == 401, (
        "a timeout on the stalled master was not retried, so the revocation check let a "
        "signed-out token through (#27210)"
    )
