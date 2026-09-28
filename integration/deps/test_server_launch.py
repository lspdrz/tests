"""Dependency smoke: `open-webui serve`, the pip install's launcher, running uvicorn.

`open-webui serve --host --port` imports the app and hands it to `uvicorn.run` by import string,
with `forwarded_allow_ips='*'`, `workers=UVICORN_WORKERS` and
`ws_per_message_deflate=UVICORN_WS_PER_MESSAGE_DEFLATE`. So a reverse proxy's `X-Forwarded-For`
and `X-Forwarded-Proto` are trusted from any address (the client address the audit log records,
the scheme of the SSO callback it sends the provider), `UVICORN_WORKERS` worker processes serve
the port, and the socket's WebSocket takes per-message compression unless it is switched off.
`open-webui dev` runs it with `reload=True`: uvicorn's reloader starts the server in a process of
its own and starts a new one when a Python file under the working directory changes.
Each boot runs the real launcher from a scratch directory; requests reach it from 127.0.0.2, an
address uvicorn trusts only when told to. The container's `start.sh`, which runs the uvicorn
command line instead, is integration/config/test_start_sh.py (twin of unit/deps/test_uvicorn.py).

Discriminates: passes on dev ef67cc3fa. In a backend copy whose launcher leaves out
`forwarded_allow_ips`, the audit log records 127.0.0.2 and the callback stays on http; one that
passes `workers=1` starts no worker processes; one that leaves out `ws_per_message_deflate` keeps
compression on with it switched off; `dev` passing `reload=False` runs no reloader.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import httpx
import pytest

from harness.instance import free_port, isolated_env, resolve_backend
from harness.oidc_provider import shared_provider, sso_env

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

LAUNCHER = "import sys; from open_webui import app; sys.argv[0] = 'open-webui'; app()"
ACCOUNT = {"name": "Admin", "email": "admin@example.com", "password": "adminpassword123"}
PROXY_ADDRESS = "127.0.0.2"
CLIENT_ADDRESS = "203.0.113.7"
BOOT_SECONDS = 300


@dataclass
class Served:
    process: subprocess.Popen
    base_url: str
    data_dir: Path
    work_dir: Path

    def client(self) -> httpx.Client:
        """A client connecting from `PROXY_ADDRESS`, where a reverse proxy would sit."""
        transport = httpx.HTTPTransport(local_address=PROXY_ADDRESS)
        return httpx.Client(base_url=self.base_url, transport=transport, timeout=60.0)


def _install_metadata(site: Path, backend: Path) -> None:
    """The record `pip install open-webui` leaves, where the launcher reads its version."""
    package = backend.parent / "package.json"
    version = json.loads(package.read_text())["version"] if package.is_file() else "0.0.0"
    record = site / f"open_webui-{version}.dist-info"
    record.mkdir()
    (record / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: open-webui\nVersion: {version}\n", encoding="utf-8"
    )


def _answers_health(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/health", timeout=3.0).status_code == 200
    except httpx.HTTPError:
        return False


@contextlib.contextmanager
def serving(directory: Path, settings: dict[str, str], command: str = "serve") -> Iterator[Served]:
    """`open-webui <command>` on 127.0.0.1 until `/health` answers, stopped when the block ends."""
    backend = resolve_backend()
    if backend is None:
        pytest.skip("open-webui backend source not found (set OPEN_WEBUI_SOURCE_DIR)")
    for name in ("data", "static", "work", "site"):
        (directory / name).mkdir()
    _install_metadata(directory / "site", backend)
    port = free_port()
    env = isolated_env(
        {
            "DATA_DIR": str(directory / "data"),
            "STATIC_DIR": str(directory / "static"),
            "FRONTEND_BUILD_DIR": str(directory / "build"),
            "WEBUI_SECRET_KEY": "launch-secret-key",
            "OFFLINE_MODE": "true",
            "ENABLE_OLLAMA_API": "false",
            "ENABLE_OPENAI_API": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join([str(backend), str(directory / "site")]),
            **settings,
        }
    )
    env.pop("FORWARDED_ALLOW_IPS", None)  # uvicorn's own default, which the launcher overrides
    log_path = directory / "server.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-c", LAUNCHER, command, "--host", "127.0.0.1", "--port", str(port)],
            cwd=directory / "work",
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    served = Served(process, f"http://127.0.0.1:{port}", directory / "data", directory / "work")
    try:
        deadline = time.monotonic() + BOOT_SECONDS
        while not _answers_health(served.base_url):
            if process.poll() is not None or time.monotonic() > deadline:
                log = log_path.read_text(encoding="utf-8", errors="replace")
                pytest.fail(
                    f"open-webui {command} never came up (exit {process.poll()}):\n{log[-4000:]}"
                )
            time.sleep(0.5)
        yield served
    finally:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def _worker_processes(pid: int) -> list[int]:
    workers = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            parent = int(stat.read_text().rsplit(") ", 1)[1].split()[1])
            command = (stat.parent / "cmdline").read_bytes()
        except OSError:
            continue
        if parent == pid and b"spawn_main" in command:
            workers.append(int(stat.parent.name))
    return workers


def _eventually(read, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    value = read()
    while not value and time.monotonic() < deadline:
        time.sleep(0.5)
        value = read()
    return value


def _offers_compression(served: Served) -> bool:
    """Whether the socket's WebSocket handshake agrees to per-message deflate."""
    from websockets.sync.client import connect

    address = (
        served.base_url.replace("http://", "ws://") + "/ws/socket.io/?EIO=4&transport=websocket"
    )
    with connect(address, compression="deflate", open_timeout=30) as socket:
        extensions = socket.response.headers.get("Sec-WebSocket-Extensions", "")
    return "permessage-deflate" in extensions


@pytest.fixture(scope="module")
def launched(tmp_path_factory):
    if sys.platform != "linux":
        pytest.skip("reaches the server from 127.0.0.2 and reads its workers from /proc")
    settings = {
        **sso_env(shared_provider()),
        "UVICORN_WORKERS": "2",
        "AUDIT_LOG_LEVEL": "METADATA",
    }
    with serving(tmp_path_factory.mktemp("serve"), settings) as served:
        yield served


def test_the_launcher_serves_the_port_it_was_given_with_the_configured_workers(launched):
    workers = _eventually(lambda: len(_worker_processes(launched.process.pid)) == 2)

    assert workers, "UVICORN_WORKERS=2 did not start two worker processes"
    with launched.client() as client:
        assert client.get("/health").json() == {"status": True}


def test_the_client_address_a_proxy_forwards_is_the_one_audited(launched):
    with launched.client() as client:
        signed_in = client.post(
            "/api/v1/auths/signin",
            json={"email": "nobody@harbour.example", "password": "wrong-password"},
            headers={"X-Forwarded-For": CLIENT_ADDRESS},
        )
    assert signed_in.status_code in (400, 401, 403), signed_in.text

    def audited_addresses() -> list[str]:
        audit_log = launched.data_dir / "audit.log"
        if not audit_log.is_file():
            return []
        entries = [json.loads(line) for line in audit_log.read_text().splitlines() if line]
        return [
            entry.get("source_ip") for entry in entries if "signin" in entry.get("request_uri", "")
        ]

    addresses = _eventually(audited_addresses)
    assert addresses == [CLIENT_ADDRESS], addresses


def test_the_scheme_a_proxy_forwards_names_the_sso_callback(launched):
    with launched.client() as client:
        started = client.get("/oauth/oidc/login", headers={"X-Forwarded-Proto": "https"})

    assert started.status_code == 302, started.text
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(started.headers["location"]).query))
    assert query["redirect_uri"].startswith("https://127.0.0.1:"), query["redirect_uri"]


def test_the_socket_takes_per_message_compression_by_default(launched):
    assert _offers_compression(launched)


def test_per_message_compression_can_be_switched_off(tmp_path):
    settings = {"UVICORN_WS_PER_MESSAGE_DEFLATE": "false"}
    with serving(tmp_path, settings) as served:
        assert not _offers_compression(served)


def test_the_development_launcher_restarts_the_server_when_code_changes(tmp_path):
    with serving(tmp_path, {}, command="dev") as served:
        first = _eventually(lambda: _worker_processes(served.process.pid))
        assert len(first) == 1, f"the reloader should run one server process: {first}"

        (served.work_dir / "harbour.py").write_text("TIDE = 1\n", encoding="utf-8")
        restarted = _eventually(
            lambda: [pid for pid in _worker_processes(served.process.pid) if pid not in first]
        )

        assert restarted, "changing a Python file did not restart the server"
        assert _eventually(lambda: _answers_health(served.base_url), timeout=120)
