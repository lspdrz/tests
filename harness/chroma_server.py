"""A Chroma server on a local port, the one chromadb ships, for an instance using it remotely.

The chromadb package installs a `chroma` command that runs the real server (`chroma run`), so
`serving_chroma()` starts one on a free port over a scratch directory, waits for its heartbeat
and yields its base URL; it stops and is removed when the block ends. `chroma_env(base_url)` is
the environment of an instance that keeps its vectors there (`CHROMA_HTTP_HOST`) in place of the
embedded store in its data directory. `collection_names(base_url)` lists what the server holds.
`recording_proxy(base_url)` stands in front of the server and keeps the headers of every request
it passes on, which is how a test reads the credentials and headers the client sent.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import httpx
import pytest

from harness.instance import free_port

COLLECTIONS = "/api/v2/tenants/default_tenant/databases/default_database/collections"


def _chroma_command() -> str | None:
    beside_python = Path(sys.executable).parent / "chroma"
    return str(beside_python) if beside_python.exists() else shutil.which("chroma")


def _wait_for_heartbeat(process: subprocess.Popen, base_url: str) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"the chroma server exited with {process.returncode}")
        try:
            if httpx.get(f"{base_url}/api/v2/heartbeat", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise AssertionError("the chroma server never answered its heartbeat")


@contextlib.contextmanager
def serving_chroma() -> Iterator[str]:
    command = _chroma_command()
    if command is None:
        pytest.skip("no `chroma` command; it comes with the chromadb package")
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="owui-chroma-") as directory:
        with open(Path(directory) / "server.log", "w") as log:
            process = subprocess.Popen(
                [command, "run", "--path", f"{directory}/data", "--host", "127.0.0.1"]
                + ["--port", str(port)],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        try:
            _wait_for_heartbeat(process, base_url)
            yield base_url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def chroma_env(base_url: str) -> dict[str, str]:
    host, port = base_url.removeprefix("http://").split(":")
    return {"CHROMA_HTTP_HOST": host, "CHROMA_HTTP_PORT": port}


def collection_names(base_url: str) -> list[str]:
    listed = httpx.get(f"{base_url}{COLLECTIONS}", timeout=10)
    listed.raise_for_status()
    return [collection["name"] for collection in listed.json()]


@contextlib.contextmanager
def recording_proxy(base_url: str) -> Iterator[tuple[str, list[dict[str, str]]]]:
    """(the proxy's base URL, the headers of each request so far, names lower-cased)."""
    seen: list[dict[str, str]] = []
    passed_on = httpx.Client(base_url=base_url, timeout=60)

    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _forward(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            headers = {name.lower(): value for name, value in self.headers.items()}
            seen.append(headers)
            forwarded = {name: value for name, value in headers.items() if name != "host"}
            answer = passed_on.request(self.command, self.path, content=body, headers=forwarded)
            self.send_response(answer.status_code)
            self.send_header("Content-Type", answer.headers.get("content-type", "text/plain"))
            self.send_header("Content-Length", str(len(answer.content)))
            self.end_headers()
            self.wfile.write(answer.content)

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _forward

    server = ThreadingHTTPServer(("127.0.0.1", 0), RequestHandler)
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        passed_on.close()
