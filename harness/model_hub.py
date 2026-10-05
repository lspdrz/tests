"""Hugging Face's file downloads played by a local proxy, for the admin's GGUF download by URL.

Open WebUI's Manage Ollama dialog downloads a model file only from `https://huggingface.co/` or
`https://github.com/`, through the shared aiohttp session, which goes through the proxy the
environment names. `serving_model_hub()` yields a `FakeModelHub` that is that proxy: it accepts
`CONNECT` to `huggingface.co` only, answers the TLS handshake with a certificate signed by its own
authority and serves `hub.files[path]` (bytes) to a GET of that path, a 404 otherwise; a list of
byte strings is sent piece by piece with a pause between, so the client reads each as a chunk.
`hub.downloads` is every path it was asked for and `hub.url(path)` the full URL to type into the
dialog. `model_hub_env(hub)` is the environment of an instance whose downloads reach the hub: the
proxy, the authority to trust (`SSL_CERT_FILE`) and loopback kept off the proxy so the Ollama
stand-in is reached directly.
"""

from __future__ import annotations

import contextlib
import socketserver
import ssl
import tempfile
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Iterator

from harness.tls_authority import issue_certificate

HUB_HOST = "huggingface.co"
PIECE_PAUSE = 0.2  # seconds between the pieces of a file served in pieces


@dataclass
class FakeModelHub:
    proxy_url: str
    ca_bundle: Path
    files: dict[str, bytes | list[bytes]] = field(default_factory=dict)
    downloads: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def url(self, path: str) -> str:
        return f"https://{HUB_HOST}{path}"


def _file_handler(hub: FakeModelHub):
    class Files(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            with hub.lock:
                hub.downloads.append(path)
            content = hub.files.get(path)
            pieces = content if isinstance(content, list) else [content or b""]
            self.send_response(200 if content is not None else 404)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(sum(len(piece) for piece in pieces)))
            self.end_headers()
            for piece in pieces:
                self.wfile.write(piece)
                self.wfile.flush()
                time.sleep(PIECE_PAUSE)

    return Files


def _proxy_handler(hub: FakeModelHub, tls: ssl.SSLContext):
    files = _file_handler(hub)

    class Proxy(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            request_line = self.rfile.readline().decode("latin-1")
            while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                pass  # the proxy's own headers
            method, target, _ = (request_line.split(" ") + ["", "", ""])[:3]
            if method != "CONNECT" or target.split(":")[0] != HUB_HOST:
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                return
            self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            self.wfile.flush()
            with contextlib.suppress(OSError, ssl.SSLError):
                wrapped = tls.wrap_socket(self.connection, server_side=True)
                files(wrapped, self.client_address, self.server)

    return Proxy


@contextlib.contextmanager
def serving_model_hub() -> Iterator[FakeModelHub]:
    with tempfile.TemporaryDirectory(prefix="fake-model-hub-") as directory:
        issued = issue_certificate(Path(directory), "Fake model hub test authority", [HUB_HOST])
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(issued.certificate, issued.key)
        server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), None)
        server.daemon_threads = True
        host, port = server.server_address
        hub = FakeModelHub(proxy_url=f"http://{host}:{port}", ca_bundle=issued.authority)
        server.RequestHandlerClass = _proxy_handler(hub, tls)
        threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
        try:
            yield hub
        finally:
            server.shutdown()
            server.server_close()


def model_hub_env(hub: FakeModelHub) -> dict[str, str]:
    return {
        "HTTPS_PROXY": hub.proxy_url,
        "SSL_CERT_FILE": str(hub.ca_bundle),
        "NO_PROXY": "127.0.0.1,localhost",
    }
