"""A forwarding HTTP proxy, the kind an operator names in `HTTP_PROXY`, that knows its own names.

Open WebUI's aiohttp sessions honour `HTTP_PROXY` and `NO_PROXY` (`trust_env=True`), and a proxied
request never resolves its target locally: aiohttp resolves the proxy's host and hands the whole
URL to it. `serving_proxy()` starts the proxy on 127.0.0.1; `proxy.known[host] = port` makes it
forward requests for that host name to that local port, as a corporate proxy resolves internal
names the instance cannot. A name it does not know is a 502 with a plain-text reason. `proxy.seen`
is every `(host, path)` it was asked for, `proxy.port` its port for the instance's environment.
"""

from __future__ import annotations

import http.client
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator
from urllib.parse import urlsplit

HOP_BY_HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "upgrade"}


@dataclass
class HttpProxy:
    port: int
    known: dict[str, int] = field(default_factory=dict)
    seen: list[tuple[str, str]] = field(default_factory=list)

    def requests_for(self, host: str) -> list[str]:
        return [path for seen_host, path in list(self.seen) if seen_host == host]


@contextmanager
def serving_proxy() -> Iterator[HttpProxy]:
    class ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _refuse(self, reason: str) -> None:
            body = reason.encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _forward(self) -> None:
            target = urlsplit(self.path)
            path = target.path + (f"?{target.query}" if target.query else "")
            proxy.seen.append((target.hostname or "", path))
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else None
            port = proxy.known.get(target.hostname or "")
            if port is None:
                self._refuse(f"proxy cannot resolve {target.hostname}")
                return
            headers = {
                name: value
                for name, value in self.headers.items()
                if name.lower() not in HOP_BY_HOP and not name.lower().startswith("proxy-")
            }
            upstream = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
            try:
                upstream.request(self.command, path, body=body, headers=headers)
                answer = upstream.getresponse()
                content = answer.read()
            finally:
                upstream.close()
            self.send_response(answer.status)
            for name, value in answer.getheaders():
                if name.lower() not in HOP_BY_HOP and name.lower() != "content-length":
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _forward

    server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
    server.daemon_threads = True
    proxy = HttpProxy(port=server.server_port)
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    try:
        yield proxy
    finally:
        server.shutdown()
        server.server_close()
