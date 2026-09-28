"""DuckDuckGo's HTML search played by a local proxy, for the web search engine `duckduckgo`.

Open WebUI's DuckDuckGo engine is the `ddgs` library, which with the `duckduckgo` backend posts
the query to `https://html.duckduckgo.com/html/` and reads the result list out of the page. It
goes through the proxy the environment names. `serving_duckduckgo()` yields a `FakeDuckDuckGo`
that is that proxy: it accepts `CONNECT` to DuckDuckGo only, answers the TLS handshake with a
certificate signed by its own authority and serves a result page listing `fake.results`
(`Result(link, title, snippet)`), or the empty 202 DuckDuckGo gives a client it rate-limits when
`fake.rate_limited` is set. `fake.searches` records each search: its form fields and headers.

`duckduckgo_env(fake)` is the environment of an instance whose searches reach the fake: the proxy,
the authority to trust (`SSL_CERT_FILE`, which Python's default TLS context reads) and loopback
kept off the proxy. `duckduckgo_settings()` is the admin's web search settings that use it.
"""

from __future__ import annotations

import contextlib
import socketserver
import ssl
import tempfile
import threading
from dataclasses import dataclass, field
from html import escape
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs

from harness.tls_authority import issue_certificate

DUCKDUCKGO_HOST = "html.duckduckgo.com"


@dataclass
class Result:
    link: str
    title: str
    snippet: str


@dataclass
class Search:
    form: dict[str, str]
    headers: dict[str, str]


@dataclass
class FakeDuckDuckGo:
    proxy_url: str
    ca_bundle: Path
    results: list[Result] = field(default_factory=list)
    rate_limited: bool = False
    searches: list[Search] = field(default_factory=list)
    refused_hosts: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def queries(self) -> list[str]:
        with self.lock:
            return [search.form.get("q", "") for search in self.searches]


def _result_page(results: list[Result]) -> str:
    """The markup of DuckDuckGo's HTML results, down to the classes ddgs reads."""
    entries = "".join(
        '<div class="result results_links web-result"><div class="links_main result__body">'
        f'<h2 class="result__title"><a class="result__a" href="{escape(result.link)}">'
        f"{escape(result.title)}</a></h2>"
        f'<a class="result__snippet" href="{escape(result.link)}">{escape(result.snippet)}</a>'
        "</div></div>"
        for result in results
    )
    return f'<html><body><div class="results">{entries}</div></body></html>'


def _search_handler(fake: FakeDuckDuckGo):
    class SearchPage(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _answer(self, status: int, body: str) -> None:
            encoded = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=UTF-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            fields = parse_qs(self.rfile.read(length).decode(), keep_blank_values=True)
            search = Search(
                form={name: values[0] for name, values in fields.items()},
                headers=dict(self.headers.items()),
            )
            with fake.lock:
                fake.searches.append(search)
            if self.path.split("?")[0] != "/html/":
                self._answer(404, "")
            elif fake.rate_limited:
                self._answer(202, "")
            else:
                self._answer(200, _result_page(fake.results))

    return SearchPage


def _proxy_handler(fake: FakeDuckDuckGo, tls: ssl.SSLContext):
    search_page = _search_handler(fake)

    class Proxy(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            request_line = self.rfile.readline().decode("latin-1")
            while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                pass  # the proxy's own headers
            method, target, _ = (request_line.split(" ") + ["", "", ""])[:3]
            host = target.split(":")[0]
            if method != "CONNECT" or host != DUCKDUCKGO_HOST:
                with fake.lock:
                    fake.refused_hosts.append(host)
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                return
            self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            self.wfile.flush()
            with contextlib.suppress(OSError, ssl.SSLError):
                wrapped = tls.wrap_socket(self.connection, server_side=True)
                search_page(wrapped, self.client_address, self.server)

    return Proxy


@contextlib.contextmanager
def serving_duckduckgo() -> Iterator[FakeDuckDuckGo]:
    with tempfile.TemporaryDirectory(prefix="fake-duckduckgo-") as directory:
        issued = issue_certificate(
            Path(directory), "Fake DuckDuckGo test authority", [DUCKDUCKGO_HOST]
        )
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(issued.certificate, issued.key)
        server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), None)
        server.daemon_threads = True
        host, port = server.server_address
        fake = FakeDuckDuckGo(proxy_url=f"http://{host}:{port}", ca_bundle=issued.authority)
        server.RequestHandlerClass = _proxy_handler(fake, tls)
        threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
        try:
            yield fake
        finally:
            server.shutdown()
            server.server_close()


def duckduckgo_env(fake: FakeDuckDuckGo) -> dict[str, str]:
    return {
        "HTTPS_PROXY": fake.proxy_url,
        "SSL_CERT_FILE": str(fake.ca_bundle),
        "NO_PROXY": "127.0.0.1,localhost",
    }


def duckduckgo_settings(result_count: int) -> dict:
    """Web search on DuckDuckGo's own backend, returning its snippets without loading the pages."""
    return {
        "ENABLE_WEB_SEARCH": True,
        "WEB_SEARCH_ENGINE": "duckduckgo",
        "DDGS_BACKEND": "duckduckgo",
        "WEB_SEARCH_RESULT_COUNT": result_count,
        "WEB_SEARCH_CONCURRENT_REQUESTS": 1,
        "BYPASS_WEB_SEARCH_WEB_LOADER": True,
        "BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL": True,
    }
