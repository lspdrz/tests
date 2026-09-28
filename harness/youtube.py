"""YouTube played by a local proxy, for the transcript loader behind the admin's Youtube Proxy URL.

The transcript loader talks to `https://www.youtube.com` only, through the proxy the admin sets.
`serving_youtube()` yields a `FakeYouTube` that is that proxy: it accepts `CONNECT`, answers the
TLS handshake with a certificate for www.youtube.com signed by its own authority, and plays the
three pages the loader reads (the watch page, the player API and the transcript itself). An
instance trusts the authority through `youtube_env(fake)`, which sets `REQUESTS_CA_BUNDLE`.

`fake.videos[video_id]` says what YouTube answers for a video, built with the helpers here:
`with_transcript(*lines)`, `unplayable(status, reason)`, `without_captions()`, `blocked_page()`
(the recaptcha page a blocked server gets), `rate_limited()` (HTTP 429) or
`needs_verification()` (a transcript URL that asks for a proof-of-origin token). `fake.requests`
lists every `(method, path)` it was sent.
"""

from __future__ import annotations

import contextlib
import json
import socketserver
import ssl
import tempfile
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, urlsplit
from xml.sax.saxutils import escape

from harness.tls_authority import issue_certificate

YOUTUBE_HOSTS = ["www.youtube.com", "youtube.com"]
API_KEY = "fake-innertube-key"
# the reasons YouTube gives, which the transcript API matches word for word
BOT_CHECK = "Sign in to confirm you’re not a bot"
AGE_CHECK = "This video may be inappropriate for some users."
UNAVAILABLE = "This video is unavailable"


@dataclass
class Video:
    player: dict | None = None
    watch_page: str | None = None
    status: int = 200
    transcript: tuple[str, ...] = ()


def with_transcript(*lines: str, language_code: str = "en", generated: bool = False) -> Video:
    track = {
        "baseUrl": "https://www.youtube.com/api/timedtext?lang=" + language_code,
        "name": {"runs": [{"text": language_code}]},
        "languageCode": language_code,
        "kind": "asr" if generated else "",
    }
    player = {
        "playabilityStatus": {"status": "OK"},
        "captions": {"playerCaptionsTracklistRenderer": {"captionTracks": [track]}},
    }
    return Video(player=player, transcript=lines)


def needs_verification() -> Video:
    video = with_transcript("never fetched")
    track = video.player["captions"]["playerCaptionsTracklistRenderer"]["captionTracks"][0]
    track["baseUrl"] += "&exp=xpe"
    return video


def unplayable(status: str, reason: str) -> Video:
    return Video(player={"playabilityStatus": {"status": status, "reason": reason}})


def without_captions() -> Video:
    return Video(player={"playabilityStatus": {"status": "OK"}})


def blocked_page() -> Video:
    return Video(watch_page='<html><form><div class="g-recaptcha"></div></form></html>')


def rate_limited() -> Video:
    return Video(status=429)


@dataclass
class FakeYouTube:
    proxy_url: str
    ca_bundle: Path
    videos: dict[str, Video] = field(default_factory=dict)
    requests: list[tuple[str, str]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def requested(self, path_prefix: str) -> list[str]:
        with self.lock:
            return [path for _, path in self.requests if path.startswith(path_prefix)]


def _youtube_handler(fake: FakeYouTube):
    class YouTubePage(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _answer(self, status: int, content_type: str, body: str) -> None:
            encoded = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _serve(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            with fake.lock:
                fake.requests.append((self.command, self.path))
            address = urlsplit(self.path)
            query = parse_qs(address.query)
            if address.path == "/watch":
                self._watch_page(query.get("v", [""])[0])
            elif address.path == "/youtubei/v1/player":
                self._player(json.loads(body or b"{}").get("videoId", ""))
            elif address.path == "/api/timedtext":
                self._transcript(query)
            else:
                self._answer(404, "text/plain", "")

        def _watch_page(self, video_id: str) -> None:
            video = fake.videos.get(video_id, without_captions())
            if video.status != 200:
                self._answer(video.status, "text/html", "")
                return
            page = video.watch_page or f'<script>ytcfg.set({{"INNERTUBE_API_KEY": "{API_KEY}"}})'
            self._answer(200, "text/html", page)

        def _player(self, video_id: str) -> None:
            video = fake.videos.get(video_id, without_captions())
            player = json.loads(json.dumps(video.player or {}))
            captions = player.get("captions", {}).get("playerCaptionsTracklistRenderer", {})
            for track in captions.get("captionTracks", []):
                track["baseUrl"] += f"&v={video_id}"
            self._answer(200, "application/json", json.dumps(player))

        def _transcript(self, query: dict) -> None:
            video = fake.videos.get(query.get("v", [""])[0], Video())
            snippets = "".join(
                f'<text start="{index}.0" dur="1.0">{escape(line)}</text>'
                for index, line in enumerate(video.transcript)
            )
            self._answer(200, "text/xml", f"<transcript>{snippets}</transcript>")

        do_GET = do_POST = _serve

    return YouTubePage


def _proxy_handler(fake: FakeYouTube, tls: ssl.SSLContext):
    youtube_page = _youtube_handler(fake)

    class Proxy(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            request_line = self.rfile.readline().decode("latin-1")
            while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                pass  # the proxy's own headers
            method, target, _ = (request_line.split(" ") + ["", "", ""])[:3]
            host = target.split(":")[0]
            if method != "CONNECT" or host not in YOUTUBE_HOSTS:
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                return
            self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            self.wfile.flush()
            with contextlib.suppress(OSError, ssl.SSLError):
                wrapped = tls.wrap_socket(self.connection, server_side=True)
                youtube_page(wrapped, self.client_address, self.server)

    return Proxy


@contextlib.contextmanager
def serving_youtube() -> Iterator[FakeYouTube]:
    with tempfile.TemporaryDirectory(prefix="fake-youtube-") as directory:
        issued = issue_certificate(Path(directory), "Fake YouTube test authority", YOUTUBE_HOSTS)
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(issued.certificate, issued.key)
        server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), None)
        server.daemon_threads = True
        host, port = server.server_address
        fake = FakeYouTube(proxy_url=f"http://{host}:{port}", ca_bundle=issued.authority)
        server.RequestHandlerClass = _proxy_handler(fake, tls)
        threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
        try:
            yield fake
        finally:
            server.shutdown()
            server.server_close()


def youtube_env(fake: FakeYouTube) -> dict[str, str]:
    """The environment of an instance that trusts the fake's certificate authority."""
    return {"REQUESTS_CA_BUNDLE": str(fake.ca_bundle)}
