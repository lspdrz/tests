"""A Model Context Protocol tool server on a local port, built with the `mcp` SDK's own FastMCP.

`serving_mcp()` runs one over Streamable HTTP in a thread of the test process and yields its
URL, the one an admin enters for an MCP tool server connection. It offers a single tool, `echo`,
and stops again when the block ends. With `media=True` it also offers `snapshot`, answering
with `SNAPSHOT_PNG` as an image, and `chime`, answering with `CHIME_WAV` as audio.
`mcp_connection(...)` is the admin's connection to it without auth; save it through
`TOOL_SERVERS` after `preserve(TOOL_SERVERS)`. Given FastMCP's `auth` settings and a
`token_verifier`, the SDK guards it the way a real OAuth-protected MCP server is guarded: a
request without a token the verifier accepts gets a 401 pointing at the protected-resource
metadata it also serves.
"""

from __future__ import annotations

import base64
import contextlib
import io
import threading
import time
import wave
from typing import Any, Iterator

import uvicorn
from mcp.server.fastmcp import Audio, FastMCP, Image

from harness.instance import free_port

ECHO_DESCRIPTION = "Repeat the text back."
SNAPSHOT_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _silence_wav() -> bytes:
    recording = io.BytesIO()
    with wave.open(recording, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(b"\x00\x00" * 800)
    return recording.getvalue()


CHIME_WAV = _silence_wav()


def _echo_server(media: bool = False, **auth: Any) -> FastMCP:
    server = FastMCP("harness-mcp", log_level="WARNING", **auth)

    @server.tool(description=ECHO_DESCRIPTION)
    def echo(text: str) -> str:
        return text

    if media:

        @server.tool(description="Take a snapshot with the camera.")
        def snapshot() -> Image:
            return Image(data=SNAPSHOT_PNG, format="png")

        @server.tool(description="Play the door chime.")
        def chime() -> Audio:
            return Audio(data=CHIME_WAV, format="wav")

    return server


@contextlib.contextmanager
def serving_mcp(port: int | None = None, media: bool = False, **auth: Any) -> Iterator[str]:
    """Serve the echo server on `port` (a free one by default); `auth` goes to FastMCP."""
    port = port or free_port()
    app = _echo_server(media, **auth).streamable_http_app()
    # log_config=None keeps uvicorn from reconfiguring the test process's logging
    runner = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, log_level="warning")
    )
    thread = threading.Thread(target=runner.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not runner.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("the MCP server did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        runner.should_exit = True
        thread.join(timeout=10)


TOOL_SERVERS = ("/api/v1/configs/tool_servers", "/api/v1/configs/tool_servers")


def mcp_connection(url: str, server_id: str, access_grants: list[dict]) -> dict:
    """An MCP tool server connection without auth, the way the admin panel saves one."""
    return {
        "url": url,
        "path": "",
        "type": "mcp",
        "auth_type": "none",
        "key": "",
        "config": {"enable": True, "access_grants": access_grants},
        "info": {"id": server_id, "name": server_id},
    }
