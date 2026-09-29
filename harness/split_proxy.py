"""One address in front of two instances: a browser's socket goes to one, its requests to the other.

Behind a load balancer with several workers or instances, a browser's Socket.IO connection and
its HTTP requests can land on different ones. `splitting(socket_instance, http_instance)` opens
a local port that reads the first request line of every incoming connection: a path under
`/ws/` goes to the socket instance, anything else to the HTTP instance, and the bytes are then
piped both ways. A WebSocket always opens a connection of its own, and the frontend uses the
websocket transport only, so routing per connection is exact.
"""

from __future__ import annotations

import contextlib
import socket
import threading
from typing import Iterator

from harness.instance import LaunchedInstance, free_port

SOCKET_PREFIX = b"/ws/"
CHUNK = 65536


def _first_request_path(connection: socket.socket) -> tuple[bytes, bytes]:
    """The first request's path and the bytes read so far, which still have to be forwarded."""
    received = b""
    while b"\r\n" not in received:
        more = connection.recv(CHUNK)
        if not more:
            break
        received += more
    parts = received.split(b" ", 2)
    return (parts[1] if len(parts) > 2 else b""), received


def _pipe(source: socket.socket, target: socket.socket) -> None:
    try:
        while data := source.recv(CHUNK):
            target.sendall(data)
    except OSError:
        pass
    finally:
        with contextlib.suppress(OSError):
            target.shutdown(socket.SHUT_WR)


def _serve_connection(connection: socket.socket, socket_port: int, http_port: int) -> None:
    with connection:
        path, received = _first_request_path(connection)
        port = socket_port if path.startswith(SOCKET_PREFIX) else http_port
        try:
            upstream = socket.create_connection(("127.0.0.1", port))
        except OSError:
            return
        with upstream:
            upstream.sendall(received)
            back = threading.Thread(target=_pipe, args=(upstream, connection), daemon=True)
            back.start()
            _pipe(connection, upstream)
            back.join()


def _accept_loop(listener: socket.socket, socket_port: int, http_port: int) -> None:
    while True:
        try:
            connection, _ = listener.accept()
        except OSError:
            return  # the listener was closed
        threading.Thread(
            target=_serve_connection, args=(connection, socket_port, http_port), daemon=True
        ).start()


def _port_of(instance: LaunchedInstance) -> int:
    return int(instance.base_url.rsplit(":", 1)[1])


@contextlib.contextmanager
def splitting(socket_instance: LaunchedInstance, http_instance: LaunchedInstance) -> Iterator[str]:
    """Yields the base URL of an address that splits the two instances."""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", free_port()))
    listener.listen()
    threading.Thread(
        target=_accept_loop,
        args=(listener, _port_of(socket_instance), _port_of(http_instance)),
        daemon=True,
    ).start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        listener.close()
