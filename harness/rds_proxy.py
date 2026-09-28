"""A Postgres front door that asks every client for its password, the way RDS does, and records it.

Stands between Open WebUI and an embedded Postgres (`pgserver`, which trusts every local
connection): each client is asked for a cleartext password, which is recorded with the user and
database it signed in as, and the connection is then handed through to the real server over its
Unix socket. So a test reads exactly which password each engine sent, and the instance still runs
on a real database. `serving_rds_proxy(socket_path)` runs one on a free port, or on `port`.
"""

from __future__ import annotations

import socket
import struct
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

SSL_REQUEST = 80877103
GSS_REQUEST = 80877104
CLEARTEXT_PASSWORD_REQUEST = b"R" + struct.pack("!ii", 8, 3)


@dataclass
class SignIn:
    user: str
    database: str
    password: str


@dataclass
class RdsProxy:
    port: int
    sign_ins: list[SignIn] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def passwords_for(self, database: str) -> set[str]:
        with self.lock:
            return {entry.password for entry in self.sign_ins if entry.database == database}


def _read_exactly(connection: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        received = connection.recv(size - len(data))
        if not received:
            raise ConnectionError("the client hung up during sign-in")
        data += received
    return data


def _startup_packet(client: socket.socket) -> bytes:
    """The client's startup packet, after declining its SSL and GSS encryption requests."""
    while True:
        length = struct.unpack("!i", _read_exactly(client, 4))[0]
        body = _read_exactly(client, length - 4)
        if struct.unpack("!i", body[:4])[0] in (SSL_REQUEST, GSS_REQUEST):
            client.sendall(b"N")
            continue
        return struct.pack("!i", length) + body


def _parameters(startup: bytes) -> dict[str, str]:
    fields = startup[8:].split(b"\0")
    return {
        fields[index].decode(): fields[index + 1].decode()
        for index in range(0, len(fields) - 1, 2)
        if fields[index]
    }


def _pipe(source: socket.socket, target: socket.socket) -> None:
    try:
        while data := source.recv(65536):
            target.sendall(data)
    except OSError:
        pass
    finally:
        for end in (source, target):
            try:
                end.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _serve_client(proxy: RdsProxy, client: socket.socket, socket_path: str) -> None:
    with client:
        try:
            startup = _startup_packet(client)
            client.sendall(CLEARTEXT_PASSWORD_REQUEST)
            _, length = struct.unpack("!ci", _read_exactly(client, 5))
            password = _read_exactly(client, length - 4).rstrip(b"\0").decode()
        except (ConnectionError, OSError, struct.error):
            return
        parameters = _parameters(startup)
        with proxy.lock:
            proxy.sign_ins.append(
                SignIn(parameters.get("user", ""), parameters.get("database", ""), password)
            )
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.connect(socket_path)
            server.sendall(startup)
            relay = threading.Thread(target=_pipe, args=(server, client), daemon=True)
            relay.start()
            _pipe(client, server)
            relay.join(timeout=5)


@contextmanager
def serving_rds_proxy(socket_path: str, port: int = 0) -> Iterator[RdsProxy]:
    listening = socket.create_server(("127.0.0.1", port))
    proxy = RdsProxy(port=listening.getsockname()[1])

    def accept() -> None:
        while True:
            try:
                client, _ = listening.accept()
            except OSError:
                return
            threading.Thread(
                target=_serve_client, args=(proxy, client, socket_path), daemon=True
            ).start()

    threading.Thread(target=accept, daemon=True).start()
    try:
        yield proxy
    finally:
        listening.close()
