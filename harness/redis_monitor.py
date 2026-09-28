"""Every command a real Redis runs, read off its MONITOR stream while a block runs.

`recording(url)` opens a MONITOR connection to the Redis at `url` and records each command it
reports, arguments decoded, until the block ends. `sent(name, key)` lists the arguments of every
command of that name (on that key, when given) so far; commands a Lua script ran are left out,
so an EVAL shows up once, as the EVAL. The recorder is how a test reads what an instance asked
its Redis for, on a server the instance really uses.
"""

from __future__ import annotations

import contextlib
import re
import socket
import threading
from dataclasses import dataclass, field
from typing import Iterator
from urllib.parse import urlparse

# `1700000000.123456 [0 127.0.0.1:50000] "SET" "key" "value"`
MONITOR_LINE = re.compile(r"^\+?[\d.]+ \[\d+ (?P<client>[^\]]*)\] (?P<command>.*)$")
QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')


@dataclass
class RedisCommand:
    name: str
    args: list[str]


@dataclass
class RedisRecorder:
    commands: list[RedisCommand] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def sent(self, name: str, key: str | None = None) -> list[list[str]]:
        with self.lock:
            return [
                command.args
                for command in self.commands
                if command.name == name.upper() and (key is None or command.args[:1] == [key])
            ]

    def clear(self) -> None:
        with self.lock:
            self.commands.clear()


def _unescape(token: str) -> str:
    return token.encode("latin-1", "backslashreplace").decode("unicode_escape")


def _record(connection: socket.socket, recorder: RedisRecorder) -> None:
    buffer = b""
    while True:
        try:
            chunk = connection.recv(65536)
        except OSError:
            return
        if not chunk:
            return
        buffer += chunk
        *lines, buffer = buffer.split(b"\r\n")
        for raw in lines:
            match = MONITOR_LINE.match(raw.decode("utf-8", "replace"))
            if not match or match["client"] == "lua":
                continue
            tokens = [_unescape(token) for token in QUOTED.findall(match["command"])]
            if tokens:
                with recorder.lock:
                    recorder.commands.append(RedisCommand(tokens[0].upper(), tokens[1:]))


@contextlib.contextmanager
def recording(url: str) -> Iterator[RedisRecorder]:
    address = urlparse(url)
    connection = socket.create_connection((address.hostname, address.port or 6379), timeout=10)
    connection.settimeout(None)
    connection.sendall(b"MONITOR\r\n")
    recorder = RedisRecorder()
    reader = threading.Thread(target=_record, args=(connection, recorder), daemon=True)
    reader.start()
    try:
        yield recorder
    finally:
        with contextlib.suppress(OSError):
            connection.shutdown(socket.SHUT_RDWR)
        connection.close()
        reader.join(timeout=5)
