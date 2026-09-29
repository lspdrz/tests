"""Host names for a local service, as an operator writes them into a setting.

aiohttp resolves a host name through the resolver `AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` picks (the
threaded OS resolver by default, c-ares through aiodns when it is on) and connects to an IP literal
without asking either, so a test of that switch reaches its fakes by name. A `NameForm` is one way
of naming a local service: `localhost` answers over both address families, `ipv6-only` is a hosts-
file name for `::1` alone and `hosts-file` a hosts-file name for another local address.
`name_forms()` gives the three as pytest params; the last two skip where the machine's hosts file
has no such name. `serving_by_name(form)` starts a `listener` on the form's address and names it by
the form's host. `UNRESOLVABLE` is a name no resolver answers (RFC 6761 reserves `.invalid`); either
resolver refuses it well inside `FAILS_WITHIN` seconds where the DNS server answers, which
`timed(call)` measures. `resolver_reason(text)` is the resolver's own wording of that failure, the
bracketed tail aiohttp ends a connection error with: c-ares words it differently from the OS
(`C_ARES_REASONS`), which is how a test tells which resolver answered.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import os
import re
import socket
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, TypeVar
from urllib.parse import urlsplit, urlunsplit

import pytest

from harness.listener import Listener, listening

ASYNC_DNS = "AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER"
RESOLVERS = {"threaded": {ASYNC_DNS: "false"}, "c-ares": {ASYNC_DNS: "true"}}
UNRESOLVABLE = "owui-unresolvable.invalid"
FAILS_WITHIN = 5.0  # seconds; the OS resolver refuses an unknown name in milliseconds

# pycares' messages for a name it cannot answer, which the OS resolver never uses
C_ARES_REASONS = (
    "Domain name not found",
    "DNS server returned answer with no data",
    "Could not contact DNS servers",
    "Timeout while contacting DNS servers",
    "DNS server returned general failure",
    "DNS server refused query",
)

T = TypeVar("T")

_BRACKETED_TAIL = re.compile(r"\[([^\[\]]*)\]\s*$")


@dataclass(frozen=True)
class NameForm:
    label: str
    host: str
    address: str


def _hosts_file() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/drivers/etc/hosts"
    return Path("/etc/hosts")


def _hosts_file_entries() -> list[tuple[str, str]]:
    """(address, name) pairs from the hosts file, comments dropped."""
    try:
        lines = _hosts_file().read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    entries = []
    for line in lines:
        fields = line.split("#", 1)[0].split()
        if len(fields) >= 2:
            entries.extend((fields[0], name) for name in fields[1:])
    return entries


def _os_addresses(host: str) -> set[str]:
    try:
        return {info[4][0] for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)}
    except OSError:
        return set()


def _bindable(address: str) -> bool:
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    try:
        with socket.socket(family) as sock:
            sock.bind((address, 0))
    except OSError:
        return False
    return True


def _hosts_file_name(wanted_family: int) -> NameForm | None:
    """A hosts-file name the OS answers with one bindable address of that family, not localhost."""
    for address, name in _hosts_file_entries():
        if name.lower() == "localhost" or name.lower().endswith(".localhost"):
            continue
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        family = socket.AF_INET6 if parsed.version == 6 else socket.AF_INET
        if family != wanted_family or _os_addresses(name) != {address}:
            continue
        if _bindable(address):
            label = "ipv6-only" if family == socket.AF_INET6 else "hosts-file"
            return NameForm(label, name, address)
    return None


def find_name_form(label: str) -> NameForm | None:
    if label == "localhost":
        return NameForm("localhost", "localhost", "127.0.0.1")
    family = socket.AF_INET6 if label == "ipv6-only" else socket.AF_INET
    return _hosts_file_name(family)


def name_forms() -> list:
    """The three name forms as pytest params, ids by label, resolved when a test asks for one."""
    return [pytest.param(label, id=label) for label in ("localhost", "ipv6-only", "hosts-file")]


def name_form(label: str) -> NameForm:
    form = find_name_form(label)
    if form is None:
        pytest.skip(f"the hosts file names no {label} address this machine can listen on")
    return form


def by_name(url: str, host: str) -> str:
    """`url` with its host swapped for `host`, port and path kept."""
    parts = urlsplit(url)
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit(parts._replace(netloc=netloc))


@contextmanager
def serving_by_name(label: str) -> Iterator[Listener]:
    """A `listener` on the name form's address whose `base_url` names it by the form's host."""
    form = name_form(label)
    with listening(form.address) as service:
        yield dataclasses.replace(service, base_url=f"http://{form.host}:{service.port}")


def timed(call: Callable[..., T], *args, **kwargs) -> tuple[T, float]:
    """What `call` returned and how many seconds it took."""
    started = time.monotonic()
    result = call(*args, **kwargs)
    return result, time.monotonic() - started


def resolver_reason(text: str) -> str | None:
    """The resolver's wording at the end of an aiohttp connection error, if the text ends in one."""
    tail = _BRACKETED_TAIL.search(text.strip())
    return tail.group(1) if tail else None


def without_resolver_reason(text: str) -> str:
    """The error text with the resolver's wording replaced, so both resolvers' errors compare."""
    return _BRACKETED_TAIL.sub("[<resolver reason>]", text.strip())
