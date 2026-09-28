"""Dependency contract: aiodns, the resolver that stays off unless an operator asks for it.

0.11.1 (c5ec01b1f, PR #28242) made aiodns resolution opt-in: c-ares broke name resolution on
some hosts (#28013, #28215), so `env.py` sets aiohttp's `DefaultResolver` aliases back to
`ThreadedResolver` unless `AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER=true`. With the flag on, a provider
reached by host name through aiodns is covered from outside in
integration/deps/test_connection_stack.py. That the flag is off by default stays here: both
resolvers answer the same names, so no request shows which one did. env.py is imported in a
fresh interpreter with and without the flag and the aliases it leaves behind are read back,
however env.py happens to spell the gate.

Discriminates: in a backend copy, a gate defaulting to on, a gate removed and one alias no
longer rewritten each fail the opt-in test.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.depcheck

OPT_IN = "AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER"

# The three aliases env.py rewrites: connectors read `aiohttp.connector.DefaultResolver`,
# plugin code the top-level one, so an alias left out keeps that path on c-ares.
RESOLVER_PROBE = """
import sys
sys.path.insert(0, sys.argv[1])
import aiohttp
import open_webui.env  # noqa: F401
modules = (aiohttp, aiohttp.resolver, aiohttp.connector)
print("RESOLVERS", " ".join(module.DefaultResolver.__name__ for module in modules))
"""


def _resolvers_once_env_ran(backend, opt_in: str | None) -> list[str]:
    """aiohttp's DefaultResolver aliases after importing env.py in a fresh interpreter."""
    env = {name: value for name, value in os.environ.items() if name != OPT_IN}
    if opt_in is not None:
        env[OPT_IN] = opt_in
    probe = [sys.executable, "-c", RESOLVER_PROBE, str(backend)]
    probed = subprocess.run(probe, capture_output=True, text=True, timeout=120, env=env)
    assert probed.returncode == 0, f"importing env.py failed\n{probed.stderr[-3000:]}"
    printed = [line for line in probed.stdout.splitlines() if line.startswith("RESOLVERS ")]
    assert printed, f"the probe printed no resolvers\n{probed.stdout[-3000:]}"
    return printed[-1].split()[1:]


def test_async_dns_is_opt_in_and_off_by_default(depcheck, open_webui_backend):
    """Importing env.py leaves every DefaultResolver alias on the threaded resolver unless the
    flag is set, and the flag hands them back to aiohttp's own aiodns default."""
    depcheck.load("aiodns")

    assert _resolvers_once_env_ran(open_webui_backend, None) == ["ThreadedResolver"] * 3, (
        "c-ares resolution is back on by default (#28013, #28215)"
    )
    assert _resolvers_once_env_ran(open_webui_backend, "true") == ["AsyncResolver"] * 3
