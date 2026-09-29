"""One instance per resolver, shared by every module in this package.

`AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` is read once at boot, so each value needs an instance of its
own. `resolver` parametrizes a test over both values (`threaded` for off, `c-ares` for on) and
`resolving_instance` is the instance booted with it; both boot once for the whole package, since
every module here drives the same pair. `resolving_admin` is that instance's admin.
"""

from __future__ import annotations

from typing import Callable

import pytest

from harness.actors import Actor, admin_of
from harness.host_names import RESOLVERS
from harness.instance import LaunchedInstance


@pytest.fixture(scope="package", params=list(RESOLVERS))
def resolver(request: pytest.FixtureRequest) -> str:
    if request.param == "c-ares":
        pytest.importorskip("aiodns", reason="aiodns not installed, so the flag has no resolver")
    return request.param


@pytest.fixture
def resolving_instance(
    resolver: str, package_instance_with: Callable[[dict[str, str]], LaunchedInstance]
) -> LaunchedInstance:
    return package_instance_with(RESOLVERS[resolver])


@pytest.fixture
def resolving_admin(resolving_instance: LaunchedInstance) -> Actor:
    return admin_of(resolving_instance)
