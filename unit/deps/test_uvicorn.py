"""Dependency contract: the uvicorn version the checkout pins, a packaging audit.

Open WebUI runs on uvicorn through two launchers, both driven end to end from outside:
`open-webui serve` and `open-webui dev` (``uvicorn.run`` with the forwarded-header, worker,
compression and reload options) in integration/deps/test_server_launch.py, and the container's
start.sh (the uvicorn command line) in integration/config/test_start_sh.py. What no request can
see is which uvicorn a release ships: live connections were cut by a routine keepalive check on
the legacy websocket implementation (#27553, issue #27550), fixed by moving to 0.51.0, and the
requirements file must not fall below that.

Discriminates: a requirements.txt pinning uvicorn==0.50.0 fails the floor test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

pytestmark = pytest.mark.depcheck

DIST_NAME = "uvicorn"


def test_requirements_pin_is_at_or_above_the_security_floor(open_webui_backend):
    """The checkout under test must not pin uvicorn below 0.51.0.

    Live connections were cut by a routine keepalive check on the legacy websocket
    implementation (#27553, issue #27550), fixed by moving to 0.51.0. A floor rather than an exact
    version, so an ordinary bump stays quiet and only a downgrade past the fix
    is reported. The file is parsed the way pip reads it, so extras, spacing,
    case and trailing comments do not matter.
    """
    pinned = _pinned_version(open_webui_backend / "requirements.txt", DIST_NAME)

    assert pinned >= Version("0.51.0"), (
        f"uvicorn is pinned at {pinned}, below the 0.51.0 that fixed "
        "the keepalive check cutting live connections (#27553)"
    )


def _pinned_version(requirements_txt: Path, dist_name: str) -> Version:
    """The exact `==` pin of `dist_name` in a requirements file, parsed as pip reads it."""
    for line in requirements_txt.read_text(encoding="utf-8").splitlines():
        spec = line.split("#", 1)[0].strip()
        if not spec or spec.startswith("-"):
            continue
        try:
            requirement = Requirement(spec)
        except InvalidRequirement:
            continue  # a URL or an option line, never a pin by name
        if canonicalize_name(requirement.name) != canonicalize_name(dist_name):
            continue
        exact = [pin.version for pin in requirement.specifier if pin.operator in ("==", "===")]
        assert exact, f"{dist_name} is not pinned exactly in requirements.txt: {spec!r}"
        return Version(exact[0])
    raise AssertionError(f"{dist_name} is not in requirements.txt")
