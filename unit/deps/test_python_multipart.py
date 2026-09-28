"""Packaging audit: the checkout pins python-multipart at or above its security fix.

python-multipart parses every upload and form: how it reads a file of any bytes and any name,
the fields beside it, a body it cannot parse and an urlencoded form is driven from outside in
integration/deps/test_request_bodies.py, integration/deps/test_audio_stack.py (the language
field of a recording) and integration/auth/test_sso_account_sync.py (the back-channel logout
form). What stays here is the pin itself, which no request can see.

Discriminates: a requirements.txt pinning python-multipart==0.0.31 fails the floor check.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

pytestmark = pytest.mark.depcheck

DIST_NAME = "python-multipart"


def test_requirements_pin_is_at_or_above_the_security_floor(open_webui_backend):
    """The checkout under test must not pin python-multipart below 0.0.32.

    A security advisory in the multipart parsing path (#26991) was fixed by moving to
    0.0.32. A floor rather than an exact
    version, so an ordinary bump stays quiet and only a downgrade past the fix
    is reported. The file is parsed the way pip reads it, so extras, spacing,
    case and trailing comments do not matter.
    """
    pinned = _pinned_version(open_webui_backend / "requirements.txt", DIST_NAME)

    assert pinned >= Version("0.0.32"), (
        f"python-multipart is pinned at {pinned}, below the 0.0.32 that fixed "
        "a security advisory in the multipart parsing path (#26991)"
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
