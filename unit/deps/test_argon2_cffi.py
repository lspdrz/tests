"""Dependency contract: argon2-cffi (import name ``argon2``), the calls Open WebUI never makes.

With `PASSWORD_HASH_ALGORITHM=argon2`, `utils/auth.py` hashes with `PasswordHasher().hash` and
verifies with `PasswordHasher().verify`, catching `InvalidHashError` and `VerificationError`.
Those are driven from outside in integration/deps/test_auth_stack.py: sign-up and sign-in on an
argon2 instance, accounts signing in across a switch of the algorithm in both directions,
salted hashes and damaged hashes of both error kinds. What stays here is the rest of the public
surface a swap-in or passlib would use (cost parameters, `check_needs_rehash`, the low-level
API), which no Open WebUI feature calls.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck


def test_password_hasher_has_check_needs_rehash(depcheck):
    mod = depcheck.load("argon2")
    assert "check_needs_rehash" in dir(mod.PasswordHasher)


def test_password_hasher_constructs_with_cost_params(depcheck):
    """PasswordHasher accepts the time, memory and parallelism cost knobs as keywords."""
    mod = depcheck.load("argon2")
    depcheck.assert_params(
        mod.PasswordHasher.__init__,
        ["time_cost", "memory_cost", "parallelism"],
    )
    hasher = mod.PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)
    assert hasher.verify(hasher.hash("low cost"), "low cost") is True


def test_low_level_api_present(depcheck):
    """argon2.low_level exposes hash_secret / verify_secret and the Type enum, the primitives
    passlib's argon2 handler uses."""
    mod = depcheck.load("argon2")
    low = mod.low_level
    for name in ("hash_secret", "verify_secret"):
        assert callable(getattr(low, name, None)), f"argon2.low_level.{name} missing"
    assert hasattr(mod.Type, "ID")
