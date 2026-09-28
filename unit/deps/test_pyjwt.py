"""Dependency contract: every PyJWT name and call the backend's source uses, read with `ast`.

The behaviour Open WebUI relies on from PyJWT is driven from outside: session tokens (minted
elsewhere with the server's key, another algorithm, unsigned, another key, not a JWT, expired) in
integration/deps/test_auth_stack.py, the user-info JWT forwarded to a provider there too, and
the back-channel logout token (the key its `kid` names, audience, issuer, events, expiry) in
integration/auth/test_sso_account_sync.py and integration/security/test_oauth_identity.py.

What stays here is a sweep over the source: every `jwt.<name>` the backend writes must exist in
the installed PyJWT, and every call it makes must bind to that name's signature. It also covers
a call upstream adds later, which no request can find. A hand-kept list once pinned
`PyJWKClient` after the logout path had moved to `PyJWKSet`.

Discriminates: in a backend copy, a `jwt` name PyJWT lacks fails the inventory and a keyword
`jwt.encode` does not take fails the call check.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import NamedTuple

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "jwt"


class JwtUse(NamedTuple):
    where: str
    name: str  # "PyJWKSet.from_dict" for `jwt.PyJWKSet.from_dict`
    call: ast.Call | None  # the call, when the name is called right there


def _dotted(node: ast.AST) -> list[str]:
    """`jwt.PyJWKSet.from_dict` as ["jwt", "PyJWKSet", "from_dict"]; [] for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.insert(0, node.attr)
        node = node.value
    return [node.id, *parts] if isinstance(node, ast.Name) and parts else []


def _backend_jwt_uses(backend: Path) -> list[JwtUse]:
    """Every `jwt.<...>` the backend writes, where `jwt` is PyJWT."""
    uses = []
    for path in sorted((backend / "open_webui").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "import jwt" not in source:
            continue
        tree = ast.parse(source)
        aliases = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name == IMPORT_NAME
        }
        calls = {id(node.func): node for node in ast.walk(tree) if isinstance(node, ast.Call)}
        for node in ast.walk(tree):
            chain = _dotted(node)
            if chain and chain[0] in aliases:
                where = f"{path.relative_to(backend)}:{node.lineno}"
                uses.append(JwtUse(where, ".".join(chain[1:]), calls.get(id(node))))
    assert uses, f"nothing under {backend} uses `import jwt`; retarget this contract"
    return uses


def test_every_jwt_name_the_backend_uses_exists(depcheck, open_webui_backend):
    mod = depcheck.load(IMPORT_NAME)
    missing = sorted(
        f"{use.where} jwt.{use.name}"
        for use in _backend_jwt_uses(open_webui_backend)
        if not depcheck.has(mod, use.name)
    )
    assert not missing, f"PyJWT lacks names the backend uses: {missing}"


def test_every_jwt_call_the_backend_makes_binds(depcheck, open_webui_backend):
    """`jwt.encode(payload, key, algorithm=...)`, `jwt.decode(token, key, ...)` and the rest,
    bound the way the backend passes them, so what it passes by position is not pinned by name."""
    mod = depcheck.load(IMPORT_NAME)
    problems = []
    for use in _backend_jwt_uses(open_webui_backend):
        if use.call is None or not depcheck.has(mod, use.name):
            continue
        positional = [None for argument in use.call.args if not isinstance(argument, ast.Starred)]
        keywords = {keyword.arg: None for keyword in use.call.keywords if keyword.arg}
        try:
            inspect.signature(depcheck.resolve(mod, use.name)).bind_partial(*positional, **keywords)
        except TypeError as error:
            problems.append(f"{use.where} jwt.{use.name}(...): {error}")
        except ValueError:
            continue  # no introspectable signature (a builtin exception class)
    assert not problems, f"backend calls PyJWT no longer accepts: {problems}"
