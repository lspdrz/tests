"""Guard: no router is mounted twice, and no two routers share a prefix.

A router is only reachable because main.py hands it to `app.include_router`. A copy-pasted mount
line serves a router twice on one prefix, and two routers mounted on one prefix share it until a
later route collides and FastAPI serves only the one it matches first. Neither shows in a request
yet, which is why this stays a source audit; integration/imports/test_router_wiring.py covers
what a request sees (an unmounted router, one served under two prefixes, a collision).

An `ast` audit of main.py; nothing is imported. The name a mount uses is resolved through
main.py's own imports, so aliases and formatting do not matter, and a commented-out mount does
not count.

Discriminates: passes on dev bbfa876af; in a copy of it, repeating the `utils` mount line fails
the first test and mounting a second router on an existing prefix fails the second.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROUTERS_PACKAGE = "open_webui.routers"


def _imported_routers(tree: ast.Module) -> dict[str, str]:
    """Local name -> router module, for every `from open_webui.routers import x as y`."""
    return {
        alias.asname or alias.name: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == ROUTERS_PACKAGE
        for alias in node.names
    }


def _prefix(mount: ast.Call) -> str:
    prefix = next((keyword.value for keyword in mount.keywords if keyword.arg == "prefix"), None)
    if prefix is None:
        return ""
    return prefix.value if isinstance(prefix, ast.Constant) else ast.unparse(prefix)


def _mounts(backend: Path) -> list[tuple[str, str]]:
    """(mounted name, prefix) for every `include_router(<name>.router, prefix=...)`."""
    tree = ast.parse((backend / "open_webui" / "main.py").read_text(encoding="utf-8"))
    local_names = _imported_routers(tree)
    mounts = []
    for node in ast.walk(tree):
        is_mount = isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "include_router"
        router = node.args[0] if is_mount and node.args else None
        if not (isinstance(router, ast.Attribute) and isinstance(router.value, ast.Name)):
            continue
        mounts.append((local_names.get(router.value.id, router.value.id), _prefix(node)))
    if not mounts:
        pytest.fail("main.py contains no include_router calls; the audit needs retargeting")
    return mounts


@pytest.fixture(scope="module")
def mounted(open_webui_backend: Path) -> list[tuple[str, str]]:
    return _mounts(open_webui_backend)


def test_no_router_is_mounted_twice(mounted) -> None:
    names = [name for name, _ in mounted]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"routers mounted more than once: {duplicates}"


def test_every_prefix_is_unique(mounted) -> None:
    """Two routers on one prefix leave the second one's colliding endpoints unreachable."""
    prefixes = [prefix for _, prefix in mounted]
    collisions = {
        prefix: sorted(name for name, other in mounted if other == prefix)
        for prefix in prefixes
        if prefixes.count(prefix) > 1
    }
    assert not collisions, f"routers sharing a prefix: {collisions}"
