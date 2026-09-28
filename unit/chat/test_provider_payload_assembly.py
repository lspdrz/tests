"""Every memory sort key breaks timestamp ties on the memory id, including ones added later.

`ff74bfa6a1` (#28292): every memory `sort_key` gained `memory.id` as the final tiebreak, so rows
sharing a timestamp come out in one order. The integration twin,
integration/chat/test_provider_payload_assembly.py, shows today's two sort keys doing so through
the memory search and path routes. This sweep stays because it also holds a sort key upstream
adds to `utils/memory.py` later, which no request of today reaches.

Discriminates: passes on bbfa876af; fails with `memory.id` dropped from a `sort_key`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression


def _functions(path: Path, name: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert found, f"no `{name}` in {path.name}; retarget this audit"
    return found


def _reads_id(node: ast.AST) -> bool:
    return any(isinstance(inner, ast.Attribute) and inner.attr == "id" for inner in ast.walk(node))


def test_every_memory_sort_key_breaks_ties_on_the_memory_id(open_webui_backend):
    memory_py = open_webui_backend / "open_webui" / "utils" / "memory.py"

    for sort_key in _functions(memory_py, "sort_key"):
        for returned in [node for node in ast.walk(sort_key) if isinstance(node, ast.Return)]:
            assert isinstance(returned.value, ast.Tuple), ast.unparse(returned)
            assert _reads_id(returned.value.elts[-1]), (
                f"line {returned.lineno}: `{ast.unparse(returned)}` has no id tiebreak, so rows "
                "sharing a timestamp come out in arrival order (#28292)"
            )
