"""The model pool is read as one snapshot, `243a39dc9` (#27821, open-webui v0.11.1).

The direct-connection branches merged `request.app.state.MODELS` with `{**pool}`. On a
`RedisDict` that is HKEYS then one HGET per key, so a refresh deleting a key in between raised
`KeyError`; `dict(pool.items())` is one HGETALL. The race cannot be timed over HTTP, so an `ast`
sweep holds every merge in the source, today's and any added later, to the snapshot form.

The other fixes this file covered (the Ollama backend check under
`BYPASS_MODEL_ACCESS_CONTROL` and the api.anthropic.com headers) are pinned by
integration/models/test_model_registry.py.

Discriminates: passes on dev `bbfa876af`; fails with one task endpoint splatting the pool again.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import create_autospec, patch

import pytest

redis = pytest.importorskip("redis")

pytestmark = pytest.mark.regression


def _model_pool_splats(tree: ast.AST) -> list[int]:
    """Lines of dict literals that unpack an attribute named MODELS directly."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if key is None and isinstance(value, ast.Attribute) and value.attr == "MODELS"
    ]


def _model_pool_snapshots(tree: ast.AST) -> int:
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "items"
        and getattr(node.func.value, "attr", None) == "MODELS"
    )


def test_no_merge_splats_the_model_pool(open_webui_backend: Path):
    package = open_webui_backend / "open_webui"
    offenders, snapshots = [], 0
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders += [f"{path.relative_to(package)}:{line}" for line in _model_pool_splats(tree)]
        snapshots += _model_pool_snapshots(tree)

    assert snapshots, "no code reads MODELS.items() any more: retarget this audit at the merges"
    assert offenders == [], (
        f"these merges unpack the model pool directly: {offenders}. On a RedisDict that is one "
        "HGET per key, and a refresh deleting a key in between raises KeyError; use "
        "dict(MODELS.items()), a single HGETALL (#27821)"
    )


def test_a_pool_snapshot_survives_a_key_evicted_mid_read(owui_module):
    """Why the audit asks for items(): the splat walks keys the refresh already dropped."""
    socket_utils = owui_module("open_webui.socket.utils")
    client = create_autospec(redis.Redis, instance=True)
    client.hkeys.return_value = ["gpt-4o", "evicted-model"]
    client.hget.side_effect = lambda _name, key: '{"id": "gpt-4o"}' if key == "gpt-4o" else None
    client.hgetall.return_value = {"gpt-4o": '{"id": "gpt-4o"}'}
    with patch.object(socket_utils, "get_redis_connection", return_value=client):
        pool = socket_utils.RedisDict("models", redis_url="redis://unused")

    with pytest.raises(KeyError):
        {**pool}
    assert dict(pool.items()) == {"gpt-4o": {"id": "gpt-4o"}}
