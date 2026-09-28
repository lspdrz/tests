"""Firecrawl checks that no route reaches (`open_webui/retrieval/web/firecrawl.py`).

- #23966 Bug 1 (NameError): in v0.9.1 `SafeFireCrawlLoader.lazy_load` called `requests.post`
  without importing `requests`, and `continue_on_failure` swallowed the NameError into an empty
  result. The Firecrawl path itself is pinned over HTTP; this audit holds every module under
  `retrieval/web/` to it.
- The timeout helpers also read a decimal or non-numeric setting and take a fallback, but the
  loader converts the setting to an integer first and nothing passes a fallback, so no route
  reaches those cases.

Request shapes, both search answer shapes (#23966 Bug 2), the header without a key, retries,
`Retry-After`, dropped connections, the reachable timeout settings and the client timeout moved
to integration/retrieval/test_firecrawl.py.

Discriminates: passes on dev bbfa876af; deleting `import requests` from firecrawl.py fails the
audit, and dropping the float parse or the fallback fails the helper cases.
"""

from __future__ import annotations

import ast

import pytest


def module_level_names(tree: ast.Module) -> set[str]:
    """Names bound by imports at module scope, including inside top-level `try` and `if`."""
    names: set[str] = set()
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.If, ast.Try)):
            pending += node.body + node.orelse + getattr(node, "finalbody", [])
            pending += [
                child for handler in getattr(node, "handlers", []) for child in handler.body
            ]
    return names


def calls_requests(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "requests"
        for node in ast.walk(tree)
    )


@pytest.mark.regression
def test_every_web_module_calling_requests_imports_it(open_webui_backend):
    web_dir = open_webui_backend / "open_webui" / "retrieval" / "web"
    modules = sorted(web_dir.rglob("*.py"))
    assert (web_dir / "firecrawl.py") in modules, f"retarget: firecrawl.py is gone from {web_dir}"

    offenders = []
    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        if calls_requests(tree) and "requests" not in module_level_names(tree):
            offenders.append(module.relative_to(open_webui_backend).as_posix())

    assert not offenders, f"#23966: these call requests.<x>() without importing it: {offenders}"


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("45.5", 45.5), ("soon", None)],
)
def test_a_setting_the_loader_never_passes_still_parses(firecrawl_module, value, seconds):
    assert firecrawl_module.get_firecrawl_timeout_seconds(timeout=value) == seconds


def test_the_client_timeout_falls_back_ten_seconds_past_the_default(firecrawl_module):
    assert (
        firecrawl_module.get_firecrawl_client_timeout_seconds(timeout=None, fallback=120) == 130.0
    )
