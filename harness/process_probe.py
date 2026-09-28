"""Read what the server process holds, through a pipe function the admin installs.

Some leaks are a few hundred bytes per request, far below what the process size can show. The
probe is a pipe model: asked anything, it walks the globals of every loaded `open_webui` module
and answers with the length of each module-level dict, list and set, and with how many strings
reachable from those globals contain the text of the question. It knows no container by name,
so a registry upstream moves or renames is still measured.

`probing(admin)` installs the probe for the length of a test; `probe.containers()` and
`probe.strings_containing(marker)` ask it, and `grown(before, after)` names what got bigger.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from harness.actors import Actor
from harness.plugins import installed_function

PROBE = """
import json
import sys
import types

MAX_DEPTH = 6
MAX_VISITS = 500_000


def _globals():
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("open_webui"):
            continue
        for name, value in list(vars(module).items()):
            if not name.startswith("__"):
                yield f"{module_name}.{name}", value


def _containers():
    return {
        name: len(value)
        for name, value in _globals()
        if isinstance(value, (dict, list, set))
    }


def _children(value):
    if isinstance(value, dict):
        return [*value.keys(), *value.values()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    skipped = (types.ModuleType, type, types.FunctionType, types.MethodType)
    if hasattr(value, "__dict__") and not isinstance(value, skipped):
        return list(vars(value).values())
    return []


def _strings_containing(marker):
    seen, count = set(), 0
    pending = [(value, 0) for _, value in _globals()]
    while pending and len(seen) < MAX_VISITS:
        value, depth = pending.pop()
        if isinstance(value, str):
            count += marker in value
            continue
        if id(value) in seen or depth >= MAX_DEPTH:
            continue
        seen.add(id(value))
        pending.extend((child, depth + 1) for child in _children(value))
    return count


class Pipe:
    def pipe(self, body: dict) -> str:
        marker = body["messages"][-1]["content"]
        return json.dumps(
            {"containers": _containers(), "strings": _strings_containing(marker)}
        )
"""


@dataclass
class Probe:
    admin: Actor
    model_id: str

    def _ask(self, marker: str) -> dict:
        with self.admin.client() as client:
            answer = client.post(
                "/api/chat/completions",
                json={
                    "model": self.model_id,
                    "stream": False,
                    "messages": [{"role": "user", "content": marker}],
                },
            )
        assert answer.status_code == 200, f"the probe did not answer: {answer.text}"
        return json.loads(answer.json()["choices"][0]["message"]["content"])

    def containers(self) -> dict[str, int]:
        """The length of every module-level container, by `module.name`."""
        return self._ask("no marker")["containers"]

    def strings_containing(self, marker: str) -> int:
        """How many strings the process holds, reachable from module globals, contain `marker`."""
        return self._ask(marker)["strings"]


def grown(before: dict[str, int], after: dict[str, int], by: int = 1) -> dict[str, int]:
    """The containers that gained at least `by` entries, with their growth."""
    growth = {name: size - before.get(name, 0) for name, size in after.items()}
    return {name: amount for name, amount in growth.items() if amount >= by}


@contextmanager
def probing(admin: Actor) -> Iterator[Probe]:
    with installed_function(admin, PROBE) as function_id:
        with admin.client() as client:
            client.get("/api/models").raise_for_status()  # registers the pipe as a model
        yield Probe(admin, function_id)
