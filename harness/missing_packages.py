"""An instance whose Python cannot import some installed packages, as if they were never installed.

`pip install open-webui` leaves out the optional `unstructured` extra, and Open WebUI then reads
spreadsheets and slides with its own fallback loaders. `without_packages_env(directory, names)`
writes a `sitecustomize` there that makes every import of those packages (and their submodules)
raise `ModuleNotFoundError` before Open WebUI is imported, and returns the environment of an
instance booted on it.
"""

from __future__ import annotations

import os
from pathlib import Path

SITECUSTOMIZE = """
import importlib.abc
import sys

HIDDEN = {names!r}


class _NotInstalled(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in HIDDEN:
            raise ModuleNotFoundError(f"No module named {{fullname!r}}", name=fullname)
        return None


sys.meta_path.insert(0, _NotInstalled())
"""


def without_packages_env(directory: Path, names: list[str]) -> dict[str, str]:
    (directory / "sitecustomize.py").write_text(
        SITECUSTOMIZE.format(names=sorted(names)), encoding="utf-8"
    )
    inherited = os.environ.get("PYTHONPATH")
    search_path = f"{directory}{os.pathsep}{inherited}" if inherited else str(directory)
    return {"PYTHONPATH": search_path}
