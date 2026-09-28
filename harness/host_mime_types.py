"""An instance whose Python reads the host's MIME table the way a Windows registry leaves it.

Windows hosts carry registry entries that `mimetypes` reads at init, and they commonly map `.js`,
`.mjs` and `.wasm` to `text/plain`. `mistyped_host_env(directory)` writes a `sitecustomize` there
that loads such a table when the interpreter starts, before Open WebUI is imported, and returns
the environment of an instance booted on it. The table also types `CONTROL_EXTENSION`, which Open
WebUI never registers, as `CONTROL_TYPE`: a file of that kind served back shows the table is live.
"""

from __future__ import annotations

import os
from pathlib import Path

HOST_WRONG_TYPE = "text/plain"
MISTYPED_EXTENSIONS = (".js", ".mjs", ".wasm")
CONTROL_EXTENSION = ".owui-mime-probe"
CONTROL_TYPE = "application/x-host-registry"

SITECUSTOMIZE = f"""
import mimetypes

mimetypes.init()
for extension in {MISTYPED_EXTENSIONS!r}:
    mimetypes.add_type({HOST_WRONG_TYPE!r}, extension)
mimetypes.add_type({CONTROL_TYPE!r}, {CONTROL_EXTENSION!r})
"""


def mistyped_host_env(directory: Path) -> dict[str, str]:
    (directory / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    inherited = os.environ.get("PYTHONPATH")
    search_path = f"{directory}{os.pathsep}{inherited}" if inherited else str(directory)
    return {"PYTHONPATH": search_path}
