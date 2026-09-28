"""Dependency contract: alembic, the one behaviour no Open WebUI command reaches.

Alembic builds and evolves the schema: every boot runs `alembic upgrade head` through
`config.py`, the migrations use `op` and `batch_alter_table`, and operators run the manual
commands from the migration guide. All of that is driven from outside in
integration/migrations/test_lifecycle.py (a new install on SQLite and Postgres, a restart, the
manual upgrade, downgrade and SQL preview) and integration/migrations/test_upgrade_from_release.py.
What stays here is the `alembic.context` proxy outside a migration run: `env.py` only calls it
inside one, so no command shows that it raises there.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.depcheck


def test_context_proxy_requires_active_environment(depcheck):
    """The `alembic.context` proxies are bound to an EnvironmentContext that only exists during
    a migration run, which is why `env.py` calls them from its run path. Run in a clean
    subprocess: importing open_webui.config elsewhere in the suite binds the proxy."""
    depcheck.load("alembic")
    code = (
        "import alembic.context as c\n"
        "try:\n"
        "    c.is_offline_mode()\n"
        "    print('NO_RAISE')\n"
        "except NameError:\n"
        "    print('RAISED_NAMEERROR')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert "RAISED_NAMEERROR" in out.stdout, (
        "alembic.context.is_offline_mode() must raise NameError outside a migration "
        f"run (stdout={out.stdout!r}, stderr={out.stderr[-500:]!r})"
    )
