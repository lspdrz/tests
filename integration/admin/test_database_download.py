"""Open bug: the admin's database download leaves out what is still in SQLite's write-ahead log.

Settings > Database > Download database (`GET /api/v1/utils/db/download`) serves `webui.db` as it
lies on disk. SQLite runs in WAL mode by default (`DATABASE_ENABLE_SQLITE_WAL`), so every write
since the last checkpoint sits in `webui.db-wal` and is missing from the download: on a fresh
instance not even the admin account is in it. A backup taken this way silently loses the latest
accounts, chats and settings. No fix yet; the account test stays red until the download
checkpoints first or serves a consistent copy.

Discriminates: fails on dev ef67cc3fa (the download holds no account at all); passes on an
instance booted with `DATABASE_ENABLE_SQLITE_WAL=false`, where every write lands in `webui.db`.
"""

from __future__ import annotations

import contextlib
import sqlite3
import tempfile

import pytest

from harness.actors import create_user

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def _emails_in_download(instance) -> set[str]:
    if not instance.database_url.startswith("sqlite"):
        pytest.skip("the database download is SQLite only")
    with instance.client() as client:
        downloaded = client.get("/api/v1/utils/db/download")
    assert downloaded.status_code == 200, downloaded.text
    with tempfile.NamedTemporaryFile(suffix=".db") as copy:
        copy.write(downloaded.content)
        copy.flush()
        with contextlib.closing(sqlite3.connect(copy.name)) as database:
            return {email for (email,) in database.execute("SELECT email FROM auth")}


def test_the_database_download_holds_the_accounts_just_added(instance, make_user):
    account = make_user()

    emails = _emails_in_download(instance)

    assert account.email in emails, (
        f"the downloaded database holds {len(emails)} accounts and not the one just added: "
        "writes still in webui.db-wal are left out"
    )


@pytest.mark.slow
def test_without_the_write_ahead_log_the_download_holds_the_accounts(instance_with):
    without_wal = instance_with({"DATABASE_ENABLE_SQLITE_WAL": "false"})
    account = create_user(without_wal)

    assert account.email in _emails_in_download(without_wal)
