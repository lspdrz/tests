"""Dependency smoke: the database drivers every request reads and writes through.

Every request's query runs on SQLAlchemy's async engine. On SQLite that drives aiosqlite
(`sqlite://` becomes `sqlite+aiosqlite://`); on Postgres it drives psycopg 3
(`postgresql+psycopg://`), while the synchronous engine that migrates and loads the settings at
boot drives psycopg2. A note saved, changed and deleted over the API must read back from the next
request exactly as written, and writes sent at once must all land. A bump that breaks a driver's
commit, parameter binding or row access loses the write or garbles what reads back. Each test runs
on SQLite (skipped when the whole run is on Postgres) and on a Postgres database of its own on the
embedded server, whose URL names SSL the way some ORMs write it (`ssl=disable`): Open WebUI hands
it to psycopg2 and psycopg as libpq's `sslmode`, which neither would take in the bare form.
pgvector's use of psycopg2 is driven in test_vector_stores.py, Alembic's migrations in
integration/migrations. Twin of unit/deps/test_psycopg.py and unit/deps/test_psycopg2_binary.py.

Discriminates: passes on dev ef67cc3fa; in a backend copy with `aiosqlite.Connection.commit`
made a no-op every write is lost, down to the admin account the boot signs up, so both SQLite
cases fail. With `psycopg.AsyncConnection.commit` made a no-op both Postgres cases fail, and
with the bare `ssl` key passed on untranslated the Postgres instance no longer boots.
"""

from __future__ import annotations

import concurrent.futures
import uuid

import pytest

from harness import backends
from harness.actors import Actor, create_user

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source]

# quotes, a placeholder look-alike and characters beyond ASCII, so binding and decoding show
AWKWARD_TEXT = 'O\'Brien said "?" and :name; Grüße, 日本語, emoji \U0001f99c'
PARALLEL_WRITES = 12


def _create_note(client, title: str, markdown: str) -> dict:
    created = client.post(
        "/api/v1/notes/create", json={"title": title, "data": {"content": {"md": markdown}}}
    )
    assert created.status_code == 200, created.text
    return created.json()


def _stored_markdown(client, note_id: str) -> str:
    stored = client.get(f"/api/v1/notes/{note_id}")
    assert stored.status_code == 200, stored.text
    return stored.json()["data"]["content"]["md"]


@pytest.fixture(scope="module")
def postgres_url():
    pytest.importorskip("pgserver", reason="the Postgres case runs on the embedded server")
    with backends.postgres_database() as url:
        yield url


@pytest.fixture(params=["sqlite", "postgres"])
def author(request, instance_with) -> Actor:
    """A fresh account on an instance keeping its data in the parametrised database."""
    if request.param == "sqlite":
        if backends.DATABASE == "postgres":
            pytest.skip("this run keeps every instance on Postgres")
        return create_user(request.getfixturevalue("instance"))
    url = request.getfixturevalue("postgres_url")
    separator = "&" if "?" in url else "?"
    return create_user(instance_with({"DATABASE_URL": f"{url}{separator}ssl=disable"}))


def test_a_note_reads_back_changed_and_deleted_as_written(author):
    with author.client() as client:
        note = _create_note(client, f"awkward {uuid.uuid4().hex[:6]}", AWKWARD_TEXT)
        assert _stored_markdown(client, note["id"]) == AWKWARD_TEXT

        updated = client.post(
            f"/api/v1/notes/{note['id']}/update",
            json={"title": note["title"], "data": {"content": {"md": AWKWARD_TEXT * 2}}},
        )
        assert updated.status_code == 200, updated.text
        assert _stored_markdown(client, note["id"]) == AWKWARD_TEXT * 2

        deleted = client.delete(f"/api/v1/notes/{note['id']}/delete")
        assert deleted.status_code == 200 and deleted.json() is True, deleted.text
        assert client.get(f"/api/v1/notes/{note['id']}").status_code == 404


def test_notes_written_at_once_all_land(author):
    titles = [f"parallel {index} {uuid.uuid4().hex[:6]}" for index in range(PARALLEL_WRITES)]

    with author.client() as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            created = list(pool.map(lambda title: _create_note(client, title, title), titles))
        listed = client.get("/api/v1/notes/")

    assert listed.status_code == 200, listed.text
    assert {note["title"] for note in listed.json()} == set(titles)
    assert len({note["id"] for note in created}) == PARALLEL_WRITES
