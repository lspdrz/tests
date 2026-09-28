"""Dependency smoke: the async SQLite driver every request reads and writes through.

On the default database every request's query runs on SQLAlchemy's async engine, which drives
SQLite through aiosqlite (`sqlite://` becomes `sqlite+aiosqlite://`). A note saved, changed and
deleted over the API must read back from the next request exactly as written, and writes sent
at once must all land. A bump that breaks aiosqlite's commit, parameter binding or row access
loses the write or garbles what reads back. On Postgres the engine uses psycopg instead, so the
module skips there. Alembic's migrations are driven by integration/migrations.

Discriminates: passes on dev ef67cc3fa; in a backend copy with `aiosqlite.Connection.commit`
made a no-op every write is lost, down to the admin account the boot signs up, so both fail.
"""

from __future__ import annotations

import concurrent.futures
import uuid

import pytest

from harness import backends

pytestmark = [
    pytest.mark.depcheck,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.skipif(
        backends.DATABASE == "postgres", reason="aiosqlite only drives the SQLite database"
    ),
]

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


def test_a_note_reads_back_changed_and_deleted_as_written(make_user):
    author = make_user()
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


def test_notes_written_at_once_all_land(make_user):
    author = make_user()
    titles = [f"parallel {index} {uuid.uuid4().hex[:6]}" for index in range(PARALLEL_WRITES)]

    with author.client() as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            created = list(pool.map(lambda title: _create_note(client, title, title), titles))
        listed = client.get("/api/v1/notes/")

    assert listed.status_code == 200, listed.text
    assert {note["title"] for note in listed.json()} == set(titles)
    assert len({note["id"] for note in created}) == PARALLEL_WRITES
