"""Regression: editing a knowledge file on Elasticsearch left its old text searchable (#31523).

Fix 7cf35bedc (#31530). Editing a knowledge file's content adds the new chunks and then deletes
the old ones, which it finds with the store's `query`. On Elasticsearch `query` filtered on the
bare field name (`file_id`) while the chunks keep it under `metadata`, so it found none of them,
and its default page held 10 hits, so a file with more chunks would have lost only the first 10.
Elasticsearch is played by `harness/elasticsearch_server.py`, whose `term` on a missing field
matches nothing as the real one does.

Discriminates: passes on dev 015dbc861, fails with the fix reverted (the old text stays stored and
found after the edit; with only the page size reverted to 10, the chunks past the tenth stay).
"""

from __future__ import annotations

import pytest

from harness.actors import admin_of
from harness.elasticsearch_server import elasticsearch_env, serving_elasticsearch
from harness.knowledge_bases import add_text_file, knowledge_base

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
HARBOUR = "The harbour master is Ingrid."
LIGHTHOUSE = "The lighthouse keeper is Tomas."


@pytest.fixture(scope="module")
def elasticsearch():
    with serving_elasticsearch() as fake:
        yield fake


@pytest.fixture
def client(instance_with, elasticsearch, preserve):
    on_elasticsearch = instance_with(elasticsearch_env(elasticsearch))
    preserve(RETRIEVAL_CONFIG, on=on_elasticsearch)
    with admin_of(on_elasticsearch).client() as admin_client:
        saved = admin_client.post(RETRIEVAL_CONFIG[1], json={"CHUNK_SIZE": 60, "CHUNK_OVERLAP": 0})
        assert saved.status_code == 200, saved.text
        yield admin_client


def _stored_texts(fake, collection: str) -> list[str]:
    with fake.lock:
        return sorted(
            source["text"]
            for documents in fake.indices.values()
            for source in documents.values()
            if source.get("collection") == collection
        )


def _found(client, knowledge_id: str, query: str, k: int = 50) -> list[str]:
    searched = client.post(
        "/api/v1/retrieval/query/collection",
        json={"collection_names": [knowledge_id], "query": query, "k": k},
    )
    assert searched.status_code == 200, searched.text
    return sorted(searched.json()["documents"][0])


def _edit(client, file_id: str, content: str) -> None:
    edited = client.post(f"/api/v1/files/{file_id}/data/content/update", json={"content": content})
    assert edited.status_code == 200, edited.text


def test_editing_a_knowledge_file_replaces_its_old_text(client, elasticsearch):
    with knowledge_base(client) as knowledge_id:
        file_id = add_text_file(client, knowledge_id, "harbour.txt", HARBOUR)

        _edit(client, file_id, LIGHTHOUSE)

        assert _stored_texts(elasticsearch, knowledge_id) == [LIGHTHOUSE]
        assert _found(client, knowledge_id, "who works here?") == [LIGHTHOUSE]


def test_editing_a_file_with_more_than_ten_chunks_removes_them_all(client, elasticsearch):
    old_notes = "\n\n".join(f"Old note {n:02d} about the harbour." for n in range(14))
    new_notes = "\n\n".join(f"New note {n:02d} about the lighthouse." for n in range(3))
    with knowledge_base(client) as knowledge_id:
        file_id = add_text_file(client, knowledge_id, "notes.txt", old_notes)
        assert len(_stored_texts(elasticsearch, knowledge_id)) == 14

        _edit(client, file_id, new_notes)

        stored = _stored_texts(elasticsearch, knowledge_id)
        assert stored == [f"New note {n:02d} about the lighthouse." for n in range(3)]
        assert _found(client, knowledge_id, "harbour") == stored


def test_editing_one_file_leaves_the_other_files_chunks(client, elasticsearch):
    with knowledge_base(client) as knowledge_id:
        edited_id = add_text_file(client, knowledge_id, "harbour.txt", HARBOUR)
        add_text_file(client, knowledge_id, "ferry.txt", "The ferry leaves at noon.")

        _edit(client, edited_id, LIGHTHOUSE)

        assert _stored_texts(elasticsearch, knowledge_id) == [
            "The ferry leaves at noon.",
            LIGHTHOUSE,
        ]
