"""Dependency smoke: a knowledge base kept on Elasticsearch, through the official client.

With `VECTOR_DB=elasticsearch` Open WebUI builds one `Elasticsearch` client from the connection
settings and keeps every collection in an index per embedding size: the client's `indices`
calls create and find those indexes, the `bulk` helper writes the chunks, `search` runs the
vector and filtered queries, `count` tells whether a collection exists, `delete_by_query` removes
a file or a whole knowledge base, the `scan` helper reads a collection back for hybrid search
and the admin's vector reset drops the indexes. Elasticsearch is played by
`harness/elasticsearch_server.py`; the instance is this module's own.

Editing a knowledge file's content is pinned in
integration/retrieval/test_elasticsearch_knowledge_edit.py.

Discriminates: passes on dev 015dbc861. In a backend copy, `bulk` handed a keyword the client does
not take (a bump renaming one) fails every test, `delete_collection` without its collection term
empties the other knowledge base too and `scan` given `q=` in place of `query=` fails the hybrid
search.
"""

from __future__ import annotations

import pytest

from harness.actors import admin_of
from harness.elasticsearch_server import INDEX_PREFIX, elasticsearch_env, serving_elasticsearch
from harness.knowledge_bases import add_text_file, knowledge_base

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
HARBOUR = "The harbour master is Ingrid."
LIGHTHOUSE = "The lighthouse keeper is Tomas."


@pytest.fixture(scope="module")
def elasticsearch():
    with serving_elasticsearch() as fake:
        yield fake


@pytest.fixture
def on_elasticsearch(instance_with, elasticsearch):
    return instance_with(elasticsearch_env(elasticsearch))


@pytest.fixture
def client(on_elasticsearch):
    with admin_of(on_elasticsearch).client() as admin_client:
        yield admin_client


def _stored_texts(fake, collection: str) -> list[str]:
    with fake.lock:
        return sorted(
            source["text"]
            for documents in fake.indices.values()
            for source in documents.values()
            if source.get("collection") == collection
        )


def _found(client, knowledge_id: str, query: str) -> list[str]:
    searched = client.post(
        "/api/v1/retrieval/query/collection",
        json={"collection_names": [knowledge_id], "query": query, "k": 10},
    )
    assert searched.status_code == 200, searched.text
    return sorted(searched.json()["documents"][0])


def test_a_knowledge_file_is_stored_in_an_index_and_found(client, elasticsearch):
    with knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "harbour.txt", HARBOUR)

        assert _stored_texts(elasticsearch, knowledge_id) == [HARBOUR]
        assert _found(client, knowledge_id, "who is the harbour master?") == [HARBOUR]

    assert any(name.startswith(f"{INDEX_PREFIX}_d") for name in elasticsearch.indices)


def test_a_search_keeps_to_its_own_knowledge_base(client):
    with knowledge_base(client) as harbour_id, knowledge_base(client) as lighthouse_id:
        add_text_file(client, harbour_id, "harbour.txt", HARBOUR)
        add_text_file(client, lighthouse_id, "lighthouse.txt", LIGHTHOUSE)

        assert _found(client, harbour_id, "who works here?") == [HARBOUR]
        assert _found(client, lighthouse_id, "who works here?") == [LIGHTHOUSE]


def test_removing_a_file_and_deleting_the_knowledge_base_clear_their_chunks(client, elasticsearch):
    with knowledge_base(client) as other_id:
        add_text_file(client, other_id, "lighthouse.txt", LIGHTHOUSE)
        with knowledge_base(client) as knowledge_id:
            harbour_file = add_text_file(client, knowledge_id, "harbour.txt", HARBOUR)
            add_text_file(client, knowledge_id, "lighthouse.txt", LIGHTHOUSE)

            removed = client.post(
                f"/api/v1/knowledge/{knowledge_id}/file/remove", json={"file_id": harbour_file}
            )
            assert removed.status_code == 200, removed.text
            assert _stored_texts(elasticsearch, knowledge_id) == [LIGHTHOUSE]

        assert _stored_texts(elasticsearch, knowledge_id) == []
        assert _stored_texts(elasticsearch, other_id) == [LIGHTHOUSE]


def test_hybrid_search_reads_the_collection_back(client, on_elasticsearch, preserve):
    preserve(RETRIEVAL_CONFIG, on=on_elasticsearch)
    saved = client.post(
        RETRIEVAL_CONFIG[1],
        json={"ENABLE_RAG_HYBRID_SEARCH": True, "CHUNK_SIZE": 60, "CHUNK_OVERLAP": 0},
    )
    assert saved.status_code == 200, saved.text
    notes = "\n\n".join([HARBOUR, LIGHTHOUSE, "The ferry leaves at noon.", "Gulls nest here."])
    with knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "notes.txt", notes)
        best = client.post(
            "/api/v1/retrieval/query/collection",
            json={
                "collection_names": [knowledge_id],
                "query": "lighthouse keeper",
                "k": 1,
                "k_reranker": 1,
                "hybrid_bm25_weight": 1,
            },
        )

    assert best.status_code == 200, best.text
    assert best.json()["documents"][0] == [LIGHTHOUSE]


def test_the_admin_reset_drops_every_index(client, elasticsearch):
    with knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "harbour.txt", HARBOUR)
        assert elasticsearch.indices

        reset = client.post("/api/v1/retrieval/reset/db")

    assert reset.status_code == 200, reset.text
    assert elasticsearch.indices == {}
