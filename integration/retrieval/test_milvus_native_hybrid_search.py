"""Journey: hybrid search on Milvus runs BM25 inside Milvus, and existing collections are migrated.

PRs #31645 and #31660 (merges 75bff4bcd and aee7c4727). On Milvus 2.5 or newer each collection
carries a `sparse` field filled by a BM25 function over its `text` column, the hybrid search asks
Milvus for the keyword hits and merges them with the vector hits, and at startup (with
`ENABLE_DB_MIGRATIONS`) every existing collection without the field is copied into one that has
it, in both store layouts (a collection per knowledge base or file, and the shared multitenant
collections). A server older than 2.5 keeps its collections and the BM25 that runs in Python.
Milvus is played by `harness.milvus_server`, which scores text searches on the `sparse` field with
BM25 and records them.

The scripted embedding is one constant vector, so vector search cannot prefer a chunk. The
keyword chunk is spelled `LIGHTHOUSE-keeper`, which Milvus's analyzer splits and lowercases and
the Python path's whitespace split does not, so only Milvus's BM25 finds it by `lighthouse keeper`.

Discriminates: passes on dev 015dbc861. In a backend copy, `_supports_bm25` returning False fails
the migration, native-search and weight tests in both layouts (nothing is migrated, no `sparse`
search reaches Milvus and the Python path misses the `LIGHTHOUSE-keeper` chunk); skipping the
migration's swap fails the migration tests in both layouts and the native ones in the shared
collections; the BM25 branch of `hybrid_search` switched off fails every native-search test.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import pytest

from harness.actors import admin_of
from harness.knowledge_bases import add_text_file, knowledge_base

pytest.importorskip("pymilvus", reason="pymilvus not installed in this env")

from pymilvus.grpc_gen import schema_pb2  # noqa: E402

from harness.milvus_server import milvus_env, serving_milvus  # noqa: E402

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
CONSTANT_EMBEDDING = [0.1, 0.2, 0.3]
SEEDED_ID = "seeded-notes"
SEEDED_TEXTS = {
    "row-1": "The harbour master is Ingrid.",
    "row-2": "Night shift: LIGHTHOUSE-keeper, Tomas.",
    "row-3": "The ferry leaves at noon.",
}
NOTES = [
    "The harbour master is Ingrid.",
    "Night shift: LIGHTHOUSE-keeper, Tomas.",
    "The ferry keeper sleeps.",
]
KEYWORD_CHUNK = NOTES[1]
PARTIAL_CHUNK = NOTES[2]


def _seed_collection(fake, multitenancy: bool) -> None:
    """A collection as the release before native hybrid search wrote it."""
    rows = []
    for row_id, text in SEEDED_TEXTS.items():
        vector = [value + len(row_id) / 100 for value in CONSTANT_EMBEDDING]
        row = {"id": row_id, "vector": vector, "metadata": {"name": "seed"}}
        if multitenancy:
            row |= {"text": text, "resource_id": SEEDED_ID}
        else:
            row["data"] = {"text": text}
        rows.append(row)
    fake.hold_collection(
        "open_webui_knowledge" if multitenancy else "open_webui_seeded_notes", rows
    )


def _seeded_name(multitenancy: bool) -> str:
    return "open_webui_knowledge" if multitenancy else "open_webui_seeded_notes"


def _booted(instance_with, version: str, multitenancy: bool):
    with serving_milvus() as fake:
        fake.version = version
        _seed_collection(fake, multitenancy)
        yield fake, instance_with(milvus_env(fake, multitenancy))


@pytest.fixture(scope="module", params=[True, False], ids=["multitenant", "per-collection"])
def multitenancy(request) -> bool:
    return request.param


@pytest.fixture(scope="module")
def native(instance_with, multitenancy):
    yield from _booted(instance_with, "v2.6.0", multitenancy)


@pytest.fixture(scope="module")
def older_server(instance_with, multitenancy):
    yield from _booted(instance_with, "v2.4.7", multitenancy)


@pytest.fixture
def native_client(native, preserve):
    _, instance = native
    preserve(RETRIEVAL_CONFIG, on=instance)
    with admin_of(instance).client() as client:
        _enable_hybrid_search(client)
        yield client


@pytest.fixture
def older_client(older_server, preserve):
    _, instance = older_server
    preserve(RETRIEVAL_CONFIG, on=instance)
    with admin_of(instance).client() as client:
        _enable_hybrid_search(client)
        yield client


def _enable_hybrid_search(client) -> None:
    saved = client.post(
        RETRIEVAL_CONFIG[1],
        json={"ENABLE_RAG_HYBRID_SEARCH": True, "CHUNK_SIZE": 60, "CHUNK_OVERLAP": 0},
    )
    assert saved.status_code == 200, saved.text


def _search(client, collection: str, query: str, weight: float = 0.5, k: int = 3) -> list[str]:
    searched = client.post(
        "/api/v1/retrieval/query/collection",
        json={
            "collection_names": [collection],
            "query": query,
            "k": k,
            "k_reranker": k,
            "hybrid_bm25_weight": weight,
        },
    )
    assert searched.status_code == 200, searched.text
    return searched.json()["documents"][0]


def _fields(fake, name: str) -> dict[str, int]:
    return {column.name: column.data_type for column in fake.collections[name].schema.fields}


def _collection_of(fake, knowledge_id: str, multitenancy: bool) -> str:
    return (
        "open_webui_knowledge" if multitenancy else f"open_webui_{knowledge_id}".replace("-", "_")
    )


def _searched_fields(fake, collection: str) -> set[str]:
    return {field for name, field, *_ in fake.searches if name == collection}


@contextlib.contextmanager
def _notes_base(client, fake) -> Iterator[str]:
    """A knowledge base holding NOTES, one chunk each, with the fake's search records cleared."""
    with knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "notes.txt", "\n\n".join(NOTES))
        fake.searches.clear()
        fake.sparse_searches.clear()
        yield knowledge_id


def test_startup_migrates_a_collection_without_the_sparse_field_and_keeps_every_row(
    native, multitenancy
):
    fake, _ = native
    name = _seeded_name(multitenancy)

    fields = _fields(fake, name)
    collection = fake.collections[name]
    assert "sparse" in fields, "the existing collection was not migrated to native hybrid search"
    assert collection.bm25_input("sparse") == "text"
    assert collection.indexes["sparse"] == "SPARSE_INVERTED_INDEX"
    assert collection.loaded
    assert {key: row["text"] for key, row in collection.rows.items()} == SEEDED_TEXTS
    for row_id, row in collection.rows.items():
        assert row["vector"] == pytest.approx(
            [value + len(row_id) / 100 for value in CONSTANT_EMBEDDING]
        )
        assert row["metadata"] == {"name": "seed"}
    assert not [other for other in fake.collections if other.endswith("_bm25_staging")]


def test_a_migrated_collection_finds_a_keyword_through_milvus_bm25(native, native_client):
    fake, _ = native
    fake.sparse_searches.clear()

    found = _search(native_client, SEEDED_ID, "lighthouse keeper")

    assert found[0] == SEEDED_TEXTS["row-2"]
    assert [query for _, query, *_ in fake.sparse_searches] == ["lighthouse keeper"]


def test_a_keyword_only_match_is_ranked_first_by_milvus_bm25(native, native_client, multitenancy):
    fake, _ = native
    with _notes_base(native_client, fake) as knowledge_id:
        found = _search(native_client, knowledge_id, "lighthouse")

        collection = _collection_of(fake, knowledge_id, multitenancy)
        assert found[0] == KEYWORD_CHUNK
        assert [query for name, query, *_ in fake.sparse_searches if name == collection] == [
            "lighthouse"
        ], "no BM25 search reached Milvus: the hybrid search ran in Python"
        assert _searched_fields(fake, collection) == {"sparse", "vector"}
        assert _fields(fake, collection)["sparse"] == schema_pb2.SparseFloatVector


@pytest.mark.parametrize(
    ("weight", "searched"),
    [(1.0, {"sparse"}), (0.0, {"vector"})],
    ids=["bm25-only", "vector-only"],
)
def test_the_bm25_weight_decides_which_searches_milvus_gets(
    native, native_client, multitenancy, weight, searched
):
    fake, _ = native
    with _notes_base(native_client, fake) as knowledge_id:
        found = _search(native_client, knowledge_id, "lighthouse keeper", weight=weight)

        assert found
        assert _searched_fields(fake, _collection_of(fake, knowledge_id, multitenancy)) == searched


def test_milvus_bm25_returns_only_the_chunks_sharing_a_term_strongest_first(native, native_client):
    fake, _ = native
    with _notes_base(native_client, fake) as knowledge_id:
        found = _search(native_client, knowledge_id, "lighthouse keeper", weight=1.0)

        assert found == [KEYWORD_CHUNK, PARTIAL_CHUNK]


def test_an_older_server_keeps_its_collection_as_it_was(older_server, multitenancy):
    fake, _ = older_server
    name = _seeded_name(multitenancy)

    assert "sparse" not in _fields(fake, name)
    assert len(fake.collections[name].rows) == len(SEEDED_TEXTS)
    assert not [other for other in fake.collections if other.endswith("_bm25_staging")]


def test_an_older_server_ranks_a_keyword_through_the_python_hybrid_search(
    older_server, older_client
):
    fake, _ = older_server
    with _notes_base(older_client, fake) as knowledge_id:
        found = _search(older_client, knowledge_id, "LIGHTHOUSE-keeper,", weight=1.0, k=1)

        assert found == [KEYWORD_CHUNK]
        assert fake.sparse_searches == []
        assert all("sparse" not in _fields(fake, name) for name in fake.collections)
