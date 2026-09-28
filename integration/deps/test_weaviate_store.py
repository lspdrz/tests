"""Dependency smoke: knowledge bases and memories kept on Weaviate, through the v4 client.

With `VECTOR_DB=weaviate` Open WebUI opens one client with `weaviate.connect_to_custom` (REST and
gRPC hosts and ports, an API key through `Auth.api_key`) and keeps each collection as a Weaviate
collection whose name is made valid for Weaviate (capitalised, hyphens to underscores). The
first write creates it with self-provided vectors and a `text` property; chunks go in through a
fixed-size batch, searches are `near_vector` queries with a `MetadataQuery(distance=True)` whose
cosine distance (0 best, 2 worst) becomes a score from 0 to 1, reads are `fetch_objects` under
`Filter.by_property(...).equal(...)` joined with `Filter.all_of`, hybrid search walks the
collection with its iterator, a removed file goes with `delete_many`, an edited memory is written
again under its id and a deleted one goes with `delete_by_id`, and the admin's reset lists every
collection and deletes it.
Weaviate is played by `harness/weaviate_server.py`, embeddings follow keywords
(`harness.keyword_embeddings`) and the instance is this module's own (twin of
unit/deps/test_weaviate_client.py).

The store takes a missing distance for the worst one and tests it with `and
obj.metadata.distance`, so a chunk Weaviate puts at distance exactly 0 (the query's own vector)
scores 0 in place of 1; that test stays red until the store checks for `None`.

Discriminates: passes on dev ef67cc3fa apart from the distance-zero test. In a backend copy, a
search that passes the raw distance on as the score fails the ranking test; a filtered delete that
never runs, or a collection delete that does nothing, fails the removal test; a `fetch_objects` call
that fails accepts the duplicate and keeps the edited file's old text; an iterator that yields
nothing leaves hybrid search on the vector match; an upsert under a fresh id keeps the memory's
first text and a `delete_by_id` that does nothing keeps the deleted one; a reset that lists no
collection leaves them all; and a client built without `auth_credentials` sends no key.
"""

from __future__ import annotations

import pytest

from harness.actors import admin_of, create_user
from harness.keyword_embeddings import keyword_embedding_env, serve_keyword_embeddings
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.listener import listening

pytest.importorskip("weaviate", reason="weaviate-client not installed in this env")

from harness.weaviate_server import API_KEY, serving_weaviate, weaviate_env  # noqa: E402

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
KEYWORDS = ["lighthouse", "ferry", "tides", "keeper"]
LIGHTHOUSE = "The lighthouse stands on the northern breakwater."
FERRY = "The ferry leaves the harbour every hour."
TIDES = "The tides turn twice a day at the harbour mouth."


@pytest.fixture(scope="module")
def weaviate():
    with serving_weaviate() as fake:
        yield fake


@pytest.fixture(scope="module")
def on_weaviate(instance_with, weaviate):
    with listening() as embeddings:
        serve_keyword_embeddings(embeddings, KEYWORDS)
        yield instance_with({**keyword_embedding_env(embeddings), **weaviate_env(weaviate)})


@pytest.fixture
def client(on_weaviate):
    with admin_of(on_weaviate).client() as admin_client:
        yield admin_client


def _collection(knowledge_id: str) -> str:
    """The Weaviate name of a knowledge base's collection."""
    name = knowledge_id.replace("-", "_")
    name = name if name[0].isalpha() else f"C{name}"
    return name[0].upper() + name[1:]


def _query(client, knowledge_id: str, query: str, k: int = 10) -> dict:
    answered = client.post(
        "/api/v1/retrieval/query/doc",
        json={"collection_name": knowledge_id, "query": query, "k": k},
    )
    assert answered.status_code == 200, answered.text
    return answered.json()


def _documents(client, knowledge_id: str, query: str) -> list[str]:
    return sorted(text.strip() for text in _query(client, knowledge_id, query)["documents"][0])


def _filled(client, knowledge_id: str) -> None:
    for name, text in (("ferry.txt", FERRY), ("lighthouse.txt", LIGHTHOUSE), ("tides.txt", TIDES)):
        add_text_file(client, knowledge_id, name, text)


def test_a_query_ranks_the_nearest_chunk_first(client, weaviate):
    with knowledge_base(client) as knowledge_id:
        _filled(client, knowledge_id)
        assert weaviate.texts(_collection(knowledge_id)) == sorted([FERRY, LIGHTHOUSE, TIDES])

        found = _query(client, knowledge_id, "Where is the lighthouse?", k=2)

    [documents], [scores] = found["documents"], found["distances"]
    assert [document.strip() for document in documents][0] == LIGHTHOUSE, documents
    assert len(documents) == 2, documents
    # a chunk sharing no keyword sits at a right angle: distance 1, score one half
    assert scores[1] == pytest.approx(0.5, abs=0.01)
    assert found["metadatas"][0][0]["name"] == "lighthouse.txt"


def test_a_chunk_at_distance_zero_scores_one(client):
    with knowledge_base(client) as knowledge_id:
        _filled(client, knowledge_id)

        found = _query(client, knowledge_id, "Where is the lighthouse?", k=1)

    [documents], [scores] = found["documents"], found["distances"]
    assert [document.strip() for document in documents] == [LIGHTHOUSE]
    assert scores[0] == pytest.approx(1.0, abs=1e-3), (
        "the chunk whose vector matches the query exactly scored "
        f"{scores[0]}: the Weaviate store reads a distance of 0 as missing and scores it as the "
        "worst match (`obj.metadata.distance if ... and obj.metadata.distance else 2.0`)"
    )


def test_a_search_keeps_to_its_own_knowledge_base(client):
    with knowledge_base(client) as ferry_id, knowledge_base(client) as tides_id:
        add_text_file(client, ferry_id, "ferry.txt", FERRY)
        add_text_file(client, tides_id, "tides.txt", TIDES)

        assert _documents(client, ferry_id, "tides") == [FERRY]
        assert _documents(client, tides_id, "ferry") == [TIDES]


def test_the_same_content_twice_in_a_knowledge_base_is_refused(client):
    with knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "ferry.txt", FERRY)
        again = client.post(
            "/api/v1/files/",
            params={"process": "true", "process_in_background": "false"},
            files={"file": ("ferry-copy.txt", FERRY.encode(), "text/plain")},
        )
        refused = client.post(
            f"/api/v1/knowledge/{knowledge_id}/file/add", json={"file_id": again.json()["id"]}
        )

    assert refused.status_code == 400, refused.text
    assert "Duplicate content" in refused.text


def test_removing_a_file_and_deleting_the_knowledge_base_clear_their_chunks(client, weaviate):
    with knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "ferry.txt", FERRY)
        tides_id = add_text_file(client, knowledge_id, "tides.txt", TIDES)

        removed = client.post(
            f"/api/v1/knowledge/{knowledge_id}/file/remove", json={"file_id": tides_id}
        )
        assert removed.status_code == 200, removed.text
        assert weaviate.texts(_collection(knowledge_id)) == [FERRY]
        assert _documents(client, knowledge_id, "tides") == [FERRY]

        deleted = client.delete(f"/api/v1/knowledge/{knowledge_id}/delete")
        assert deleted.status_code == 200, deleted.text

    assert _collection(knowledge_id) not in weaviate.collections


def test_editing_a_knowledge_file_replaces_its_old_text(client, weaviate):
    with knowledge_base(client) as knowledge_id:
        file_id = add_text_file(client, knowledge_id, "notes.txt", FERRY)

        edited = client.post(
            f"/api/v1/files/{file_id}/data/content/update", json={"content": TIDES}
        )
        assert edited.status_code == 200, edited.text

        assert weaviate.texts(_collection(knowledge_id)) == [TIDES]


def test_hybrid_search_reads_the_collection_back(client, on_weaviate, preserve):
    preserve(RETRIEVAL_CONFIG, on=on_weaviate)
    saved = client.post(
        RETRIEVAL_CONFIG[1],
        json={"ENABLE_RAG_HYBRID_SEARCH": True, "CHUNK_SIZE": 60, "CHUNK_OVERLAP": 0},
    )
    assert saved.status_code == 200, saved.text
    notes = "\n\n".join([FERRY, LIGHTHOUSE, TIDES, "Gulls nest on the old pier."])
    with knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "notes.txt", notes)
        best = client.post(
            "/api/v1/retrieval/query/collection",
            json={
                "collection_names": [knowledge_id],
                "query": "lighthouse nest old pier",
                "k": 1,
                "k_reranker": 1,
                "hybrid_bm25_weight": 1,
            },
        )

    assert best.status_code == 200, best.text
    assert best.json()["documents"][0] == ["Gulls nest on the old pier."]


def _remember(client, content: str) -> str:
    added = client.post("/api/v1/memories/add", json={"content": content})
    assert added.status_code == 200, added.text
    return added.json()["id"]


def _recalled(client, query: str) -> list[str]:
    found = client.post("/api/v1/memories/query", json={"content": query, "k": 5})
    assert found.status_code == 200, found.text
    return found.json()["documents"][0]


def test_an_edited_memory_replaces_its_stored_text_and_a_deleted_one_is_gone(on_weaviate):
    with create_user(on_weaviate).client() as client:
        memory_id = _remember(client, "The keeper rows out at dawn.")
        _remember(client, "The ferry is painted red.")

        edited = client.post(
            f"/api/v1/memories/{memory_id}/update",
            json={"content": "The keeper watches the tides."},
        )
        assert edited.status_code == 200, edited.text
        recalled = _recalled(client, "keeper")

        deleted = client.delete(f"/api/v1/memories/{memory_id}")
        assert deleted.status_code == 200, deleted.text
        after_delete = _recalled(client, "keeper")

    assert len(recalled) == 2, recalled
    assert "The keeper watches the tides." in recalled[0]
    assert not any("rows out at dawn" in text for text in recalled)
    assert len(after_delete) == 1 and "ferry" in after_delete[0], after_delete


def test_the_admin_reset_drops_every_collection(client, weaviate):
    with knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "ferry.txt", FERRY)
        assert weaviate.collections

        reset = client.post("/api/v1/retrieval/reset/db")

    assert reset.status_code == 200, reset.text
    assert weaviate.collections == {}


def test_every_call_carries_the_api_key(client, weaviate):
    with knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "ferry.txt", FERRY)
        assert _documents(client, knowledge_id, "ferry") == [FERRY]

    assert weaviate.api_keys
    assert set(weaviate.api_keys) == {API_KEY}
