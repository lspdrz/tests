"""Dependency smoke: Chroma and S3 Vectors, through knowledge bases, memories and maintenance.

Chroma is the default vector store. Without `CHROMA_HTTP_HOST` Open WebUI opens chromadb's
`PersistentClient` in its data directory; with it, an `HttpClient` to a Chroma server, here the
one the chromadb package ships (`harness.chroma_server`). Each test runs on both. A knowledge
base fills its collection with `add` through `create_batches`, refuses content it already holds
by reading the collection with a `where` filter on the content hash, forgets a removed file with
a filtered `delete` and drops the collection with `delete_collection`. `/query/doc` searches with
`query`, cutting to `k` and turning Chroma's cosine distance (0 best, 2 worst) into a score from
0 to 1. Memories are written with `upsert`, so an edit replaces the stored vector. The embeddings
follow keywords (`harness.keyword_embeddings`), so the nearest chunk is known in advance.
A server that asks for credentials gets them through chromadb's `Settings` auth provider, and
`CHROMA_HTTP_HEADERS` on every request, which a proxy in front of the server records.

With `VECTOR_DB=s3vector` the same features go through boto3's `s3vectors` client to a local
fake (`harness.s3_vectors`): `create_index` on first write, `put_vectors`, `query_vectors`,
`list_vectors` for filtered reads, `delete_vectors` and `delete_index`, and `list_indexes` when
the admin resets the vector store.

Discriminates: passes on dev ef67cc3fa; in a backend copy whose Chroma search passes the raw
distance on as the score the ranking test fails, a hash lookup that never matches accepts the
copy, a skipped filtered delete keeps the removed file's chunk, a `delete_collection` that does
nothing keeps the collection and `upsert` swapped for `add` keeps the memory's first text, each on
both stores; leaving out the auth provider and the headers fails the credentials test. In a copy
whose S3 Vectors client skips `put_vectors` on insert the three knowledge tests fail, skipping it
on upsert fails the memory test and a reset that lists no index fails the reset test.
"""

from __future__ import annotations

import base64

import httpx
import pytest

from harness.actors import Actor, admin_of, create_user
from harness.chroma_server import (
    chroma_env,
    collection_names,
    recording_proxy,
    serving_chroma,
)
from harness.instance import LaunchedInstance
from harness.keyword_embeddings import keyword_embedding_env, serve_keyword_embeddings
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.listener import listening
from harness.s3_vectors import s3_vectors_env, serving_s3_vectors

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

VECTOR_BUCKET = "owui-vectors"
KEYWORDS = ["lighthouse", "ferry", "tides", "keeper"]
LIGHTHOUSE = "The lighthouse stands on the northern breakwater."
FERRY = "The ferry leaves the harbour every hour."
TIDES = "The tides turn twice a day at the harbour mouth."


@pytest.fixture(scope="module")
def embeddings():
    with listening() as service:
        serve_keyword_embeddings(service, KEYWORDS)
        yield service


@pytest.fixture(scope="module")
def chroma_server():
    with serving_chroma() as base_url:
        yield base_url


@pytest.fixture(params=["embedded", "server"])
def store(request, instance_with, embeddings) -> tuple[str, LaunchedInstance]:
    """(which Chroma, an instance embedding by keyword and keeping its vectors there)."""
    env = keyword_embedding_env(embeddings)
    if request.param == "server":
        env.update(chroma_env(request.getfixturevalue("chroma_server")))
    return request.param, instance_with(env)


def _query(client: httpx.Client, collection_name: str, query: str, k: int = 10) -> dict:
    answered = client.post(
        "/api/v1/retrieval/query/doc",
        json={"collection_name": collection_name, "query": query, "k": k},
    )
    assert answered.status_code == 200, answered.text
    return answered.json()


def _documents(client: httpx.Client, collection_name: str, query: str) -> list[str]:
    return [text.strip() for text in _query(client, collection_name, query)["documents"][0]]


def test_a_query_ranks_the_nearest_chunk_first_and_scores_it_near_one(store):
    _, instance = store
    with admin_of(instance).client() as client, knowledge_base(client) as knowledge_id:
        for name, text in (
            ("ferry.txt", FERRY),
            ("lighthouse.txt", LIGHTHOUSE),
            ("tides.txt", TIDES),
        ):
            add_text_file(client, knowledge_id, name, text)

        found = _query(client, knowledge_id, "Where is the lighthouse?", k=2)

    [documents], [scores] = found["documents"], found["distances"]
    assert len(documents) == 2, documents
    assert documents[0].strip() == LIGHTHOUSE
    assert scores[0] == pytest.approx(1.0, abs=1e-3)
    # a chunk sharing no keyword sits at a right angle: distance 1, score one half
    assert scores[1] == pytest.approx(0.5, abs=0.01)
    assert found["metadatas"][0][0]["name"] == "lighthouse.txt"


def test_the_same_content_twice_in_a_knowledge_base_is_refused(store):
    _, instance = store
    with admin_of(instance).client() as client, knowledge_base(client) as knowledge_id:
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


def test_a_file_removed_from_a_knowledge_base_is_no_longer_found(store):
    _, instance = store
    with admin_of(instance).client() as client, knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "ferry.txt", FERRY)
        tides_id = add_text_file(client, knowledge_id, "tides.txt", TIDES)
        assert TIDES in _documents(client, knowledge_id, "tides")

        removed = client.post(
            f"/api/v1/knowledge/{knowledge_id}/file/remove", json={"file_id": tides_id}
        )
        assert removed.status_code == 200, removed.text

        assert _documents(client, knowledge_id, "tides") == [FERRY]


def test_a_deleted_knowledge_base_takes_its_collection_along(store, request):
    which, instance = store
    with admin_of(instance).client() as client, knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "ferry.txt", FERRY)
        if which == "server":
            assert knowledge_id in collection_names(request.getfixturevalue("chroma_server"))

        deleted = client.delete(f"/api/v1/knowledge/{knowledge_id}/delete")
        assert deleted.status_code == 200, deleted.text
        gone = client.post(
            "/api/v1/retrieval/query/doc",
            json={"collection_name": knowledge_id, "query": "ferry"},
        )

    assert gone.status_code == 200, gone.text
    assert not (gone.json() or {}).get("documents"), "the deleted knowledge base still answers"
    if which == "server":
        assert knowledge_id not in collection_names(request.getfixturevalue("chroma_server"))


def _remember(client: httpx.Client, content: str) -> str:
    added = client.post("/api/v1/memories/add", json={"content": content})
    assert added.status_code == 200, added.text
    return added.json()["id"]


def _recalled(client: httpx.Client, query: str) -> list[str]:
    found = client.post("/api/v1/memories/query", json={"content": query, "k": 5})
    assert found.status_code == 200, found.text
    return found.json()["documents"][0]


def test_an_edited_memory_replaces_its_stored_text(store):
    _, instance = store
    account: Actor = create_user(instance)
    with account.client() as client:
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
    assert len(after_delete) == 1 and "ferry" in after_delete[0]


CHROMA_CREDENTIALS = "harbour:master-key"


@pytest.fixture(scope="module")
def chroma_proxy(chroma_server):
    with recording_proxy(chroma_server) as proxied:
        yield proxied


def test_a_chroma_server_gets_the_configured_credentials_and_headers(
    instance_with, embeddings, chroma_proxy
):
    proxy_url, seen = chroma_proxy
    instance = instance_with(
        {
            **keyword_embedding_env(embeddings),
            **chroma_env(proxy_url),
            "CHROMA_CLIENT_AUTH_PROVIDER": "chromadb.auth.basic_authn.BasicAuthClientProvider",
            "CHROMA_CLIENT_AUTH_CREDENTIALS": CHROMA_CREDENTIALS,
            "CHROMA_HTTP_HEADERS": "X-Harbour-Tenant=north",
        }
    )
    with admin_of(instance).client() as client, knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "ferry.txt", FERRY)
        assert _documents(client, knowledge_id, "ferry") == [FERRY]

    basic = "Basic " + base64.b64encode(CHROMA_CREDENTIALS.encode()).decode()
    assert seen, "the instance never reached the Chroma server"
    assert all(headers.get("authorization") == basic for headers in seen)
    assert all(headers.get("x-harbour-tenant") == "north" for headers in seen)


# ---------------------------------------------------------------- S3 Vectors through boto3


@pytest.fixture(scope="module")
def s3_vectors():
    with serving_s3_vectors() as fake:
        yield fake


@pytest.fixture
def on_s3_vectors(instance_with, embeddings, s3_vectors) -> LaunchedInstance:
    return instance_with(
        {**keyword_embedding_env(embeddings), **s3_vectors_env(s3_vectors, VECTOR_BUCKET)}
    )


def test_s3_vectors_keeps_a_knowledge_base_and_answers_its_queries(on_s3_vectors, s3_vectors):
    with admin_of(on_s3_vectors).client() as client, knowledge_base(client) as knowledge_id:
        for name, text in (
            ("ferry.txt", FERRY),
            ("lighthouse.txt", LIGHTHOUSE),
            ("tides.txt", TIDES),
        ):
            add_text_file(client, knowledge_id, name, text)
        stored = len(s3_vectors.indexes[knowledge_id])

        found = _query(client, knowledge_id, "Where is the lighthouse?", k=2)

    assert stored == 3
    [documents] = found["documents"]
    assert len(documents) == 2, documents
    assert documents[0].strip() == LIGHTHOUSE


def test_s3_vectors_refuses_the_same_content_twice(on_s3_vectors):
    with admin_of(on_s3_vectors).client() as client, knowledge_base(client) as knowledge_id:
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


def test_s3_vectors_forgets_a_removed_file_and_a_deleted_knowledge_base(on_s3_vectors, s3_vectors):
    with admin_of(on_s3_vectors).client() as client, knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "ferry.txt", FERRY)
        tides_id = add_text_file(client, knowledge_id, "tides.txt", TIDES)

        removed = client.post(
            f"/api/v1/knowledge/{knowledge_id}/file/remove", json={"file_id": tides_id}
        )
        assert removed.status_code == 200, removed.text
        assert _documents(client, knowledge_id, "tides") == [FERRY]
        assert len(s3_vectors.indexes[knowledge_id]) == 1

        deleted = client.delete(f"/api/v1/knowledge/{knowledge_id}/delete")
        assert deleted.status_code == 200, deleted.text

    assert knowledge_id not in s3_vectors.indexes


def test_s3_vectors_replaces_an_edited_memory(on_s3_vectors):
    with create_user(on_s3_vectors).client() as client:
        memory_id = _remember(client, "The keeper rows out at dawn.")
        edited = client.post(
            f"/api/v1/memories/{memory_id}/update",
            json={"content": "The keeper watches the tides."},
        )
        assert edited.status_code == 200, edited.text

        recalled = _recalled(client, "keeper")

    assert len(recalled) == 1 and "The keeper watches the tides." in recalled[0]


def test_resetting_the_vector_store_deletes_every_s3_vectors_index(on_s3_vectors, s3_vectors):
    with admin_of(on_s3_vectors).client() as client, knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "ferry.txt", FERRY)
        assert s3_vectors.indexes

        reset = client.post("/api/v1/retrieval/reset/db")

    assert reset.status_code == 200, reset.text
    assert s3_vectors.indexes == {}
