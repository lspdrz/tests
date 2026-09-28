"""Dependency smoke: the vector stores, through knowledge bases, memories and upkeep.

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

With `VECTOR_DB=pgvector` on a Postgres app database the same tests run on pgvector: its
SQLAlchemy `Vector` column holds the chunks, written and read through the app's synchronous
psycopg2 session, and a search orders them by `cosine_distance` (pgvector's `<=>`) while the rest
of the app talks to the same database through psycopg. The embedded Postgres (`pgserver`) ships
the vector extension. A second instance booted on that database reflects the existing column
back through pgvector and still answers from it.

With `VECTOR_DB=opensearch` they run once more through opensearch-py against a local stand-in
of an OpenSearch cluster (`harness.opensearch_server`): an index per collection created on first
write, `helpers.bulk` to index, upsert and delete, `search` with a cosine `script_score` and with
`size` for the hash lookup, `delete_by_query` for a removed file and `indices.delete` for a
deleted knowledge base, every request signed with the configured user. A reset lists the
indices by pattern and deletes each.

With `VECTOR_DB=pinecone` they run through the pinecone package's gRPC client against local
stand-ins (`harness.pinecone_server`): the first boot finds no index in `list_indexes` and creates
one serverless with the configured dimension, metric, cloud and region, `describe_index` names the
index host, reached over gRPC and TLS, and chunks are upserted with the collection name in their
metadata, queried with a metadata filter and deleted by filter or id, every call carrying the API
key. A reset deletes every vector.

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
on upsert fails the memory test and a reset that lists no index fails the reset test. On dev
ef67cc3fa, ranking by `l2_distance` in place of `cosine_distance` fails the pgvector ranking case
and a Pinecone score passed on as it comes fails the Pinecone one; an OpenSearch upsert that never
sends its bulk fails its memory case and a skipped `delete_by_query` its removed-file case. With
`vector` dropped from SQLAlchemy's reflected Postgres types the restarted instance no longer
boots; an OpenSearch client without `http_auth` or a reset that lists no index fails the
OpenSearch test, and an index created in another region fails the Pinecone test.
"""

from __future__ import annotations

import base64

import httpx
import pytest

from harness import backends
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
from harness.opensearch_server import PASSWORD, USERNAME, opensearch_env, serving_opensearch
from harness.pinecone_server import API_KEY, INDEX_NAME, REGION, pinecone_env, serving_pinecone
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


@pytest.fixture(scope="module")
def pgvector_database():
    pytest.importorskip("pgserver", reason="pgvector runs on the embedded Postgres (pgserver)")
    with backends.postgres_database() as url:
        yield url


def pgvector_env(database_url: str) -> dict[str, str]:
    return {"DATABASE_URL": database_url, "VECTOR_DB": "pgvector"}


@pytest.fixture(scope="module")
def opensearch():
    with serving_opensearch() as fake:
        yield fake


@pytest.fixture(scope="module")
def pinecone():
    with serving_pinecone() as fake:
        yield fake


@pytest.fixture(scope="module")
def on_pinecone_env(pinecone, tmp_path_factory) -> dict[str, str]:
    directory = tmp_path_factory.mktemp("pinecone-trust")
    return pinecone_env(pinecone, directory, dimension=len(KEYWORDS) + 1)


@pytest.fixture(params=["embedded", "server", "pgvector", "opensearch", "pinecone"])
def store(request, instance_with, embeddings) -> tuple[str, LaunchedInstance]:
    """(which store, an instance embedding by keyword and keeping its vectors there)."""
    env = keyword_embedding_env(embeddings)
    if request.param == "server":
        env.update(chroma_env(request.getfixturevalue("chroma_server")))
    if request.param == "pgvector":
        env.update(pgvector_env(request.getfixturevalue("pgvector_database")))
    if request.param == "opensearch":
        env.update(opensearch_env(request.getfixturevalue("opensearch")))
    if request.param == "pinecone":
        env.update(request.getfixturevalue("on_pinecone_env"))
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
    documents = (gone.json() or {}).get("documents") or []
    assert not any(documents), "the deleted knowledge base still answers"
    if which == "server":
        assert knowledge_id not in collection_names(request.getfixturevalue("chroma_server"))
    if which == "opensearch":
        assert f"open_webui_{knowledge_id}" not in request.getfixturevalue("opensearch").indices


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


def test_a_restarted_instance_finds_what_pgvector_holds(
    instance_with, embeddings, pgvector_database
):
    env = {**keyword_embedding_env(embeddings), **pgvector_env(pgvector_database)}
    first = instance_with(env)
    with admin_of(first).client() as client, knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "lighthouse.txt", LIGHTHOUSE)

        # a second boot reflects the existing vector column before it serves
        restarted = instance_with({**env, "WEBUI_NAME": "Restarted"})
        with admin_of(restarted).client() as again:
            found = _documents(again, knowledge_id, "lighthouse")

    assert found == [LIGHTHOUSE]


def test_opensearch_gets_the_configured_user_and_a_reset_drops_every_index(
    instance_with, embeddings, opensearch
):
    instance = instance_with({**keyword_embedding_env(embeddings), **opensearch_env(opensearch)})
    with admin_of(instance).client() as client, knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "ferry.txt", FERRY)
        assert f"open_webui_{knowledge_id}" in opensearch.indices

        reset = client.post("/api/v1/retrieval/reset/db")

    assert reset.status_code == 200, reset.text
    assert not [name for name in opensearch.indices if name.startswith("open_webui_")]
    basic = "Basic " + base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()
    assert opensearch.authorizations
    assert set(opensearch.authorizations) == {basic}


def test_pinecone_gets_its_index_created_signed_calls_and_a_reset(
    instance_with, embeddings, pinecone, on_pinecone_env
):
    instance = instance_with({**keyword_embedding_env(embeddings), **on_pinecone_env})
    with admin_of(instance).client() as client, knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "ferry.txt", FERRY)
        collections = {metadata["collection_name"] for _, metadata in pinecone.vectors.values()}
        assert f"open-webui_{knowledge_id}" in collections

        reset = client.post("/api/v1/retrieval/reset/db")

    assert reset.status_code == 200, reset.text
    assert pinecone.vectors == {}
    assert pinecone.indexes[INDEX_NAME] == {
        "dimension": len(KEYWORDS) + 1,
        "metric": "cosine",
        "cloud": "aws",
        "region": REGION,
    }
    assert pinecone.api_keys and set(pinecone.api_keys) == {API_KEY}


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
