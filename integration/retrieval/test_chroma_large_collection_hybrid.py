"""Regression: hybrid search failed on a Chroma collection of more than about 32,000 chunks.

Fix fe56ab24f (open-webui/open-webui#30368, issue open-webui/open-webui#30351). Hybrid search
reads the whole collection for its keyword ranking, and Chroma's unpaged `get()` asks SQLite for
every id in one statement, which passes SQLite's limit of 32,766 bound variables once the
collection is larger. The search then failed with "Error querying knowledge base". The read is
now paged. The collection here is one uploaded file cut into 33,000 one-line chunks, on the
default Chroma store, embedded in large batches so the upload takes seconds.

Discriminates: passes on dev efe63bd34, fails with fe56ab24f reverted (both hybrid queries on
the large file answer 400 while vector search and the small file still answer).
"""

from __future__ import annotations

import httpx
import pytest

from harness.actors import admin_of
from harness.web_retrieval import RETRIEVAL_CONFIG

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

LARGE_CHUNK_COUNT = 33_000  # past SQLite's 32,766 bound variables
LINE_CHUNKS = {
    "TEXT_SPLITTER": "",
    "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER": False,
    "CHUNK_SIZE": 12,
    "CHUNK_OVERLAP": 0,
    "CHUNK_MIN_SIZE_TARGET": 0,
    "ENABLE_RAG_HYBRID_SEARCH": True,
    "RAG_FULL_CONTEXT": False,
    "BYPASS_EMBEDDING_AND_RETRIEVAL": False,
}


def tide_lines(count: int) -> str:
    return "\n".join(f"tide {number:05d}" for number in range(count))


@pytest.fixture(scope="module")
def batched(instance_with):
    """An instance that embeds thousands of chunks per request."""
    return instance_with({"RAG_EMBEDDING_BATCH_SIZE": "4096"})


@pytest.fixture(scope="module")
def admin_client(batched):
    with admin_of(batched).client() as client:
        client.timeout = httpx.Timeout(600.0)
        original = client.get(RETRIEVAL_CONFIG[0]).json()
        saved = client.post(RETRIEVAL_CONFIG[1], json=LINE_CHUNKS)
        assert saved.status_code == 200, saved.text
        try:
            yield client
        finally:
            client.post(RETRIEVAL_CONFIG[1], json=original)


def upload(client: httpx.Client, filename: str, text: str) -> str:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (filename, text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    file_id = uploaded.json()["id"]
    stored = client.get(f"/api/v1/files/{file_id}").json()
    assert stored["data"].get("status") == "completed", stored["data"].get("error")
    return file_id


@pytest.fixture(scope="module")
def large_file(admin_client) -> str:
    return upload(admin_client, "tides.txt", tide_lines(LARGE_CHUNK_COUNT))


@pytest.fixture(scope="module")
def small_file(admin_client) -> str:
    return upload(admin_client, "few-tides.txt", tide_lines(50))


def query_doc(client: httpx.Client, file_id: str, query: str, **options) -> httpx.Response:
    return client.post(
        "/api/v1/retrieval/query/doc",
        json={"collection_name": f"file-{file_id}", "query": query, "k": 3, **options},
    )


def test_hybrid_search_answers_on_a_collection_past_the_sqlite_limit(admin_client, large_file):
    answered = query_doc(admin_client, large_file, "tide 31337")

    assert answered.status_code == 200, (
        f"hybrid search on {LARGE_CHUNK_COUNT} chunks failed (#30351): {answered.text}"
    )
    assert "tide 31337" in answered.json()["documents"][0]


def test_hybrid_search_over_collections_answers_on_the_large_collection(
    admin_client, large_file, small_file
):
    answered = admin_client.post(
        "/api/v1/retrieval/query/collection",
        json={
            "collection_names": [f"file-{large_file}", f"file-{small_file}"],
            "query": "tide 32999",
            "k": 3,
        },
    )

    assert answered.status_code == 200, answered.text
    assert "tide 32999" in answered.json()["documents"][0]


def test_vector_search_answers_on_the_large_collection(admin_client, large_file):
    answered = query_doc(admin_client, large_file, "tide 00001", hybrid=False)

    assert answered.status_code == 200, answered.text
    assert len(answered.json()["documents"][0]) == 3


def test_hybrid_search_answers_on_a_small_collection(admin_client, small_file):
    answered = query_doc(admin_client, small_file, "tide 00042")

    assert answered.status_code == 200, answered.text
    assert "tide 00042" in answered.json()["documents"][0]
