"""A Qdrant or Milvus server played by a local service, and the external knowledge it serves.

`serve_qdrant(listener, collection, points)` routes a listener as a Qdrant REST server whose
`collection` answers a vector query with the first `limit` of `points` (built with `point(...)`)
and returns the admin's connection form for it. `queries_to(listener, collection)` is every query
body it got.
`serve_milvus(fake, collection, rows)` does the same for a `FakeMilvus` (harness/milvus_server.py)
holding `collection` with `rows` (built with `milvus_row(...)`), ranked by cosine against the
embedded question, and `searches_of(fake, collection)` is every vector search it got.
`external_connection(client, form)` saves that connection and `external_knowledge_base(...)` a
read-only knowledge base on one of its collections, each removed again afterwards.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import httpx

from harness.listener import Answer, Listener, ReceivedRequest, json_answer

CONNECTIONS = "/api/v1/knowledge/external/connections"
QDRANT_API_KEY = "qdrant-key-0123456789"
MILVUS_TOKEN = "milvus-token-0123456789"
# where the payload keeps what `point` writes
SOURCE_CONFIG = {"content_field": "payload.text", "metadata_field": "payload.metadata"}
# the columns `milvus_row` fills, as the form offers them for Milvus
MILVUS_SOURCE_CONFIG = {
    "content_field": "data.text",
    "vector_field": "vector",
    "metadata_field": "metadata",
    "document_id_field": "id",
}
# the mock provider embeds every text as this
EMBEDDING = [0.1, 0.2, 0.3]


def point(point_id: int, text: str, score: float, **metadata) -> dict:
    """A scored point as Qdrant's query API returns it."""
    payload = {"text": text, "metadata": metadata}
    return {"id": point_id, "version": 1, "score": score, "payload": payload}


def serve_qdrant(listener: Listener, collection: str, points: list[dict]) -> dict:
    """Answer as Qdrant holding `collection`; returns the connection form that uses it."""
    listener.route("GET", "/", json_answer({"title": "qdrant", "version": "1.15.0"}))

    def query(request: ReceivedRequest) -> Answer:
        nearest = points[: request.json().get("limit", 10)]
        return json_answer({"result": {"points": nearest}, "status": "ok", "time": 0.001})

    listener.route("POST", f"/collections/{collection}/points/query", query)
    return {
        "name": "Test Qdrant",
        "provider": "qdrant",
        "endpoint": listener.base_url,
        "auth_config": {"api_key": QDRANT_API_KEY},
    }


def milvus_row(row_id: str, text: str, vector: list[float], **metadata) -> dict:
    """A row of a Milvus collection: the text sits in the JSON column `data`."""
    return {"id": row_id, "vector": vector, "data": {"text": text}, "metadata": metadata}


def serve_milvus(fake, collection: str, rows: list[dict], db_name: str = "") -> dict:
    """Hold `collection` in `fake`; returns the connection form that reaches it."""
    fake.hold_collection(collection, rows)
    return {
        "name": "Test Milvus",
        "provider": "milvus",
        "endpoint": fake.address,
        "auth_config": {"type": "bearer", "api_key": MILVUS_TOKEN},
        "config": {"db_name": db_name} if db_name else {},
    }


def searches_of(fake, collection: str) -> list[tuple[str, str, int, list[str]]]:
    return [search for search in fake.searches if search[0] == collection]


def queries_to(listener: Listener, collection: str) -> list[dict]:
    return [call.json() for call in listener.requests_to(f"/collections/{collection}/points/query")]


@contextlib.contextmanager
def external_connection(client: httpx.Client, form: dict) -> Iterator[str]:
    created = client.post(CONNECTIONS, json=form)
    assert created.status_code == 200, f"saving the external connection failed: {created.text}"
    connection_id = created.json()["id"]
    try:
        yield connection_id
    finally:
        client.delete(f"{CONNECTIONS}/{connection_id}")


@contextlib.contextmanager
def external_knowledge_base(
    client: httpx.Client,
    connection_id: str,
    collection: str,
    name: str = "Team vectors",
    config: dict = SOURCE_CONFIG,
) -> Iterator[str]:
    created = client.post(
        "/api/v1/knowledge/external/knowledge/create",
        json={
            "name": name,
            "connection_id": connection_id,
            "source": {"name": collection, "config": config},
        },
    )
    assert created.status_code == 200, f"creating the external knowledge failed: {created.text}"
    knowledge_id = created.json()["id"]
    try:
        yield knowledge_id
    finally:
        client.delete(f"/api/v1/knowledge/{knowledge_id}/delete")
