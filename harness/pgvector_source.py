"""A pgvector table on the embedded Postgres, and the external knowledge that reads from it.

`serving_pgvector(chunks)` creates a database of its own on the session's embedded Postgres
(`pgserver`, which ships the vector extension), a `document_chunk` table in the layout Open WebUI
itself writes, and one row per `chunk(...)`. It yields the admin's connection form for it, whose
endpoint is the database URL. The mock provider embeds every query as `QUERY_VECTOR`, so a row's
distance from a query is set by the vector it was given.
"""

from __future__ import annotations

import contextlib
import json
from typing import Iterator

import psycopg

from harness import backends

QUERY_VECTOR = [0.1, 0.2, 0.3]
COLLECTION = "harbour_docs"
# where the table keeps what `chunk` writes; the defaults are Open WebUI's own layout
SOURCE_CONFIG = {
    "table_name": "document_chunk",
    "collection_field": "collection_name",
    "content_field": "text",
    "vector_field": "vector",
    "metadata_field": "vmetadata",
}


def chunk(
    chunk_id: str, text: str, vector: list[float], collection: str = COLLECTION, **metadata
) -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "vector": vector,
        "collection": collection,
        "metadata": metadata,
    }


@contextlib.contextmanager
def serving_pgvector(chunks: list[dict]) -> Iterator[dict]:
    """Yield the connection form of a database holding `chunks`."""
    with backends.postgres_database() as url, psycopg.connect(url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION vector")
        conn.execute(
            "CREATE TABLE document_chunk (id text PRIMARY KEY, collection_name text,"
            " text text, vector vector(3), vmetadata jsonb)"
        )
        for row in chunks:
            conn.execute(
                "INSERT INTO document_chunk VALUES (%s, %s, %s, %s::vector, %s)",
                (
                    row["id"],
                    row["collection"],
                    row["text"],
                    json.dumps(row["vector"]),
                    json.dumps(row["metadata"]),
                ),
            )
        yield {"name": "Test pgvector", "provider": "pgvector", "endpoint": url}
