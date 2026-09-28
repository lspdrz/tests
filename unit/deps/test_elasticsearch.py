"""Dependency contract: elasticsearch, the client surface no Open WebUI request reaches.

Open WebUI's Elasticsearch vector store (`retrieval/vector/dbs/elasticsearch.py`) is driven from
outside in integration/deps/test_elasticsearch_store.py against the stand-in of
`harness/elasticsearch_server.py`: the client built with the connection settings, its `indices`
calls, `search`, `count`, `delete_by_query` and the `bulk` and `scan` helpers.

Kept as a unit contract: the client's single-document `index` and `delete`, which the store does
not call, and `BadRequestError` sitting under `ApiError`, which it imports but never catches, so
no request reaches either. The client constructs lazily, so nothing here opens a socket. Uses the
`depcheck` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "elasticsearch"

# Client methods the store does not call, kept for callers of the whole client.
UNUSED_CLIENT_METHODS = ["index", "delete"]


def test_single_document_methods_exist_and_are_callable(depcheck):
    es = depcheck.load(IMPORT_NAME)
    client = es.Elasticsearch(hosts=["http://localhost:9200"])
    for name in UNUSED_CLIENT_METHODS:
        assert callable(getattr(client, name, None)), f"client.{name} missing/not callable"


def test_bad_request_error_subclasses_api_error(depcheck):
    """BadRequestError sits under the client's ``ApiError`` base; pin that so a
    broad ``except ApiError`` stays valid across the 8 -> 9 exception reshuffle."""
    mod = depcheck.load(IMPORT_NAME)
    assert hasattr(mod, "ApiError"), "elasticsearch.ApiError missing"
    assert issubclass(mod.BadRequestError, mod.ApiError), (
        "BadRequestError no longer subclasses ApiError"
    )
