"""An Elasticsearch server played by a local service that keeps documents, for an instance on it.

`serving_elasticsearch()` yields a `FakeElasticsearch` speaking the part of the REST API Open
WebUI's Elasticsearch store uses: index create and exists, `_bulk` index, update-as-upsert and
delete, `_count`, `_search` (a `bool` filter of `term` and `terms` clauses, optionally inside a
`script_score` query scored by cosine similarity), `_delete_by_query` and a one-page scroll.
Every answer carries `X-Elastic-Product: Elasticsearch`, which the client insists on, and
`searches` holds every search body it got. `elasticsearch_env(fake)` is the environment of an
instance that stores its vectors there.
"""

from __future__ import annotations

import contextlib
import fnmatch
import json
import math
import threading
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

INDEX_PREFIX = "open_webui_collections"


@dataclass
class FakeElasticsearch:
    base_url: str = ""
    indices: dict[str, dict[str, dict]] = field(default_factory=dict)  # index -> id -> source
    searches: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def elasticsearch_env(fake: FakeElasticsearch) -> dict[str, str]:
    return {
        "VECTOR_DB": "elasticsearch",
        "ELASTICSEARCH_URL": fake.base_url,
        "ELASTICSEARCH_INDEX_PREFIX": INDEX_PREFIX,
    }


def _field_value(document_id: str, source: dict, name: str) -> Any:
    if name == "_id":
        return document_id
    value: Any = source
    for part in name.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _clause_holds(document_id: str, source: dict, clause: dict) -> bool:
    ((kind, condition),) = clause.items()
    if kind == "bool":
        return _filter_holds(document_id, source, condition.get("filter", []))
    ((name, expected),) = condition.items()
    value = _field_value(document_id, source, name)
    if kind == "term":
        return value == expected
    if kind == "terms":
        return value in expected
    raise ValueError(f"the stand-in does not know the {kind!r} query")


def _filter_holds(document_id: str, source: dict, clauses: list | dict) -> bool:
    clauses = clauses if isinstance(clauses, list) else [clauses]
    return all(_clause_holds(document_id, source, clause) for clause in clauses)


def _cosine(left: list[float], right: list[float]) -> float:
    norms = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right)) / norms if norms else 0.0


class _Api:
    def __init__(self, fake: FakeElasticsearch) -> None:
        self.fake = fake

    def _matching_indices(self, pattern: str) -> list[str]:
        names = pattern.split(",")
        return [
            index for index in self.fake.indices if any(fnmatch.fnmatch(index, n) for n in names)
        ]

    def _documents(self, pattern: str, query: dict | None) -> list[tuple[str, str, dict]]:
        """(index, id, source) of every document the query's filter keeps."""
        clauses: list | dict = []
        if query:
            if "bool" in query:
                clauses = query["bool"].get("filter", [])
            else:
                clauses = query
        return [
            (index, document_id, source)
            for index in self._matching_indices(pattern)
            for document_id, source in self.fake.indices[index].items()
            if _filter_holds(document_id, source, clauses)
        ]

    def handle(self, method: str, path: str, params: dict, body: Any) -> tuple[int, Any]:
        parts = [urllib.parse.unquote(part) for part in path.strip("/").split("/") if part]
        with self.fake.lock:
            if not parts:
                return 200, {
                    "name": "stand-in",
                    "cluster_name": "stand-in",
                    "version": {"number": "9.0.0", "build_flavor": "default"},
                    "tagline": "You Know, for Search",
                }
            if parts == ["_bulk"] or parts[-1:] == ["_bulk"]:
                return self._bulk(body)
            if parts[:2] == ["_search", "scroll"]:
                empty = {"_scroll_id": "done", "hits": {"hits": [], "total": {"value": 0}}}
                return 200, {"succeeded": True} if method == "DELETE" else empty
            index = parts[0]
            action = parts[1] if len(parts) > 1 else ""
            if action == "" and method == "HEAD":
                return (200 if index in self.fake.indices else 404), None
            if action == "" and method == "PUT":
                self.fake.indices.setdefault(index, {})
                return 200, {"acknowledged": True, "index": index}
            if action == "" and method == "GET":
                return 200, {name: {} for name in self._matching_indices(index)}
            if action == "" and method == "DELETE":
                for name in self._matching_indices(index):
                    del self.fake.indices[name]
                return 200, {"acknowledged": True}
            if action == "_refresh":
                return 200, {"_shards": {"total": 1, "successful": 1, "failed": 0}}
            if action == "_count":
                return 200, {"count": len(self._documents(index, (body or {}).get("query")))}
            if action == "_search":
                return self._search(index, params, body or {})
            if action == "_delete_by_query":
                doomed = self._documents(index, (body or {}).get("query"))
                for name, document_id, _ in doomed:
                    self.fake.indices[name].pop(document_id, None)
                return 200, {"deleted": len(doomed), "failures": []}
        return 404, {"error": {"type": "stand_in_unknown_route", "reason": f"{method} {path}"}}

    def _bulk(self, lines: list[dict]) -> tuple[int, Any]:
        items = []
        pending = iter(lines)
        for header in pending:
            ((operation, target),) = header.items()
            index, document_id = target["_index"], target["_id"]
            documents = self.fake.indices.setdefault(index, {})
            if operation == "delete":
                documents.pop(document_id, None)
            else:
                payload = next(pending)
                if operation == "update":
                    documents[document_id] = {**documents.get(document_id, {}), **payload["doc"]}
                else:
                    documents[document_id] = payload
            items.append({operation: {"_index": index, "_id": document_id, "status": 200}})
        return 200, {"took": 1, "errors": False, "items": items}

    def _search(self, index: str, params: dict, body: dict) -> tuple[int, Any]:
        self.fake.searches.append(body)
        query = body.get("query") or {}
        vector = None
        if "script_score" in query:
            vector = query["script_score"]["script"]["params"]["vector"]
            query = query["script_score"]["query"]
        size = int(body.get("size", params.get("size", 10)))
        hits = [
            {
                "_index": name,
                "_id": document_id,
                "_score": (_cosine(vector, source["vector"]) + 1.0) if vector else 1.0,
                "_source": source,
            }
            for name, document_id, source in self._documents(index, query)
        ]
        hits.sort(key=lambda hit: hit["_score"], reverse=True)
        answer = {"took": 1, "hits": {"total": {"value": len(hits)}, "hits": hits[:size]}}
        if "scroll" in params:
            answer["_scroll_id"] = "done"
        return 200, answer


@contextlib.contextmanager
def serving_elasticsearch() -> Iterator[FakeElasticsearch]:
    fake = FakeElasticsearch()
    api = _Api(fake)

    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _body(self) -> Any:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode() if length else ""
            if self.path.split("?")[0].endswith("_bulk"):
                return [json.loads(line) for line in raw.splitlines() if line.strip()]
            return json.loads(raw) if raw else None

        def _serve(self) -> None:
            url = urllib.parse.urlsplit(self.path)
            params = dict(urllib.parse.parse_qsl(url.query))
            status, result = api.handle(self.command, url.path, params, self._body())
            encoded = b"" if result is None else json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/vnd.elasticsearch+json;compatible-with=9")
            self.send_header("X-Elastic-Product", "Elasticsearch")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(encoded)

        do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = _serve

    server = ThreadingHTTPServer(("127.0.0.1", 0), RequestHandler)
    fake.base_url = f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    try:
        yield fake
    finally:
        server.shutdown()
        server.server_close()
