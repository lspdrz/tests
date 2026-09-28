"""An OpenSearch cluster played by a local service that keeps documents, for an instance on it.

`serving_opensearch()` yields a `FakeOpenSearch` speaking the part of the REST API Open WebUI's
OpenSearch store drives through opensearch-py: index exists, create, list by pattern, refresh and
delete, `_bulk` index, update-as-upsert and delete, `_search` with `size` (a `bool` filter of
`term` and `terms` clauses on `.keyword` fields, `match_all`, optionally inside a `script_score`
query) and `_delete_by_query`. A `script_score` runs only the scoring script Open WebUI sends,
cosine similarity mapped onto 0 to 1; any other script is refused the way a cluster refuses one
it cannot compile. `authorizations` keeps the `Authorization` header of every request.
`opensearch_env(fake)` is the environment of an instance that stores its vectors there.
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

USERNAME = "harbour"
PASSWORD = "opensearch-secret"
COSINE_SCRIPT = "(cosineSimilarity(params.query_value, doc[params.field]) + 1.0) / 2.0"


@dataclass
class FakeOpenSearch:
    base_url: str = ""
    indices: dict[str, dict[str, dict]] = field(default_factory=dict)  # index -> id -> source
    authorizations: list[str | None] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def opensearch_env(fake: FakeOpenSearch) -> dict[str, str]:
    return {
        "VECTOR_DB": "opensearch",
        "OPENSEARCH_URI": fake.base_url,
        "OPENSEARCH_SSL": "false",
        "OPENSEARCH_USERNAME": USERNAME,
        "OPENSEARCH_PASSWORD": PASSWORD,
    }


def _field_value(source: dict, name: str) -> Any:
    # a string field is matched exactly through the keyword subfield dynamic mapping gives it
    name = name.removesuffix(".keyword")
    value: Any = source
    for part in name.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _clause_holds(source: dict, clause: dict) -> bool:
    ((kind, condition),) = clause.items()
    if kind == "match_all":
        return True
    if kind == "bool":
        return all(_clause_holds(source, inner) for inner in condition.get("filter", []))
    ((name, expected),) = condition.items()
    value = _field_value(source, name)
    if kind == "term":
        return value == expected
    if kind == "terms":
        return value in expected
    raise ValueError(f"the stand-in does not know the {kind!r} query")


def _cosine(left: list[float], right: list[float]) -> float:
    norms = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right)) / norms if norms else 0.0


class _Api:
    def __init__(self, fake: FakeOpenSearch) -> None:
        self.fake = fake

    def _matching_indices(self, pattern: str) -> list[str]:
        names = pattern.split(",")
        return [
            index for index in self.fake.indices if any(fnmatch.fnmatch(index, n) for n in names)
        ]

    def _documents(self, index: str, query: dict) -> list[tuple[str, str, dict]]:
        return [
            (name, document_id, source)
            for name in self._matching_indices(index)
            for document_id, source in self.fake.indices[name].items()
            if _clause_holds(source, query)
        ]

    def handle(self, method: str, path: str, params: dict, body: Any) -> tuple[int, Any]:
        parts = [urllib.parse.unquote(part) for part in path.strip("/").split("/") if part]
        with self.fake.lock:
            if not parts:
                return 200, {"version": {"distribution": "opensearch", "number": "2.19.0"}}
            if parts[-1] == "_bulk":
                return self._bulk(body)
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
            if action == "_search":
                return self._search(index, params, body or {})
            if action == "_delete_by_query":
                doomed = self._documents(index, (body or {}).get("query") or {"match_all": {}})
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
        query = body.get("query") or {"match_all": {}}
        script = None
        if "script_score" in query:
            script = query["script_score"]["script"]
            if script["source"] != COSINE_SCRIPT:
                return 400, {"error": {"type": "script_exception", "reason": script["source"]}}
            query = query["script_score"]["query"]
        size = int(body.get("size", params.get("size", 10)))
        hits = []
        for name, document_id, source in self._documents(index, query):
            score = 1.0
            if script:
                vector = source[script["params"]["field"]]
                score = (_cosine(script["params"]["query_value"], vector) + 1.0) / 2.0
            hits.append({"_index": name, "_id": document_id, "_score": score, "_source": source})
        hits.sort(key=lambda hit: hit["_score"], reverse=True)
        return 200, {"took": 1, "hits": {"total": {"value": len(hits)}, "hits": hits[:size]}}


@contextlib.contextmanager
def serving_opensearch() -> Iterator[FakeOpenSearch]:
    fake = FakeOpenSearch()
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
            with fake.lock:
                fake.authorizations.append(self.headers.get("Authorization"))
            status, result = api.handle(self.command, url.path, params, self._body())
            encoded = b"" if result is None else json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=UTF-8")
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
