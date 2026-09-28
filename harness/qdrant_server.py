"""A Qdrant server played by a local service that keeps points, for an instance booted on Qdrant.

`serving_qdrant()` yields a `FakeQdrant` speaking the REST API `qdrant-client` uses: collections,
payload indexes, upserts, deletes, counts, scrolls and vector queries with payload filters, with
cosine scores worked out in Python. Set `max_query_limit` to act like strict mode: a scroll or
query asking for more points than that is refused with Qdrant's "Limit exceeded" error. A
collection whose name contains one of `unavailable` answers every vector query with a 503, as a
shard that is down does.
The same store also answers Qdrant's gRPC API on `grpc_port`, for a client that prefers gRPC:
each call is turned into the REST body above with qdrant-client's own conversions and back,
and counted in `grpc_calls`.
`qdrant_env(fake, multitenancy, grpc)` is the environment of an instance that stores its
vectors there.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import threading
from concurrent import futures
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

OK = "ok"


@dataclass
class FakeQdrant:
    base_url: str = ""
    grpc_port: int = 0
    grpc_calls: int = 0
    collections: dict[str, dict[Any, dict]] = field(default_factory=dict)
    max_query_limit: int | None = None
    refused_limits: list[int] = field(default_factory=list)
    unavailable: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)


def qdrant_env(fake: FakeQdrant, multitenancy: bool = True, grpc: bool = False) -> dict[str, str]:
    return {
        "VECTOR_DB": "qdrant",
        "QDRANT_URI": fake.base_url,
        "QDRANT_PREFER_GRPC": "true" if grpc else "false",
        "QDRANT_GRPC_PORT": str(fake.grpc_port),
        "ENABLE_QDRANT_MULTITENANCY_MODE": "true" if multitenancy else "false",
    }


def _payload_value(payload: dict, key: str) -> Any:
    value: Any = payload
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _condition_holds(point: dict, condition: dict) -> bool:
    if "has_id" in condition:
        return point["id"] in condition["has_id"]
    if any(clause in condition for clause in ("must", "should", "must_not")):
        return _filter_holds(point, condition)
    value = _payload_value(point["payload"], condition["key"])
    match = condition.get("match") or {}
    candidates = value if isinstance(value, list) else [value]
    if "value" in match:
        return match["value"] in candidates
    if "any" in match:
        return any(candidate in match["any"] for candidate in candidates)
    return False


def _filter_holds(point: dict, point_filter: dict | None) -> bool:
    if not point_filter:
        return True
    must = point_filter.get("must") or []
    should = point_filter.get("should") or []
    must_not = point_filter.get("must_not") or []
    return (
        all(_condition_holds(point, condition) for condition in must)
        and (not should or any(_condition_holds(point, condition) for condition in should))
        and not any(_condition_holds(point, condition) for condition in must_not)
    )


def _cosine(left: list[float], right: list[float]) -> float:
    norms = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right)) / norms if norms else 0.0


def _incoming_points(body: dict) -> list[dict]:
    if "points" in body:
        return body["points"]
    batch = body["batch"]
    payloads = batch.get("payloads") or [None] * len(batch["ids"])
    return [
        {"id": point_id, "vector": vector, "payload": payload}
        for point_id, vector, payload in zip(batch["ids"], batch["vectors"], payloads)
    ]


class _Api:
    def __init__(self, fake: FakeQdrant) -> None:
        self.fake = fake

    def _collection(self, name: str) -> dict[Any, dict]:
        return self.fake.collections.setdefault(name, {})

    def _matching(self, name: str, point_filter: dict | None) -> list[dict]:
        return [p for p in self._collection(name).values() if _filter_holds(p, point_filter)]

    def _refuses(self, limit: int | None) -> bool:
        ceiling = self.fake.max_query_limit
        if ceiling is not None and limit is not None and limit > ceiling:
            self.fake.refused_limits.append(limit)
            return True
        return False

    def handle(self, method: str, path: str, body: dict) -> tuple[int, Any]:
        fake = self.fake
        if method == "GET" and path == "/":
            return 200, {"title": "qdrant - vector search engine", "version": "1.18.0"}
        if method == "GET" and path == "/collections":
            return 200, {"collections": [{"name": name} for name in fake.collections]}
        matched = re.fullmatch(r"/collections/([^/]+)(/.*)?", path)
        if not matched:
            return 404, None
        name, action = matched.group(1), matched.group(2) or ""
        with fake.lock:
            if action == "/exists":
                return 200, {"exists": name in fake.collections}
            if action == "" and method == "PUT":
                fake.collections.setdefault(name, {})
                return 200, True
            if action == "" and method == "DELETE":
                return 200, fake.collections.pop(name, None) is not None
            if name not in fake.collections:
                return 404, f"Not found: Collection `{name}` doesn't exist!"
            if action == "/index":
                return 200, {"operation_id": 0, "status": "completed"}
            if action == "/points" and method == "PUT":
                for point in _incoming_points(body):
                    self._collection(name)[point["id"]] = point
                return 200, {"operation_id": 0, "status": "completed"}
            if action == "/points/delete":
                stored = self._collection(name)
                if "points" in body:
                    doomed = set(body["points"])
                else:
                    doomed = {
                        point_id
                        for point_id, point in stored.items()
                        if _filter_holds(point, body.get("filter"))
                    }
                for point_id in doomed:
                    stored.pop(point_id, None)
                return 200, {"operation_id": 0, "status": "completed"}
            if action == "/points/count":
                return 200, {"count": len(self._matching(name, body.get("filter")))}
            if action == "/points/scroll":
                return self._scroll(name, body)
            if action == "/points/query":
                if any(part in name for part in fake.unavailable):
                    return 503, f"Service unavailable: {name} has no active replica"
                return self._query(name, body)
        return 404, None

    def _scroll(self, name: str, body: dict) -> tuple[int, Any]:
        limit = body.get("limit", 10)
        if self._refuses(limit):
            return 400, f"Bad request: Limit exceeded {limit} > {self.fake.max_query_limit}"
        points = self._matching(name, body.get("filter"))
        ids = [point["id"] for point in points]
        start = ids.index(body["offset"]) if body.get("offset") in ids else 0
        page = points[start : start + limit]
        following = points[start + limit : start + limit + 1]
        records = [{"id": p["id"], "payload": p["payload"], "vector": None} for p in page]
        next_offset = following[0]["id"] if following else None
        return 200, {"points": records, "next_page_offset": next_offset}

    def _query(self, name: str, body: dict) -> tuple[int, Any]:
        limit = body.get("limit", 10)
        if self._refuses(limit):
            return 400, f"Bad request: Limit exceeded {limit} > {self.fake.max_query_limit}"
        query = body.get("query")
        vector = query.get("nearest", query) if isinstance(query, dict) else query
        points = self._matching(name, body.get("filter"))
        scored = sorted(
            ({**p, "score": _cosine(vector, p["vector"])} for p in points),
            key=lambda point: point["score"],
            reverse=True,
        )[:limit]
        return 200, {
            "points": [
                {"id": p["id"], "version": 0, "score": p["score"], "payload": p["payload"]}
                for p in scored
            ]
        }


def _grpc_server(api: _Api):
    """The gRPC face of the store: every call becomes the REST request `api` answers."""
    import grpc
    from qdrant_client import grpc as qdrant_grpc
    from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

    refusals = {404: grpc.StatusCode.NOT_FOUND, 503: grpc.StatusCode.UNAVAILABLE}

    def answer(context, method: str, path: str, body: dict | None = None) -> Any:
        api.fake.grpc_calls += 1
        status, result = api.handle(method, path, body or {})
        if status != 200:
            context.abort(refusals.get(status, grpc.StatusCode.INVALID_ARGUMENT), str(result))
        return result

    def filter_of(request) -> dict | None:
        if not request.HasField("filter"):
            return None
        return GrpcToRest.convert_filter(request.filter).model_dump(exclude_none=True)

    def point_id(rest_id):
        return RestToGrpc.convert_extended_point_id(rest_id)

    def done():
        completed = qdrant_grpc.UpdateResult(
            operation_id=0, status=qdrant_grpc.UpdateStatus.Completed
        )
        return qdrant_grpc.PointsOperationResponse(result=completed, time=0.0)

    class Health(qdrant_grpc.QdrantServicer):
        def HealthCheck(self, request, context):
            return qdrant_grpc.HealthCheckReply(title="qdrant", version="1.18.0")

    class Collections(qdrant_grpc.CollectionsServicer):
        def CollectionExists(self, request, context):
            found = answer(context, "GET", f"/collections/{request.collection_name}/exists")
            exists = qdrant_grpc.CollectionExists(exists=found["exists"])
            return qdrant_grpc.CollectionExistsResponse(result=exists, time=0.0)

        def Create(self, request, context):
            answer(context, "PUT", f"/collections/{request.collection_name}")
            return qdrant_grpc.CollectionOperationResponse(result=True, time=0.0)

        def Delete(self, request, context):
            dropped = answer(context, "DELETE", f"/collections/{request.collection_name}")
            return qdrant_grpc.CollectionOperationResponse(result=dropped, time=0.0)

        def List(self, request, context):
            listed = answer(context, "GET", "/collections")["collections"]
            descriptions = [qdrant_grpc.CollectionDescription(name=c["name"]) for c in listed]
            return qdrant_grpc.ListCollectionsResponse(collections=descriptions, time=0.0)

    class Points(qdrant_grpc.PointsServicer):
        def CreateFieldIndex(self, request, context):
            answer(context, "PUT", f"/collections/{request.collection_name}/index")
            return done()

        def Upsert(self, request, context):
            points = [GrpcToRest.convert_point_struct(p).model_dump() for p in request.points]
            answer(
                context, "PUT", f"/collections/{request.collection_name}/points", {"points": points}
            )
            return done()

        def Delete(self, request, context):
            selector = request.points
            if selector.HasField("points"):
                body = {"points": [GrpcToRest.convert_point_id(i) for i in selector.points.ids]}
            else:
                body = {"filter": filter_of(selector)}
            answer(context, "POST", f"/collections/{request.collection_name}/points/delete", body)
            return done()

        def Count(self, request, context):
            path = f"/collections/{request.collection_name}/points/count"
            counted = answer(context, "POST", path, {"filter": filter_of(request)})
            result = qdrant_grpc.CountResult(count=counted["count"])
            return qdrant_grpc.CountResponse(result=result, time=0.0)

        def Scroll(self, request, context):
            body = {
                "filter": filter_of(request),
                "limit": request.limit if request.HasField("limit") else 10,
            }
            if request.HasField("offset"):
                body["offset"] = GrpcToRest.convert_point_id(request.offset)
            path = f"/collections/{request.collection_name}/points/scroll"
            page = answer(context, "POST", path, body)
            records = [
                qdrant_grpc.RetrievedPoint(
                    id=point_id(p["id"]), payload=RestToGrpc.convert_payload(p["payload"])
                )
                for p in page["points"]
            ]
            following = page["next_page_offset"]
            if following is None:
                return qdrant_grpc.ScrollResponse(result=records, time=0.0)
            return qdrant_grpc.ScrollResponse(
                result=records, next_page_offset=point_id(following), time=0.0
            )

        def Query(self, request, context):
            body = {
                "query": list(request.query.nearest.dense.data),
                "limit": request.limit if request.HasField("limit") else 10,
                "filter": filter_of(request),
            }
            path = f"/collections/{request.collection_name}/points/query"
            found = answer(context, "POST", path, body)["points"]
            scored = [
                qdrant_grpc.ScoredPoint(
                    id=point_id(p["id"]),
                    payload=RestToGrpc.convert_payload(p["payload"]),
                    score=p["score"],
                    version=0,
                )
                for p in found
            ]
            return qdrant_grpc.QueryResponse(result=scored, time=0.0)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    qdrant_grpc.add_QdrantServicer_to_server(Health(), server)
    qdrant_grpc.add_CollectionsServicer_to_server(Collections(), server)
    qdrant_grpc.add_PointsServicer_to_server(Points(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    return server, port


@contextlib.contextmanager
def serving_qdrant() -> Iterator[FakeQdrant]:
    fake = FakeQdrant()
    api = _Api(fake)

    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _serve(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            path = self.path.split("?")[0]
            status, result = api.handle(self.command, path, json.loads(raw or "{}"))
            if path == "/":
                answer = result
            elif status == 200:
                answer = {"result": result, "status": OK, "time": 0.0}
            else:
                answer = {"status": {"error": result or "Not found"}, "time": 0.0}
            encoded = json.dumps(answer).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _serve

    server = ThreadingHTTPServer(("127.0.0.1", 0), RequestHandler)
    fake.base_url = f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    grpc_server, fake.grpc_port = _grpc_server(api)
    grpc_server.start()
    try:
        yield fake
    finally:
        grpc_server.stop(grace=None)
        server.shutdown()
        server.server_close()
