"""A Weaviate server played by local services that keep objects, for an instance booted on it.

The v4 weaviate-client speaks REST for the schema and gRPC for data. `serving_weaviate()` runs
both and yields a `FakeWeaviate`: REST answers `/v1/meta`, readiness, collections (create, read,
list, delete) and deleting one object by id; gRPC answers the health check, `BatchObjects`
(insert, or replace an object with the same id), `Search` (a `near_vector` query ranked by cosine
distance, or a plain fetch paged by id after a cursor, each under a filter of property equalities
joined by AND and OR) and `BatchDelete` by filter. `collections` holds each collection's class
definition and `objects` its objects by id, each with its properties and vector. `api_keys`
lists the bearer key of every REST request and gRPC call. `weaviate_env(fake)` is the
environment of an instance that stores its vectors there.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import struct
import threading
import uuid
from concurrent import futures
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import grpc
from google.protobuf import struct_pb2
from weaviate.proto.v1 import (
    base_pb2,
    batch_delete_pb2,
    batch_pb2,
    health_weaviate_pb2,
    properties_pb2,
    search_get_pb2,
    weaviate_pb2_grpc,
)

API_KEY = "weaviate-harbour-key"
VERSION = "1.30.0"
OBJECT_PATH = re.compile(r"^/v1/objects/(?P<collection>[^/]+)/(?P<id>[0-9a-fA-F-]{36})$")
SCHEMA_PATH = re.compile(r"^/v1/schema/(?P<collection>[^/]+)$")


@dataclass
class StoredObject:
    properties: dict
    vector: list[float]


@dataclass
class FakeWeaviate:
    http_port: int = 0
    grpc_port: int = 0
    collections: dict[str, dict] = field(default_factory=dict)
    objects: dict[str, dict[str, StoredObject]] = field(default_factory=dict)
    api_keys: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_key(self, authorization: str) -> None:
        with self.lock:
            self.api_keys.append(authorization.removeprefix("Bearer ").strip())

    def texts(self, collection: str) -> list[str]:
        with self.lock:
            stored = self.objects.get(collection, {})
            return sorted(entry.properties.get("text", "") for entry in stored.values())


# ---------------------------------------------------------------- values in and out


def _from_struct(value: struct_pb2.Value):
    kind = value.WhichOneof("kind")
    if kind == "string_value":
        return value.string_value
    if kind == "number_value":
        return value.number_value
    if kind == "bool_value":
        return value.bool_value
    if kind == "list_value":
        return [_from_struct(item) for item in value.list_value.values]
    if kind == "struct_value":
        return {key: _from_struct(item) for key, item in value.struct_value.fields.items()}
    return None


def _batch_properties(properties: batch_pb2.BatchObject.Properties) -> dict:
    values = {key: _from_struct(item) for key, item in properties.non_ref_properties.fields.items()}
    for array in properties.text_array_properties:
        values[array.prop_name] = list(array.values)
    for array in properties.int_array_properties:
        values[array.prop_name] = list(array.values)
    for array in properties.boolean_array_properties:
        values[array.prop_name] = list(array.values)
    for array in properties.number_array_properties:
        raw = array.values_bytes
        values[array.prop_name] = list(struct.unpack(f"<{len(raw) // 8}d", raw)) if raw else []
    for name in properties.empty_list_props:
        values[name] = []
    return values


def _to_value(value) -> properties_pb2.Value:
    if value is None:
        return properties_pb2.Value(null_value=struct_pb2.NULL_VALUE)
    if isinstance(value, bool):
        return properties_pb2.Value(bool_value=value)
    if isinstance(value, int):
        return properties_pb2.Value(int_value=value)
    if isinstance(value, float):
        return properties_pb2.Value(number_value=value)
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            texts = properties_pb2.TextValues(values=value)
            return properties_pb2.Value(list_value=properties_pb2.ListValue(text_values=texts))
        packed = struct.pack(f"<{len(value)}d", *value)
        numbers = properties_pb2.NumberValues(values=packed)
        return properties_pb2.Value(list_value=properties_pb2.ListValue(number_values=numbers))
    if isinstance(value, dict):
        fields = {key: _to_value(item) for key, item in value.items()}
        return properties_pb2.Value(object_value=properties_pb2.Properties(fields=fields))
    return properties_pb2.Value(text_value=str(value))


def _vector(raw: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def _cosine_distance(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return 1.0 - (dot / norms if norms else 0.0)


# ---------------------------------------------------------------- filters


def _filter_value(condition: base_pb2.Filters):
    kind = next(
        name
        for name in ("value_text", "value_int", "value_boolean", "value_number")
        if condition.HasField(name)
    )
    return getattr(condition, kind)


def _matches(properties: dict, condition: base_pb2.Filters | None) -> bool:
    if condition is None:
        return True
    operator = condition.operator
    if operator == base_pb2.Filters.OPERATOR_AND:
        return all(_matches(properties, part) for part in condition.filters)
    if operator == base_pb2.Filters.OPERATOR_OR:
        return any(_matches(properties, part) for part in condition.filters)
    name = condition.target.property if condition.HasField("target") else condition.on[0]
    expected = _filter_value(condition)
    if operator == base_pb2.Filters.OPERATOR_EQUAL:
        return properties.get(name) == expected
    if operator == base_pb2.Filters.OPERATOR_NOT_EQUAL:
        return properties.get(name) != expected
    raise NotImplementedError(f"filter operator {operator} is not played by the fake")


# ---------------------------------------------------------------- gRPC


def _key_from(context) -> str:
    return dict(context.invocation_metadata()).get("authorization", "")


class _Weaviate(weaviate_pb2_grpc.WeaviateServicer):
    def __init__(self, fake: FakeWeaviate):
        self.fake = fake

    def BatchObjects(self, request, context):
        self.fake.record_key(_key_from(context))
        with self.fake.lock:
            for entry in request.objects:
                raw = entry.vector_bytes or (
                    entry.vectors[0].vector_bytes if entry.vectors else b""
                )
                stored = self.fake.objects.setdefault(entry.collection, {})
                object_id = entry.uuid or str(uuid.uuid4())
                stored[object_id] = StoredObject(_batch_properties(entry.properties), _vector(raw))
        return batch_pb2.BatchObjectsReply(took=0.001)

    def BatchDelete(self, request, context):
        self.fake.record_key(_key_from(context))
        condition = request.filters if request.HasField("filters") else None
        with self.fake.lock:
            stored = self.fake.objects.get(request.collection, {})
            doomed = [key for key, entry in stored.items() if _matches(entry.properties, condition)]
            if not request.dry_run:
                for key in doomed:
                    del stored[key]
        return batch_delete_pb2.BatchDeleteReply(
            took=0.001, matches=len(doomed), successful=len(doomed), failed=0
        )

    def Search(self, request, context):
        self.fake.record_key(_key_from(context))
        condition = request.filters if request.HasField("filters") else None
        with self.fake.lock:
            stored = dict(self.fake.objects.get(request.collection, {}))
        candidates = [
            (key, entry) for key, entry in stored.items() if _matches(entry.properties, condition)
        ]
        distances: dict[str, float] = {}
        if request.HasField("near_vector"):
            near = request.near_vector
            raw = near.vector_bytes or (near.vectors[0].vector_bytes if near.vectors else b"")
            query = _vector(raw) if raw else list(near.vector)
            distances = {key: _cosine_distance(query, entry.vector) for key, entry in candidates}
            candidates.sort(key=lambda pair: distances[pair[0]])
        else:
            candidates.sort(key=lambda pair: pair[0])
            if request.after:
                candidates = [pair for pair in candidates if pair[0] > request.after]
        if request.limit:
            candidates = candidates[: request.limit]
        return search_get_pb2.SearchReply(
            took=0.001, results=[self._result(key, entry, distances) for key, entry in candidates]
        )

    @staticmethod
    def _result(key: str, entry: StoredObject, distances: dict[str, float]):
        metadata = search_get_pb2.MetadataResult(id=key, id_as_bytes=uuid.UUID(key).bytes)
        if key in distances:
            metadata.distance = distances[key]
            metadata.distance_present = True
        fields = {name: _to_value(value) for name, value in entry.properties.items()}
        properties = search_get_pb2.PropertiesResult(
            non_ref_props=properties_pb2.Properties(fields=fields)
        )
        return search_get_pb2.SearchResult(metadata=metadata, properties=properties)


def _health_check(request, context):
    return health_weaviate_pb2.WeaviateHealthCheckResponse(
        status=health_weaviate_pb2.WeaviateHealthCheckResponse.SERVING
    )


def _health_service() -> grpc.GenericRpcHandler:
    check = grpc.unary_unary_rpc_method_handler(
        _health_check,
        request_deserializer=health_weaviate_pb2.WeaviateHealthCheckRequest.FromString,
        response_serializer=health_weaviate_pb2.WeaviateHealthCheckResponse.SerializeToString,
    )
    return grpc.method_handlers_generic_handler("grpc.health.v1.Health", {"Check": check})


# ---------------------------------------------------------------- REST


HNSW_DEFAULTS = {
    "cleanupIntervalSeconds": 300,
    "distance": "cosine",
    "dynamicEfMin": 100,
    "dynamicEfMax": 500,
    "dynamicEfFactor": 8,
    "ef": -1,
    "efConstruction": 128,
    "flatSearchCutoff": 40000,
    "maxConnections": 32,
    "skip": False,
    "vectorCacheMaxObjects": 1000000000000,
}


def _named_vector(sent: dict) -> dict:
    return {
        **sent,
        "vectorIndexType": sent.get("vectorIndexType", "hnsw"),
        "vectorIndexConfig": {**HNSW_DEFAULTS, **sent.get("vectorIndexConfig", {})},
    }


def _class_definition(created: dict) -> dict:
    """The class as Weaviate returns it: what was sent, with the defaults filled in."""
    return {
        "invertedIndexConfig": {},
        "replicationConfig": {"factor": 1},
        "shardingConfig": {},
        "multiTenancyConfig": {"enabled": False},
        "moduleConfig": {},
        **created,
        "vectorConfig": {
            name: _named_vector(sent) for name, sent in created.get("vectorConfig", {}).items()
        },
        "properties": [
            {"indexFilterable": True, "indexSearchable": True, "tokenization": "word", **prop}
            for prop in created.get("properties", [])
        ],
    }


def _rest_handler(fake: FakeWeaviate):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _answer(self, status: int, body=None) -> None:
            encoded = b"" if body is None else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}") if length else {}

        def _record(self) -> None:
            if not self.path.startswith("/v1/.well-known"):
                fake.record_key(self.headers.get("Authorization", ""))

        def do_GET(self) -> None:
            self._record()
            schema = SCHEMA_PATH.match(self.path)
            if self.path == "/v1/meta":
                self._answer(200, {"version": VERSION, "modules": {}, "hostname": "fake"})
            elif self.path in ("/v1/.well-known/ready", "/v1/.well-known/live"):
                self._answer(200, {})
            elif self.path == "/v1/schema":
                with fake.lock:
                    classes = list(fake.collections.values())
                self._answer(200, {"classes": classes})
            elif schema:
                with fake.lock:
                    found = fake.collections.get(schema["collection"])
                self._answer(200, found) if found else self._answer(404)
            else:
                self._answer(404, {"error": [{"message": f"no route {self.path}"}]})

        def do_POST(self) -> None:
            self._record()
            if self.path == "/v1/schema":
                created = _class_definition(self._body())
                with fake.lock:
                    fake.collections[created["class"]] = created
                    fake.objects.setdefault(created["class"], {})
                self._answer(200, created)
            else:
                self._answer(404, {"error": [{"message": f"no route {self.path}"}]})

        def do_DELETE(self) -> None:
            self._record()
            schema = SCHEMA_PATH.match(self.path)
            stored_object = OBJECT_PATH.match(self.path.split("?")[0])
            if schema:
                with fake.lock:
                    fake.collections.pop(schema["collection"], None)
                    fake.objects.pop(schema["collection"], None)
                self._answer(200)
            elif stored_object:
                with fake.lock:
                    stored = fake.objects.get(stored_object["collection"], {})
                    removed = stored.pop(stored_object["id"], None)
                self._answer(204) if removed else self._answer(404)
            else:
                self._answer(404, {"error": [{"message": f"no route {self.path}"}]})

    return Handler


@contextlib.contextmanager
def serving_weaviate() -> Iterator[FakeWeaviate]:
    fake = FakeWeaviate()
    rest = ThreadingHTTPServer(("127.0.0.1", 0), _rest_handler(fake))
    rest.daemon_threads = True
    fake.http_port = rest.server_address[1]
    threading.Thread(target=rest.serve_forever, args=(0.05,), daemon=True).start()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    weaviate_pb2_grpc.add_WeaviateServicer_to_server(_Weaviate(fake), server)
    server.add_generic_rpc_handlers((_health_service(),))
    fake.grpc_port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield fake
    finally:
        server.stop(grace=None)
        rest.shutdown()
        rest.server_close()


def weaviate_env(fake: FakeWeaviate) -> dict[str, str]:
    """The environment of an instance whose vectors live in `fake`, signing in with `API_KEY`."""
    dead_proxy = "http://127.0.0.1:9"
    loopback = "127.0.0.1,localhost"
    return {
        "VECTOR_DB": "weaviate",
        "WEAVIATE_HTTP_HOST": "127.0.0.1",
        "WEAVIATE_HTTP_PORT": str(fake.http_port),
        "WEAVIATE_GRPC_HOST": "127.0.0.1",
        "WEAVIATE_GRPC_PORT": str(fake.grpc_port),
        "WEAVIATE_API_KEY": API_KEY,
        # the client's start-up asks PyPI for its latest release; fail that at once
        "HTTPS_PROXY": dead_proxy,
        "https_proxy": dead_proxy,
        "NO_PROXY": loopback,
        "no_proxy": loopback,
    }
