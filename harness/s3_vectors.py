"""Amazon S3 Vectors played by a local service that keeps vectors, for an instance storing there.

`serving_s3_vectors()` yields a `FakeS3Vectors` speaking the JSON API boto3's `s3vectors` client
uses: create, get, list and delete indexes, and put, list (paged by `nextToken`), query and delete
vectors. A query ranks by cosine distance (0 for the same direction), nearest first, keeps to
`topK` and to an equality or `$in` metadata filter, the way the service does. A missing index
answers `NotFoundException`. `indexes` is what it keeps, by index name, and `calls` the name of
every operation it was asked for.

`s3_vectors_env(fake, bucket)` is the environment of an instance that keeps its vectors there;
boto3 reaches the fake through its per-service endpoint variable.
"""

from __future__ import annotations

import contextlib
import json
import math
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator


@dataclass
class FakeS3Vectors:
    base_url: str = ""
    indexes: dict[str, dict[str, dict]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def s3_vectors_env(fake: FakeS3Vectors, bucket: str) -> dict[str, str]:
    return {
        "VECTOR_DB": "s3vector",
        "S3_VECTOR_BUCKET_NAME": bucket,
        "S3_VECTOR_REGION": "us-east-1",
        "AWS_ENDPOINT_URL_S3VECTORS": fake.base_url,
        "AWS_ACCESS_KEY_ID": "AKIAOPENWEBUITEST",
        "AWS_SECRET_ACCESS_KEY": "open-webui-test-secret",
    }


def _cosine_distance(left: list[float], right: list[float]) -> float:
    norms = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return 1.0 - (sum(x * y for x, y in zip(left, right)) / norms if norms else 0.0)


def _matches(metadata: dict, vector_filter: dict | None) -> bool:
    for key, expected in (vector_filter or {}).items():
        if key == "$and":
            if not all(_matches(metadata, part) for part in expected):
                return False
        elif key == "$or":
            if not any(_matches(metadata, part) for part in expected):
                return False
        elif isinstance(expected, dict) and "$in" in expected:
            if metadata.get(key) not in expected["$in"]:
                return False
        elif isinstance(expected, dict) and "$eq" in expected:
            if metadata.get(key) != expected["$eq"]:
                return False
        elif metadata.get(key) != expected:
            return False
    return True


class _NotFound(Exception):
    pass


class _Api:
    def __init__(self, fake: FakeS3Vectors) -> None:
        self.fake = fake

    def _index(self, body: dict) -> dict[str, dict]:
        vectors = self.fake.indexes.get(body["indexName"])
        if vectors is None:
            raise _NotFound(f"The specified index could not be found: {body['indexName']}")
        return vectors

    def handle(self, operation: str, body: dict) -> dict:
        fake = self.fake
        if operation == "CreateIndex":
            fake.indexes.setdefault(body["indexName"], {})
            return {}
        if operation == "GetIndex":
            self._index(body)
            return {
                "index": {
                    "vectorBucketName": body["vectorBucketName"],
                    "indexName": body["indexName"],
                    "dataType": "float32",
                    "distanceMetric": "cosine",
                }
            }
        if operation == "ListIndexes":
            return {"indexes": [{"indexName": name} for name in fake.indexes]}
        if operation == "DeleteIndex":
            self._index(body)
            del fake.indexes[body["indexName"]]
            return {}
        vectors = self._index(body)
        if operation == "PutVectors":
            for vector in body["vectors"]:
                vectors[vector["key"]] = vector
            return {}
        if operation == "DeleteVectors":
            for key in body["keys"]:
                vectors.pop(key, None)
            return {}
        if operation == "ListVectors":
            return self._list(vectors, body)
        if operation == "QueryVectors":
            return self._query(vectors, body)
        raise _NotFound(f"unknown operation {operation}")

    @staticmethod
    def _list(vectors: dict[str, dict], body: dict) -> dict:
        keys = list(vectors)
        start = int(body.get("nextToken") or 0)
        page = keys[start : start + body.get("maxResults", 500)]
        listed = []
        for key in page:
            entry = {"key": key}
            if body.get("returnMetadata"):
                entry["metadata"] = vectors[key].get("metadata", {})
            if body.get("returnData"):
                entry["data"] = vectors[key]["data"]
            listed.append(entry)
        answer: dict[str, Any] = {"vectors": listed}
        if start + len(page) < len(keys):
            answer["nextToken"] = str(start + len(page))
        return answer

    @staticmethod
    def _query(vectors: dict[str, dict], body: dict) -> dict:
        query = body["queryVector"]["float32"]
        candidates = [
            vector
            for vector in vectors.values()
            if _matches(vector.get("metadata", {}), body.get("filter"))
        ]
        ranked = sorted(
            ((_cosine_distance(query, vector["data"]["float32"]), vector) for vector in candidates),
            key=lambda pair: pair[0],
        )[: body["topK"]]
        found = []
        for distance, vector in ranked:
            entry: dict[str, Any] = {"key": vector["key"]}
            if body.get("returnMetadata"):
                entry["metadata"] = vector.get("metadata", {})
            if body.get("returnDistance"):
                entry["distance"] = distance
            found.append(entry)
        return {"vectors": found, "distanceMetric": "cosine"}


@contextlib.contextmanager
def serving_s3_vectors() -> Iterator[FakeS3Vectors]:
    fake = FakeS3Vectors()
    api = _Api(fake)

    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            operation = self.path.split("?")[0].strip("/")
            headers = {"Content-Type": "application/json"}
            with fake.lock:
                fake.calls.append(operation)
                try:
                    status, answer = 200, api.handle(operation, body)
                except _NotFound as missing:
                    status, answer = 404, {"message": str(missing)}
                    headers["x-amzn-ErrorType"] = "NotFoundException"
            encoded = json.dumps(answer).encode()
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), RequestHandler)
    fake.base_url = f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    try:
        yield fake
    finally:
        server.shutdown()
        server.server_close()
