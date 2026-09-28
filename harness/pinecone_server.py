"""Pinecone played by local services: the control plane over REST, the index over gRPC and TLS.

Open WebUI's Pinecone store uses the gRPC client when the package has it: `list_indexes`,
`create_index` and `describe_index` go to the control plane (`PINECONE_CONTROLLER_HOST`, here a
local REST service), and the index host that `describe_index` names is reached over gRPC with
TLS. `serving_pinecone()` runs both and yields a `FakePinecone`: `indexes` holds the indexes by
name (their dimension and metric), `vectors` every stored vector by id with its values and
metadata, and `api_keys` the key each call carried. The index answers `Upsert`, `Query` (cosine
similarity, a metadata filter of `$eq`, `$ne`, `$in`, `$nin`, `$and` and `$or`, `top_k`) and
`Delete` (by ids, by filter or all), and refuses a vector of the wrong dimension as Pinecone does.

The index's certificate is issued by a scratch authority. `pinecone_env(fake, directory,
dimension)` is the environment of an instance on the fake with an index of that dimension, whose
Python trusts the authority the way a host with a private CA does: a `sitecustomize` in
`directory` points certifi, where the client reads its roots, at a bundle holding it.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import math
import os
import tempfile
import threading
from concurrent import futures
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from harness.instance import free_port

API_KEY = "pinecone-key-0123"
INDEX_NAME = "open-webui-harbour"
REGION = "us-east-1"


@dataclass
class FakePinecone:
    control_url: str = ""
    index_host: str = ""
    ca_bundle: str = ""
    dimension: int = 0
    indexes: dict[str, dict] = field(default_factory=dict)
    vectors: dict[str, tuple[list[float], dict]] = field(default_factory=dict)
    api_keys: list[str | None] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def pinecone_env(fake: FakePinecone, directory: Path, dimension: int) -> dict[str, str]:
    (directory / "sitecustomize.py").write_text(
        f"import certifi\ncertifi.where = lambda: {fake.ca_bundle!r}\n", encoding="utf-8"
    )
    inherited = os.environ.get("PYTHONPATH")
    search_path = f"{directory}{os.pathsep}{inherited}" if inherited else str(directory)
    fake.dimension = dimension
    return {
        "PYTHONPATH": search_path,
        "VECTOR_DB": "pinecone",
        "PINECONE_CONTROLLER_HOST": fake.control_url,
        "PINECONE_API_KEY": API_KEY,
        "PINECONE_ENVIRONMENT": REGION,
        "PINECONE_CLOUD": "aws",
        "PINECONE_INDEX_NAME": INDEX_NAME,
        "PINECONE_DIMENSION": str(dimension),
        "PINECONE_METRIC": "cosine",
    }


# ---------------------------------------------------------------- the metadata filter


def _condition_holds(value: Any, condition: Any) -> bool:
    if not isinstance(condition, dict):
        return value == condition
    for operator, operand in condition.items():
        if operator == "$eq" and value != operand:
            return False
        if operator == "$ne" and value == operand:
            return False
        if operator == "$in" and value not in operand:
            return False
        if operator == "$nin" and value in operand:
            return False
        if operator not in ("$eq", "$ne", "$in", "$nin"):
            raise ValueError(f"the stand-in does not know the {operator!r} operator")
    return True


def filter_holds(metadata: dict, pinecone_filter: dict) -> bool:
    for key, condition in pinecone_filter.items():
        if key == "$and":
            if not all(filter_holds(metadata, inner) for inner in condition):
                return False
        elif key == "$or":
            if not any(filter_holds(metadata, inner) for inner in condition):
                return False
        elif not _condition_holds(metadata.get(key), condition):
            return False
    return True


def _cosine(left: list[float], right: list[float]) -> float:
    norms = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right)) / norms if norms else 0.0


# ---------------------------------------------------------------- the control plane


def _index_model(fake: FakePinecone, name: str) -> dict:
    index = fake.indexes[name]
    return {
        "name": name,
        "dimension": index["dimension"],
        "metric": index["metric"],
        "host": fake.index_host,
        "spec": {"serverless": {"cloud": index["cloud"], "region": index["region"]}},
        "status": {"ready": True, "state": "Ready"},
        "deletion_protection": "disabled",
        "vector_type": "dense",
    }


def _control_answer(fake: FakePinecone, method: str, path: str, body: dict | None):
    parts = [part for part in path.split("?")[0].strip("/").split("/") if part]
    if parts == ["indexes"] and method == "GET":
        return 200, {"indexes": [_index_model(fake, name) for name in fake.indexes]}
    if parts == ["indexes"] and method == "POST":
        serverless = body["spec"]["serverless"]
        fake.indexes[body["name"]] = {
            "dimension": body["dimension"],
            "metric": body.get("metric", "cosine"),
            "cloud": serverless["cloud"],
            "region": serverless["region"],
        }
        return 201, _index_model(fake, body["name"])
    if len(parts) == 2 and parts[0] == "indexes" and method == "GET":
        if parts[1] not in fake.indexes:
            return 404, {"error": {"code": "NOT_FOUND", "message": f"{parts[1]} not found"}}
        return 200, _index_model(fake, parts[1])
    return 404, {"error": {"code": "NOT_FOUND", "message": f"{method} {path}"}}


@contextlib.contextmanager
def _serving_control_plane(fake: FakePinecone) -> Iterator[str]:
    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _serve(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else None
            with fake.lock:
                fake.api_keys.append(self.headers.get("Api-Key"))
                status, answer = _control_answer(fake, self.command, self.path, body)
            encoded = json.dumps(answer).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        do_GET = do_POST = do_DELETE = do_PATCH = _serve

    server = ThreadingHTTPServer(("127.0.0.1", 0), RequestHandler)
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------- the index over gRPC


def _certificates(directory: Path) -> tuple[bytes, bytes, str]:
    """(server key, server certificate chain, path of the CA bundle) for `localhost`."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = datetime.datetime.now(datetime.timezone.utc)
    validity = (now - datetime.timedelta(days=1), now + datetime.timedelta(days=1))

    def certificate(subject: str, key, issuer: str, signer, *, authority: bool):
        builder = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer)]))
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(validity[0])
            .not_valid_after(validity[1])
            .add_extension(x509.BasicConstraints(ca=authority, path_length=None), critical=True)
        )
        if not authority:
            names = x509.SubjectAlternativeName([x509.DNSName("localhost")])
            builder = builder.add_extension(names, critical=False)
        return builder.sign(signer, hashes.SHA256())

    authority_key = ec.generate_private_key(ec.SECP256R1())
    authority = certificate(
        "harness authority", authority_key, "harness authority", authority_key, authority=True
    )
    server_key = ec.generate_private_key(ec.SECP256R1())
    server = certificate(
        "localhost", server_key, "harness authority", authority_key, authority=False
    )

    bundle = directory / "authority.pem"
    bundle.write_bytes(authority.public_bytes(serialization.Encoding.PEM))
    key_pem = server_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return key_pem, server.public_bytes(serialization.Encoding.PEM), str(bundle)


@contextlib.contextmanager
def _serving_index(fake: FakePinecone, key: bytes, chain: bytes) -> Iterator[str]:
    grpc = pytest.importorskip("grpc")
    from google.protobuf import json_format
    from pinecone.core.grpc.protos import db_data_2025_01_pb2 as messages
    from pinecone.core.grpc.protos import db_data_2025_01_pb2_grpc as services

    def as_dict(struct) -> dict:
        return json_format.MessageToDict(struct) if struct is not None else {}

    class Index(services.VectorServiceServicer):
        def _checked(self, context) -> None:
            key_sent = dict(context.invocation_metadata()).get("api-key")
            with fake.lock:
                fake.api_keys.append(key_sent)

        def Upsert(self, request, context):  # noqa: N802
            self._checked(context)
            for vector in request.vectors:
                if len(vector.values) != fake.dimension:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"Vector dimension {len(vector.values)} does not match the dimension "
                        f"of the index {fake.dimension}",
                    )
            with fake.lock:
                for vector in request.vectors:
                    fake.vectors[vector.id] = (list(vector.values), as_dict(vector.metadata))
            return messages.UpsertResponse(upserted_count=len(request.vectors))

        def Query(self, request, context):  # noqa: N802
            self._checked(context)
            wanted = as_dict(request.filter) if request.HasField("filter") else {}
            with fake.lock:
                stored = list(fake.vectors.items())
            scored = [
                (_cosine(list(request.vector), values), vector_id, metadata)
                for vector_id, (values, metadata) in stored
                if filter_holds(metadata, wanted)
            ]
            scored.sort(key=lambda entry: entry[0], reverse=True)
            matches = [
                messages.ScoredVector(
                    id=vector_id,
                    score=score,
                    metadata=metadata if request.include_metadata else None,
                )
                for score, vector_id, metadata in scored[: request.top_k]
            ]
            return messages.QueryResponse(matches=matches, namespace=request.namespace)

        def Delete(self, request, context):  # noqa: N802
            self._checked(context)
            wanted = as_dict(request.filter) if request.HasField("filter") else None
            with fake.lock:
                if request.delete_all:
                    fake.vectors.clear()
                for vector_id in request.ids:
                    fake.vectors.pop(vector_id, None)
                if wanted is not None:
                    for vector_id, (_, metadata) in list(fake.vectors.items()):
                        if filter_holds(metadata, wanted):
                            del fake.vectors[vector_id]
            return messages.DeleteResponse()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    services.add_VectorServiceServicer_to_server(Index(), server)
    port = free_port()
    server.add_secure_port(f"localhost:{port}", grpc.ssl_server_credentials([(key, chain)]))
    server.start()
    try:
        yield f"localhost:{port}"
    finally:
        server.stop(grace=None)


@contextlib.contextmanager
def serving_pinecone() -> Iterator[FakePinecone]:
    pytest.importorskip("pinecone.grpc", reason="Open WebUI talks to Pinecone over gRPC")
    fake = FakePinecone()
    with tempfile.TemporaryDirectory(prefix="owui-pinecone-") as directory:
        key, chain, fake.ca_bundle = _certificates(Path(directory))
        with _serving_control_plane(fake) as control_url, _serving_index(fake, key, chain) as host:
            fake.control_url, fake.index_host = control_url, host
            yield fake
