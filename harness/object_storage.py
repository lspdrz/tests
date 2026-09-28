"""S3, Azure Blob Storage and Google Cloud Storage played by local services that keep objects.

`serving_s3()` yields a `FakeS3` speaking the part of the S3 REST API boto3 uses for Open WebUI's
file storage, with path-style addressing: put, head, get, delete and tag an object, and list a
bucket (`list-type=2`). `serving_azure_blob()` yields a `FakeAzureBlob` speaking the Blob service
REST API azure-storage-blob uses: put, get (ranged), delete and list the blobs of a container,
under an account path the way the Azurite emulator serves it. Neither checks signatures, and
both refuse a request without the `Authorization` header the SDK signs with, as the real
services do. `serving_gcs()` yields a `FakeGcs` speaking the part of the Cloud Storage JSON API
google-cloud-storage uses: a multipart upload, an object's metadata and media, delete and list.
Like the emulator the SDK is pointed at with `STORAGE_EMULATOR_HOST`, it takes anonymous
requests; with `token_endpoint=True` it also answers a service account's OAuth token exchange
at `/token` and refuses storage requests without the token it issued there (`GCS_ACCESS_TOKEN`).
`objects` is what each keeps, by bucket or container and key, `tags` the S3 tag set of each key,
`requests` every `(method, path)` they got and `authorizations` the `Authorization` header of each.

`s3_env(fake, bucket)`, `azure_blob_env(fake, container)` and `gcs_env(fake, bucket)` are the
environment of an instance that keeps its uploads there.
"""

from __future__ import annotations

import base64
import contextlib
import datetime
import email.parser
import email.policy
import hashlib
import ipaddress
import json
import ssl
import tempfile
import threading
import urllib.parse
from dataclasses import dataclass, field
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree
from xml.sax.saxutils import escape

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

AZURE_ACCOUNT = "devstoreaccount1"
# the Azurite emulator's well-known key; any base64 key signs, the fake never checks it
AZURE_ACCOUNT_KEY = (
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
)
S3_NAMESPACE = "http://s3.amazonaws.com/doc/2006-03-01/"
GCS_ACCESS_TOKEN = "service-account-storage-token"

Answer = tuple[int, dict[str, str], bytes]


@dataclass
class _Store:
    base_url: str = ""
    objects: dict[str, dict[str, bytes]] = field(default_factory=dict)
    requests: list[tuple[str, str]] = field(default_factory=list)
    authorizations: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def keys(self, bucket: str) -> list[str]:
        with self.lock:
            return sorted(self.objects.get(bucket, {}))

    def sent(self, method: str) -> list[str]:
        """The paths of every request of that method, query string included."""
        with self.lock:
            return [path for sent_method, path in self.requests if sent_method == method]


@dataclass
class FakeS3(_Store):
    tags: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class FakeAzureBlob(_Store):
    # the certificate of a service served over TLS, for the client to trust
    ca_bundle: str = ""


@dataclass
class FakeGcs(_Store):
    # a service account's token exchange: the JWT assertions it was sent, newest last
    assertions: list[str] = field(default_factory=list)
    token_endpoint: bool = False


def s3_env(fake: FakeS3, bucket: str, **settings: str) -> dict[str, str]:
    return {
        "STORAGE_PROVIDER": "s3",
        "S3_ENDPOINT_URL": fake.base_url,
        "S3_BUCKET_NAME": bucket,
        "S3_REGION_NAME": "us-east-1",
        "S3_ACCESS_KEY_ID": "AKIAOPENWEBUITEST",
        "S3_SECRET_ACCESS_KEY": "open-webui-test-secret",
        "S3_ADDRESSING_STYLE": "path",
        **settings,
    }


def azure_blob_env(fake: FakeAzureBlob, container: str, with_key: bool = True) -> dict[str, str]:
    """Without the key the instance signs in with its managed identity, which needs TLS."""
    env = {
        "STORAGE_PROVIDER": "azure",
        "AZURE_STORAGE_ENDPOINT": f"{fake.base_url}/{AZURE_ACCOUNT}",
        "AZURE_STORAGE_CONTAINER_NAME": container,
    }
    if with_key:
        env["AZURE_STORAGE_KEY"] = AZURE_ACCOUNT_KEY
    if fake.ca_bundle:
        env["REQUESTS_CA_BUNDLE"] = fake.ca_bundle
    return env


def gcs_env(fake: FakeGcs, bucket: str, credentials: dict | None = None) -> dict[str, str]:
    """Without `credentials` (a service account's JSON key) the SDK goes anonymous."""
    env = {
        "STORAGE_PROVIDER": "gcs",
        "GCS_BUCKET_NAME": bucket,
        "STORAGE_EMULATOR_HOST": fake.base_url,
    }
    if credentials:
        env["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = json.dumps(credentials)
    return env


def _etag(body: bytes) -> str:
    return f'"{hashlib.md5(body).hexdigest()}"'


def _object_headers(body: bytes) -> dict[str, str]:
    return {
        "ETag": _etag(body),
        "Last-Modified": formatdate(usegmt=True),
        "Content-Type": "application/octet-stream",
    }


def _xml(body: str, status: int = 200, extra: dict[str, str] | None = None) -> Answer:
    headers = {"Content-Type": "application/xml", **(extra or {})}
    return status, headers, f'<?xml version="1.0" encoding="UTF-8"?>{body}'.encode()


def _s3_error(status: int, code: str) -> Answer:
    return _xml(f"<Error><Code>{code}</Code><Message>{code}</Message></Error>", status)


def _s3_tag_set(body: bytes) -> dict[str, str]:
    root = ElementTree.fromstring(body)
    return {
        tag.findtext(f"{{{S3_NAMESPACE}}}Key") or tag.findtext("Key"): (
            tag.findtext(f"{{{S3_NAMESPACE}}}Value") or tag.findtext("Value") or ""
        )
        for tag in root.iter()
        if tag.tag.endswith("Tag") and not tag.tag.endswith("TagSet")
    }


def _s3_answer(fake: FakeS3, method: str, path: str, query: dict, body: bytes) -> Answer:
    bucket, _, key = urllib.parse.unquote(path).lstrip("/").partition("/")
    with fake.lock:
        stored = fake.objects.setdefault(bucket, {})
        if not key and method == "GET":
            # boto3 asks for url-encoded keys and decodes them again
            encode = urllib.parse.quote if query.get("encoding-type") == "url" else str
            contents = "".join(
                f"<Contents><Key>{escape(encode(name))}</Key><Size>{len(data)}</Size>"
                f"<ETag>{escape(_etag(data))}</ETag><StorageClass>STANDARD</StorageClass></Contents>"
                for name, data in sorted(stored.items())
            )
            listing = (
                f'<ListBucketResult xmlns="{S3_NAMESPACE}"><Name>{escape(bucket)}</Name>'
                f"<KeyCount>{len(stored)}</KeyCount><MaxKeys>1000</MaxKeys>"
                f"<IsTruncated>false</IsTruncated>{contents}</ListBucketResult>"
            )
            return _xml(listing)
        if "tagging" in query and method == "PUT":
            if key not in stored:
                return _s3_error(404, "NoSuchKey")
            fake.tags[key] = _s3_tag_set(body)
            return 200, {}, b""
        if method == "PUT":
            stored[key] = body
            return 200, {"ETag": _etag(body)}, b""
        if key not in stored:
            return _s3_error(404, "NoSuchKey") if method != "DELETE" else (204, {}, b"")
        if method in ("GET", "HEAD"):
            return 200, _object_headers(stored[key]), stored[key]
        if method == "DELETE":
            del stored[key]
            fake.tags.pop(key, None)
            return 204, {}, b""
    return _s3_error(405, "MethodNotAllowed")


def _azure_error(status: int, code: str) -> Answer:
    body = f"<Error><Code>{code}</Code><Message>{code}</Message></Error>"
    return _xml(body, status, {"x-ms-error-code": code})


def _azure_ranged(data: bytes, requested: str | None) -> Answer:
    headers = {**_object_headers(data), "x-ms-blob-type": "BlockBlob"}
    if not requested:
        return 200, headers, data
    start, _, end = requested.removeprefix("bytes=").partition("-")
    first = int(start)
    last = min(int(end) if end else len(data) - 1, len(data) - 1)
    headers["Content-Range"] = f"bytes {first}-{last}/{len(data)}"
    return 206, headers, data[first : last + 1]


def _azure_answer(
    fake: FakeAzureBlob, method: str, path: str, query: dict, headers: dict, body: bytes
) -> Answer:
    account, _, rest = urllib.parse.unquote(path).lstrip("/").partition("/")
    container, _, blob = rest.partition("/")
    if account != AZURE_ACCOUNT:
        return _azure_error(400, "InvalidUri")
    with fake.lock:
        stored = fake.objects.setdefault(container, {})
        if not blob and query.get("comp") == "list" and method == "GET":
            blobs = "".join(
                f"<Blob><Name>{escape(name)}</Name><Properties>"
                f"<Content-Length>{len(data)}</Content-Length><BlobType>BlockBlob</BlobType>"
                f"</Properties></Blob>"
                for name, data in sorted(stored.items())
            )
            listing = (
                f'<EnumerationResults ServiceEndpoint="{fake.base_url}/{AZURE_ACCOUNT}/" '
                f'ContainerName="{escape(container)}"><Blobs>{blobs}</Blobs>'
                f"<NextMarker /></EnumerationResults>"
            )
            return _xml(listing)
        if method == "PUT" and blob:
            stored[blob] = body
            return 201, {"ETag": _etag(body), "Last-Modified": formatdate(usegmt=True)}, b""
        if blob not in stored:
            return _azure_error(404, "BlobNotFound")
        if method == "GET":
            return _azure_ranged(stored[blob], headers.get("x-ms-range") or headers.get("range"))
        if method == "DELETE":
            del stored[blob]
            return 202, {}, b""
    return _azure_error(405, "UnsupportedHttpVerb")


def _gcs_json(payload: dict, status: int = 200) -> Answer:
    return status, {"Content-Type": "application/json"}, json.dumps(payload).encode()


def _gcs_error(status: int, message: str) -> Answer:
    return _gcs_json({"error": {"code": status, "message": message}}, status)


def _gcs_resource(bucket: str, name: str, data: bytes) -> dict:
    return {
        "kind": "storage#object",
        "id": f"{bucket}/{name}/1",
        "name": name,
        "bucket": bucket,
        "generation": "1",
        "metageneration": "1",
        "size": str(len(data)),
        "md5Hash": base64.b64encode(hashlib.md5(data).digest()).decode(),
    }


def _multipart_upload(headers: dict, body: bytes) -> tuple[str, bytes]:
    """The object name from the metadata part and the bytes of the media part."""
    envelope = f"Content-Type: {headers['content-type']}\r\n\r\n".encode() + body
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(envelope)
    metadata_part, media_part = message.iter_parts()
    return json.loads(metadata_part.get_content())["name"], media_part.get_payload(decode=True)


def _gcs_answer(fake: FakeGcs, method: str, path: str, query: dict, headers: dict, body: bytes):
    if fake.token_endpoint and path == "/token" and method == "POST":
        form = dict(urllib.parse.parse_qsl(body.decode()))
        with fake.lock:
            fake.assertions.append(form.get("assertion", ""))
        token = {"access_token": GCS_ACCESS_TOKEN, "expires_in": 3600, "token_type": "Bearer"}
        return _gcs_json(token)
    if fake.token_endpoint and headers.get("authorization") != f"Bearer {GCS_ACCESS_TOKEN}":
        return _gcs_error(401, "Invalid Credentials")
    route, _, rest = urllib.parse.unquote(path).lstrip("/").partition("storage/v1/b/")
    bucket, _, name = rest.partition("/o")
    name = name.removeprefix("/")
    with fake.lock:
        stored = fake.objects.setdefault(bucket, {})
        if route == "upload/" and method == "POST" and query.get("uploadType") == "multipart":
            name, data = _multipart_upload(headers, body)
            stored[name] = data
            return _gcs_json(_gcs_resource(bucket, name, data))
        if route == "" and not name and method == "GET":
            items = [_gcs_resource(bucket, key, data) for key, data in sorted(stored.items())]
            return _gcs_json({"kind": "storage#objects", "items": items})
        if name not in stored:
            return _gcs_error(404, "No such object")
        if route == "download/" and method == "GET":
            return 200, {"Content-Type": "application/octet-stream"}, stored[name]
        if route == "" and method == "GET":
            return _gcs_json(_gcs_resource(bucket, name, stored[name]))
        if route == "" and method == "DELETE":
            del stored[name]
            return 204, {}, b""
    return _gcs_error(405, "Method not allowed")


def self_signed_certificate(directory: Path) -> tuple[Path, Path]:
    """A certificate for 127.0.0.1 that is its own authority, and its key."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    certificate_path, key_path = directory / "certificate.pem", directory / "key.pem"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return certificate_path, key_path


def _serve(fake: _Store, answer, anonymous: bool) -> tuple[ThreadingHTTPServer, str]:
    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _serve(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            with fake.lock:
                fake.requests.append((self.command, self.path))
                fake.authorizations.append(self.headers.get("Authorization", ""))
            path, _, raw_query = self.path.partition("?")
            query = dict(urllib.parse.parse_qsl(raw_query, keep_blank_values=True))
            headers = {name.lower(): value for name, value in self.headers.items()}
            if "authorization" not in headers and not anonymous:
                status, answer_headers, answer_body = 403, {}, b"AuthenticationFailed"
            else:
                status, answer_headers, answer_body = answer(
                    self.command, path, query, headers, body
                )
            self.send_response(status)
            for name, value in answer_headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(answer_body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(answer_body)

        do_GET = do_PUT = do_POST = do_DELETE = do_HEAD = _serve

    server = ThreadingHTTPServer(("127.0.0.1", 0), RequestHandler)
    return server, f"http://127.0.0.1:{server.server_port}"


@contextlib.contextmanager
def _running(fake: _Store, answer, tls: bool = False, anonymous: bool = False) -> Iterator[None]:
    server, fake.base_url = _serve(fake, answer, anonymous)
    with tempfile.TemporaryDirectory(prefix="owui-object-storage-") as directory:
        if tls:
            certificate_path, key_path = self_signed_certificate(Path(directory))
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certificate_path, key_path)
            server.socket = context.wrap_socket(server.socket, server_side=True)
            fake.base_url = fake.base_url.replace("http://", "https://")
            fake.ca_bundle = str(certificate_path)
        threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
        try:
            yield
        finally:
            server.shutdown()
            server.server_close()


@contextlib.contextmanager
def serving_s3() -> Iterator[FakeS3]:
    fake = FakeS3()

    def answer(method, path, query, _headers, body):
        return _s3_answer(fake, method, path, query, body)

    with _running(fake, answer):
        yield fake


@contextlib.contextmanager
def serving_azure_blob(tls: bool = False) -> Iterator[FakeAzureBlob]:
    fake = FakeAzureBlob()

    def answer(method, path, query, headers, body):
        return _azure_answer(fake, method, path, query, headers, body)

    with _running(fake, answer, tls):
        yield fake


@contextlib.contextmanager
def serving_gcs(token_endpoint: bool = False) -> Iterator[FakeGcs]:
    fake = FakeGcs(token_endpoint=token_endpoint)

    def answer(method, path, query, headers, body):
        return _gcs_answer(fake, method, path, query, headers, body)

    with _running(fake, answer, anonymous=True):
        yield fake
