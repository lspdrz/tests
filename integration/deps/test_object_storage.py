"""Dependency smoke: uploads kept in S3, Azure Blob Storage and Google Cloud Storage via their SDKs.

With `STORAGE_PROVIDER=s3` every upload is written to the bucket by boto3's `upload_file`, tagged
with its owner when `S3_ENABLE_TAGGING` is on, read back with `download_fileobj` whenever the file
is served or processed, removed with `delete_object`, and the admin's "delete all files" lists the
bucket and deletes what lies under `S3_KEY_PREFIX`. With `STORAGE_PROVIDER=azure` and an account
key, azure-storage-blob does the same with `upload_blob`, `download_blob`, `delete_blob` and
`list_blobs`. Without the key the blob client signs in with azure-identity's
`DefaultAzureCredential`, here through an App Service managed identity played by a `listener`;
the SDK only sends such a token over TLS, so that container is served with a certificate of its
own. With `STORAGE_PROVIDER=gcs` google-cloud-storage uploads with `upload_from_filename`, finds
the object with `get_blob` to `download_to_filename` it or `delete` it, and deletes every object
`list_blobs` names. Without `GOOGLE_APPLICATION_CREDENTIALS_JSON` it goes anonymous against the
emulator host; with it, `Client.from_service_account_info` signs a JWT with the account's key,
trades it at the account's token endpoint and sends the token it got. The services are local
fakes (`harness.object_storage`) that keep the objects, so a test reads what reached the store
and changes it behind the instance's back to show a download comes from there.

Discriminates: passes on dev ef67cc3fa; in a backend copy whose S3 provider skips `upload_file`,
downloads the key into nothing, skips `delete_object` or deletes every key of the bucket, the
upload, download, delete and delete-all S3 tests fail in turn, and the same four edits to the
Azure provider (`upload_blob`, `download_blob`, `delete_blob`, `list_blobs`) fail the Azure four
and to the GCS provider (`upload_from_filename`, `download_to_filename`, `blob.delete`,
`list_blobs`) the GCS four; building the keyless Azure client with an anonymous credential fails
the managed identity test, and a GCS client built with `storage.Client()` in place of
`from_service_account_info` fails the service account test.
"""

from __future__ import annotations

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from harness.actors import Actor, admin_of, create_user
from harness.instance import LaunchedInstance
from harness.listener import json_answer, listening
from harness.object_storage import (
    GCS_ACCESS_TOKEN,
    azure_blob_env,
    gcs_env,
    s3_env,
    serving_azure_blob,
    serving_gcs,
    serving_s3,
)

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

BUCKET = "owui-uploads"
KEY_PREFIX = "open-webui"
CONTAINER = "owui-uploads"
TEXT = "The harbour lighthouse keeps its logbook in the cloud."
STORAGE_TOKEN = "managed-identity-storage-token"
GCS_BUCKET = "owui-uploads"
SERVICE_ACCOUNT_EMAIL = "open-webui@harbour-project.iam.gserviceaccount.com"
# google-auth names Google's token endpoint as the audience whatever `token_uri` says
GOOGLE_TOKEN_AUDIENCE = "https://oauth2.googleapis.com/token"
SERVICE_ACCOUNT_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def s3():
    with serving_s3() as fake:
        yield fake


@pytest.fixture(scope="module")
def azure():
    with serving_azure_blob() as fake:
        yield fake


@pytest.fixture(scope="module")
def azure_over_tls():
    with serving_azure_blob(tls=True) as fake:
        yield fake


@pytest.fixture(scope="module")
def managed_identity():
    with listening() as endpoint:
        token = {
            "access_token": STORAGE_TOKEN,
            "expires_on": "4102444800",
            "resource": "https://storage.azure.com",
            "token_type": "Bearer",
        }
        endpoint.route("GET", "/msi/token", json_answer(token))
        yield endpoint


@pytest.fixture(scope="module")
def gcs():
    with serving_gcs() as fake:
        yield fake


@pytest.fixture(scope="module")
def gcs_with_token_endpoint():
    with serving_gcs(token_endpoint=True) as fake:
        yield fake


@pytest.fixture
def on_s3(instance_with, s3) -> LaunchedInstance:
    return instance_with(s3_env(s3, BUCKET, S3_KEY_PREFIX=KEY_PREFIX, S3_ENABLE_TAGGING="true"))


@pytest.fixture
def on_azure(instance_with, azure) -> LaunchedInstance:
    return instance_with(azure_blob_env(azure, CONTAINER))


@pytest.fixture
def on_azure_identity(instance_with, azure_over_tls, managed_identity) -> LaunchedInstance:
    return instance_with(
        {
            **azure_blob_env(azure_over_tls, CONTAINER, with_key=False),
            "IDENTITY_ENDPOINT": f"{managed_identity.base_url}/msi/token",
            "IDENTITY_HEADER": "managed-identity-secret",
        }
    )


@pytest.fixture
def on_gcs(instance_with, gcs) -> LaunchedInstance:
    return instance_with(gcs_env(gcs, GCS_BUCKET))


def _service_account(token_uri: str) -> dict:
    """A service account's JSON key, the way the Google Cloud console downloads it."""
    private_key = SERVICE_ACCOUNT_KEY.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return {
        "type": "service_account",
        "project_id": "harbour-project",
        "private_key_id": "harbour-key-1",
        "private_key": private_key.decode(),
        "client_email": SERVICE_ACCOUNT_EMAIL,
        "client_id": "104",
        "token_uri": token_uri,
    }


@pytest.fixture
def on_gcs_service_account(instance_with, gcs_with_token_endpoint) -> LaunchedInstance:
    fake = gcs_with_token_endpoint
    account = _service_account(f"{fake.base_url}/token")
    return instance_with(gcs_env(fake, GCS_BUCKET, credentials=account))


def _upload(account: Actor, filename: str, text: str = TEXT) -> str:
    """Upload and process a text file the way the chat input does; returns its id."""
    with account.client() as client:
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "true", "process_in_background": "false"},
            files={"file": (filename, text.encode(), "text/plain")},
        )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def _extracted_text(account: Actor, file_id: str) -> str:
    with account.client() as client:
        read = client.get(f"/api/v1/files/{file_id}/data/content")
    assert read.status_code == 200, read.text
    return read.json()["content"]


def _download(account: Actor, file_id: str) -> httpx.Response:
    with account.client() as client:
        return client.get(f"/api/v1/files/{file_id}/content")


def _delete(account: Actor, file_id: str) -> None:
    with account.client() as client:
        deleted = client.delete(f"/api/v1/files/{file_id}")
    assert deleted.status_code == 200, deleted.text


def _delete_every_file(instance: LaunchedInstance) -> None:
    with admin_of(instance).client() as client:
        deleted = client.delete("/api/v1/files/all")
    assert deleted.status_code == 200, deleted.text


# ---------------------------------------------------------------- S3 through boto3


def test_an_upload_is_stored_in_the_bucket_tagged_with_its_owner(on_s3, s3):
    owner = create_user(on_s3)
    file_id = _upload(owner, "logbook.txt")
    key = f"{KEY_PREFIX}/{file_id}_logbook.txt"

    assert key in s3.keys(BUCKET), s3.keys(BUCKET)
    assert s3.objects[BUCKET][key] == TEXT.encode()
    assert s3.tags[key]["OpenWebUI-User-Email"] == owner.email
    assert s3.tags[key]["OpenWebUI-File-Id"] == file_id
    assert _extracted_text(owner, file_id) == TEXT


def test_a_download_is_read_from_the_bucket(on_s3, s3):
    owner = create_user(on_s3)
    file_id = _upload(owner, "tides.txt")
    s3.objects[BUCKET][f"{KEY_PREFIX}/{file_id}_tides.txt"] = b"changed in the bucket"

    downloaded = _download(owner, file_id)

    assert downloaded.status_code == 200, downloaded.text
    assert downloaded.content == b"changed in the bucket"


def test_a_deleted_file_leaves_the_bucket(on_s3, s3):
    owner = create_user(on_s3)
    file_id = _upload(owner, "ferries.txt")
    key = f"{KEY_PREFIX}/{file_id}_ferries.txt"
    assert key in s3.keys(BUCKET)

    _delete(owner, file_id)

    assert key not in s3.keys(BUCKET)


def test_deleting_every_file_empties_the_prefix_and_keeps_the_rest(on_s3, s3):
    s3.objects.setdefault(BUCKET, {})["backups/keep.txt"] = b"not Open WebUI's"
    _upload(create_user(on_s3), "weather.txt")

    _delete_every_file(on_s3)

    assert s3.keys(BUCKET) == ["backups/keep.txt"]


# ---------------------------------------------------------------- Azure Blob Storage


def test_an_upload_is_stored_in_the_container(on_azure, azure):
    owner = create_user(on_azure)
    file_id = _upload(owner, "logbook.txt")
    blob = f"{file_id}_logbook.txt"

    assert blob in azure.keys(CONTAINER), azure.keys(CONTAINER)
    assert azure.objects[CONTAINER][blob] == TEXT.encode()
    assert _extracted_text(owner, file_id) == TEXT


def test_a_download_is_read_from_the_container(on_azure, azure):
    owner = create_user(on_azure)
    file_id = _upload(owner, "tides.txt")
    azure.objects[CONTAINER][f"{file_id}_tides.txt"] = b"changed in the container"

    downloaded = _download(owner, file_id)

    assert downloaded.status_code == 200, downloaded.text
    assert downloaded.content == b"changed in the container"


def test_a_deleted_file_leaves_the_container(on_azure, azure):
    owner = create_user(on_azure)
    file_id = _upload(owner, "ferries.txt")
    blob = f"{file_id}_ferries.txt"
    assert blob in azure.keys(CONTAINER)

    _delete(owner, file_id)

    assert blob not in azure.keys(CONTAINER)


def test_deleting_every_file_empties_the_container(on_azure, azure):
    _upload(create_user(on_azure), "weather.txt")
    assert azure.keys(CONTAINER)

    _delete_every_file(on_azure)

    assert azure.keys(CONTAINER) == []


def test_without_a_key_the_container_is_reached_with_the_managed_identity(
    on_azure_identity, azure_over_tls, managed_identity
):
    owner = create_user(on_azure_identity)
    file_id = _upload(owner, "logbook.txt")

    assert azure_over_tls.objects[CONTAINER][f"{file_id}_logbook.txt"] == TEXT.encode()
    assert _download(owner, file_id).content == TEXT.encode()
    assert set(azure_over_tls.authorizations) == {f"Bearer {STORAGE_TOKEN}"}
    asked = managed_identity.requests_to("/msi/token")[-1]
    assert "resource=https://storage.azure.com" in asked.path


# ---------------------------------------------------------------- Google Cloud Storage


def test_an_upload_is_stored_in_the_gcs_bucket(on_gcs, gcs):
    owner = create_user(on_gcs)
    file_id = _upload(owner, "logbook.txt")
    name = f"{file_id}_logbook.txt"

    assert name in gcs.keys(GCS_BUCKET), gcs.keys(GCS_BUCKET)
    assert gcs.objects[GCS_BUCKET][name] == TEXT.encode()
    assert _extracted_text(owner, file_id) == TEXT


def test_a_download_is_read_from_the_gcs_bucket(on_gcs, gcs):
    owner = create_user(on_gcs)
    file_id = _upload(owner, "tides.txt")
    gcs.objects[GCS_BUCKET][f"{file_id}_tides.txt"] = b"changed in the bucket"

    downloaded = _download(owner, file_id)

    assert downloaded.status_code == 200, downloaded.text
    assert downloaded.content == b"changed in the bucket"


def test_a_deleted_file_leaves_the_gcs_bucket(on_gcs, gcs):
    owner = create_user(on_gcs)
    file_id = _upload(owner, "ferries.txt")
    name = f"{file_id}_ferries.txt"
    assert name in gcs.keys(GCS_BUCKET)

    _delete(owner, file_id)

    assert name not in gcs.keys(GCS_BUCKET)


def test_deleting_every_file_empties_the_gcs_bucket(on_gcs, gcs):
    _upload(create_user(on_gcs), "weather.txt")
    assert gcs.keys(GCS_BUCKET)

    _delete_every_file(on_gcs)

    assert gcs.keys(GCS_BUCKET) == []


def test_with_a_service_account_the_bucket_is_reached_with_its_token(
    on_gcs_service_account, gcs_with_token_endpoint
):
    fake = gcs_with_token_endpoint
    owner = create_user(on_gcs_service_account)
    file_id = _upload(owner, "logbook.txt")

    assert fake.objects[GCS_BUCKET][f"{file_id}_logbook.txt"] == TEXT.encode()
    assert _download(owner, file_id).content == TEXT.encode()
    storage_calls = [
        authorization
        for (_, path), authorization in zip(fake.requests, fake.authorizations)
        if path != "/token"
    ]
    assert set(storage_calls) == {f"Bearer {GCS_ACCESS_TOKEN}"}
    # the assertion traded for the token is signed with the account's own key
    claims = jwt.decode(
        fake.assertions[-1],
        SERVICE_ACCOUNT_KEY.public_key(),
        algorithms=["RS256"],
        audience=GOOGLE_TOKEN_AUDIENCE,
    )
    assert claims["iss"] == SERVICE_ACCOUNT_EMAIL
    assert "devstorage" in claims["scope"]
