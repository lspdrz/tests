"""Dependency smoke: uploads kept in S3 through boto3 and in Azure Blob Storage through its SDK.

With `STORAGE_PROVIDER=s3` every upload is written to the bucket by boto3's `upload_file`, tagged
with its owner when `S3_ENABLE_TAGGING` is on, read back with `download_fileobj` whenever the file
is served or processed, removed with `delete_object`, and the admin's "delete all files" lists the
bucket and deletes what lies under `S3_KEY_PREFIX`. With `STORAGE_PROVIDER=azure` and an account
key, azure-storage-blob does the same with `upload_blob`, `download_blob`, `delete_blob` and
`list_blobs`. Without the key the blob client signs in with azure-identity's
`DefaultAzureCredential`, here through an App Service managed identity played by a `listener`;
the SDK only sends such a token over TLS, so that container is served with a certificate of its
own. Both services are local fakes (`harness.object_storage`) that keep the objects, so a test
reads what reached the store and changes it behind the instance's back to show a download comes
from there.

Discriminates: passes on dev ef67cc3fa; in a backend copy whose S3 provider skips `upload_file`,
downloads the key into nothing, skips `delete_object` or deletes every key of the bucket, the
upload, download, delete and delete-all S3 tests fail in turn, and the same four edits to the
Azure provider (`upload_blob`, `download_blob`, `delete_blob`, `list_blobs`) fail the Azure four;
building the keyless client with an anonymous credential fails the managed identity test.
"""

from __future__ import annotations

import httpx
import pytest

from harness.actors import Actor, admin_of, create_user
from harness.instance import LaunchedInstance
from harness.listener import json_answer, listening
from harness.object_storage import azure_blob_env, s3_env, serving_azure_blob, serving_s3

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

BUCKET = "owui-uploads"
KEY_PREFIX = "open-webui"
CONTAINER = "owui-uploads"
TEXT = "The harbour lighthouse keeps its logbook in the cloud."
STORAGE_TOKEN = "managed-identity-storage-token"


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
