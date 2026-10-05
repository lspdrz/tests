"""Dependency smoke: how request bodies are read and checked, through the routes that take them.

python-multipart parses every upload: the file itself, however its bytes look and whatever its
name is written in, and the form field beside it. A body it cannot parse is refused and the
server carries on. An urlencoded form (the provider's back-channel logout) goes through it too,
which integration/auth/test_sso_account_sync.py drives.

pydantic checks every JSON body and query parameter and shapes every answer. A body missing
fields is refused with one entry per field, a value is coerced where it can be (digits, a "yes")
and refused where it cannot, a validator refuses a dangerous profile image, the default avatar
fills in an empty one, and a date of birth is parsed from ISO text and written back as it.
Settings kept per user keep keys the model does not name, a folder refuses them, and a SCIM
group writes its members with SCIM's `$ref` name. `HttpUrl` refuses what is not a URL and hands
the importer the URL it took. The Ollama proxy forwards a form without its unset fields and with
the ones it does not name (integration/models/test_ollama_model_management.py), and a workspace
tool's spec comes from a model pydantic builds from the method
(integration/deps/test_tool_specs.py).

A SCIM group member's `$ref` was always null: the member was built with `ref=`, which pydantic
ignores on a field that only takes its alias. PR #31529 (open-webui/open-webui#31525) fixed it, and
that test fails on dev 176d31d1d.

Since 24e30d1cb the sign-in routes answer a refused body with one generic message, so the
per-field entries are read from creating a knowledge base and the profile image's reason from the
admin's edit of the account.

Discriminates: on dev ef67cc3fa, one backend copy with a default for the sign-in password, a
`StrictBool` expanded state, `FolderForm` ignoring extra fields, `UserSettings` without
`extra="allow"`, `LoadUrlForm.url` typed `str`, the SCIM member field without its alias, the
upload's filename read as latin-1 and its metadata field dropped fails exactly those tests;
another without the profile image validator, without `_ensure_profile_image` and with the date
of birth typed `str` fails the three profile tests, and with `model_dump_json()` in place of
`exclude_none=True` the Ollama create test; a chat list whose page is typed `str` fails the
numeric parameter test. On dev b859124f9 a default for a knowledge base's description fails the
missing fields test. The byte-for-byte and unparseable-body tests pin python-multipart's own
parsing, which no backend edit reaches. Twin of unit/deps/test_pydantic.py and the parser half
of unit/deps/test_python_multipart.py.
"""

from __future__ import annotations

import json
import os
import uuid

import httpx
import pytest

from harness.scim import SCIM_ENV, provision, scim_client

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source]

GROUP_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:Group"


# ---------------------------------------------------------------- uploads


def _upload(client: httpx.Client, filename: str, content: bytes, **form) -> dict:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "false"},
        files={"file": (filename, content, "application/octet-stream")},
        data=form,
    )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()


def test_an_upload_arrives_byte_for_byte_across_many_parts(make_user):
    # random bytes, CRLFs and the start of a boundary, over several of the parser's reads
    content = os.urandom(200_000) + b"\r\n--" + b"\r\n" * 1000 + os.urandom(200_000)
    with make_user().client() as client:
        stored = _upload(client, "cargo.bin", content)
        downloaded = client.get(f"/api/v1/files/{stored['id']}/content")

    assert downloaded.status_code == 200, downloaded.text
    assert downloaded.content == content
    assert stored["meta"]["size"] == len(content)


def test_a_filename_outside_ascii_is_kept(make_user):
    filename = "Hafenbücher 灯台 Ωμέγα.txt"
    with make_user().client() as client:
        stored = _upload(client, filename, b"logbook")

    assert stored["filename"] == filename
    assert stored["meta"]["name"] == filename


def test_the_form_field_beside_the_file_is_kept_with_it(make_user):
    metadata = {"harbour": "north", "berth": 7}
    with make_user().client() as client:
        stored = _upload(client, "cargo.txt", b"manifest", metadata=json.dumps(metadata))
        read = client.get(f"/api/v1/files/{stored['id']}")

    assert read.json()["meta"]["data"] == metadata


def test_a_body_that_is_not_multipart_is_refused(make_user):
    headers = {"Content-Type": "multipart/form-data; boundary=harbour"}
    with make_user().client() as client:
        refused = client.post(
            "/api/v1/files/", content=b"--harbour\r\nno headers, no end", headers=headers
        )
        still_up = client.get("/api/v1/files/")

    assert refused.status_code in (400, 422), refused.text
    assert still_up.status_code == 200


# ---------------------------------------------------------------- JSON bodies


def test_a_body_missing_fields_is_refused_with_one_entry_per_field(make_user):
    with make_user().client() as client:
        refused = client.post("/api/v1/knowledge/create", json={})

    assert refused.status_code == 422, refused.text
    errors = refused.json()["detail"]
    assert sorted(tuple(error["loc"]) for error in errors) == [
        ("body", "description"),
        ("body", "name"),
    ]
    assert all(error["msg"] and error["type"] == "missing" for error in errors)


def _folder(client: httpx.Client, **extra) -> httpx.Response:
    return client.post("/api/v1/folders/", json={"name": f"berths {uuid.uuid4().hex[:6]}", **extra})


@pytest.mark.parametrize(
    ("sent", "stored"),
    [
        pytest.param("yes", True, id="yes"),
        pytest.param("off", False, id="off"),
        pytest.param(1, True, id="one"),
        pytest.param("maybe", None, id="refused"),
    ],
)
def test_a_true_or_false_field_takes_what_reads_as_one(make_user, sent, stored):
    with make_user().client() as client:
        folder = _folder(client)
        assert folder.status_code == 200, folder.text
        folder_id = folder.json()["id"]
        updated = client.post(
            f"/api/v1/folders/{folder_id}/update/expanded", json={"is_expanded": sent}
        )
        read = client.get(f"/api/v1/folders/{folder_id}")

    if stored is None:
        assert updated.status_code == 422, updated.text
        assert read.json()["is_expanded"] is False
    else:
        assert updated.status_code == 200, updated.text
        assert read.json()["is_expanded"] is stored


def test_a_numeric_query_parameter_takes_digits_and_refuses_words(make_user):
    with make_user().client() as client:
        taken = client.get("/api/v1/chats/list", params={"page": "1"})
        refused = client.get("/api/v1/chats/list", params={"page": "first"})

    assert taken.status_code == 200, taken.text
    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"][0]["loc"] == ["query", "page"]


def test_a_folder_refuses_a_field_it_does_not_name(make_user):
    with make_user().client() as client:
        refused = _folder(client, colour="teal")

    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"][0]["loc"][-1] == "colour"


def test_user_settings_keep_the_keys_the_model_does_not_name(make_user):
    settings = {"ui": {"theme": "dark"}, "harbour": {"berth": 7}}
    with make_user().client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json=settings)
        read = client.get("/api/v1/users/user/settings")

    assert saved.status_code == 200, saved.text
    assert read.json()["harbour"] == {"berth": 7}
    assert read.json()["ui"]["theme"] == "dark"


def _update_profile(client: httpx.Client, **fields) -> httpx.Response:
    profile = {"name": "Harbour Keeper", "profile_image_url": "", **fields}
    return client.post("/api/v1/auths/update/profile", json=profile)


def test_a_date_of_birth_is_read_and_written_as_an_iso_date(make_user):
    with make_user().client() as client:
        saved = _update_profile(client, date_of_birth="1990-05-17")
        refused = _update_profile(client, date_of_birth="1990-02-30")
        session = client.get("/api/v1/auths/")

    assert saved.status_code == 200, saved.text
    assert refused.status_code == 422, refused.text
    assert session.json()["date_of_birth"] == "1990-05-17"


def test_a_dangerous_profile_image_is_refused(admin, make_user):
    account = make_user()
    with account.client() as client:
        before = client.get("/api/v1/auths/").json()["profile_image_url"]
        refused = _update_profile(client, profile_image_url="javascript:alert(1)")
        after = client.get("/api/v1/auths/").json()["profile_image_url"]
    with admin.client() as client:
        refused_for_the_admin = client.post(
            f"/api/v1/users/{account.id}/update", json={"profile_image_url": "javascript:alert(1)"}
        )

    assert refused.status_code == 422, refused.text
    assert after == before
    # the sign-in routes answer one generic message since 24e30d1cb, so the reason shows here
    assert refused_for_the_admin.status_code == 422, refused_for_the_admin.text
    assert "image" in json.dumps(refused_for_the_admin.json()["detail"]).lower()


def test_an_empty_profile_image_becomes_the_default_avatar(make_user):
    account = make_user()
    with account.client() as client:
        saved = _update_profile(client, profile_image_url="")
        session = client.get("/api/v1/auths/")

    assert saved.status_code == 200, saved.text
    assert session.json()["profile_image_url"] == f"/api/v1/users/{account.id}/profile/image"


@pytest.fixture(scope="module")
def scim_instance(instance_with):
    return instance_with(SCIM_ENV)


def _scim_group_member(scim_instance) -> tuple[dict, dict]:
    """(the provisioned account, how the group it was put in lists it)."""
    with scim_client(scim_instance) as client:
        member = provision(client)
        created = client.post(
            "/Groups",
            json={
                "schemas": [GROUP_SCHEMA],
                "displayName": f"Harbour crew {uuid.uuid4().hex[:6]}",
                "members": [{"value": member["id"]}],
            },
        )
        assert created.status_code == 201, created.text
        read = client.get(f"/Groups/{created.json()['id']}")
    [listed] = read.json()["members"]
    return member, listed


def test_a_scim_group_member_is_written_with_the_attribute_names_of_scim(scim_instance):
    member, listed = _scim_group_member(scim_instance)

    assert listed["value"] == member["id"]
    assert "$ref" in listed and "ref" not in listed, listed


def test_a_scim_group_member_links_to_its_user(scim_instance):
    # red on dev fccd75568: `ref=` is ignored on an alias-only field, so `$ref` is always null
    member, listed = _scim_group_member(scim_instance)

    assert (listed.get("$ref") or "").endswith(f"/api/v1/scim/v2/Users/{member['id']}"), (
        f"the member's $ref does not name its user: {listed}"
    )


def test_an_import_url_must_be_a_url_and_is_fetched_as_given(admin, listener):
    listener.route("GET", "/tools/berth_planner.py", (200, {}, b"class Tools:\n    pass\n"))
    with admin.client() as client:
        refused = client.post("/api/v1/tools/load/url", json={"url": "not a url"})
        loaded = client.post(
            "/api/v1/tools/load/url", json={"url": f"{listener.base_url}/tools/berth_planner.py"}
        )

    assert refused.status_code == 422, refused.text
    assert loaded.status_code == 200, loaded.text
    assert loaded.json() == {"name": "berth_planner", "content": "class Tools:\n    pass\n"}
    assert listener.requests_to("/tools/berth_planner.py")
