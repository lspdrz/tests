"""Regressions for three v0.11.1 repairs of data persisted in the wrong shape.

1. Default model settings (commit 5c05608e3a). Instances whose `ui.default_models` or
   `ui.default_pinned_models` row was stored as a list instead of the comma-separated string the
   app reads had their defaults silently ignored. The boot's config repair now joins such a row.
   The instance here boots on a data directory whose legacy `config.json` holds the rows: its
   import writes each key verbatim before the repair runs, so the boot meets the old shapes
   (`harness.prepared_data.LEGACY_CONFIG_ROWS`).
2. Connection tags saved as plain strings (commit 8be4c5fa6a, issue #28749). `/api/models` did
   `[tag.get('name') for tag in ...]` inside a bare `try/except`, so a connection whose tags or
   listed `info.meta.tags` were strings lost every tag, and a listed model with `info: null`
   took the whole listing down. The fix normalises both.
3. SSO links stored double-encoded (commit bd8378f643, PR #28107, issue #28101). Releases before
   0.11.1 could hold `user.oauth` as a JSON string of the object, so the provider-and-sub lookup
   missed and those accounts could not sign in through SSO. Migration `6d09d1bf1f23` decodes such
   rows. The server boots on the v0.10.2 data sets of `upgrade_data/` with one account's link
   written that way, one as the object and one without a link.

Twin of unit/migrations/test_startup_repairs.py.
Discriminates: passes on dev ef67cc3fa; with the default-model pass of
`Config.repair_config_rows` removed from a copy the list comes back as a list, with the
`main.py` half of 8be4c5fa6a reverted the string tags come back empty and the `info: null`
model fails the listing, and with the upgrade of `6d09d1bf1f23` emptied the account with the
double-encoded link is refused at SSO sign-in and the admin's user list answers 500 on both
engines.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import httpx
import pytest

from harness.listener import json_answer
from harness.oidc_provider import session_user, shared_provider, sign_in, sso_env
from harness.prepared_data import (
    ReleaseData,
    RunningBackend,
    release_data,
    run_sql,
    serving,
    with_legacy_config,
)
from harness.second_provider import OPENAI_CONFIG, attach
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

TAGGED = {"id": "tagged-model", "info": {"meta": {"tags": ["alpha", "beta"]}}}
BARE = {"id": "bare-model", "info": None}
DICT_TAGGED = {
    "id": "dict-tagged-model",
    "info": {"meta": {"tags": [{"name": "delta"}], "profile_image_url": "data:image/png;base64,A"}},
}


@pytest.fixture(scope="module")
def repaired_config(instance_with, tmp_path_factory) -> dict:
    with with_legacy_config(instance_with, tmp_path_factory).client() as client:
        config = client.get("/api/config")
    config.raise_for_status()
    return config.json()


@pytest.mark.slow
def test_a_list_shaped_default_models_row_reads_back_as_the_string(repaired_config):
    assert repaired_config["default_models"] == f"{MOCK_MODEL_ID},second-model", (
        "a list-shaped default models row was not repaired, so the defaults stay ignored"
    )


@pytest.mark.slow
def test_a_string_shaped_default_models_row_is_left_as_it_is(repaired_config):
    assert repaired_config["default_pinned_models"] == f"{MOCK_MODEL_ID},second-model"


@pytest.mark.slow
def test_flattened_permission_rows_are_reassembled(repaired_config):
    permissions = repaired_config["permissions"]
    assert permissions["chat"]["controls"] is False, permissions
    assert permissions["workspace"]["models"] is True, permissions


@pytest.fixture
def connect(admin, listener, preserve):
    """`connect(tags)` adds one more connection with its tags saved as given; returns the
    admin's client."""
    preserve(OPENAI_CONFIG)
    clients = []

    def factory(tags: list) -> httpx.Client:
        client = admin.client()
        clients.append(client)
        attach(client, listener, TAGGED["id"], tags=tags)
        return client

    yield factory
    for client in clients:
        client.close()


def _listed(client, listener, *models: dict) -> dict[str, dict]:
    """`/api/models`, by id, once the connection lists `models`."""
    listener.route("GET", "/v1/models", json_answer({"object": "list", "data": list(models)}))
    wanted = {model["id"] for model in models}
    deadline = time.monotonic() + 10
    while True:
        listed = client.get("/api/models", params={"refresh": "true"})
        assert listed.status_code == 200, f"the model listing failed: {listed.text}"
        found = {model["id"]: model for model in listed.json()["data"]}
        if wanted <= found.keys() or time.monotonic() > deadline:
            return found
        time.sleep(0.2)


def _names(tags: list[dict]) -> list[str]:
    return sorted(tag["name"] for tag in tags)


def test_plain_string_tags_come_back_as_tags(connect, listener):
    tagged = _listed(connect(["gamma"]), listener, TAGGED)[TAGGED["id"]]

    assert _names(tagged["tags"]) == ["alpha", "beta", "gamma"], tagged["tags"]
    assert tagged["info"]["meta"]["tags"] == [{"name": "alpha"}, {"name": "beta"}], (
        "the listed tags the connection editor reads back were not normalised (#28749)"
    )


def test_a_model_listed_with_null_info_does_not_break_the_listing(connect, listener):
    listed = _listed(connect([]), listener, BARE)

    assert listed[BARE["id"]]["tags"] == []


def test_tag_objects_and_the_profile_image_strip_still_work(connect, listener):
    listed = _listed(connect([{"name": "gamma"}]), listener, DICT_TAGGED)[DICT_TAGGED["id"]]

    assert _names(listed["tags"]) == ["delta", "gamma"]
    assert "profile_image_url" not in listed["info"]["meta"]


UPGRADE_DATA = Path(__file__).parent / "upgrade_data"
# linked the way releases before 0.11.1 could leave it, and the way they meant to
DOUBLE_ENCODED = {"who": "alice", "sub": "alice-sso-sub"}
AS_OBJECT = {"who": "bob", "sub": "bob-sso-sub"}


def _link(sub: str) -> dict:
    return {"oidc": {"sub": sub}}


@pytest.fixture(scope="module")
def idp():
    return shared_provider()


@dataclass
class Repaired:
    backend: RunningBackend
    release: ReleaseData

    def account(self, who: str) -> dict:
        return self.release.manifest["accounts"][who]

    def admin_token(self) -> str:
        admin = self.account("admin")
        with self.backend.client() as client:
            signed_in = client.post(
                "/api/v1/auths/signin",
                json={"email": admin["email"], "password": admin["password"]},
            )
        signed_in.raise_for_status()
        return signed_in.json()["token"]


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("v0.10.2-sqlite", id="sqlite"),
        pytest.param("v0.10.2-postgres", id="postgres", marks=pytest.mark.requires_postgres),
    ],
)
def repaired_links(request, idp, tmp_path_factory) -> Iterator[Repaired]:
    """A v0.10.2 install with SSO links in both shapes, started on the checkout."""
    archive = UPGRADE_DATA / f"{request.param}.tar.gz"
    with contextlib.ExitStack() as stack:
        release = stack.enter_context(release_data(archive, tmp_path_factory.mktemp("links")))
        accounts = release.manifest["accounts"]
        links = [
            {
                "oauth": json.dumps(json.dumps(_link(DOUBLE_ENCODED["sub"]))),
                "account": accounts[DOUBLE_ENCODED["who"]]["id"],
            },
            {
                "oauth": json.dumps(_link(AS_OBJECT["sub"])),
                "account": accounts[AS_OBJECT["who"]]["id"],
            },
        ]
        run_sql(release.database_url, 'UPDATE "user" SET oauth = :oauth WHERE id = :account', links)
        settings = {**sso_env(idp), **release.settings}
        yield Repaired(stack.enter_context(serving(release.data_dir, settings)), release)


@pytest.mark.slow
@pytest.mark.parametrize("linked", [DOUBLE_ENCODED, AS_OBJECT], ids=["double-encoded", "object"])
def test_an_sso_link_from_before_the_repair_signs_its_owner_in(repaired_links, idp, linked):
    account = repaired_links.account(linked["who"])
    idp.sign_in_as(sub=linked["sub"], email=account["email"], name=account["name"])

    result = sign_in(repaired_links.backend)

    assert result.token, (
        f"{linked['who']}'s SSO sign-in was refused after the upgrade, so the stored link was not "
        f"repaired (#28101): {result.error}"
    )
    assert session_user(repaired_links.backend, result.token)["id"] == account["id"]


@pytest.mark.slow
def test_the_admin_sees_each_link_as_an_object_and_no_link_stays_none(repaired_links):
    backend = repaired_links.backend
    token = repaired_links.admin_token()
    with backend.client(token) as client:
        listed = client.get("/api/v1/users/", params={"query": "example.com"})
    assert listed.status_code == 200, f"the admin's user list failed: {listed.text}"
    by_email = {user["email"]: user["oauth"] for user in listed.json()["users"]}

    for linked in (DOUBLE_ENCODED, AS_OBJECT):
        email = repaired_links.account(linked["who"])["email"]
        assert by_email[email] == _link(linked["sub"]), by_email[email]
    assert by_email[repaired_links.account("carol")["email"]] is None
