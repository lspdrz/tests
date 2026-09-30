"""Regression: SCIM group responses listed every member with `"$ref": null`.

Issue open-webui/open-webui#31525, fix PR open-webui/open-webui#31529. A member of a SCIM group
must carry `$ref`, the URL of its user resource, so a directory can follow it to the person. The
member model was filled by field name while its `$ref` is an alias, so the value was dropped.
Every route that returns a group is read here (get one, list, create, replace and patch), and
each `$ref` is followed over SCIM to the user it names.

Discriminates: passes on dev a5bc78300; fails with a29c969fc reverted (the member model no longer
accepts its field name, so every member of every route comes back with `$ref` null).
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.scim import PATCH_SCHEMA, SCIM_ENV, provision, scim_client

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

GROUP_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:Group"


@pytest.fixture(scope="module")
def directory(instance_with):
    with scim_client(instance_with(SCIM_ENV)) as client:
        yield client


def _members(*people: dict) -> list[dict]:
    return [{"value": person["id"]} for person in people]


def _create_group(directory, *people: dict) -> dict:
    created = directory.post(
        "/Groups",
        json={
            "schemas": [GROUP_SCHEMA],
            "displayName": f"ref-team-{uuid.uuid4().hex[:8]}",
            "members": _members(*people),
        },
    )
    assert created.status_code == 201, created.text
    return created.json()


def _patch_group(directory, group_id: str, *operations: dict) -> httpx.Response:
    return directory.patch(
        f"/Groups/{group_id}", json={"schemas": [PATCH_SCHEMA], "Operations": list(operations)}
    )


def _assert_members_reference_their_users(directory, group: dict, *people: dict) -> None:
    by_id = {member["value"]: member for member in group["members"]}
    assert set(by_id) == {person["id"] for person in people}, group
    for person in people:
        ref = by_id[person["id"]].get("$ref")
        assert ref, f"member {person['id']} has no $ref: {group}"
        assert ref.endswith(f"/Users/{person['id']}"), ref
        followed = directory.get(ref)
        assert followed.status_code == 200, followed.text
        assert followed.json()["userName"] == person["userName"]


def test_a_created_group_answers_its_members_with_refs(directory):
    alice, bob = provision(directory), provision(directory)

    group = _create_group(directory, alice, bob)

    _assert_members_reference_their_users(directory, group, alice, bob)


def test_a_fetched_group_answers_its_members_with_refs(directory):
    alice, bob = provision(directory), provision(directory)
    group = _create_group(directory, alice, bob)

    fetched = directory.get(f"/Groups/{group['id']}")

    assert fetched.status_code == 200, fetched.text
    _assert_members_reference_their_users(directory, fetched.json(), alice, bob)


def test_the_group_list_answers_members_with_refs(directory):
    alice, bob = provision(directory), provision(directory)
    group = _create_group(directory, alice, bob)

    listed = directory.get("/Groups", params={"count": 100})

    assert listed.status_code == 200, listed.text
    ours = [entry for entry in listed.json()["Resources"] if entry["id"] == group["id"]]
    assert len(ours) == 1, listed.text
    _assert_members_reference_their_users(directory, ours[0], alice, bob)


def test_a_replaced_group_answers_its_members_with_refs(directory):
    alice, bob, carol = provision(directory), provision(directory), provision(directory)
    group = _create_group(directory, alice, bob)

    replaced = directory.put(
        f"/Groups/{group['id']}",
        json={
            "schemas": [GROUP_SCHEMA],
            "displayName": group["displayName"],
            "members": _members(carol),
        },
    )

    assert replaced.status_code == 200, replaced.text
    _assert_members_reference_their_users(directory, replaced.json(), carol)


def test_a_patched_group_answers_its_members_with_refs(directory):
    alice, bob = provision(directory), provision(directory)
    group = _create_group(directory, alice)

    patched = _patch_group(
        directory, group["id"], {"op": "add", "path": "members", "value": _members(bob)}
    )

    assert patched.status_code == 200, patched.text
    _assert_members_reference_their_users(directory, patched.json(), alice, bob)


def test_a_member_keeps_its_id_and_type_next_to_the_ref(directory):
    alice = provision(directory)

    group = _create_group(directory, alice)

    (member,) = group["members"]
    assert (member["value"], member["type"]) == (alice["id"], "User")
