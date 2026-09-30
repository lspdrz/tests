"""Regression: the person who started a direct message could add or remove people over the API.

Issue #31570, fix `3d70d43a1` (#31575): the app offers member changes only in group channels,
but the members routes let the owner of a direct message (the person who opened it) and an admin
change its members too. Someone added this way read the whole earlier conversation, and since the
pair no longer matched the conversation, the next time either of the two opened it a second,
empty direct message was started. Changing the members of a direct message now answers 403 and
leaves it as it was. Changing the members of a group channel still works.

Discriminates: passes on dev a5bc78300; with 3d70d43a1 reverted in a backend copy every direct
message case answers 200 (the added person reads the history, the removed one loses it, and
opening the conversation again starts a new one). The group channel cases pass on both.
"""

from __future__ import annotations

import pytest

from harness.channel_quotes import enable_channels, group_channel, post_message

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

REFUSED = 403
HISTORY = "the launch moves to Friday"


@pytest.fixture
def channels_on(admin, preserve):
    preserve("admin_config")
    enable_channels(admin)


def _change_members(actor, channel_id: str, action: str, user_ids: list[str]):
    with actor.client() as client:
        return client.post(
            f"/api/v1/channels/{channel_id}/update/members/{action}", json={"user_ids": user_ids}
        )


def _open_direct_message(actor, other) -> str:
    with actor.client() as client:
        opened = client.get(f"/api/v1/channels/users/{other.id}")
    assert opened.status_code == 200, opened.text
    return opened.json()["id"]


def _member_ids(actor, channel_id: str) -> set[str]:
    with actor.client() as client:
        listed = client.get(f"/api/v1/channels/{channel_id}/members")
    assert listed.status_code == 200, listed.text
    return {member["id"] for member in listed.json()["users"]}


def _read_messages(actor, channel_id: str):
    with actor.client() as client:
        return client.get(f"/api/v1/channels/{channel_id}/messages")


@pytest.mark.parametrize("changed_by", ["starter", "admin"])
def test_nobody_can_be_added_to_a_direct_message(changed_by, channels_on, admin, make_user):
    starter, partner, outsider = make_user(), make_user(), make_user()
    channel_id = _open_direct_message(starter, partner)
    post_message(starter, channel_id, HISTORY)
    actor = admin if changed_by == "admin" else starter

    added = _change_members(actor, channel_id, "add", [outsider.id])

    assert added.status_code == REFUSED, (
        f"adding a third person to a direct message answered {added.status_code} (#31570)"
    )
    assert _member_ids(starter, channel_id) == {starter.id, partner.id}
    assert _read_messages(outsider, channel_id).status_code == REFUSED
    assert _open_direct_message(partner, starter) == channel_id, (
        "opening the conversation again started a second direct message"
    )


@pytest.mark.parametrize("changed_by", ["starter", "admin"])
def test_nobody_can_be_removed_from_a_direct_message(changed_by, channels_on, admin, make_user):
    starter, partner = make_user(), make_user()
    channel_id = _open_direct_message(starter, partner)
    post_message(starter, channel_id, HISTORY)
    actor = admin if changed_by == "admin" else starter

    removed = _change_members(actor, channel_id, "remove", [partner.id])

    assert removed.status_code == REFUSED, (
        f"removing a person from a direct message answered {removed.status_code} (#31570)"
    )
    assert _member_ids(starter, channel_id) == {starter.id, partner.id}
    read = _read_messages(partner, channel_id)
    assert read.status_code == 200, read.text
    assert HISTORY in read.text
    assert _open_direct_message(starter, partner) == channel_id


def test_a_group_channels_owner_still_adds_and_removes_members(channels_on, make_user):
    owner, newcomer, leaving = make_user(), make_user(), make_user()
    channel_id = group_channel(owner, leaving)
    post_message(owner, channel_id, HISTORY)

    added = _change_members(owner, channel_id, "add", [newcomer.id])
    removed = _change_members(owner, channel_id, "remove", [leaving.id])

    assert added.status_code == 200, added.text
    assert removed.status_code == 200, removed.text
    assert _member_ids(owner, channel_id) == {owner.id, newcomer.id}
    assert HISTORY in _read_messages(newcomer, channel_id).text
    assert _read_messages(leaving, channel_id).status_code == REFUSED
