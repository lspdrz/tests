"""Journey: who is in a channel over the API, who may change that, and the direct message list.

The owner of a group channel adds a member, who then reads the channel and lists it, and removes
one, who then loses both. A plain member cannot add or remove anyone and changes nothing. An
admin who is not in the channel may manage its members. A direct message is listed for both
people with the other person's account, counts what the other wrote as unread for the reader only
and refuses a third person.

Discriminates: in a backend copy, skipping the owner check in `add_members_by_id` turns the plain
member add row red (HTTP 200 and a new member), leaving the membership row in place in
`remove_members_by_id` turns the removal test red (the person still reads the channel), counting
no unread messages in `get_channels` turns the unread test red, and dropping the membership check
from the `dm` branch of `get_channel_by_id` or of `get_channel_messages` turns the third person
test red.
"""

from __future__ import annotations

import pytest

from harness.channel_quotes import enable_channels, group_channel, post_message

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

REFUSED = 403


@pytest.fixture
def channels_on(admin, preserve):
    preserve("admin_config")
    enable_channels(admin)


def _channel_path(channel_id: str, tail: str = "") -> str:
    return f"/api/v1/channels/{channel_id}{tail}"


def _status_of_reading(actor, channel_id: str) -> int:
    with actor.client() as client:
        return client.get(_channel_path(channel_id, "/messages")).status_code


def _listed_ids(actor) -> set[str]:
    with actor.client() as client:
        listed = client.get("/api/v1/channels/")
    listed.raise_for_status()
    return {channel["id"] for channel in listed.json()}


def _member_ids(actor, channel_id: str) -> set[str]:
    with actor.client() as client:
        listed = client.get(_channel_path(channel_id, "/members"))
    listed.raise_for_status()
    return {member["id"] for member in listed.json()["users"]}


def _change_members(actor, channel_id: str, action: str, user_ids: list[str]):
    with actor.client() as client:
        return client.post(
            _channel_path(channel_id, f"/update/members/{action}"), json={"user_ids": user_ids}
        )


def _direct_message(actor, other) -> str:
    with actor.client() as client:
        opened = client.get(f"/api/v1/channels/users/{other.id}")
    opened.raise_for_status()
    return opened.json()["id"]


def test_an_added_member_reads_and_lists_the_channel(channels_on, make_user):
    owner, newcomer = make_user(), make_user()
    channel_id = group_channel(owner)
    post_message(owner, channel_id, "before you arrived")
    assert _status_of_reading(newcomer, channel_id) == REFUSED
    assert channel_id not in _listed_ids(newcomer)

    added = _change_members(owner, channel_id, "add", [newcomer.id])

    assert added.status_code == 200, added.text
    assert _status_of_reading(newcomer, channel_id) == 200
    assert channel_id in _listed_ids(newcomer)
    assert newcomer.id in _member_ids(owner, channel_id)


def test_a_removed_member_no_longer_reads_or_lists_the_channel(channels_on, make_user):
    owner, leaving, staying = make_user(), make_user(), make_user()
    channel_id = group_channel(owner, leaving, staying)
    assert _status_of_reading(leaving, channel_id) == 200

    removed = _change_members(owner, channel_id, "remove", [leaving.id])

    assert removed.status_code == 200, removed.text
    assert _status_of_reading(leaving, channel_id) == REFUSED
    assert channel_id not in _listed_ids(leaving)
    assert _status_of_reading(staying, channel_id) == 200
    assert _member_ids(owner, channel_id) == {owner.id, staying.id}


@pytest.mark.parametrize("action", ["add", "remove"])
def test_a_plain_member_cannot_change_the_members(action, channels_on, make_user):
    owner, member, other = make_user(), make_user(), make_user()
    channel_id = group_channel(owner, member, other)
    before = _member_ids(owner, channel_id)
    target = other if action == "remove" else make_user()

    answered = _change_members(member, channel_id, action, [target.id])

    assert answered.status_code == REFUSED, answered.text
    assert _member_ids(owner, channel_id) == before


def test_an_admin_outside_the_channel_can_change_its_members(channels_on, admin, make_user):
    owner, newcomer = make_user(), make_user()
    channel_id = group_channel(owner)

    added = _change_members(admin, channel_id, "add", [newcomer.id])

    assert added.status_code == 200, added.text
    assert _status_of_reading(newcomer, channel_id) == 200


def test_a_direct_message_lists_for_both_people_and_counts_unread_for_the_reader(
    channels_on, make_user
):
    sender, reader = make_user(), make_user()
    channel_id = _direct_message(sender, reader)
    post_message(sender, channel_id, "are you coming?")
    post_message(sender, channel_id, "we leave at nine")

    listings = {}
    for actor in (sender, reader):
        with actor.client() as client:
            listings[actor.id] = {c["id"]: c for c in client.get("/api/v1/channels/").json()}

    assert listings[reader.id][channel_id]["unread_count"] == 2
    assert listings[sender.id][channel_id]["unread_count"] == 0
    for listing in listings.values():
        assert {user["id"] for user in listing[channel_id]["users"]} == {sender.id, reader.id}


def test_a_third_person_cannot_open_a_direct_message(channels_on, make_user):
    sender, reader, outsider = make_user(), make_user(), make_user()
    channel_id = _direct_message(sender, reader)
    post_message(sender, channel_id, "for your eyes only")

    with outsider.client() as client:
        opened = client.get(_channel_path(channel_id))

    assert opened.status_code == REFUSED, opened.text
    assert _status_of_reading(outsider, channel_id) == REFUSED
    assert _member_ids(sender, channel_id) == {sender.id, reader.id}
    assert channel_id not in _listed_ids(outsider)
