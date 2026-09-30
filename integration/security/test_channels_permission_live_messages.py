"""Regression: a user without the Channels permission still got a channel's messages live.

Fix `028dab8f1` (#31578): the socket joined an account to the rooms of its channels, where every
new message is pushed, whenever the account had the Channels permission by the built-in defaults.
The check left out the default user permissions the admin saves, so switching Channels off there
changed nothing for the socket: `user-join` and `join-channels` still joined every channel room,
and opening a direct message or creating a group channel joined the open tabs of each member
without any check. A member without the permission now joins none of those rooms. A member whose
group grants Channels and an admin still get every message.

Discriminates: passes on dev a5bc78300; with 028dab8f1 reverted in a backend copy each restricted
case receives the new message (the join events through the built-in defaults, a new direct
message or group channel through the unchecked room join). The allowed cases pass on both.
"""

from __future__ import annotations

import threading
import uuid

import pytest

from harness.access import make_group
from harness.channel_quotes import enable_channels, group_channel, post_message
from harness.socket_client import SocketSession, connected

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PERMISSIONS = "/api/v1/users/default/permissions"
CHANNELS_GRANTED = {"features": {"channels": True}}
ARRIVAL_TIMEOUT = 15.0
QUIET_PERIOD = 2.0


@pytest.fixture
def channels_for_admins_and_groups_only(admin, preserve):
    """Channels on, with the Channels permission off in the default user permissions."""
    preserve("admin_config")
    preserve("permissions")
    enable_channels(admin)
    with admin.client() as client:
        current = client.get(PERMISSIONS)
        assert current.status_code == 200, current.text
        permissions = current.json()
        permissions["features"] = {**permissions["features"], "channels": False}
        saved = client.post(PERMISSIONS, json=permissions)
    assert saved.status_code == 200, saved.text


def _arrivals(session: SocketSession, marker: str) -> threading.Event:
    """Set once a channel event carrying `marker` reaches the tab."""
    arrived = threading.Event()

    def record(*payload) -> None:
        if marker in str(payload):
            arrived.set()

    session.client.on("events:channel", record)
    return arrived


def _received(flag: threading.Event, expected: bool) -> bool:
    return flag.wait(ARRIVAL_TIMEOUT if expected else QUIET_PERIOD)


@pytest.mark.parametrize("join_event", ["user-join", "join-channels"])
def test_joining_channels_skips_a_member_without_the_permission(
    join_event, admin, make_user, channels_for_admins_and_groups_only
):
    restricted, granted = make_user(), make_user()
    make_group(admin, [granted], CHANNELS_GRANTED)
    channel_id = group_channel(admin, restricted, granted)

    with (
        connected(restricted) as restricted_tab,
        connected(granted) as granted_tab,
        connected(admin) as admins_tab,
    ):
        for account, tab in ((restricted, restricted_tab), (granted, granted_tab)):
            tab.call(join_event, {"auth": {"token": account.token}})
        admins_tab.call(join_event, {"auth": {"token": admin.token}})
        marker = f"team news {uuid.uuid4().hex[:6]}"
        to_restricted = _arrivals(restricted_tab, marker)
        to_granted = _arrivals(granted_tab, marker)
        to_admin = _arrivals(admins_tab, marker)
        post_message(admin, channel_id, marker)

        granted_received = _received(to_granted, expected=True)
        admin_received = _received(to_admin, expected=True)
        restricted_received = _received(to_restricted, expected=False)

    assert granted_received, "a member whose group grants Channels got no live message"
    assert admin_received, "an admin got no live message"
    assert not restricted_received, (
        f"a member without the Channels permission joined the channel's room through "
        f"{join_event} and got its new message live (#31578)"
    )


def _open_direct_message(admin, member) -> str:
    with admin.client() as client:
        opened = client.get(f"/api/v1/channels/users/{member.id}")
    assert opened.status_code == 200, opened.text
    return opened.json()["id"]


def test_a_new_direct_message_does_not_bring_a_member_without_the_permission_into_its_room(
    admin, make_user, channels_for_admins_and_groups_only
):
    restricted, granted = make_user(), make_user()
    make_group(admin, [granted], CHANNELS_GRANTED)

    with connected(restricted) as restricted_tab, connected(granted) as granted_tab:
        to_restricted_channel = _open_direct_message(admin, restricted)
        to_granted_channel = _open_direct_message(admin, granted)
        marker = f"just between us {uuid.uuid4().hex[:6]}"
        to_restricted = _arrivals(restricted_tab, marker)
        to_granted = _arrivals(granted_tab, marker)
        post_message(admin, to_granted_channel, marker)
        post_message(admin, to_restricted_channel, marker)

        granted_received = _received(to_granted, expected=True)
        restricted_received = _received(to_restricted, expected=False)

    assert granted_received, "a member whose group grants Channels got no live direct message"
    assert not restricted_received, (
        "opening a direct message put the open tab of a member without the Channels permission "
        "into its room, and the message arrived live (#31578)"
    )


def test_a_new_group_channel_does_not_bring_a_member_without_the_permission_into_its_room(
    admin, make_user, channels_for_admins_and_groups_only
):
    restricted, granted = make_user(), make_user()
    make_group(admin, [granted], CHANNELS_GRANTED)

    with connected(restricted) as restricted_tab, connected(granted) as granted_tab:
        channel_id = group_channel(admin, restricted, granted)
        marker = f"kick-off {uuid.uuid4().hex[:6]}"
        to_restricted = _arrivals(restricted_tab, marker)
        to_granted = _arrivals(granted_tab, marker)
        post_message(admin, channel_id, marker)

        granted_received = _received(to_granted, expected=True)
        restricted_received = _received(to_restricted, expected=False)

    assert granted_received, "a member whose group grants Channels got no live message"
    assert not restricted_received, (
        "creating a group channel put the open tab of a member without the Channels permission "
        "into its room, and the message arrived live (#31578)"
    )
