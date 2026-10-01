"""Regression: a member added to a group channel got its live feed only after a reload.

Issue #30432, fix `5e1f4d8c3` (#31113): adding people to an existing group channel sent them no
`channel:created` event (so the sidebar did not list it) and did not put their open tabs into the
channel's room, so new messages stayed invisible to them until they reloaded. The members route now
sends the event to the new members and joins their open tabs to the room. A member added again
keeps the feed, and a standard channel (access by grants) is left as it was.

Discriminates: passes on dev 015dbc861; with 5e1f4d8c3 reverted in a backend copy the added
member's open tab receives neither the event nor the message. The existing member and standard
channel cases pass on both.
"""

from __future__ import annotations

import threading
import uuid

import pytest

from harness.channel_quotes import enable_channels, group_channel, post_message
from harness.socket_client import SocketSession, connected

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

ARRIVAL_TIMEOUT = 15.0
QUIET_PERIOD = 2.0


@pytest.fixture
def channels_on(admin, preserve):
    preserve("admin_config")
    enable_channels(admin)


class _ChannelFeed:
    """Every channel event a tab receives, with a wait for one carrying a given text."""

    def __init__(self, session: SocketSession) -> None:
        self.payloads: list[str] = []
        self._changed = threading.Condition()
        session.client.on("events:channel", self._record)

    def _record(self, *payload) -> None:
        with self._changed:
            self.payloads.append(str(payload))
            self._changed.notify_all()

    def received(self, needle: str, timeout: float) -> bool:
        with self._changed:
            return self._changed.wait_for(
                lambda: any(needle in text for text in self.payloads), timeout
            )


def _add_members(owner, channel_id: str, *members):
    with owner.client() as client:
        return client.post(
            f"/api/v1/channels/{channel_id}/update/members/add",
            json={"user_ids": [member.id for member in members]},
        )


def test_a_member_added_to_a_group_channel_gets_it_and_its_messages_live(channels_on, make_user):
    owner, founder, newcomer = make_user(), make_user(), make_user()
    channel_id = group_channel(owner, founder)

    with connected(newcomer) as newcomer_tab:
        feed = _ChannelFeed(newcomer_tab)
        marker = f"welcome aboard {uuid.uuid4().hex[:6]}"
        added = _add_members(owner, channel_id, newcomer)
        assert added.status_code == 200, added.text
        sidebar_refreshed = feed.received("channel:created", ARRIVAL_TIMEOUT)
        post_message(owner, channel_id, marker)
        message_arrived = feed.received(marker, ARRIVAL_TIMEOUT)

    assert sidebar_refreshed, "the added member's open tab got no channel:created event (#30432)"
    assert message_arrived, (
        "the added member's open tab got no live message from the channel (#30432)"
    )


def test_adding_an_existing_member_again_keeps_every_member_live(channels_on, make_user):
    owner, founder = make_user(), make_user()
    channel_id = group_channel(owner, founder)

    with connected(founder) as founder_tab:
        founder_tab.call("join-channels", {"auth": {"token": founder.token}})
        feed = _ChannelFeed(founder_tab)
        marker = f"nothing changed {uuid.uuid4().hex[:6]}"
        added = _add_members(owner, channel_id, founder)
        assert added.status_code == 200, added.text
        post_message(owner, channel_id, marker)
        message_arrived = feed.received(marker, ARRIVAL_TIMEOUT)

    assert message_arrived, "adding an existing member again cut the live feed"


def test_adding_a_member_to_a_standard_channel_sends_no_channel_created_event(
    channels_on, admin, make_user
):
    newcomer = make_user()
    with admin.client() as client:
        created = client.post("/api/v1/channels/create", json={"name": f"s-{uuid.uuid4().hex[:8]}"})
    assert created.status_code == 200, created.text
    channel_id = created.json()["id"]

    with connected(newcomer) as newcomer_tab:
        feed = _ChannelFeed(newcomer_tab)
        added = _add_members(admin, channel_id, newcomer)
        refreshed = feed.received("channel:created", QUIET_PERIOD)

    assert added.status_code == 200, added.text
    assert not refreshed, "adding a member to a standard channel sent a channel:created event"
