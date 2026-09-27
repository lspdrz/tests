"""A chat streamed on one instance reaches the account's tab on another, over a shared Redis.

PR open-webui/open-webui#28818 (0.11.5): with `WEBSOCKET_MANAGER=redis`, an emit to one room
(every live chat event goes to the account's `user:{id}` room) is published on a Redis channel
of that room's own, so an instance with nobody in the room drops it by channel name instead of
decoding it. `WEBSOCKET_REDIS_ROOM_CHANNELS=false` puts every emit back on the shared channel.
An instance in that mode listens on the shared channel only, so room emits from an instance
with the switch on do not reach it: the fleet has to run one mode, and be updated together.

Four instances share one database and one real Redis, two with the switch on (one by default)
and two with it off. A chat is sent to one and read live by a tab signed in on another, and a
test client subscribed to every Redis channel records which channel carried the chat's events.
That an instance with nobody in the room skips the message is only visible as CPU and is not
pinned.

Discriminates: passes on dev ef67cc3fa; with the switch ignored (the stock manager always) the
events travel on the shared channel and reach a switched-off instance; with room emits published
on a channel no instance listens for, the stream never reaches the other switched-on instance;
with the switch-off ignored (the room manager always) switched-off instances publish on room
channels and receive a switched-on instance's emits.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from typing import Iterator

import pytest
import redis

from harness import backends
from harness import upstream as reply
from harness.actors import create_user
from harness.chat import send_message, wait_for_reply
from harness.socket_client import connected

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

SHARED_CHANNEL = "socketio"
PIECES = ["Wien ", "liegt ", "an der ", "Donau."]
ANSWER = "".join(PIECES)
# how long a tab keeps listening after the sender's own tab saw the stream end
GRACE_SECONDS = 2.0


@pytest.fixture(scope="module")
def shared_redis() -> Iterator[str]:
    with backends.redis_server() as url:
        yield url


@pytest.fixture(scope="module")
def fleet(shared_redis, instance_with) -> dict:
    """Four instances on one database and one Redis, by name: on-a, on-b, off-a and off-b."""
    shared = {"REDIS_URL": shared_redis, "WEBSOCKET_MANAGER": "redis"}
    first = instance_with({**shared, "WEBUI_NAME": "on-a"})  # the switch at its default
    joined = {**shared, "DATABASE_URL": first.database_url}
    return {
        "on-a": first,
        "on-b": instance_with(
            {**joined, "WEBUI_NAME": "on-b", "WEBSOCKET_REDIS_ROOM_CHANNELS": "true"}
        ),
        "off-a": instance_with(
            {**joined, "WEBUI_NAME": "off-a", "WEBSOCKET_REDIS_ROOM_CHANNELS": "false"}
        ),
        "off-b": instance_with(
            {**joined, "WEBUI_NAME": "off-b", "WEBSOCKET_REDIS_ROOM_CHANNELS": "false"}
        ),
    }


@contextlib.contextmanager
def watching_redis(url: str) -> Iterator[list[tuple[str, bytes]]]:
    """Every (channel, message) published while the block runs, filled in when it ends."""
    client = redis.Redis.from_url(url)
    subscription = client.pubsub(ignore_subscribe_messages=True)
    subscription.psubscribe("*")
    subscription.get_message(timeout=1.0)  # the psubscribe confirmation
    published: list[tuple[str, bytes]] = []
    try:
        yield published
    finally:
        while message := subscription.get_message(timeout=0.5):
            published.append((message["channel"].decode(), message["data"]))
        subscription.close()
        client.close()


@dataclasses.dataclass
class Streamed:
    chat_id: str
    user_id: str
    receiver_events: list[dict]
    sender_events: list[dict]
    published: list[tuple[str, bytes]]

    def channels(self) -> set[str]:
        """The Redis channels that carried an emit about this chat."""
        return {channel for channel, data in self.published if self.chat_id.encode() in data}


def stream_across(fleet: dict, shared_redis: str, sender: str, receiver: str) -> Streamed:
    """Send a chat to `sender` while the account has a tab open on each of the two instances."""
    sending, receiving = fleet[sender], fleet[receiver]
    account = create_user(sending)
    prompt = f"Where is Vienna? ({sender} to {receiver})"
    # the connection settings live in the shared database, so every instance asks on-a's provider
    fleet["on-a"].upstream.queue(reply.text(PIECES, match=reply.answering(prompt)))
    with (
        watching_redis(shared_redis) as published,
        connected(dataclasses.replace(account, base_url=receiving.base_url)) as far_tab,
        connected(account) as near_tab,
        account.client() as client,
    ):
        turn = send_message(client, prompt)
        wait_for_reply(client, turn)
        near_tab.wait_for(turn.chat_id, "chat:completion", done=True)
        deadline = time.monotonic() + GRACE_SECONDS
        while time.monotonic() < deadline and not _finished(far_tab.events_of(turn.chat_id)):
            time.sleep(0.05)
    return Streamed(
        chat_id=turn.chat_id,
        user_id=account.id,
        receiver_events=far_tab.events_of(turn.chat_id),
        sender_events=near_tab.events_of(turn.chat_id),
        published=published,
    )


def _finished(events: list[dict]) -> bool:
    return any(_is_done(event) for event in events)


def _is_done(event: dict) -> bool:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    return event.get("type") == "chat:completion" and data.get("done") is True


def _streamed_text(events: list[dict]) -> str:
    """The reply as its live text deltas spelled it out."""
    payloads = [event.get("data") for event in events]
    return "".join(
        payload["delta"]
        for payload in payloads
        if isinstance(payload, dict) and isinstance(payload.get("delta"), str)
    )


def _event_types(events: list[dict]) -> set[str]:
    return {event.get("type") for event in events}


@pytest.mark.parametrize(("sender", "receiver"), [("on-a", "on-b"), ("off-a", "off-b")])
def test_a_tab_on_another_instance_sees_the_chat_stream_live(fleet, shared_redis, sender, receiver):
    streamed = stream_across(fleet, shared_redis, sender, receiver)

    assert streamed.receiver_events, (
        f"a tab on {receiver} saw nothing of a chat streamed on {sender}; the sender's own tab "
        f"saw {sorted(_event_types(streamed.sender_events))}"
    )
    assert _streamed_text(streamed.receiver_events) == ANSWER
    assert _finished(streamed.receiver_events), "the tab never learned the reply had finished"
    assert "chat:title" in _event_types(streamed.receiver_events)
    assert streamed.receiver_events == streamed.sender_events, (
        "the tab on the other instance saw different events from the one on the sender"
    )


def test_room_emits_travel_on_the_room_channel_only(fleet, shared_redis):
    streamed = stream_across(fleet, shared_redis, "on-a", "on-b")

    channels = streamed.channels()
    assert channels, "nothing about the chat was published on Redis at all"
    assert SHARED_CHANNEL not in channels, (
        "the chat's events were published on the shared channel, which every instance decodes"
    )
    assert all(channel.endswith(f"user:{streamed.user_id}") for channel in channels), (
        f"the chat's events went to {sorted(channels)}, not the account's room channel"
    )


def test_with_room_channels_off_every_emit_travels_on_the_shared_channel(fleet, shared_redis):
    streamed = stream_across(fleet, shared_redis, "off-a", "off-b")

    assert streamed.channels() == {SHARED_CHANNEL}


def test_a_switched_on_instance_still_receives_from_a_switched_off_one(fleet, shared_redis):
    streamed = stream_across(fleet, shared_redis, "off-a", "on-b")

    assert _streamed_text(streamed.receiver_events) == ANSWER
    assert _finished(streamed.receiver_events)


def test_a_switched_off_instance_does_not_receive_room_emits(fleet, shared_redis):
    """The Changed note of 0.11.5: an instance on the old delivery misses an updated one's emits."""
    streamed = stream_across(fleet, shared_redis, "on-a", "off-b")

    assert _finished(streamed.sender_events), "the sender's own tab did not see the stream"
    assert streamed.receiver_events == [], (
        "a tab on a switched-off instance received room emits from a switched-on one; "
        "the release notes say it does not"
    )
