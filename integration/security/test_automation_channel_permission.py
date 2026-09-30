"""Regression: a channel automation kept posting after its owner lost the Channels permission.

Fix `9b45bedea` (#31577): saving an automation that targets a channel checks the Channels switch,
the owner's Channels permission and their write access to the channel, but a run only checked
the switch. An admin taking the permission away (from the default user permissions or from the
group that granted it) left the automation posting its prompt and the model's answer into the
channel. The run now fails and is recorded as an error unless the owner holds the permission;
admins are exempt.

Discriminates: passes on dev a5bc78300; with the permission check removed from
`_execute_channel_automation` the three tests that revoke the permission (default, then restored,
and group) go red (the prompt lands in the channel and the run is recorded as a success) while
the tests where the owner, the group or the admin holds the permission stay green.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness import upstream as reply
from harness.access import make_group
from harness.actors import Actor
from harness.channel_quotes import enable_channels
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PERMISSIONS = "/api/v1/users/default/permissions"
DAILY = "RRULE:FREQ=DAILY"
CHANNELS_GRANTED = {"features": {"channels": True, "automations": True}}


def _set_default_features(admin: Actor, **features: bool) -> None:
    with admin.client() as client:
        permissions = client.get(PERMISSIONS).json()
        permissions["features"].update(features)
        saved = client.post(PERMISSIONS, json=permissions)
    assert saved.status_code == 200, saved.text


@pytest.fixture
def automations_and_channels(admin, preserve):
    """Channels and automations on; every user holds both permissions by default."""
    preserve("admin_config", "permissions")
    enable_channels(admin)
    with admin.client() as client:
        config = client.get("/api/v1/auths/admin/config").json()
        client.post(
            "/api/v1/auths/admin/config", json={**config, "ENABLE_AUTOMATIONS": True}
        ).raise_for_status()
    _set_default_features(admin, channels=True, automations=True)


def _channel_written_by(admin: Actor, account: Actor) -> str:
    grants = [
        {"principal_type": "user", "principal_id": account.id, "permission": permission}
        for permission in ("read", "write")
    ]
    with admin.client() as client:
        created = client.post(
            "/api/v1/channels/create",
            json={"name": f"room-{uuid.uuid4().hex[:8]}", "access_grants": grants},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _create_channel_automation(owner: Actor, channel_id: str, prompt: str) -> str:
    form = {
        "name": f"post {uuid.uuid4().hex[:8]}",
        "is_active": False,
        "data": {
            "prompt": prompt,
            "model_id": MOCK_MODEL_ID,
            "rrule": DAILY,
            "target": {"type": "channel", "channel_id": channel_id},
        },
    }
    with owner.client() as client:
        created = client.post("/api/v1/automations/create", json=form)
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _runs(owner: Actor, automation_id: str) -> list[dict]:
    with owner.client() as client:
        runs = client.get(f"/api/v1/automations/{automation_id}/runs")
    assert runs.status_code == 200, runs.text
    return runs.json()


def _run_now(owner: Actor, automation_id: str) -> dict:
    """Start a run the way the Run now button does; returns the run it records."""
    before = len(_runs(owner, automation_id))
    with owner.client() as client:
        started = client.post(f"/api/v1/automations/{automation_id}/run")
    assert started.status_code == 200, started.text
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        runs = _runs(owner, automation_id)
        if len(runs) > before:
            return runs[0]
        time.sleep(0.2)
    raise AssertionError("the run was never recorded")


def _channel_holds(reader: Actor, channel_id: str, prompt: str) -> bool:
    with reader.client() as client:
        listed = client.get(f"/api/v1/channels/{channel_id}/messages")
        assert listed.status_code == 200, listed.text
        return any(prompt in (message.get("content") or "") for message in listed.json())


def _prompt() -> str:
    return f"nightly digest {uuid.uuid4().hex[:8]}"


def test_an_automation_posts_while_its_owner_holds_the_channels_permission(
    automations_and_channels, admin, make_user, upstream
):
    owner = make_user()
    channel_id = _channel_written_by(admin, owner)
    prompt = _prompt()
    automation_id = _create_channel_automation(owner, channel_id, prompt)
    upstream.queue(reply.text("here is the digest"))

    run = _run_now(owner, automation_id)

    assert run["status"] == "success", run
    assert _channel_holds(admin, channel_id, prompt)


def test_an_admins_channel_automation_posts_without_the_channels_permission(
    automations_and_channels, admin, make_user, upstream
):
    _set_default_features(admin, channels=False)
    owner = make_user(role="admin")
    channel_id = _channel_written_by(admin, owner)
    prompt = _prompt()
    automation_id = _create_channel_automation(owner, channel_id, prompt)
    upstream.queue(reply.text("here is the digest"))

    run = _run_now(owner, automation_id)

    assert run["status"] == "success", run
    assert _channel_holds(admin, channel_id, prompt)


def test_an_automation_stops_posting_when_the_default_channels_permission_is_taken_away(
    automations_and_channels, admin, make_user, upstream
):
    owner = make_user()
    channel_id = _channel_written_by(admin, owner)
    prompt = _prompt()
    automation_id = _create_channel_automation(owner, channel_id, prompt)
    _set_default_features(admin, channels=False)
    upstream.queue(reply.text("here is the digest"))

    run = _run_now(owner, automation_id)

    assert run["status"] == "error", (
        f"the run of a channel automation whose owner lost the Channels permission was "
        f"recorded as {run['status']} (#31577)"
    )
    assert not _channel_holds(admin, channel_id, prompt), (
        "the automation posted into the channel after its owner lost the Channels permission"
    )


def test_an_automation_posts_again_when_the_channels_permission_returns(
    automations_and_channels, admin, make_user, upstream
):
    owner = make_user()
    channel_id = _channel_written_by(admin, owner)
    prompt = _prompt()
    automation_id = _create_channel_automation(owner, channel_id, prompt)
    _set_default_features(admin, channels=False)
    assert _run_now(owner, automation_id)["status"] == "error"
    _set_default_features(admin, channels=True)
    upstream.queue(reply.text("here is the digest"))

    run = _run_now(owner, automation_id)

    assert run["status"] == "success", run
    assert _channel_holds(admin, channel_id, prompt)


def test_an_automation_stops_posting_when_the_owner_leaves_the_group_that_granted_channels(
    automations_and_channels, admin, make_user, upstream
):
    _set_default_features(admin, channels=False)
    owner = make_user()
    group_id = make_group(admin, [owner], CHANNELS_GRANTED)
    channel_id = _channel_written_by(admin, owner)
    prompt = _prompt()
    automation_id = _create_channel_automation(owner, channel_id, prompt)
    with admin.client() as client:
        left = client.post(
            f"/api/v1/groups/id/{group_id}/users/remove", json={"user_ids": [owner.id]}
        )
    assert left.status_code == 200, left.text
    upstream.queue(reply.text("here is the digest"))

    run = _run_now(owner, automation_id)

    assert run["status"] == "error", (
        f"the run of a channel automation whose owner left the group that granted Channels "
        f"was recorded as {run['status']} (#31577)"
    )
    assert not _channel_holds(admin, channel_id, prompt)


def test_an_automation_posts_while_the_group_still_grants_channels(
    automations_and_channels, admin, make_user, upstream
):
    _set_default_features(admin, channels=False)
    owner = make_user()
    make_group(admin, [owner], CHANNELS_GRANTED)
    channel_id = _channel_written_by(admin, owner)
    prompt = _prompt()
    automation_id = _create_channel_automation(owner, channel_id, prompt)
    upstream.queue(reply.text("here is the digest"))

    run = _run_now(owner, automation_id)

    assert run["status"] == "success", run
    assert _channel_holds(admin, channel_id, prompt)
