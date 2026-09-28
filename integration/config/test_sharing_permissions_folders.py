"""Regression: the admin's folder sharing toggle was dropped on save.

open-webui 0.11.0, fix `3cf3f8e18` (PR #27296, issue #27120): `sharing.folders` was in
`DEFAULT_USER_PERMISSIONS` but had no field on `SharingPermissions`, the schema the default and
group permission endpoints validate through. Pydantic drops unknown keys, so an admin enabling
folder sharing had the flag discarded on every save and the setting never took effect. The fix
adds the field.

The broad test covers every default permission, including ones added later: on an instance where
nobody has saved yet, users get the built-in defaults as they are, and the admin's first save of
the default permissions must hand every one of those flags back to them.

Discriminates: passes on bbfa876af, fails with the `folders` field removed from
`SharingPermissions` (the saved flag reads back without it, the user's session says False and the
first save drops `sharing.folders` from every user's permissions); the round trip of every other
key passes on both.
"""

from __future__ import annotations

import pytest

from harness.actors import create_user

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PERMISSIONS = "/api/v1/users/default/permissions"


def _save_and_reload(admin, permissions: dict) -> dict:
    """Save as the admin's permissions modal does, then read what it shows when reopened."""
    with admin.client() as client:
        saved = client.post(PERMISSIONS, json=permissions)
        assert saved.status_code == 200, saved.text
        stored = client.get(PERMISSIONS)
    assert stored.status_code == 200, stored.text
    return stored.json()


@pytest.fixture
def permissions(admin, preserve) -> dict:
    preserve("permissions")
    with admin.client() as client:
        current = client.get(PERMISSIONS)
    assert current.status_code == 200, current.text
    return current.json()


# narrow: the folder sharing flag survives the save and reaches the users


def test_enabled_folder_sharing_survives_a_save(admin, permissions):
    permissions["sharing"]["folders"] = True

    stored = _save_and_reload(admin, permissions)

    assert stored["sharing"].get("folders") is True, (
        "the admin enabled folder sharing and the save dropped it, so the toggle reverts (#27120)"
    )


def test_enabled_folder_sharing_reaches_the_users_session(admin, permissions, make_user):
    permissions["sharing"]["folders"] = True
    _save_and_reload(admin, permissions)

    with make_user().client() as client:
        session = client.get("/api/v1/auths/")

    assert session.status_code == 200, session.text
    assert session.json()["permissions"]["sharing"]["folders"] is True, (
        "folder sharing reads as off for users although the admin enabled it (#27120)"
    )


# broad and nearby: every permission round-trips in both directions


def test_every_permission_round_trips_both_ways(admin, permissions):
    flipped = {
        section: {key: not enabled for key, enabled in flags.items()}
        for section, flags in permissions.items()
    }

    assert _save_and_reload(admin, flipped) == flipped
    assert _save_and_reload(admin, permissions) == permissions


def _flags(permissions: dict) -> dict[str, bool]:
    return {
        f"{section}.{key}": enabled
        for section, flags in permissions.items()
        for key, enabled in flags.items()
    }


@pytest.mark.slow
def test_the_first_save_keeps_every_default_permission_users_had(instance_with):
    unsaved = instance_with({})  # nobody has saved the default permissions here yet
    member = create_user(unsaved)
    with member.client() as client:
        before = _flags(client.get("/api/v1/auths/").json()["permissions"])

    with unsaved.client() as client:
        shown = client.get(PERMISSIONS)
        assert shown.status_code == 200, shown.text
        saved = client.post(PERMISSIONS, json=shown.json())
        assert saved.status_code == 200, saved.text
    with member.client() as client:
        after = _flags(client.get("/api/v1/auths/").json()["permissions"])

    dropped = sorted(set(before) - set(after))
    assert dropped == [], (
        f"default permissions {dropped} have no field in the admin's permission form, so the "
        "first save removed them from every user (#27120)"
    )
    assert after == before
