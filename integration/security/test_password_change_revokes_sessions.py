"""Regression: changing a password left every other signed-in session working.

open-webui 0.11.1 fix `21e390561` (#28725): a password change wrote nothing to the token
revocation list, so every session issued before it kept working until its JWT expired, four
weeks by default. Both password paths, `POST /api/v1/auths/update/password` and an admin reset
through `POST /api/v1/users/{id}/update`, now revoke every earlier session. The fix kept the
revocation as a marker in Redis, for as long as a token can live, and without Redis revoked
nothing; since 24e30d1cb it is kept with the account in the database (a session stamp every token
carries), so it holds with no Redis at all and when anything Redis kept is gone.

The revocation tests run on an instance of their own backed by `StatefulRedis`; the no-Redis
test runs on the shared instance, or on a Redis-less one of its own when the run puts the shared
instance on Redis.

Fix `2062231f9` (#31621): `GET /api/config` decoded the session token but never asked the
revocation list, so a signed-out or password-revoked session still received the signed-in
configuration (permissions, default models and prompts). It now gets the logged-out shape, the
one an anonymous request gets, while a live session keeps the full one.

Twin of unit/security/test_password_change_revokes_sessions.py.

Discriminates: passes on dev b859124f9 and fails with the password write no longer revoking
(earlier sessions keep answering 200). On dev 015dbc861, before 24e30d1cb, the Redis-loss and
no-Redis tests fail (the earlier session answers 200 again once the marker is gone, and
throughout without Redis). With the admin reset revoking even when no password was written its
test fails (bbfa876af). The app
configuration tests pass on dev a5bc78300 and fail with 2062231f9 reverted (the revoked sessions
get the signed-in configuration back).
"""

from __future__ import annotations

import time
import uuid

import httpx
import pytest

from harness.actors import create_user, sign_in
from integration.stateful_redis import StatefulRedis

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

NEW_PASSWORD = "N3w-password-after-the-change"
PASSWORD_CHANGES = ["self-service", "admin reset"]
SCIM_TOKEN = "scim-provisioning-token"


@pytest.fixture(scope="session")
def revocation_store():
    store = StatefulRedis()
    yield store
    store.close()


@pytest.fixture
def redis_instance(revocation_store, instance_with):
    return instance_with(
        {
            "REDIS_URL": revocation_store.url,
            # Tokens that outlive the fixed 30 days the marker used to be kept for.
            "JWT_EXPIRES_IN": "8w",
            # Provisions accounts without a password row, as single sign-on does.
            "ENABLE_SCIM": "true",
            "SCIM_TOKEN": SCIM_TOKEN,
            "SCIM_AUTH_PROVIDER": "oidc",
        }
    )


def _session_status(instance, token: str) -> int:
    with instance.client(token) as client:
        return client.get("/api/v1/auths/").status_code


def _config_keys(instance, token: str | None = None) -> set[str]:
    """The top-level keys of `/api/config`; no token means an anonymous request."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = httpx.get(f"{instance.base_url}/api/config", headers=headers, timeout=60.0)
    assert response.status_code == 200, response.text
    return set(response.json())


def _change_own_password(instance, account, current_password: str) -> httpx.Response:
    with instance.client(account.token) as client:
        return client.post(
            "/api/v1/auths/update/password",
            json={"password": current_password, "new_password": NEW_PASSWORD},
        )


def _change_password(instance, account, change: str) -> None:
    if change == "self-service":
        response = _change_own_password(instance, account, account.password)
    else:
        with instance.client() as admin_client:
            response = admin_client.post(
                f"/api/v1/users/{account.id}/update", json={"password": NEW_PASSWORD}
            )
    assert response.status_code == 200, f"{change} failed: {response.text}"


def _keys_naming(store: StatefulRedis, user_id: str) -> list[str]:
    return [key for key in store.keys() if user_id in key]


@pytest.mark.parametrize("change", PASSWORD_CHANGES)
def test_a_password_change_signs_out_every_earlier_session(redis_instance, change):
    account = create_user(redis_instance)
    other_device = sign_in(redis_instance, account.email, account.password)
    bystander = create_user(redis_instance)

    _change_password(redis_instance, account, change)

    assert _session_status(redis_instance, other_device) == 401, (
        f"a session issued before the {change} still authenticates: a token stolen under the "
        "old password keeps working for the whole JWT lifetime (#28725)"
    )
    assert _session_status(redis_instance, account.token) == 401
    assert _session_status(redis_instance, bystander.token) == 200, (
        f"the {change} of one account signed another account out"
    )


@pytest.mark.parametrize("change", PASSWORD_CHANGES)
def test_the_revocation_outlives_anything_redis_kept(redis_instance, revocation_store, change):
    account = create_user(redis_instance)
    other_device = sign_in(redis_instance, account.email, account.password)

    _change_password(redis_instance, account, change)
    for key in _keys_naming(revocation_store, account.id):
        revocation_store.forget(key)

    assert _session_status(redis_instance, other_device) == 401, (
        f"the session revoked by the {change} authenticates again once Redis forgets the "
        "revocation: it expires before the tokens it revokes (#28725)"
    )


def test_a_wrong_current_password_is_refused_and_signs_nothing_out(
    redis_instance, revocation_store
):
    account = create_user(redis_instance)
    other_device = sign_in(redis_instance, account.email, account.password)

    refused = _change_own_password(redis_instance, account, "not-the-current-password")

    assert refused.status_code == 400
    assert _session_status(redis_instance, account.token) == 200
    assert _session_status(redis_instance, other_device) == 200, (
        "a refused password change signed the account out: a stolen session could log the "
        "real user out at will"
    )
    assert _keys_naming(revocation_store, account.id) == []


def test_a_reset_that_writes_no_password_revokes_nothing(redis_instance, revocation_store):
    """Nothing changed for an account without a password row, so its sessions must survive."""
    suffix = uuid.uuid4().hex[:8]
    email = f"sso-{suffix}@example.com"
    provisioned = httpx.post(
        f"{redis_instance.base_url}/api/v1/scim/v2/Users",
        headers={"Authorization": f"Bearer {SCIM_TOKEN}"},
        json={
            "userName": email,
            "displayName": f"SSO {suffix}",
            "emails": [{"value": email, "primary": True}],
        },
        timeout=60.0,
    )
    assert provisioned.status_code == 201, provisioned.text
    account_id = provisioned.json()["id"]

    with redis_instance.client() as admin_client:
        reset = admin_client.post(
            f"/api/v1/users/{account_id}/update", json={"password": NEW_PASSWORD}
        )

    assert reset.status_code == 200, reset.text
    assert _keys_naming(revocation_store, account_id) == [], (
        "an admin reset that wrote no password still revoked the account's sessions, signing a "
        "single sign-on user out of everything for nothing"
    )


def test_signing_in_with_the_new_password_works_after_the_change(redis_instance):
    account = create_user(redis_instance)
    _change_password(redis_instance, account, "self-service")
    changed_in_second = int(time.time())

    # The marker has one-second resolution and also rejects a token issued in its own second.
    time.sleep(max(0.0, changed_in_second + 1 - time.time()))
    fresh = sign_in(redis_instance, account.email, NEW_PASSWORD)
    stale = httpx.post(
        f"{redis_instance.base_url}/api/v1/auths/signin",
        json={"email": account.email, "password": account.password},
        timeout=60.0,
    )

    assert _session_status(redis_instance, fresh) == 200, (
        "the session from signing in with the new password was rejected: the account is locked "
        "out after its own password change"
    )
    assert stale.status_code == 400


def test_signing_out_one_session_leaves_the_others(redis_instance):
    account = create_user(redis_instance)
    other_device = sign_in(redis_instance, account.email, account.password)

    with redis_instance.client(account.token) as client:
        client.post("/api/v1/auths/signout").raise_for_status()

    assert _session_status(redis_instance, account.token) == 401
    assert _session_status(redis_instance, other_device) == 200, (
        "signing out one device signed out the account's other sessions too"
    )


def test_without_redis_the_change_still_signs_out_earlier_sessions(instance, instance_with):
    if instance.redis_url:
        instance = instance_with({"REDIS_URL": ""})
    account = create_user(instance)
    other_device = sign_in(instance, account.email, account.password)

    _change_password(instance, account, "self-service")

    assert _session_status(instance, other_device) == 401, (
        "without Redis a session issued before the password change still authenticates (#28725)"
    )


def test_a_signed_out_session_gets_the_logged_out_configuration(redis_instance):
    account = create_user(redis_instance)
    other_device = sign_in(redis_instance, account.email, account.password)
    anonymous = _config_keys(redis_instance)
    signed_in = _config_keys(redis_instance, account.token)
    assert signed_in > anonymous, "a live session must get more than the logged-out configuration"

    with redis_instance.client(account.token) as client:
        client.post("/api/v1/auths/signout").raise_for_status()

    assert _config_keys(redis_instance, account.token) == anonymous, (
        "a signed-out token still received the signed-in configuration (#31621)"
    )
    assert _config_keys(redis_instance, other_device) == signed_in, (
        "signing out one device changed the configuration the account's other session gets"
    )


@pytest.mark.parametrize("change", PASSWORD_CHANGES)
def test_a_session_revoked_by_a_password_change_gets_the_logged_out_configuration(
    redis_instance, change
):
    account = create_user(redis_instance)
    other_device = sign_in(redis_instance, account.email, account.password)
    bystander = create_user(redis_instance)
    anonymous = _config_keys(redis_instance)
    signed_in = _config_keys(redis_instance, bystander.token)

    _change_password(redis_instance, account, change)

    assert _config_keys(redis_instance, other_device) == anonymous, (
        f"a session revoked by the {change} still received the signed-in configuration (#31621)"
    )
    assert _config_keys(redis_instance, account.token) == anonymous
    assert _config_keys(redis_instance, bystander.token) == signed_in
