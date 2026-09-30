"""Regression: the audit log recorded new passwords in plain text.

Fix `d3dde3609` (#31622): with request auditing on, the logged request body was
masked only for a field named exactly `password`, so `new_password` (a user changing their own
password), the LDAP `app_dn_password` and the YaCy `YACY_PASSWORD` were written to the audit
log as typed. Any JSON field whose name ends in `password`, in any letter case, is now replaced
with asterisks before the entry is logged; the rest of the request still is.

The tests run on an instance of their own with request auditing on, and read the audit log
file in its data directory.

Discriminates: passes on dev a5bc78300; with the masking pattern reverted to the exact
`"password"` field, the self-service change, the LDAP and the YaCy tests go red (the new value
is in the log), while the admin reset and the add-user tests stay green.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

from harness.actors import create_user
from harness.instance import ADMIN_EMAIL

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

MASK = "********"
AUDIT_ENV = {"AUDIT_LOG_LEVEL": "REQUEST", "AUDIT_EXCLUDED_PATHS": ""}
RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
LDAP_SERVER = "/api/v1/auths/admin/config/ldap/server"


@pytest.fixture
def audited(instance_with):
    return instance_with(AUDIT_ENV)


def _secret(label: str) -> str:
    return f"Sec-{label}-{uuid.uuid4().hex[:12]}"


def _entries_for(instance, path_suffix: str, user_email: str) -> list[dict]:
    audit_log = instance.data_dir / "audit.log"
    if not audit_log.exists():
        return []
    lines = [line for line in audit_log.read_text().splitlines() if line.strip()]
    entries = [json.loads(line) for line in lines]
    return [
        entry
        for entry in entries
        if entry["request_uri"].split("?")[0].endswith(path_suffix)
        and entry["user"].get("email") == user_email
    ]


def _logged_request(instance, path_suffix: str, user_email: str) -> str:
    """The request body of the newest audit entry for this route and account, once it lands."""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        entries = _entries_for(instance, path_suffix, user_email)
        if entries:
            return entries[-1]["request_object"]
        time.sleep(0.2)
    raise AssertionError(f"no audit entry for {path_suffix} by {user_email}")


def test_a_users_own_password_change_logs_neither_password(audited):
    account = create_user(audited)
    new_password = _secret("new")

    with audited.client(account.token) as client:
        changed = client.post(
            "/api/v1/auths/update/password",
            json={"password": account.password, "new_password": new_password},
        )
    assert changed.status_code == 200, changed.text

    logged = _logged_request(audited, "/auths/update/password", account.email)
    assert new_password not in logged, f"the new password reached the audit log (#31622): {logged}"
    assert account.password not in logged
    assert json.loads(logged) == {"password": MASK, "new_password": MASK}


def test_an_admin_reset_logs_the_name_but_not_the_new_password(audited):
    account = create_user(audited)
    new_password = _secret("reset")
    new_name = f"Renamed {uuid.uuid4().hex[:8]}"

    with audited.client() as client:
        reset = client.post(
            f"/api/v1/users/{account.id}/update",
            json={"name": new_name, "password": new_password},
        )
    assert reset.status_code == 200, reset.text

    logged = _logged_request(audited, f"/users/{account.id}/update", ADMIN_EMAIL)
    assert new_password not in logged
    assert json.loads(logged) == {"name": new_name, "password": MASK}


def test_adding_a_user_still_masks_the_password_and_logs_the_email(audited):
    password = _secret("added")
    email = f"added-{uuid.uuid4().hex[:8]}@example.com"

    with audited.client() as client:
        added = client.post(
            "/api/v1/auths/add",
            json={"name": "Added", "email": email, "password": password, "role": "user"},
        )
    assert added.status_code == 200, added.text

    logged = _logged_request(audited, "/auths/add", ADMIN_EMAIL)
    assert password not in logged
    assert json.loads(logged)["email"] == email
    assert json.loads(logged)["password"] == MASK


def test_an_ldap_app_password_is_masked_in_the_settings_save(audited):
    # Unset LDAP settings cannot be posted back, so the module's own instance is left as saved.
    app_password = _secret("ldap")
    settings = {
        "label": "Audit directory",
        "host": "ldap.audit.invalid",
        "app_dn": "cn=reader,dc=example,dc=com",
        "app_dn_password": app_password,
        "search_base": "dc=example,dc=com",
        "use_tls": False,
    }

    with audited.client() as client:
        saved = client.post(LDAP_SERVER, json=settings)
    assert saved.status_code == 200, saved.text

    logged = _logged_request(audited, "/admin/config/ldap/server", ADMIN_EMAIL)
    assert app_password not in logged, f"the LDAP password reached the audit log (#31622): {logged}"
    assert json.loads(logged)["app_dn_password"] == MASK
    assert json.loads(logged)["host"] == "ldap.audit.invalid"


def test_a_yacy_password_in_capitals_is_masked_in_the_retrieval_settings_save(audited, preserve):
    preserve(RETRIEVAL_CONFIG, on=audited)
    yacy_password = _secret("yacy")
    yacy_user = f"yacy-{uuid.uuid4().hex[:8]}"

    with audited.client() as client:
        saved = client.post(
            RETRIEVAL_CONFIG[1],
            json={"web": {"YACY_USERNAME": yacy_user, "YACY_PASSWORD": yacy_password}},
        )
    assert saved.status_code == 200, saved.text

    logged = _logged_request(audited, "/retrieval/config/update", ADMIN_EMAIL)
    assert yacy_password not in logged, (
        f"the YaCy password reached the audit log (#31622): {logged}"
    )
    assert json.loads(logged)["web"] == {"YACY_USERNAME": yacy_user, "YACY_PASSWORD": MASK}
