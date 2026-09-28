"""Regression: SSO settings kept the values of an earlier boot and ignored the environment.

open-webui fix `e398ba350` (#26928). With `ENABLE_OAUTH_PERSISTENT_CONFIG` off the SSO settings
come from the environment on every boot and are never stored, yet the boot still seeded a row
for every `oauth.*` setting with the value it had then. Once an operator turned persistence on,
those stale rows won over the environment for good, so an SSO setting changed in the
environment never took effect. Seeding now skips the settings the database is not in charge of.

Each test boots the checkout twice on one data directory: first without an account, the way a
deployment starts, then with the environment changed. The admin reads the SSO settings from the
second boot.

Twin of unit/security/test_oauth_session_lifecycle.py (its settings seeding part).

Discriminates: passes on dev ef67cc3fa, fails with the persistence check removed from
`Config.seed_defaults` (the first boot's row wins and OAuth sign-up stays off).
"""

from __future__ import annotations

import pytest

from harness.prepared_data import booted_again

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

OAUTH_CONFIG = "/api/v1/auths/admin/config/oauth"
USERS_CONFIG = "/api/v1/auths/admin/config"


def test_turning_persistence_on_reads_sso_settings_from_the_environment(instance_with, tmp_path):
    """Narrow: a boot with persistence off stores nothing that later shadows the environment."""
    launched = booted_again(
        instance_with,
        tmp_path / "data",
        first={"ENABLE_OAUTH_PERSISTENT_CONFIG": "false", "ENABLE_OAUTH_SIGNUP": "false"},
        second={"ENABLE_OAUTH_PERSISTENT_CONFIG": "true", "ENABLE_OAUTH_SIGNUP": "true"},
    )

    with launched.client() as admin:
        settings = admin.get(OAUTH_CONFIG).json()

    assert settings["ENABLE_OAUTH_PERSISTENT_CONFIG"] is True
    assert settings["ENABLE_OAUTH_SIGNUP"] is True, (
        "OAuth sign-up stayed off although the environment now turns it on: the first boot "
        "stored the setting while persistence was off and the row shadows the environment (#26928)"
    )


def test_with_persistence_on_from_the_start_the_stored_setting_wins(instance_with, tmp_path):
    """Nearby: once the database is in charge, its value outlives a changed environment."""
    launched = booted_again(
        instance_with,
        tmp_path / "data",
        first={"ENABLE_OAUTH_PERSISTENT_CONFIG": "true", "ENABLE_OAUTH_SIGNUP": "false"},
        second={"ENABLE_OAUTH_PERSISTENT_CONFIG": "true", "ENABLE_OAUTH_SIGNUP": "true"},
    )

    with launched.client() as admin:
        assert admin.get(OAUTH_CONFIG).json()["ENABLE_OAUTH_SIGNUP"] is False


def test_other_settings_are_still_stored_while_sso_persistence_is_off(instance_with, tmp_path):
    """Nearby: skipping the SSO settings leaves every other setting seeded at the first boot."""
    launched = booted_again(
        instance_with,
        tmp_path / "data",
        first={"ENABLE_OAUTH_PERSISTENT_CONFIG": "false", "DEFAULT_USER_ROLE": "pending"},
        second={"ENABLE_OAUTH_PERSISTENT_CONFIG": "false", "DEFAULT_USER_ROLE": "user"},
    )

    with launched.client() as admin:
        assert admin.get(USERS_CONFIG).json()["DEFAULT_USER_ROLE"] == "pending"
