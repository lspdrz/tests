"""Accounts as a release from before the user-table migration (`b10670c03dd5`) stored them.

Releases up to 0.6.x kept `user.settings` and `user.info` as JSON text, a single SSO link in
`user.oauth_sub` (`provider@sub`, or a bare sub meaning `oidc`) and one API key in
`user.api_key`. No data set under `integration/migrations/upgrade_data/` is that old, so
`seed_legacy_accounts(data_dir, database_url)` builds the database the way such an install left
it: the manual `alembic upgrade` to the revision before that migration, then the rows of
`LEGACY_ACCOUNTS` written in the old shapes. Booting the checkout on it runs the rest of the
chain. Each account signs in with `LEGACY_PASSWORD`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import bcrypt
import sqlalchemy

from harness.prepared_data import manual_alembic

# the revision a 0.6.x install sits on before the user-table migration
BEFORE_USER_TABLE_MIGRATION = "2f1211949ecc"
LEGACY_PASSWORD = "legacy-password-123"
LEGACY_API_KEY = "sk-legacy0123456789abcdef0123456789"

# settings as the web client saved them; double-encoded they come back as a string
RICH_SETTINGS = {
    "ui": {"widescreenMode": True, "showChangelog": False, "chatBubble": False},
    "notifications": True,
    "models": ["gpt-4o", "claude"],
}
RICH_INFO = {"organization": "acme", "seats": 3}

LEGACY_ACCOUNTS = {
    "admin": {"role": "admin", "api_key": LEGACY_API_KEY, "settings": {"ui": {}}},
    "alice": {"settings": RICH_SETTINGS, "info": RICH_INFO},
    "bob": {"settings": {}},
    "carol": {"settings": None},
    "dana": {"oauth_sub": "oidc@legacy-dana-sub"},
    "erin": {"oauth_sub": "legacy-erin-sub"},
}


def legacy_email(who: str) -> str:
    return f"{who}@legacy.example.com"


def legacy_id(who: str) -> str:
    return f"legacy-{who}"


def _as_text(value) -> str | None:
    return None if value is None else json.dumps(value)


def seed_legacy_accounts(data_dir: Path, database_url: str | None = None) -> None:
    """Build the pre-migration database in `data_dir` (or at `database_url`) with the accounts."""
    migrated = manual_alembic(
        data_dir, "upgrade", BEFORE_USER_TABLE_MIGRATION, database_url=database_url
    )
    assert migrated.returncode == 0, f"building the old schema failed:\n{migrated.stderr[-3000:]}"

    hashed = bcrypt.hashpw(LEGACY_PASSWORD.encode(), bcrypt.gensalt()).decode()
    now = int(time.time())
    user = sqlalchemy.table(
        "user",
        *(sqlalchemy.column(name) for name in ("id", "name", "email", "role", "profile_image_url")),
        *(sqlalchemy.column(name) for name in ("last_active_at", "updated_at", "created_at")),
        *(sqlalchemy.column(name) for name in ("api_key", "settings", "info", "oauth_sub")),
    )
    auth = sqlalchemy.table(
        "auth", *(sqlalchemy.column(name) for name in ("id", "email", "password", "active"))
    )
    engine = sqlalchemy.create_engine(database_url or f"sqlite:///{data_dir / 'webui.db'}")
    try:
        with engine.begin() as connection:
            for who, account in LEGACY_ACCOUNTS.items():
                row = {
                    "id": legacy_id(who),
                    "name": who.title(),
                    "email": legacy_email(who),
                    "role": account.get("role", "user"),
                    "profile_image_url": "/user.png",
                    "last_active_at": now,
                    "updated_at": now,
                    "created_at": now,
                    "api_key": account.get("api_key"),
                    "settings": _as_text(account.get("settings")),
                    "info": _as_text(account.get("info")),
                    "oauth_sub": account.get("oauth_sub"),
                }
                connection.execute(sqlalchemy.insert(user).values(row))
                credentials = {"id": row["id"], "email": row["email"], "password": hashed}
                connection.execute(sqlalchemy.insert(auth).values(active=True, **credentials))
    finally:
        engine.dispose()
