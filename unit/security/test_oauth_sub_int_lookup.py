"""Regression: the OAuth sub lookup still coerces an integer it is handed.

Fix `a6834f089` (open-webui/open-webui#28954, issue open-webui/open-webui#27760) made the lookup
match a numeric sub whether it is stored as a JSON number or as text, and coerce an int argument.
Every caller turns the sub into text first, so no request can hand the lookup an int; this case
stays a unit test. The stored-number and text cases are covered over HTTP in
integration/security/test_oauth_identity.py.

Discriminates: passes on dev ef67cc3fa; with the lookup's text coercion of the sub removed in a
copy the int lookup misses the account.
"""

from __future__ import annotations

import uuid

import pytest

from unit.security.memory_db import memory_database

pytestmark = pytest.mark.regression


@pytest.fixture
def users(owui_module):
    return owui_module("open_webui.models.users")


@pytest.mark.asyncio
async def test_an_int_sub_finds_the_account_storing_it_as_text(owui_module, users):
    account_id = str(uuid.uuid4())
    async with memory_database(owui_module, users.User):
        await users.Users.insert_new_user(
            id=account_id,
            name=account_id,
            email=f"{account_id}@example.com",
            oauth={"github": {"sub": "888"}},
        )
        found = await users.Users.get_user_by_oauth_sub(provider="github", sub=888)
    assert found is not None and found.id == account_id
