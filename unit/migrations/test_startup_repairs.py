"""Guard: the repair of double-encoded `user.oauth` rows leaves every other string alone.

Migration `6d09d1bf1f23` (PR #28107, commit bd8378f643, issue #28101, v0.11.1) decodes an
`oauth` value stored as a JSON string of an object, which the user-table migration used to
write. A string that is not an encoded object (plain text, an encoded number) must stay as it
was. Runs `upgrade head` over seeded rows in a fresh interpreter and reads them back through a
JSON column.

Stays a unit test: no release ever wrote such a string, and any string in the column already
makes the admin's user list answer 500, so no request tells a left-alone string from a mangled
one. The
repair of real double-encoded rows is pinned from outside in
`integration/migrations/test_startup_repairs.py`.

Discriminates: passes on dev ef67cc3fa; with the repair writing any decodable value (the
`isinstance(decoded, dict)` check dropped) in a copy the encoded number comes back an int.
"""

from __future__ import annotations

import pytest

from .conftest import sqlite_at

pytestmark = pytest.mark.regression

# Head of the chain on v0.11.0 and the parent of the repair on v0.11.1.
PRE_REPAIR_REVISION = "f0bd01a18a3d"

REPAIR = f"""
import sqlalchemy as sa
from sqlalchemy import create_engine

engine = create_engine(os.environ['DATABASE_URL'])
user = sa.table(
    'user',
    sa.column('id', sa.Text),
    sa.column('name', sa.Text),
    sa.column('email', sa.Text),
    sa.column('oauth', sa.JSON),
)
command.upgrade(cfg, {PRE_REPAIR_REVISION!r})
seeds = {{'u_plain': 'not-json-at-all', 'u_scalar': '12345'}}
with engine.begin() as connection:
    for user_id, value in seeds.items():
        connection.execute(
            sa.insert(user).values(id=user_id, name=user_id, email=user_id + '@x.io', oauth=value)
        )
command.upgrade(cfg, 'head')
rows = {{}}
with engine.connect() as connection:
    for user_id in seeds:
        value = connection.execute(sa.select(user.c.oauth).where(user.c.id == user_id)).scalar()
        rows[user_id] = {{'type': type(value).__name__, 'value': value}}
print('RESULT:' + json.dumps(rows))
"""


@pytest.fixture(scope="module")
def repaired_rows(open_webui_backend, tmp_path_factory) -> dict:
    database = sqlite_at(open_webui_backend, tmp_path_factory.mktemp("oauth-repair"))
    return database.run(REPAIR, what="upgrading seeded user.oauth rows to head")


@pytest.mark.parametrize(
    ("user_id", "stored"), [("u_plain", "not-json-at-all"), ("u_scalar", "12345")]
)
def test_strings_that_are_not_an_encoded_object_are_left_alone(repaired_rows, user_id, stored):
    assert repaired_rows[user_id] == {"type": "str", "value": stored}
