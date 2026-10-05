"""Nested groups as the admin panel builds them: a chain of three with a member at each level.

`build_chain(admin, make_user)` makes a top group, a middle group inside it and a leaf group
inside that, with `top_member` in the top group only, `middle_member` in the middle group only
and `leaf_member` in the leaf group only. `move_group(...)` sends the group editor's update with
a new parent (or `None` for the top level) and `group_of(...)` reads the stored group back.
`share_with_group(...)` replaces a resource's access grants with one read grant for a group.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from harness.access import grant, make_group
from harness.actors import Actor


@dataclass
class Chain:
    top: str
    middle: str
    leaf: str
    top_member: Actor
    middle_member: Actor
    leaf_member: Actor


def build_chain(admin: Actor, make_user) -> Chain:
    top_member, middle_member, leaf_member = make_user(), make_user(), make_user()
    top = make_group(admin, [top_member])
    middle = make_group(admin, [middle_member], parent_id=top)
    leaf = make_group(admin, [leaf_member], parent_id=middle)
    return Chain(top, middle, leaf, top_member, middle_member, leaf_member)


def group_of(admin: Actor, group_id: str) -> dict:
    with admin.client() as client:
        stored = client.get(f"/api/v1/groups/id/{group_id}")
    assert stored.status_code == 200, stored.text
    return stored.json()


def move_group(admin: Actor, group_id: str, parent_id: str | None, **changes) -> httpx.Response:
    """The editor's save: the group's own name and description with the parent chosen."""
    stored = group_of(admin, group_id)
    form = {
        "name": stored["name"],
        "description": stored["description"],
        "parent_group_id": parent_id,
        **changes,
    }
    with admin.client() as client:
        return client.post(f"/api/v1/groups/id/{group_id}/update", json=form)


def share_with_group(admin: Actor, path: str, group_id: str, **fields: object) -> httpx.Response:
    with admin.client() as client:
        return client.post(
            path,
            json={**fields, "access_grants": [grant("group", group_id, "read")]},
        )
