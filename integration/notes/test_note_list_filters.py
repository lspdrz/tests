"""Journey: which notes the Notes page lists under each filter, seen through the API.

The page asks `GET /api/v1/notes/search` for its list, with `view_option` for All (none),
Created by you (`created`) and Shared with you (`shared`), and `permission` for Write (none) or
Read Only (`read_only`). An account sees its own notes and the ones it may edit under Write, the
ones it may only read under Read Only, and never a note nobody shared with it; a grant to a
group it is in counts like one to the account. Twin of the filter tests in
e2e/notes/test_notes_page.py.

Discriminates: passes on dev 176d31d1d; in a backend copy, the search ignoring `view_option`
together with the read-only filter listing every readable note turns every row but All with Write
red, and the list defaulting to read in place of write access turns that row, the shared one and
the total red.
"""

from __future__ import annotations

import uuid

import pytest

from harness.access import grant, make_group
from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def _create_note(owner: Actor, title: str, grants: list[dict] | None = None) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={
                "title": title,
                "data": {"content": {"md": title}},
                "access_grants": grants or [],
            },
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


@pytest.fixture
def notes_around(make_user, admin) -> tuple[Actor, dict[str, str]]:
    """An account with its own note, notes shared with it to edit and to read, and a stranger's."""
    viewer, other = make_user(), make_user()
    group_id = make_group(admin, [viewer])
    tag = uuid.uuid4().hex[:6]
    to_edit = [grant("user", viewer.id, "read"), grant("user", viewer.id, "write")]
    group_edit = [grant("group", group_id, "read"), grant("group", group_id, "write")]
    ids = {
        "own": _create_note(viewer, f"own {tag}"),
        "edit": _create_note(other, f"edit {tag}", to_edit),
        "group edit": _create_note(other, f"group edit {tag}", group_edit),
        "read": _create_note(other, f"read {tag}", [grant("user", viewer.id, "read")]),
        "group read": _create_note(other, f"group read {tag}", [grant("group", group_id, "read")]),
        "private": _create_note(other, f"private {tag}"),
    }
    return viewer, ids


def _listed(viewer: Actor, ids: dict[str, str], **params: str) -> set[str]:
    with viewer.client() as client:
        found = client.get("/api/v1/notes/search", params=params)
    assert found.status_code == 200, found.text
    names = {note_id: name for name, note_id in ids.items()}
    return {names[item["id"]] for item in found.json()["items"] if item["id"] in names}


@pytest.mark.parametrize(
    "params, expected",
    [
        ({}, {"own", "edit", "group edit"}),
        ({"permission": "read_only"}, {"read", "group read"}),
        ({"view_option": "created"}, {"own"}),
        ({"view_option": "shared"}, {"edit", "group edit"}),
        ({"view_option": "shared", "permission": "read_only"}, {"read", "group read"}),
    ],
    ids=["all-write", "all-read-only", "created", "shared-write", "shared-read-only"],
)
def test_each_filter_lists_the_notes_it_names(notes_around, params, expected):
    viewer, ids = notes_around

    assert _listed(viewer, ids, **params) == expected


def test_the_total_counts_what_the_filter_keeps(notes_around):
    viewer, ids = notes_around

    with viewer.client() as client:
        shared = client.get("/api/v1/notes/search", params={"view_option": "shared"}).json()

    assert shared["total"] == len(shared["items"]) == 2
