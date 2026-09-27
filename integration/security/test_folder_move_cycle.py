"""Regression: a folder can never become its own ancestor, and no walk follows a loop forever.

open-webui 0.11.1, fix `23b3a69bc` (#28748): `POST /api/v1/folders/{id}/update/parent` accepted a
move of a folder under itself or under one of its own descendants. A folder whose parent chain
loops is never a root, so it and everything below it vanished from the sidebar with no way back.
The fix answers 400 for such a move and has `GET /api/v1/folders/` put a looping folder back at
the top level, so a loop already in the database becomes reachable again. Loops that predate the
fix are seeded straight into the scratch instance's database.

Every walk over the folder tree followed such a loop with no record of where it had been: the
access check behind opening a shared folder, the child walk behind a folder's chat list, the
subtree walk behind marking a folder read and deleting it, and the delete cascade. The fix bounds
each walk with the ids it has seen. Those walks run on a throwaway instance of their own, with a
time limit on every request, because a walk that follows the loop spins until the process goes.

Discriminates: passes on ef67cc3fa; with the subtree check removed from the move route the moves
into the own subtree answer 200 and write the loop, and with the cycle check removed from the
listing the seeded loops are listed unchanged. With the visited-id sets of 23b3a69bc removed the
access check and the subtree walk never answer, and the chat list and the delete leave out the
folder the loop leads back through. The other tests pass on both.
"""

from __future__ import annotations

import httpx
import pytest

from harness.actors import create_user
from harness.backends import write_rows

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def _create_folder(client, name: str, parent_id: str | None = None) -> str:
    created = client.post("/api/v1/folders/", json={"name": name, "parent_id": parent_id})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _move(client, folder_id: str, parent_id: str | None):
    return client.post(f"/api/v1/folders/{folder_id}/update/parent", json={"parent_id": parent_id})


def _listed_parents(client) -> dict[str, str | None]:
    """Each listed folder's name mapped to its parent's name (or raw id when not listed)."""
    listed = client.get("/api/v1/folders/")
    assert listed.status_code == 200, listed.text
    name_by_id = {entry["id"]: entry["name"] for entry in listed.json()}
    return {
        entry["name"]: name_by_id.get(entry["parent_id"], entry["parent_id"])
        for entry in listed.json()
    }


def _unreachable(parents: dict[str, str | None]) -> set[str]:
    """Folders whose parent chain never reaches the top level."""
    stuck = set()
    for name in parents:
        current, steps = name, 0
        while current is not None and steps <= len(parents):
            current, steps = parents.get(current), steps + 1
        if current is not None:
            stuck.add(name)
    return stuck


def _write_parents(instance, parent_by_folder_id: dict[str, str | None]) -> None:
    """Write parent ids straight into the database, the state data from before the fix is in."""
    rows = [{"parent": parent, "folder": folder} for folder, parent in parent_by_folder_id.items()]
    write_rows(instance, "UPDATE folder SET parent_id = :parent WHERE id = :folder", rows)


@pytest.fixture
def owner(make_user):
    return make_user()


@pytest.fixture
def tree(owner):
    """projects > 2026 > q1, plus archive at the top level."""
    with owner.client() as client:
        projects = _create_folder(client, "projects")
        year = _create_folder(client, "2026", parent_id=projects)
        quarter = _create_folder(client, "q1", parent_id=year)
        archive = _create_folder(client, "archive")
    return {"projects": projects, "2026": year, "q1": quarter, "archive": archive}


HEALTHY_TREE = {"projects": None, "2026": "projects", "q1": "2026", "archive": None}


# narrow: a move into the folder's own subtree is refused and changes nothing


@pytest.mark.parametrize("target", ["2026", "q1", "projects"], ids=["child", "grandchild", "self"])
def test_moving_a_folder_into_its_own_subtree_is_refused(owner, tree, target):
    with owner.client() as client:
        moved = _move(client, tree["projects"], tree[target])
        parents = _listed_parents(client)

    assert moved.status_code == 400, (
        f"moving 'projects' under '{target}' in its own subtree answered HTTP "
        f"{moved.status_code}; the parent chain then loops and the folder vanishes (#28748)"
    )
    detail = moved.json()["detail"].lower()
    assert "itself" in detail or "subfolder" in detail, (
        f"the refusal reads {moved.json()['detail']!r}, which does not say the target is in the "
        "folder's own subtree (#28748)"
    )
    assert parents == HEALTHY_TREE, f"the refused move still changed the tree: {parents}"


# narrow: a loop already in the database is broken on the next listing


@pytest.mark.parametrize(
    "loop_closer", ["2026", "q1"], ids=["two-folder-loop", "three-folder-loop"]
)
def test_listing_brings_a_looping_folder_back_to_the_top_level(owner, tree, instance, loop_closer):
    _write_parents(instance, {tree["projects"]: tree[loop_closer]})

    with owner.client() as client:
        parents = _listed_parents(client)
        stored = {
            name: client.get(f"/api/v1/folders/{tree[name]}").json()["parent_id"]
            for name in ("projects", "2026", "q1")
        }

    assert _unreachable(parents) == set(), (
        f"folders in a parent loop were listed without a way to the top level: {parents}; "
        "the sidebar cannot show them and the user cannot get them back (#28748)"
    )
    assert None in stored.values(), (
        f"the listing did not save the repair, the loop is still in the database: {stored} (#28748)"
    )
    assert parents["archive"] is None


def test_listing_repair_keeps_the_healthy_child_below_a_loop(owner, tree, instance):
    _write_parents(instance, {tree["projects"]: tree["2026"]})

    with owner.client() as client:
        parents = _listed_parents(client)

    assert parents["q1"] == "2026", (
        f"a healthy child below the loop was moved as well: {parents}; only the loop should be "
        "broken (#28748)"
    )


# nearby: ordinary moves and the existing repairs still work


@pytest.mark.parametrize(
    ("mover", "target"),
    [("q1", "archive"), ("q1", None), ("q1", "projects"), ("projects", "archive")],
    ids=["to-unrelated", "to-top-level", "under-grandparent", "subtree-to-unrelated"],
)
def test_moves_outside_the_folders_own_subtree_still_work(owner, tree, mover, target):
    with owner.client() as client:
        moved = _move(client, tree[mover], tree.get(target))
        parents = _listed_parents(client)

    assert moved.status_code == 200, (
        f"moving '{mover}' under {target!r} is not a loop but was refused: {moved.text}"
    )
    assert parents == {**HEALTHY_TREE, mover: target}


def test_listing_a_healthy_tree_changes_nothing(owner, tree):
    with owner.client() as client:
        assert _listed_parents(client) == HEALTHY_TREE
        assert _listed_parents(client) == HEALTHY_TREE


def test_listing_still_recovers_a_folder_whose_parent_is_gone(owner, tree, instance):
    _write_parents(instance, {tree["q1"]: "deleted-folder"})

    with owner.client() as client:
        parents = _listed_parents(client)

    assert parents["q1"] is None, f"a folder with a missing parent stayed unreachable: {parents}"


def test_access_granted_on_an_ancestor_still_reaches_a_nested_folder(owner, tree, make_user):
    reader = make_user()
    grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with owner.client() as client:
        shared = client.post(
            f"/api/v1/folders/{tree['projects']}/access/update", json={"access_grants": [grant]}
        )
    assert shared.status_code == 200, shared.text

    with reader.client() as client:
        nested = client.get(f"/api/v1/folders/{tree['q1']}")
        unrelated = client.get(f"/api/v1/folders/{tree['archive']}")

    assert nested.status_code == 200, (
        f"a read grant two levels up no longer reaches the nested folder: HTTP {nested.status_code}"
    )
    assert unrelated.status_code == 404


# Every walk over a looping tree, on a throwaway instance a spinning walk cannot take down.

WALK_TIME_LIMIT = 15.0


@pytest.fixture(scope="module")
def throwaway(instance_with):
    return instance_with({"REGRESSION_THROWAWAY_INSTANCE": "folder-parent-loops"})


@pytest.fixture
def loop(throwaway):
    """Folders 'a' and 'b', each the other's parent, with a chat in each; and their owner."""
    owner = create_user(throwaway)
    with owner.client() as client:
        a = _create_folder(client, "a")
        b = _create_folder(client, "b", parent_id=a)
        chats = {
            folder_id: client.post(
                "/api/v1/chats/new", json={"chat": {"title": f"in {name}"}, "folder_id": folder_id}
            ).json()["id"]
            for name, folder_id in (("a", a), ("b", b))
        }
    _write_parents(throwaway, {a: b})
    return {"owner": owner, "a": a, "b": b, "chats": chats}


def _within_the_limit(send, what: str) -> httpx.Response:
    try:
        return send(timeout=WALK_TIME_LIMIT)
    except httpx.TimeoutException:
        pytest.fail(
            f"{what} gave no answer within {WALK_TIME_LIMIT:g}s over a two-folder parent loop: "
            "the walk follows the loop and never stops (#28748)"
        )


@pytest.mark.slow
def test_opening_a_looping_folder_as_a_stranger_is_refused_in_time(throwaway, loop):
    with create_user(throwaway).client() as client:
        response = _within_the_limit(
            lambda **limit: client.get(f"/api/v1/folders/{loop['a']}", **limit),
            "a stranger opening the folder",
        )

    assert response.status_code == 404, response.text


@pytest.mark.slow
def test_a_looping_folders_chat_list_reaches_both_folders(loop):
    with loop["owner"].client() as client:
        response = _within_the_limit(
            lambda **limit: client.get(f"/api/v1/chats/folder/{loop['a']}", **limit),
            "listing the folder's chats",
        )

    assert response.status_code == 200, response.text
    assert {chat["id"] for chat in response.json()} == set(loop["chats"].values()), (
        "the chat list of a folder in a parent loop lost the other folder's chat: the child walk "
        "followed the loop until it failed (#28748)"
    )


@pytest.mark.slow
def test_marking_a_looping_folder_read_covers_both_folders(loop):
    with loop["owner"].client() as client:
        response = _within_the_limit(
            lambda **limit: client.post(f"/api/v1/folders/{loop['a']}/read", **limit),
            "marking the folder read",
        )

    assert response.status_code == 200, response.text
    assert set(response.json()["folder_ids"]) == {loop["a"], loop["b"]}


@pytest.mark.slow
def test_deleting_a_looping_folder_deletes_both_folders(loop):
    with loop["owner"].client() as client:
        response = _within_the_limit(
            lambda **limit: client.delete(f"/api/v1/folders/{loop['a']}", **limit),
            "deleting the folder",
        )
        leftovers = {
            folder_id: client.get(f"/api/v1/folders/{folder_id}").status_code
            for folder_id in (loop["a"], loop["b"])
        }

    assert response.status_code == 200, response.text
    assert leftovers == {loop["a"]: 404, loop["b"]: 404}, (
        f"deleting a folder in a parent loop left folders behind: {leftovers} (#28748)"
    )


@pytest.mark.slow
def test_every_walk_still_reaches_a_whole_acyclic_tree(throwaway):
    owner = create_user(throwaway)
    with owner.client() as client:
        top = _create_folder(client, "top")
        middle = _create_folder(client, "middle", parent_id=top)
        bottom = _create_folder(client, "bottom", parent_id=middle)
        chat_in_bottom = client.post(
            "/api/v1/chats/new", json={"chat": {"title": "deep"}, "folder_id": bottom}
        ).json()["id"]

        listed_chats = client.get(f"/api/v1/chats/folder/{top}").json()
        marked = client.post(f"/api/v1/folders/{top}/read").json()
        deleted = client.delete(f"/api/v1/folders/{top}")
        bottom_after = client.get(f"/api/v1/folders/{bottom}")

    assert [chat["id"] for chat in listed_chats] == [chat_in_bottom]
    assert set(marked["folder_ids"]) == {top, middle, bottom}
    assert deleted.status_code == 200, deleted.text
    assert bottom_after.status_code == 404
