"""Regression: a private arena model with no access grants vanished for every admin, #30013.

Fix 8c34af303 (PR #31858). With `BYPASS_ADMIN_ACCESS_CONTROL` off, an arena model left at its
default access (Private, no grants) was checked like any user's model: nobody holds a grant, so
the model list, the chat request and the profile image refused it to every admin as well, and
nobody on the instance could use it. An arena model without grants is now admin-only, as the docs
describe; arena models with grants and regular users are checked as before.

Discriminates: passes on dev b859124f9, fails with 8c34af303 reverted (the admin's model list lacks
the arena model, a chat with it is refused and its profile image is the default one).
"""

from __future__ import annotations

import base64
import uuid
from typing import Iterator

import httpx
import pytest

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.chat import ask
from harness.instance import LaunchedInstance
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

EVALUATION_CONFIG = ("/api/v1/evaluations/config", "/api/v1/evaluations/config")
# a 1x1 transparent PNG
PROFILE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
PROFILE_IMAGE_URL = "data:image/png;base64," + base64.b64encode(PROFILE_PNG).decode()


@pytest.fixture
def restricted(instance_with) -> LaunchedInstance:
    """An instance where admins are held to access grants like everyone else."""
    return instance_with({"BYPASS_ADMIN_ACCESS_CONTROL": "false"})


def save_arena(restricted: LaunchedInstance, access_grants: list[dict]) -> str:
    arena_id = f"arena-{uuid.uuid4().hex[:8]}"
    arena = {
        "id": arena_id,
        "name": f"Arena {arena_id}",
        "meta": {
            "model_ids": [MOCK_MODEL_ID],
            "access_grants": access_grants,
            "profile_image_url": PROFILE_IMAGE_URL,
        },
    }
    with admin_of(restricted).client() as client:
        saved = client.post(
            "/api/v1/evaluations/config",
            json={"ENABLE_EVALUATION_ARENA_MODELS": True, "EVALUATION_ARENA_MODELS": [arena]},
        )
    assert saved.status_code == 200, saved.text
    return arena_id


@pytest.fixture
def private_arena(restricted, preserve) -> Iterator[str]:
    """An arena model left at its default access: Private, with no grants."""
    preserve(EVALUATION_CONFIG, on=restricted)
    yield save_arena(restricted, access_grants=[])


def listed_ids(client: httpx.Client) -> set[str]:
    models = client.get("/api/models", params={"refresh": True})
    assert models.status_code == 200, models.text
    return {model["id"] for model in models.json()["data"]}


def profile_image(client: httpx.Client, model_id: str) -> httpx.Response:
    return client.get("/api/v1/models/model/profile/image", params={"id": model_id})


# ---------------------------------------------------------------- narrow


def test_an_admin_sees_a_private_arena_model_without_grants(restricted, private_arena):
    with admin_of(restricted).client() as client:
        assert private_arena in listed_ids(client), "the arena model is gone for admins (#30013)"


def test_an_admin_chats_with_a_private_arena_model_without_grants(restricted, private_arena):
    restricted.upstream.queue(reply.text("from the arena", match=reply.answering("hello arena")))
    with admin_of(restricted).client() as client:
        listed_ids(client)
        _, answer = ask(client, "hello arena", model=private_arena)

    assert answer["content"] == "from the arena"


def test_an_admin_gets_the_profile_image_of_a_private_arena_model(restricted, private_arena):
    with admin_of(restricted).client() as client:
        image = profile_image(client, private_arena)

    assert image.status_code == 200, image.text
    assert image.content == PROFILE_PNG, "the admin got the default image (#30013)"


# ---------------------------------------------------------------- nearby


def test_a_user_does_not_see_a_private_arena_model_without_grants(restricted, private_arena):
    with create_user(restricted).client() as client:
        assert private_arena not in listed_ids(client)
        image = profile_image(client, private_arena)

    assert image.content != PROFILE_PNG


def test_a_user_cannot_chat_with_a_private_arena_model_without_grants(restricted, private_arena):
    with create_user(restricted).client() as client:
        listed_ids(client)
        refused = client.post(
            "/api/chat/completions",
            json={
                "model": private_arena,
                "messages": [{"role": "user", "content": "let me in"}],
                "stream": False,
            },
        )

    assert refused.status_code >= 400, refused.text
    assert not restricted.upstream.chat_requests()


def test_an_arena_model_shared_with_a_user_is_theirs_and_not_the_admins(restricted, preserve):
    preserve(EVALUATION_CONFIG, on=restricted)
    member = create_user(restricted)
    grant = {"principal_type": "user", "principal_id": member.id, "permission": "read"}
    arena_id = save_arena(restricted, access_grants=[grant])

    with member.client() as client:
        assert arena_id in listed_ids(client)
    with admin_of(restricted).client() as client:
        assert arena_id not in listed_ids(client)
