"""Editing an arena model in Settings > Models dropped its access and its model pool, #29564.

Fix commit `b8de508db` (open-webui/open-webui#31309). Saving an arena model on the admin models
page (to give it tools, say) stores a model row under the arena id. That row's metadata replaced
the arena model's wholesale, so the access grants, `model_ids` and `filter_mode` set under
Settings > Evaluations were gone: non-admin users lost the arena model even when it was public,
and its chats picked from every model instead of the pool. The override now keeps those three
keys from the evaluation config and applies everything else it sets.

Discriminates: passes on dev efe63bd34; with b8de508db reverted the edited arena model lists
without its grants and pool, the user no longer sees it or may chat with it, and its chats leave
the pool; the unedited arena model passes on both.
"""

from __future__ import annotations

import uuid
from typing import Iterator

import httpx
import pytest

from harness import upstream as reply
from harness.actors import Actor
from harness.chat import ask
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

EVALUATION_CONFIG = ("/api/v1/evaluations/config", "/api/v1/evaluations/config")
PUBLIC_READ = [{"principal_type": "user", "principal_id": "*", "permission": "read"}]
POOL_SYSTEM_PROMPT = "You are the only model in the arena pool."


@pytest.fixture
def pool_model(admin) -> Iterator[str]:
    """A preset readable by everyone, whose system prompt marks requests routed to it."""
    model_id = f"arena-pool-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": model_id,
                "meta": {},
                "params": {"system": POOL_SYSTEM_PROMPT},
                "access_grants": PUBLIC_READ,
            },
        )
        assert created.status_code == 200, created.text
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def arena_model(admin, preserve, pool_model) -> Iterator[str]:
    """A public arena model whose pool is `pool_model` alone."""
    preserve(EVALUATION_CONFIG)
    arena_id = f"arena-{uuid.uuid4().hex[:8]}"
    arena = {
        "id": arena_id,
        "name": f"Arena {arena_id}",
        "meta": {"model_ids": [pool_model], "filter_mode": "include", "access_grants": PUBLIC_READ},
    }
    with admin.client() as client:
        saved = client.post(
            "/api/v1/evaluations/config",
            json={"ENABLE_EVALUATION_ARENA_MODELS": True, "EVALUATION_ARENA_MODELS": [arena]},
        )
        assert saved.status_code == 200, saved.text
        yield arena_id
        client.post("/api/v1/models/model/delete", json={"id": arena_id})


def _edit_in_settings_models(admin: Actor, arena_id: str) -> None:
    """What saving the arena model on the admin models page stores."""
    with admin.client() as client:
        saved = client.post(
            "/api/v1/models/create",
            json={
                "id": arena_id,
                "base_model_id": None,
                "name": f"Arena {arena_id}",
                "meta": {"description": "Edited on the models page", "toolIds": []},
                "params": {},
            },
        )
    assert saved.status_code == 200, saved.text


def _listed(client: httpx.Client, model_id: str) -> dict | None:
    models = client.get("/api/models", params={"refresh": True})
    assert models.status_code == 200, models.text
    return next((model for model in models.json()["data"] if model["id"] == model_id), None)


# ---------------------------------------------------------------- narrow


def test_an_edited_arena_model_keeps_its_access_and_pool(admin, arena_model, pool_model):
    _edit_in_settings_models(admin, arena_model)

    with admin.client() as client:
        listed = _listed(client, arena_model)

    assert listed is not None
    meta = listed["info"]["meta"]
    assert meta.get("model_ids") == [pool_model], "the edit dropped the arena pool (#29564)"
    assert meta.get("filter_mode") == "include"
    assert meta.get("access_grants"), "the edit dropped the arena access grants (#29564)"
    assert meta.get("description") == "Edited on the models page", "the edit itself was lost"


def test_a_user_still_sees_and_chats_with_an_edited_public_arena_model(
    admin, make_user, upstream, arena_model
):
    _edit_in_settings_models(admin, arena_model)
    member = make_user()
    upstream.queue(reply.text("from the arena"))

    with member.client() as client:
        assert _listed(client, arena_model) is not None, "the user lost the arena model (#29564)"
        _, answer = ask(client, "hello arena", model=arena_model)

    assert answer["content"] == "from the arena"


def test_chats_on_an_edited_arena_model_stay_in_its_pool(admin, upstream, arena_model, pool_model):
    _edit_in_settings_models(admin, arena_model)
    upstream.reset()

    picked = set()
    with admin.client() as client:
        _listed(client, arena_model)
        for turn in range(6):
            _, answer = ask(client, f"arena round {turn}", model=arena_model)
            picked.add(answer.get("selectedModelId"))

    assert picked == {pool_model}, "the arena picked outside its configured pool (#29564)"
    system_prompts = [
        message["content"]
        for request in upstream.chat_requests()
        for message in request["messages"]
        if message["role"] == "system"
    ]
    assert system_prompts.count(POOL_SYSTEM_PROMPT) == 6


# ---------------------------------------------------------------- nearby


def test_an_unedited_arena_model_is_listed_for_a_user(make_user, arena_model, pool_model):
    member = make_user()
    with member.client() as client:
        listed = _listed(client, arena_model)

    assert listed is not None
    assert listed["info"]["meta"]["model_ids"] == [pool_model]


def test_a_private_arena_model_stays_hidden_after_an_edit(admin, make_user, preserve):
    preserve(EVALUATION_CONFIG)
    arena_id = f"arena-private-{uuid.uuid4().hex[:8]}"
    arena = {"id": arena_id, "name": arena_id, "meta": {"model_ids": None, "access_grants": []}}
    with admin.client() as client:
        saved = client.post(
            "/api/v1/evaluations/config",
            json={"ENABLE_EVALUATION_ARENA_MODELS": True, "EVALUATION_ARENA_MODELS": [arena]},
        )
        assert saved.status_code == 200, saved.text
    try:
        _edit_in_settings_models(admin, arena_id)
        with make_user().client() as client:
            assert _listed(client, arena_id) is None
    finally:
        with admin.client() as client:
            client.post("/api/v1/models/model/delete", json={"id": arena_id})
