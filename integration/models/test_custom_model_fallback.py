"""Regression: a web client chat on a custom model whose base model is gone never fell back.

With `ENABLE_CUSTOM_MODEL_FALLBACK=true` a custom model whose base model no connection serves any
more is answered by the first default model instead of failing (docs: env-configuration,
workspace/models). `chat_completion` rebound the model to that fallback, but a request carrying a
socket session and a chat id (every send from the web client) was then fanned out per entry of
`message_ids`, and each entry still named the custom model. The provider router resolved the
custom model's missing base again and the reply was stored as the error "Model '' was not
found". An API client without a session never reaches the fan-out, and its request did fall
back. PR #31353 (open-webui/open-webui#31345) hands the fallback to each fanned-out entry.

Discriminates: passes on dev 015dbc861; the web client test fails on dev a5bc78300, before PR
#31353 (the send fails with "Model '' was not found"). The API client test passes on both and
fails with the rebind to the fallback model removed (the request fails the same way).
"""

from __future__ import annotations

import uuid

import pytest

from harness import upstream as reply
from harness.chat import send_message, wait_for_reply
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

FALLBACK_ENV = {"ENABLE_CUSTOM_MODEL_FALLBACK": "true", "DEFAULT_MODELS": MOCK_MODEL_ID}


@pytest.fixture(scope="module")
def fallback_instance(instance_with):
    return instance_with(FALLBACK_ENV)


@pytest.fixture
def orphaned_model(fallback_instance) -> str:
    """A custom model on a base model that no connection serves."""
    model_id = f"orphaned-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "name": "Orphaned preset",
        "base_model_id": "retired-base-model",
        "meta": {},
        "params": {},
    }
    with fallback_instance.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        client.get("/api/models").raise_for_status()
    fallback_instance.upstream.reset()
    return model_id


def test_a_web_client_chat_on_the_orphaned_model_is_answered_by_the_default_model(
    fallback_instance, orphaned_model
):
    prompt = "is anyone there?"
    fallback_instance.upstream.queue(
        reply.text("the default model here", match=reply.answering(prompt))
    )
    with fallback_instance.client() as client:
        message = wait_for_reply(client, send_message(client, prompt, model=orphaned_model))

    assert not message.get("error"), (
        f"the web client's chat failed instead of falling back: {message['error']}"
    )
    assert message["content"] == "the default model here"


def test_an_api_client_on_the_orphaned_model_is_answered_by_the_default_model(
    fallback_instance, orphaned_model
):
    prompt = "is anyone there?"
    fallback_instance.upstream.queue(
        reply.text("the default model here", match=reply.answering(prompt))
    )
    with fallback_instance.client() as client:
        answered = client.post(
            "/api/chat/completions",
            json={"model": orphaned_model, "messages": [{"role": "user", "content": prompt}]},
        )

    assert answered.status_code == 200, answered.text
    assert answered.json()["choices"][0]["message"]["content"] == "the default model here"
    assert fallback_instance.upstream.chat_requests()[-1]["model"] == MOCK_MODEL_ID
