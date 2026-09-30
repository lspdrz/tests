"""Regression: a GPT-5-class model behind an Amazon Bedrock prefix was sent `max_tokens`.

Issue open-webui/open-webui#30510, fix PR open-webui/open-webui#30976. On an OpenAI-compatible
connection that is not api.openai.com, a new model (o-series, gpt-5 and later) is sent
`max_completion_tokens` in place of `max_tokens`. Bedrock ids carry a provider prefix
(`us.openai.gpt-6-sol`, `openai.gpt-5`), so the model was not recognised as new and Bedrock
refused the request. Ids that are not new models keep `max_tokens`.

Discriminates: passes on dev a5bc78300, fails with 9d2c3965f reverted (the prefixed ids arrive with
`max_tokens` and no `max_completion_tokens`).
"""

from __future__ import annotations

import pytest

from harness import second_provider
from harness.chat import send_message, wait_for_reply

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

TOKEN_LIMIT = 64
COMPLETIONS = "/v1/chat/completions"


def _chat_with_limit(admin, listener, preserve, model_id: str, **extra) -> dict:
    """The request the provider received for one chat that limits the reply to TOKEN_LIMIT."""
    preserve(second_provider.OPENAI_CONFIG)
    with admin.client() as client:
        second_provider.attach(client, listener, model_id)
        listener.route("POST", COMPLETIONS, second_provider.sse({"content": "ok"}))
        wait_for_reply(client, send_message(client, "hello", model=model_id, **extra))
    return listener.requests_to(COMPLETIONS)[-1].json()


@pytest.mark.parametrize(
    "model_id", ["us.openai.gpt-6-sol", "openai.gpt-5", "global.openai.gpt-5.6"]
)
def test_a_prefixed_new_model_is_sent_max_completion_tokens(admin, listener, preserve, model_id):
    params = {"max_tokens": TOKEN_LIMIT}
    sent = _chat_with_limit(admin, listener, preserve, model_id, params=params)

    assert sent.get("max_completion_tokens") == TOKEN_LIMIT, sent
    assert "max_tokens" not in sent, "Bedrock refuses max_tokens for a GPT-5 class model (#30510)"


@pytest.mark.parametrize("model_id", ["openai.gpt-oss-120b-1:0", "gpt-4o", "us.openai.gpt-4o"])
def test_a_model_that_is_not_new_keeps_max_tokens(admin, listener, preserve, model_id):
    params = {"max_tokens": TOKEN_LIMIT}
    sent = _chat_with_limit(admin, listener, preserve, model_id, params=params)

    assert sent.get("max_tokens") == TOKEN_LIMIT, sent
    assert "max_completion_tokens" not in sent


def test_a_bare_gpt_5_id_is_sent_max_completion_tokens(admin, listener, preserve):
    params = {"max_tokens": TOKEN_LIMIT}
    sent = _chat_with_limit(admin, listener, preserve, "gpt-5", params=params)

    assert sent.get("max_completion_tokens") == TOKEN_LIMIT, sent
    assert "max_tokens" not in sent
