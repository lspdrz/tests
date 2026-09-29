"""A local OpenAI-shaped provider a browser can call directly, for a user's own connection.

A direct connection is fetched by the page itself, so the provider must answer the browser's
CORS preflight as well as the calls. `serve(listener, model_id)` routes `/v1/models` and
`/v1/chat/completions` on a `listener` with the headers a cross-origin page needs;
`provider.reply_with(text)` sets what the next chats answer. `provider.models_requests()` and
`provider.chat_requests()` are what the browser sent, headers included.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from harness.listener import Answer, Listener, ReceivedRequest, json_answer, text_answer

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "authorization, content-type",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}


@dataclass
class BrowserProvider:
    listener: Listener
    model_id: str
    text: str = "hello from the browser side"

    @property
    def base_url(self) -> str:
        return f"{self.listener.base_url}/v1"

    def reply_with(self, text: str) -> None:
        self.text = text

    def models_requests(self) -> list[ReceivedRequest]:
        return [r for r in self.listener.requests_to("/v1/models") if r.method == "GET"]

    def chat_requests(self) -> list[ReceivedRequest]:
        return [r for r in self.listener.requests_to("/v1/chat/completions") if r.method == "POST"]

    def _models(self, _request: ReceivedRequest) -> Answer:
        status, headers, body = json_answer({"object": "list", "data": [{"id": self.model_id}]})
        return status, {**headers, **CORS}, body

    def _chat(self, _request: ReceivedRequest) -> Answer:
        delta = {"index": 0, "delta": {"role": "assistant", "content": self.text}}
        stop = {"index": 0, "delta": {}, "finish_reason": "stop"}
        lines = [
            f"data: {json.dumps({'object': 'chat.completion.chunk', 'choices': [choice]})}\n\n"
            for choice in (delta, stop)
        ]
        status, headers, body = text_answer(
            "".join(lines) + "data: [DONE]\n\n", content_type="text/event-stream"
        )
        return status, {**headers, **CORS}, body


def _preflight(_request: ReceivedRequest) -> Answer:
    return 204, CORS, b""


def serve(listener: Listener, model_id: str) -> BrowserProvider:
    provider = BrowserProvider(listener, model_id)
    listener.route("GET", "/v1/models", provider._models)
    listener.route("POST", "/v1/chat/completions", provider._chat)
    for path in ("/v1/models", "/v1/chat/completions"):
        listener.route("OPTIONS", path, _preflight)
    return provider
