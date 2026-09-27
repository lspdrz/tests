"""Regression: an image prompt sent to ComfyUI stays out of the server log.

open-webui 0.10.2 fix `64b92ff08` (#26400). `comfyui_create_image` and `comfyui_edit_image`
logged the whole workflow at INFO, the default level, and the workflow carries the user's prompt,
so every prompt landed in operator-visible logs. The fix logs it at DEBUG.

Twin of unit/security/test_image_log_privacy.py, whose source audit only caught the f-string
form; this reads the log a running instance writes, so any INFO line carrying the prompt fails.
ComfyUI is the local stand-in from harness/comfyui.py.

Discriminates: passes on dev `bbfa876af`; with either workflow log line back at INFO (in the
`%s` form dev now uses) the prompt is in the log and the matching test fails.
"""

from __future__ import annotations

import base64
import json
import uuid
from typing import Iterator

import pytest

from harness.comfyui import PNG, FakeComfyUI, comfyui_settings, serving, workflow

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

IMAGES_CONFIG = ("/api/v1/images/config", "/api/v1/images/config/update")


@pytest.fixture
def comfyui(admin, preserve) -> Iterator[FakeComfyUI]:
    """ComfyUI as the image engine for generation and editing, restored afterwards."""
    fake = FakeComfyUI()
    preserve(IMAGES_CONFIG)
    with serving(fake) as base_url, admin.client() as client:
        current = client.get(IMAGES_CONFIG[0]).json()
        engine = comfyui_settings(base_url, workflow())
        updated = client.post(IMAGES_CONFIG[1], json={**current, **engine})
        assert updated.status_code == 200, updated.text
        yield fake


def _png_data_url() -> str:
    return f"data:image/png;base64,{base64.b64encode(PNG).decode()}"


REQUESTS = {
    "generate": lambda prompt: ("/api/v1/images/generations", {"prompt": prompt}),
    "edit": lambda prompt: ("/api/v1/images/edit", {"prompt": prompt, "image": _png_data_url()}),
}


@pytest.mark.parametrize("action", REQUESTS)
def test_the_prompt_sent_to_comfyui_stays_out_of_the_log(instance, admin, comfyui, action):
    prompt = f"a lighthouse at dusk {uuid.uuid4().hex}"
    route, payload = REQUESTS[action](prompt)
    offset = instance.log_size()

    with admin.client() as client:
        response = client.post(route, json=payload)

    assert response.status_code == 200, response.text
    assert prompt in json.dumps(comfyui.queued), "the workflow ComfyUI got lacks the prompt"
    logged = instance.log_since(offset)
    assert logged, "nothing was logged at all, so the log level hides what this checks"
    assert prompt not in logged, (
        f"the {action} prompt was written to the server log at the default level, so every "
        "image prompt is readable by whoever reads the logs (#26400)"
    )
