"""Regression: a ComfyUI workflow ending in "Save Image (Advanced)" produced no image.

Fix f7294161f (open-webui/open-webui#30420, issue open-webui/open-webui#30404). Open WebUI only
collected the images of output nodes of class `SaveImage` and `PreviewImage`, so a workflow whose
output is the core `SaveImageAdvanced` node, such as the Qwen Image Edit template, finished in
ComfyUI and came back empty for both generation and editing. That class is now collected too.
ComfyUI is the local stand-in from harness/comfyui.py.

Discriminates: passes on dev efe63bd34, fails with f7294161f reverted (generation and editing
through the advanced node answer an empty list).
"""

from __future__ import annotations

import base64

import pytest

from harness.comfyui import PNG, FakeComfyUI, comfyui_settings, serving, workflow
from harness.image_engines import IMAGES_CONFIG, save_image_settings

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PNG_DATA_URL = f"data:image/png;base64,{base64.b64encode(PNG).decode()}"
REQUESTS = {
    "generate": ("/api/v1/images/generations", {"prompt": "a lighthouse at dusk"}),
    "edit": ("/api/v1/images/edit", {"prompt": "add fog", "image": PNG_DATA_URL}),
}


@pytest.fixture
def comfyui(admin, preserve):
    """`comfyui(save_node_class)` runs a stand-in whose workflow ends in that node."""
    preserve(IMAGES_CONFIG)
    fake = FakeComfyUI()
    with serving(fake) as base_url, admin.client() as client:

        def configure(save_node_class: str) -> FakeComfyUI:
            save_image_settings(client, **comfyui_settings(base_url, workflow(save_node_class)))
            return fake

        yield configure


def images_from(actor, action: str) -> list[dict]:
    route, payload = REQUESTS[action]
    with actor.client() as client:
        answered = client.post(route, json=payload)
    assert answered.status_code == 200, answered.text
    return answered.json()


def stored_bytes(actor, image: dict) -> bytes:
    with actor.client() as client:
        fetched = client.get(image["url"])
    assert fetched.status_code == 200, fetched.text
    return fetched.content


@pytest.mark.parametrize("action", REQUESTS)
def test_an_image_saved_by_save_image_advanced_reaches_the_chat(comfyui, make_user, action):
    comfyui("SaveImageAdvanced")
    painter = make_user()

    images = images_from(painter, action)

    assert len(images) == 1, (
        f"ComfyUI finished the {action} workflow but its Save Image (Advanced) output was "
        f"dropped: {images} (#30404)"
    )
    assert stored_bytes(painter, images[0]) == PNG


@pytest.mark.parametrize("save_node_class", ["SaveImage", "PreviewImage"])
@pytest.mark.parametrize("action", REQUESTS)
def test_the_standard_output_nodes_still_deliver(comfyui, make_user, action, save_node_class):
    comfyui(save_node_class)
    painter = make_user()

    images = images_from(painter, action)

    assert len(images) == 1, images
    assert stored_bytes(painter, images[0]) == PNG


@pytest.mark.parametrize("action", REQUESTS)
def test_a_node_that_is_not_an_output_node_delivers_nothing(comfyui, make_user, action):
    comfyui("VAEDecode")

    assert images_from(make_user(), action) == []
