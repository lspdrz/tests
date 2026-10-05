"""Regression: Manage Ollama's "Upload a GGUF model" never produced a model, in either mode.

Issue open-webui/open-webui#31861, fix PR open-webui/open-webui#31862. The admin opens a
connection's Manage dialog (Admin > Settings > Connections), shows the experimental upload section
and picks a GGUF file (File Mode) or types its URL (URL Mode). The dialog now shows "Model created
successfully!" and the model appears in the model selector; the server creates it, so the dialog no
longer follows up with a create of its own and the Modelfile Content box that call used is gone. A
create Ollama refuses shows its error.

The Ollama stand-in adds a model only for a create that is read to the end (or asked not to stream),
as Ollama does. The file URL is served by `harness/model_hub.py` on an instance of its own. Twin of
integration/models/test_ollama_gguf_upload.py.

Discriminates: passes on dev b859124f9; with 1b2ceedd6 reverted in a backend copy, the File Mode and
URL Mode tests fail (no model on the server, no success toast); in a frontend build with its dialog
half reverted, every test here fails (no success toast, no error toast and the Modelfile Content box
shows).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import admin_of
from harness.listener import json_answer
from harness.model_hub import model_hub_env, serving_model_hub
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, serve_ollama
from utils.manage_ollama import open_manage_ollama
from utils.model_selector import model_options
from utils.tooltips import tooltip_button

pytestmark = [
    pytest.mark.regression,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

FILE_CONTENT = b"GGUF" + bytes(range(256)) * 100
DOWNLOAD_PATH = "/ggml-org/tiny-gguf/resolve/main/tiny-url.gguf"
CREATED = "Model created successfully!"


@pytest.fixture(scope="module")
def hub():
    with serving_model_hub() as serving:
        serving.files[DOWNLOAD_PATH] = FILE_CONTENT
        yield serving


@pytest.fixture
def managed(instance_with, hub, preserve, listener):
    instance = instance_with(model_hub_env(hub))
    if not instance.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    preserve(OLLAMA_CONFIG, on=instance)
    server = serve_ollama(listener)
    admin = admin_of(instance)
    with admin.client() as client:
        connect_ollama(client, listener)
    return admin, server


def open_upload_section(page: Page, url: str) -> Locator:
    dialog = open_manage_ollama(page, url)
    dialog.get_by_role("button", name="Show").click()
    expect(dialog.get_by_text("Upload a GGUF model")).to_be_visible()
    return dialog


def choose_file(page: Page, dialog: Locator, name: str) -> None:
    with page.expect_file_chooser() as chooser:
        dialog.get_by_role("button", name="Click here to select").click()
    chooser.value.set_files(
        {"name": name, "mimeType": "application/octet-stream", "buffer": FILE_CONTENT}
    )
    expect(dialog.get_by_role("button", name=name)).to_be_visible()


def upload(dialog: Locator) -> None:
    tooltip_button(dialog, "Upload Model").click()


def test_an_uploaded_gguf_file_creates_the_model(page_for, managed):
    admin, server = managed
    page = page_for(admin)
    dialog = open_upload_section(page, server.listener.base_url)

    choose_file(page, dialog, "tiny-file.gguf")
    upload(dialog)

    expect(page.get_by_text(CREATED)).to_be_visible()
    expect(dialog.get_by_text("Modelfile Content")).to_have_count(0)
    assert "tiny-file:latest" in server.models
    page.goto("/")
    expect(model_options(page, "tiny-file:latest")).to_have_count(1)


def test_a_gguf_downloaded_by_url_creates_the_model(page_for, managed, hub):
    admin, server = managed
    page = page_for(admin)
    dialog = open_upload_section(page, server.listener.base_url)

    dialog.get_by_role("button", name="File Mode").click()
    dialog.get_by_placeholder("Type Hugging Face Resolve (Download) URL").fill(
        hub.url(DOWNLOAD_PATH)
    )
    upload(dialog)

    expect(page.get_by_text(CREATED)).to_be_visible()
    expect(dialog.get_by_text("Modelfile Content")).to_have_count(0)
    assert "tiny-url:latest" in server.models
    page.goto("/")
    expect(model_options(page, "tiny-url:latest")).to_have_count(1)


def test_a_create_ollama_refuses_shows_its_error(page_for, managed):
    admin, server = managed
    server.listener.route("POST", "/api/create", json_answer({"error": "bad GGUF"}, status=500))
    page = page_for(admin)
    dialog = open_upload_section(page, server.listener.base_url)

    choose_file(page, dialog, "broken.gguf")
    upload(dialog)

    expect(page.get_by_text("Failed to create model in Ollama.")).to_be_visible()
    expect(page.get_by_text(CREATED)).to_have_count(0)
    assert "broken:latest" not in server.models
