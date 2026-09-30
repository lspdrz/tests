"""Regression: Eject in the model selector failed on a runner behind an untrusted certificate.

Issue open-webui/open-webui#31371, fix PR open-webui/open-webui#31391 (fab58bd35). On an instance
started with `AIOHTTP_CLIENT_SESSION_SSL=false`, chat with a llama.cpp or Ollama connection served
over HTTPS with a self-signed certificate worked, but ejecting its loaded model from the model
selector showed "Error unloading model" with a certificate error. Here the admin ejects a loaded
llama.cpp model from the selector and the model is unloaded.

Twin of integration/models/test_unload_over_https_with_ssl_disabled.py.

Discriminates: passes on dev a5bc78300; fails with fab58bd35 reverted (the selector shows the
unload error and the runner keeps the model).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness import second_provider
from harness.actors import admin_of
from harness.listener import listening
from harness.model_runners import connect_runner, serve_llama_cpp
from utils.chat_ui import chat_input
from utils.model_selector import model_options

pytestmark = [
    pytest.mark.regression,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

MODEL = "tiny-model"


@pytest.fixture
def relaxed(instance_with):
    launched = instance_with({"AIOHTTP_CLIENT_SESSION_SSL": "false"})
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    return launched


def test_eject_unloads_a_llama_cpp_model_served_over_https(relaxed, preserve, page_for):
    with listening(tls=True) as https_listener:
        runner = serve_llama_cpp(https_listener, MODEL)
        runner.loaded.append(MODEL)
        preserve(second_provider.OPENAI_CONFIG, on=relaxed)
        with relaxed.client() as client:
            connect_runner(client, runner)
        page = page_for(admin_of(relaxed))
        expect(chat_input(page)).to_be_visible()

        option = model_options(page, MODEL)
        option.hover()
        option.get_by_role("button", name="Eject model").click()

        expect(page.get_by_text("Model unloaded successfully")).to_be_visible()
        expect(page.get_by_text("Error unloading model")).to_have_count(0)
        assert runner.sent("/models/unload") == [{"model": MODEL}]
