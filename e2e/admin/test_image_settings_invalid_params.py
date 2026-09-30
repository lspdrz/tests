"""Regression: invalid JSON in the image Additional Parameters left Save spinning with no message.

open-webui issue #31381, fixed by PR #31382: in Admin Settings > Images, saving with text that is
no JSON object in the Additional Parameters of the OpenAI engine or of AUTOMATIC1111 kept the
Save button spinning and showed nothing. Save now reports "Invalid JSON format for Parameters"
and stays usable, so the parameters can be corrected and saved.

Discriminates: passes on the dev a5bc78300 build; with the check for the two parameter fields
missing from the save handler the message never shows and Save stays disabled.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.image_engines import IMAGES_CONFIG, save_image_settings, serve_automatic1111

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PARAMS_PLACEHOLDER = "Enter additional parameters in JSON format"
OPENAI_PARAMS = {"quality": "hd"}
AUTOMATIC1111_PARAMS = {"cfg_scale": 7}


def open_images_tab(page: Page) -> Locator:
    page.goto("/admin/settings/images")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("switch", name="Image Generation")).to_be_visible(timeout=20_000)
    return settings


def save_refused_then_fixed(page: Page, settings: Locator, fixed: dict) -> None:
    params = settings.get_by_placeholder(PARAMS_PLACEHOLDER)
    save = settings.get_by_role("button", name="Save", exact=True)
    # the tab rewrites the field once its settings load, so fill only after that
    expect(params).to_have_value("{}")
    params.fill("{quality: hd")
    save.click()
    expect(
        page.get_by_text("Invalid JSON format for Parameters"),
        "#31381: no message for the invalid parameters",
    ).to_be_visible()
    expect(save, "#31381: Save stayed disabled after the error").to_be_enabled()
    params.fill(json.dumps(fixed))
    save.click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def saved_settings(admin) -> dict:
    with admin.client() as client:
        return client.get(IMAGES_CONFIG[0]).json()


def test_invalid_openai_parameters_are_reported_and_save_works_once_fixed(
    page_for, make_user, admin, preserve
):
    preserve(IMAGES_CONFIG)
    with admin.client() as client:
        save_image_settings(
            client,
            ENABLE_IMAGE_GENERATION=True,
            IMAGE_GENERATION_ENGINE="openai",
            IMAGE_GENERATION_MODEL="gpt-image-1",
            IMAGES_OPENAI_API_KEY="sk-image-test",
            IMAGES_OPENAI_API_BASE_URL="http://127.0.0.1:9/v1",
        )
    page = page_for(make_user(role="admin"))
    settings = open_images_tab(page)

    save_refused_then_fixed(page, settings, OPENAI_PARAMS)

    assert saved_settings(admin)["IMAGES_OPENAI_API_PARAMS"] == OPENAI_PARAMS


def test_invalid_automatic1111_parameters_are_reported_and_save_works_once_fixed(
    page_for, make_user, admin, preserve, listener
):
    preserve(IMAGES_CONFIG)
    with admin.client() as client:
        save_image_settings(client, **serve_automatic1111(listener))
    page = page_for(make_user(role="admin"))
    settings = open_images_tab(page)

    save_refused_then_fixed(page, settings, AUTOMATIC1111_PARAMS)

    assert saved_settings(admin)["AUTOMATIC1111_PARAMS"] == AUTOMATIC1111_PARAMS
