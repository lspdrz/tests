"""Opening the Manage Ollama dialog of one connection, the way an admin does."""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect

from utils.tooltips import tooltip_button

URL_PLACEHOLDER = "Enter URL (e.g. http://localhost:11434)"


def open_manage_ollama(page: Page, url: str) -> Locator:
    """The Manage dialog of the Ollama connection saved with `url`, opened from Connections."""
    page.goto("/admin/settings/connections")
    settings = page.get_by_role("dialog")
    addresses = settings.get_by_placeholder(URL_PLACEHOLDER)
    expect(addresses.first).to_be_visible()
    index = addresses.evaluate_all("(inputs, url) => inputs.findIndex((i) => i.value === url)", url)
    assert index >= 0, f"no connection row reads {url}"
    row = addresses.nth(index).locator("xpath=ancestor::div[.//button][1]")
    tooltip_button(row, "Manage").click()
    dialog = page.get_by_role("dialog").filter(has_text="Pull a model from Ollama.com")
    expect(dialog).to_be_visible()
    return dialog
