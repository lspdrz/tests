"""Journey: models, prompts, tools and skills leave the workspace as a JSON file and return.

A fresh admin owns one of each. Exporting from a workspace list downloads a file that holds the
item; after the item is deleted, importing that file from the same list brings it back, with the
tool asking for a confirmation first. A file that is not JSON is met with an error message and
adds nothing.

Discriminates: passes on dev 176d31d1d apart from the prompts and tools cases of the bad file
test, which fail there on purpose: both lists parse the file without a guard, so a file that is
not JSON is dropped with no message (not yet reported upstream). In a frontend copy with each
import loop emptied, the models export saving an empty list and the models and skills bad-file
messages removed, all eight tests fail.
"""

from __future__ import annotations

import json
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TOOL_SOURCE = "class Tools:\n    def ping(self) -> str:\n        return 'pong'\n"


@pytest.fixture
def keeper(make_user):
    """A fresh admin; what it owns in the workspace is deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for model in client.get("/api/v1/models/list").json().get("items", []):
            if model["user_id"] == account.id:
                client.post("/api/v1/models/model/delete", json={"id": model["id"]})
        for prompt in client.get("/api/v1/prompts/").json():
            if prompt["user_id"] == account.id:
                client.delete(f"/api/v1/prompts/id/{prompt['id']}/delete")
        for tool in client.get("/api/v1/tools/").json():
            if tool["user_id"] == account.id:
                client.delete(f"/api/v1/tools/id/{tool['id']}/delete")
        for skill in client.get("/api/v1/skills/").json():
            if skill["user_id"] == account.id:
                client.delete(f"/api/v1/skills/id/{skill['id']}/delete")


def _create(account: Actor, path: str, form: dict) -> dict:
    with account.client() as client:
        created = client.post(path, json=form)
    assert created.status_code == 200, created.text
    return created.json()


def _remove(account: Actor, path: str, **options) -> None:
    with account.client() as client:
        removed = client.request(options.pop("method", "DELETE"), path, **options)
    assert removed.status_code == 200, removed.text


def _choose_action(page: Page, section: str, action: str) -> None:
    page.goto(f"/workspace/{section}")
    page.get_by_label("Open create menu").click()
    page.get_by_role("button", name=action).click()


def _export(page: Page, section: str) -> list[dict]:
    with page.expect_download() as download:
        _choose_action(page, section, "Export JSON")
    with open(download.value.path()) as saved:
        return json.load(saved)


def _import(page: Page, section: str, content: str) -> None:
    with page.expect_file_chooser() as chooser:
        _choose_action(page, section, "Import JSON")
    chooser.value.set_files(
        files=[{"name": "import.json", "mimeType": "application/json", "buffer": content.encode()}]
    )


def _find_in_list(page: Page, label: str, name: str) -> None:
    page.get_by_role("textbox", name=label).fill(name)
    expect(page.get_by_text(name, exact=True)).to_be_visible()


def test_an_exported_model_is_restored_by_importing_the_file(page_for, keeper):
    name = f"Exported {uuid.uuid4().hex[:8]}"
    model_id = name.lower().replace(" ", "-")
    form = {"id": model_id, "base_model_id": MOCK_MODEL_ID, "name": name, "meta": {}, "params": {}}
    _create(keeper, "/api/v1/models/create", form)
    page = page_for(keeper)

    saved = [entry for entry in _export(page, "models") if entry["id"] == model_id]
    assert len(saved) == 1, "the exported file does not hold the model"
    _remove(keeper, "/api/v1/models/model/delete", method="POST", json={"id": model_id})
    _import(page, "models", json.dumps(saved))

    _find_in_list(page, "Search Models", name)


def test_an_exported_prompt_is_restored_by_importing_the_file(page_for, keeper):
    suffix = uuid.uuid4().hex[:8]
    name, command = f"Exported prompt {suffix}", f"exported{suffix}"
    prompt = _create(
        keeper,
        "/api/v1/prompts/create",
        {"command": command, "name": name, "content": f"Say hello, take {suffix}."},
    )
    page = page_for(keeper)

    saved = [entry for entry in _export(page, "prompts") if entry["command"] == command]
    assert len(saved) == 1, "the exported file does not hold the prompt"
    _remove(keeper, f"/api/v1/prompts/id/{prompt['id']}/delete")
    _import(page, "prompts", json.dumps(saved))

    _find_in_list(page, "Search Prompts", name)


def test_an_exported_tool_is_restored_by_importing_the_file(page_for, keeper):
    suffix = uuid.uuid4().hex[:8]
    name, tool_id = f"Exported tool {suffix}", f"exported_tool_{suffix}"
    form = {"id": tool_id, "name": name, "content": TOOL_SOURCE, "meta": {"description": "pings"}}
    _create(keeper, "/api/v1/tools/create", form)
    page = page_for(keeper)

    saved = [entry for entry in _export(page, "tools") if entry["id"] == tool_id]
    assert len(saved) == 1, "the exported file does not hold the tool"
    _remove(keeper, f"/api/v1/tools/id/{tool_id}/delete")
    _import(page, "tools", json.dumps(saved))
    page.get_by_role("dialog", name="Confirm your action").get_by_role(
        "button", name="Confirm"
    ).click()

    expect(page.get_by_text("Tool imported successfully")).to_be_visible()
    _find_in_list(page, "Search Tools", name)


def test_an_exported_skill_is_restored_by_importing_the_file(page_for, keeper):
    suffix = uuid.uuid4().hex[:8]
    name, skill_id = f"Exported skill {suffix}", f"exported-skill-{suffix}"
    form = {"id": skill_id, "name": name, "content": "Be brief.", "meta": {}, "is_active": True}
    _create(keeper, "/api/v1/skills/create", form)
    page = page_for(keeper)

    saved = [entry for entry in _export(page, "skills") if entry["id"] == skill_id]
    assert len(saved) == 1, "the exported file does not hold the skill"
    _remove(keeper, f"/api/v1/skills/id/{skill_id}/delete")
    _import(page, "skills", json.dumps(saved))

    expect(page.get_by_text("Skill imported successfully")).to_be_visible()
    _find_in_list(page, "Search Skills", name)


# ---------------------------------------------------------------- nearby


@pytest.mark.parametrize("section", ["models", "prompts", "tools", "skills"])
def test_a_file_that_is_not_json_shows_an_error(page_for, keeper, section):
    page = page_for(keeper)

    _import(page, section, "this is not json {")
    if section == "tools":
        page.get_by_role("dialog", name="Confirm your action").get_by_role(
            "button", name="Confirm"
        ).click()

    # prompts and tools parse without a guard, so their file is dropped without a message
    expect(page.locator("[data-sonner-toast][data-type='error']")).to_be_visible()
