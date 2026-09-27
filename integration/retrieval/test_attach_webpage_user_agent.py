"""Regression: Attach Webpage fetched the page first without the configured `USER_AGENT`.

Fix 744ce6cbf (open-webui/open-webui#30385, issue open-webui/open-webui#29617). The chat's
Attach Webpage route first fetches the link to tell a web page from a file, and that first
request went out with aiohttp's default agent. Sites that refuse unknown agents, such as
Wikipedia, answered 403 Forbidden and the attachment failed before the web loader, which does
send `USER_AGENT`, ever ran. The page here is a local service that refuses any other agent.

Discriminates: passes on dev efe63bd34, fails with 744ce6cbf reverted (the first request carries
aiohttp's agent, the page answers 403 and the attachment is refused).
"""

from __future__ import annotations

import pytest

from harness.listener import ReceivedRequest, text_answer
from harness.web_retrieval import LOCAL_WEB_FETCH, web_settings_restored

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

AGENT = "HarbourBot/1.0 (ops@example.com)"
PAGE_TEXT = "Ferries to the islands leave from the north pier"


@pytest.fixture(scope="module")
def agent_instance(instance_with):
    return instance_with({**LOCAL_WEB_FETCH, "USER_AGENT": AGENT})


@pytest.fixture
def web_admin(agent_instance):
    with agent_instance.client() as client, web_settings_restored(client):
        yield client


@pytest.fixture
def picky_page(listener) -> str:
    """A page that answers 403 to any agent but the configured one."""

    def answer(request: ReceivedRequest):
        if request.headers.get("User-Agent") != AGENT:
            return text_answer("Forbidden", "text/plain", 403)
        return text_answer(f"<html><body><p>{PAGE_TEXT}</p></body></html>")

    listener.route("GET", "/wiki/Ferries", answer)
    return f"{listener.base_url}/wiki/Ferries"


def attach_webpage(client, url: str):
    return client.post("/api/v1/retrieval/process/url", json={"url": url})


def test_a_page_that_blocks_unknown_agents_is_attached(web_admin, picky_page):
    attached = attach_webpage(web_admin, picky_page)

    assert attached.status_code == 200, (
        f"the page refused the first request's agent, so attaching failed (#29617): {attached.text}"
    )
    assert PAGE_TEXT in attached.text


def test_the_first_and_the_loading_request_carry_the_configured_agent(
    web_admin, picky_page, listener
):
    attach_webpage(web_admin, picky_page)

    agents = [
        request.headers.get("User-Agent") for request in listener.requests_to("/wiki/Ferries")
    ]
    assert agents, "the page was never fetched"
    assert agents[0] == AGENT, f"the first request went out as {agents[0]!r} (#29617)"
    assert agents[-1] == AGENT, f"the web loader fetched the page as {agents[-1]!r}"


def test_a_page_that_refuses_everyone_is_named_in_the_error(web_admin, listener):
    listener.route("GET", "/private", text_answer("Forbidden", "text/plain", 403))
    link = f"{listener.base_url}/private"

    refused = attach_webpage(web_admin, link)

    assert refused.status_code == 400, refused.text
    assert link in refused.json()["detail"], refused.json()["detail"]
