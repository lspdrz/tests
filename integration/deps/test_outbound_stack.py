"""Dependency smoke: code and services Open WebUI reaches out to, each through its feature.

The mcp SDK connects to an MCP tool server when an admin verifies the connection and lists its
tools. validators decides which links a page fetch accepts, BeautifulSoup reads the fetched page
(its text, and the title, description and language stored with it; a `.xml` link through its
XML parser), black formats code for the code
editor, and opentelemetry exports request traces to an OTLP/HTTP collector (requests carries
them there, as it carries a document to Tika). The collector's address is only read at boot, so
the tracing test boots an instance of its own.

Discriminates: passes on dev bbfa876af; in a backend copy, skipping `session.initialize()` fails
the MCP verification, dropping the `validators.url` check lets the malformed link through to the
fetch, a loader that stores no page title fails the metadata test, asking BeautifulSoup for an
unknown parser in place of "xml" fails the feed test, returning the code unformatted fails the
formatter, black without string normalisation fails the wrapping test and never adding the span
processor leaves the collector empty.
"""

from __future__ import annotations

import time

import pytest

from harness.instance import free_port
from harness.listener import listening, text_answer
from harness.mcp_server import ECHO_DESCRIPTION, serving_mcp
from harness.web_retrieval import LOCAL_WEB_FETCH

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source]

PAGE_TEXT = "Herons wait motionless in the shallows"
EXPORT_WAIT = 30.0


def _verify_mcp(admin, url: str):
    connection = {
        "type": "mcp",
        "url": url,
        "path": "",
        "auth_type": "none",
        "key": "",
        "config": {},
    }
    with admin.client() as client:
        return client.post("/api/v1/configs/tool_servers/verify", json=connection)


def test_verifying_an_mcp_server_lists_its_tools(admin):
    with serving_mcp() as url:
        verified = _verify_mcp(admin, url)

    assert verified.status_code == 200, verified.text
    [echo] = verified.json()["specs"]
    assert echo["name"] == "echo"
    assert echo["description"] == ECHO_DESCRIPTION
    assert echo["parameters"]["properties"]["text"]["type"] == "string"


def test_verifying_an_mcp_server_that_is_not_there_fails(admin):
    verified = _verify_mcp(admin, f"http://127.0.0.1:{free_port()}/mcp")

    assert verified.status_code == 400, verified.text


@pytest.fixture
def local_fetch(instance_with):
    return instance_with(LOCAL_WEB_FETCH)


def _fetch(instance, url: str):
    with instance.client() as client:
        return client.post("/api/v1/retrieval/process/web?process=false", json={"url": url})


@pytest.mark.slow
def test_a_page_link_is_fetched_and_a_malformed_one_refused_unfetched(local_fetch, listener):
    listener.route("GET", "/page", text_answer(f"<html><body><p>{PAGE_TEXT}</p></body></html>"))

    fetched = _fetch(local_fetch, f"{listener.base_url}/page")
    assert fetched.status_code == 200, fetched.text
    assert PAGE_TEXT in fetched.json()["content"]

    fetches_so_far = len(listener.received)
    # a space in the path: fetchers would quote it and load the page, validators refuses it
    refused = _fetch(local_fetch, f"{listener.base_url}/pa ge")
    assert refused.status_code == 400, refused.text
    assert len(listener.received) == fetches_so_far, "a link validators refuses was fetched"
    assert _fetch(local_fetch, "not a url").status_code == 400


PAGE = f"""<!DOCTYPE html>
<html lang="en-GB"><head><title>Harbour notices</title>
<meta name="description" content="Notices for the harbour"></head>
<body><h1>Notices</h1><p>{PAGE_TEXT}</p></body></html>"""
FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Harbour feed</title>
<item><title>Ferry timetable</title><link>https://harbour.example/ferry</link></item>
</channel></rss>"""


@pytest.mark.slow
def test_a_fetched_page_is_stored_with_its_title_description_and_language(local_fetch, listener):
    listener.route("GET", "/notices", text_answer(PAGE))

    with local_fetch.client() as client:
        fetched = client.post(
            "/api/v1/retrieval/process/web", json={"url": f"{listener.base_url}/notices"}
        )
        assert fetched.status_code == 200, fetched.text
        stored = client.post(
            "/api/v1/retrieval/query/doc",
            json={"collection_name": fetched.json()["collection_name"], "query": "herons"},
        )

    assert stored.status_code == 200, stored.text
    [[document]], [[metadata]] = stored.json()["documents"], stored.json()["metadatas"]
    assert PAGE_TEXT in document and "<p>" not in document
    assert metadata["title"] == "Harbour notices"
    assert metadata["description"] == "Notices for the harbour"
    assert metadata["language"] == "en-GB"


@pytest.mark.slow
def test_a_fetched_xml_feed_is_read_as_text(local_fetch, listener):
    listener.route("GET", "/feed.xml", text_answer(FEED, content_type="application/xml"))

    fetched = _fetch(local_fetch, f"{listener.base_url}/feed.xml")

    assert fetched.status_code == 200, fetched.text
    content = fetched.json()["content"]
    assert "Ferry timetable" in content and "https://harbour.example/ferry" in content
    assert "<item>" not in content


def test_black_formats_code_and_reports_what_it_cannot_parse(admin):
    with admin.client() as client:
        formatted = client.post("/api/v1/utils/code/format", json={"code": "x=[1,2]\nprint( x )"})
        broken = client.post("/api/v1/utils/code/format", json={"code": "def broken(:\n"})

    assert formatted.status_code == 200, formatted.text
    assert formatted.json()["code"] == "x = [1, 2]\nprint(x)\n"
    assert broken.status_code == 400, broken.text
    assert "Cannot parse" in broken.json()["detail"]


ARGUMENTS = "argument_one, argument_two, argument_three, argument_four, argument_five"
LONG_CALL = f"result = some_function({ARGUMENTS})\n"
WRAPPED_CALL = f"result = some_function(\n    {ARGUMENTS}\n)\n"


def test_black_wraps_past_88_columns_and_leaves_formatted_code_alone(admin):
    with admin.client() as client:
        quoted = client.post("/api/v1/utils/code/format", json={"code": "s = 'harbour'\n"})
        wrapped = client.post("/api/v1/utils/code/format", json={"code": LONG_CALL})
        again = client.post("/api/v1/utils/code/format", json={"code": WRAPPED_CALL})

    assert quoted.json()["code"] == 's = "harbour"\n'
    assert wrapped.json()["code"] == WRAPPED_CALL
    assert again.json()["code"] == WRAPPED_CALL


@pytest.fixture(scope="module")
def collector():
    with listening() as service:
        service.route("POST", "/v1/traces", (200, {"Content-Type": "application/x-protobuf"}, b""))
        yield service


@pytest.fixture
def traced(instance_with, collector):
    return instance_with(
        {
            "ENABLE_OTEL": "true",
            "ENABLE_OTEL_TRACES": "true",
            "OTEL_OTLP_SPAN_EXPORTER": "http",
            "OTEL_EXPORTER_OTLP_ENDPOINT": f"{collector.base_url}/v1/traces",
            "OTEL_BSP_SCHEDULE_DELAY": "200",
        }
    )


def _exported_spans(collector) -> bytes:
    return b"".join(request.body for request in collector.requests_to("/v1/traces"))


@pytest.mark.slow
def test_request_traces_are_exported_over_otlp_http(traced, collector):
    with traced.client() as client:
        client.get("/api/version").raise_for_status()

    deadline = time.monotonic() + EXPORT_WAIT
    while b"/api/version" not in _exported_spans(collector) and time.monotonic() < deadline:
        time.sleep(0.2)

    exported = _exported_spans(collector)
    assert b"/api/version" in exported, "no span for the request reached the collector"
    assert b"open-webui" in exported, "the spans do not name the service"
    sent = collector.requests_to("/v1/traces")[-1]
    assert sent.headers["Content-Type"] == "application/x-protobuf"
