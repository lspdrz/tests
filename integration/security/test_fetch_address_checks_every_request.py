"""Regression: the web fetch filter list must reach every request and every address it dials.

open-webui 0.11.1 fix `e3e4bd87d` (#27823). A `WEB_FETCH_FILTER_LIST` entry written as an address
range matched nothing at all, because entries were compared as DNS labels: `!10.0.0.0/8` blocked
nothing. Entries naming an address or a range are now matched by containment, the same rule the
admin's web search domain filter uses. The list also ran once, on the submitted URL, so a
redirect hop reached a listed host; it now runs per request on both transports, the requests
adapter behind `/process/web` and the aiohttp connector behind `/process/url`. And the addresses a
hop's host resolved to were never judged at the connection layer, so a name answering with a
listed address was dialled; every answer is now checked, including the IPv4 address an IPv6
answer carries. The redirect gaps need `AIOHTTP_CLIENT_ALLOW_REDIRECTS=true`.

A hop whose host is `2130706434` (a spelling of 127.0.0.2 only the resolver understands) or the
IPv6 form `::ffff:127.0.0.2` passes the name check, so only the check on what it resolves to can
refuse it. The matching rules are also pinned through the admin's web search domain filter.

Discriminates: passes on dev ef67cc3fa; with the containment match removed from
`_host_matches_pattern` the range tests and the range and spelling rows of the domain filter
fail (the listed address is fetched, the listed results are kept), with the per-request hooks
removed from `_SSRFSafeAdapter.send` and `_SSRFSafeConnector.connect` the listed-host redirect
tests fail, with the resolved-address checks removed from `_SSRFSafeConnector._resolve_host` and
`_ssrf_safe_new_conn` the resolved-address hops are dialled, and with the address lookup removed
from `get_filtered_results` the name resolving into a blocked range is kept.
"""

from __future__ import annotations

import socket

import pytest

from harness.listener import listening, text_answer
from harness.web_retrieval import save_web_settings, serve_search_results, web_settings_restored

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

LISTED_ADDRESS = "127.0.0.2"
LOCAL_FETCH = {
    "ENABLE_LOCAL_WEB_FETCH": "true",
    "WEB_FETCH_FILTER_LIST": f"!{LISTED_ADDRESS}/32,!localhost",
    "AIOHTTP_CLIENT_ALLOW_REDIRECTS": "true",
}
FETCH_ROUTES = ["/api/v1/retrieval/process/web", "/api/v1/retrieval/process/url"]

PAGE = "<html><body><p>public page text</p></body></html>"
SECRET_PAGE = "<html><body><p>internal secret text</p></body></html>"


@pytest.fixture(scope="module")
def local_fetch(instance_with):
    return instance_with(LOCAL_FETCH)


def _fetch(instance, route: str, url: str):
    with instance.client() as client:
        return client.post(f"{route}?process=false", json={"url": url})


def _redirect_to(location: str):
    return 302, {"Location": location}, b""


@pytest.mark.parametrize("route", FETCH_ROUTES)
def test_a_range_entry_blocks_the_addresses_inside_it(local_fetch, route):
    with listening(host=LISTED_ADDRESS) as listed_service:
        listed_service.route("GET", "/page", text_answer(SECRET_PAGE))
        response = _fetch(local_fetch, route, f"{listed_service.base_url}/page")
        reached = list(listed_service.received)

    assert reached == [], (
        f"{route} fetched {LISTED_ADDRESS} although `!{LISTED_ADDRESS}/32` lists it; a range "
        "entry matched nothing, so the operator's block did nothing (#27823)"
    )
    assert response.status_code == 400, response.text


@pytest.mark.parametrize("route", FETCH_ROUTES)
def test_a_redirect_hop_to_a_listed_host_is_never_fetched(local_fetch, listener, route):
    hop_url = f"http://localhost:{listener.port}/secret"
    listener.route("GET", "/start", _redirect_to(hop_url))
    listener.route("GET", "/secret", text_answer(SECRET_PAGE))

    response = _fetch(local_fetch, route, f"{listener.base_url}/start")

    assert listener.requests_to("/secret") == [], (
        f"{route} followed a redirect to {hop_url} although `!localhost` lists it; the filter "
        "list only ran on the submitted URL, never on a hop (#27823)"
    )
    assert "internal secret text" not in response.text


@pytest.mark.parametrize("route", FETCH_ROUTES)
def test_an_unlisted_address_is_still_fetched(local_fetch, listener, route):
    listener.route("GET", "/page", text_answer(PAGE))

    response = _fetch(local_fetch, route, f"{listener.base_url}/page")

    assert response.status_code == 200, response.text
    assert "public page text" in response.json()["content"]


@pytest.mark.parametrize("route", FETCH_ROUTES)
def test_a_redirect_to_an_unlisted_host_is_still_followed(local_fetch, listener, route):
    listener.route("GET", "/start", _redirect_to(f"{listener.base_url}/landing"))
    listener.route("GET", "/landing", text_answer(PAGE))

    response = _fetch(local_fetch, route, f"{listener.base_url}/start")

    assert response.status_code == 200, response.text
    assert "public page text" in response.json()["content"]


@pytest.mark.parametrize(
    ("route", "hop_host"),
    [
        (FETCH_ROUTES[0], "2130706434"),
        (FETCH_ROUTES[0], f"[::ffff:{LISTED_ADDRESS}]"),
        # aiohttp refuses the decimal spelling by itself
        (FETCH_ROUTES[1], f"[::ffff:{LISTED_ADDRESS}]"),
    ],
    ids=["web-decimal", "web-ipv4-in-ipv6", "url-ipv4-in-ipv6"],
)
def test_a_redirect_hop_resolving_to_a_listed_address_is_never_dialled(
    local_fetch, listener, route, hop_host
):
    with listening(host=LISTED_ADDRESS) as listed_service:
        listed_service.route("GET", "/secret", text_answer(SECRET_PAGE))
        listener.route(
            "GET", "/start", _redirect_to(f"http://{hop_host}:{listed_service.port}/secret")
        )
        response = _fetch(local_fetch, route, f"{listener.base_url}/start")
        reached = list(listed_service.received)

    assert reached == [], (
        f"{route} followed a redirect to {hop_host}, which resolves to {LISTED_ADDRESS}, and "
        f"dialled it although `!{LISTED_ADDRESS}/32` lists it; resolved addresses were never "
        "judged (#27823)"
    )
    assert "internal secret text" not in response.text


@pytest.mark.parametrize("route", FETCH_ROUTES)
@pytest.mark.parametrize("url", ["ftp://example.com/x", "file:///etc/passwd", "not a url", ""])
def test_a_url_that_is_not_http_is_refused(local_fetch, route, url):
    response = _fetch(local_fetch, route, url)

    assert response.status_code == 400, response.text


# The admin's web search domain filter matches result hosts by the same rules.

DOMAIN_FILTER_CASES = [
    # an allow entry written as a range admits the addresses inside it
    (["10.0.0.0/8"], ["10.1.2.3", "11.0.0.1"], ["10.1.2.3"]),
    # an address entry matches any spelling of that address
    (
        ["!fd00:ec2::254"],
        ["[fd00:ec2:0:0:0:0:0:254]", "[FD00:EC2::254]", "[fd00:0ec2::0254]", "[fd00:ec2::255]"],
        ["[fd00:ec2::255]"],
    ),
    (["!fd00::/8"], ["[fd00:ec2::254]", "[fe80::1]"], ["[fe80::1]"]),
    # hostname entries keep matching on DNS label boundaries
    (["!corp.com"], ["api.corp.com", "corp.com", "evilcorp.com"], ["evilcorp.com"]),
    (["corp.com"], ["api.corp.com", "other.example"], ["api.corp.com"]),
    # an address entry matches itself and nothing else
    (["!169.254.169.254"], ["169.254.169.254", "169.254.169.253"], ["169.254.169.253"]),
    # a block entry beats an allow entry for the same host
    (["corp.com", "!api.corp.com"], ["api.corp.com", "www.corp.com"], ["www.corp.com"]),
    # no filter list means no filtering
    ([], ["anything.example", "10.1.2.3"], ["anything.example", "10.1.2.3"]),
]


# RFC 6761 reserves every name under localhost for the loopback addresses.
LOOPBACK_NAME = "kestrel.localhost"


def _search_links(instance, listener, filter_list: list[str], hosts: list[str]) -> list[str]:
    """Run one web search listing a page on each host; returns the links the filter kept."""
    links = [f"http://{host}/page" for host in hosts]
    with instance.client() as client, web_settings_restored(client):
        save_web_settings(
            client,
            **serve_search_results(listener, links),
            WEB_SEARCH_DOMAIN_FILTER_LIST=filter_list,
            BYPASS_WEB_SEARCH_WEB_LOADER=True,
            BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=True,
        )
        searched = client.post("/api/v1/retrieval/process/web/search", json={"queries": ["q"]})
    if searched.status_code == 404:
        return []
    assert searched.status_code == 200, searched.text
    return searched.json()["filenames"]


@pytest.mark.parametrize("filter_list, hosts, kept", DOMAIN_FILTER_CASES)
def test_the_web_search_domain_filter_matching_rules(
    local_fetch, listener, filter_list, hosts, kept
):
    kept_links = _search_links(local_fetch, listener, filter_list, hosts)

    assert kept_links == [f"http://{host}/page" for host in kept], (
        f"the domain filter {filter_list} kept the wrong results (#27823)"
    )


@pytest.fixture
def loopback_name() -> str:
    try:
        socket.getaddrinfo(LOOPBACK_NAME, 80)
    except socket.gaierror:
        pytest.skip(f"this resolver does not answer {LOOPBACK_NAME} with a loopback address")
    return LOOPBACK_NAME


def test_a_block_range_judges_the_addresses_a_result_resolves_to(
    local_fetch, listener, loopback_name
):
    kept_links = _search_links(
        local_fetch, listener, ["!127.0.0.0/8", "!::1/128"], [loopback_name, "10.1.2.3"]
    )

    assert kept_links == ["http://10.1.2.3/page"], (
        f"a result on {loopback_name}, which resolves to loopback, survived a block of the "
        "loopback ranges; the filter never looked the name up (#27823)"
    )


def test_a_name_outside_the_blocked_range_is_kept(local_fetch, listener, loopback_name):
    kept_links = _search_links(local_fetch, listener, ["!10.0.0.0/8"], [loopback_name, "10.1.2.3"])

    assert kept_links == [f"http://{loopback_name}/page"]
