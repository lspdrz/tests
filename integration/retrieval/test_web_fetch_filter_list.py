"""Regression: malformed `WEB_FETCH_FILTER_LIST` entries must not turn into rules.

open-webui 0.11.0 fix `18719fef9` (#26910, issue #26908): `get_allow_block_lists` kept each entry
verbatim. Docker Compose list syntax passes surrounding quotes through, so `"localhost"` became
the allow entry `"localhost"`, quotes included, which matches no host; and since a non-empty
allow list refuses every host it does not match, one stray quote blocked every web address. A
quoted block entry (`'!host'`) turned into an allow entry the same way, quotes inside the `!`
(`!"host"`) blocked nothing, and an entry of quotes alone became an empty allow entry. The fix
strips quotes and whitespace, strips again after a leading `!` and drops entries left empty.

Each list is set the way Compose hands it over, on an instance of its own with local fetching
on. The listener answers on 127.0.0.1 (also reached as `localhost`) and on 127.0.0.2, so one
list can allow one name and refuse another.

Twin of unit/retrieval/test_web_fetch_filter_list.py.

Discriminates: passes on dev bbfa876af; keeping entries verbatim fails the allowed host (the
quoted entry matches nothing, so the page is refused). On dev ef67cc3fa, the pre-fix parser
(entries only stripped of whitespace, empty ones kept) fails the quoted block entry (the unlisted
host is refused), the quotes inside the `!` (the blocked host is fetched) and the quote-only
entries (every host is refused); the plain block entry and the unlisted host pass on both.
"""

from __future__ import annotations

import pytest

from harness.listener import listening, text_answer

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

PAGE_TEXT = "Local notes behind the filter list"
# each list as Compose passes it through, quotes included
QUOTED_ALLOW = '"localhost"'
QUOTED_BLOCK = "'!localhost'"
QUOTES_INSIDE_THE_BANG = '!"localhost", !127.0.0.2'
QUOTE_ONLY_ENTRIES = '\'""\', !, !""'


@pytest.fixture(scope="module")
def pages():
    """Hosts serving the page: 127.0.0.1 (and `localhost`) and 127.0.0.2, by port."""
    with listening() as first, listening("127.0.0.2") as second:
        for service in (first, second):
            service.route("GET", "/notes", text_answer(f"<p>{PAGE_TEXT}</p>"))
        yield {"localhost": first, "127.0.0.1": first, "127.0.0.2": second}


@pytest.fixture
def preview(instance_with, pages):
    """`preview(filter_list, host)` attaches the page under that host name, without storing it."""

    def attach(filter_list: str, host: str):
        filtering = instance_with(
            {"ENABLE_LOCAL_WEB_FETCH": "true", "WEB_FETCH_FILTER_LIST": filter_list}
        )
        url = f"http://{host}:{pages[host].port}/notes"
        with filtering.client() as client:
            return client.post("/api/v1/retrieval/process/web?process=false", json={"url": url})

    return attach


def assert_fetched(previewed, host: str) -> None:
    assert previewed.status_code == 200, f"{host} was refused: {previewed.text}"
    assert PAGE_TEXT in previewed.json()["content"]


def assert_refused(previewed, host: str) -> None:
    assert previewed.status_code == 400, f"{host} was fetched: {previewed.text}"


def test_a_host_on_a_quoted_allow_list_is_fetched(preview):
    assert_fetched(preview(QUOTED_ALLOW, "localhost"), "localhost")


def test_a_host_missing_from_the_allow_list_is_refused(preview, pages):
    fetches_before = len(pages["127.0.0.1"].requests_to("/notes"))

    assert_refused(preview(QUOTED_ALLOW, "127.0.0.1"), "127.0.0.1")

    assert len(pages["127.0.0.1"].requests_to("/notes")) == fetches_before


def test_a_quoted_block_entry_blocks_its_host_and_nothing_else(preview):
    assert_refused(preview(QUOTED_BLOCK, "localhost"), "localhost")
    assert_fetched(preview(QUOTED_BLOCK, "127.0.0.1"), "127.0.0.1")


def test_quotes_inside_the_bang_still_block_the_host(preview):
    assert_refused(preview(QUOTES_INSIDE_THE_BANG, "localhost"), "localhost")
    assert_fetched(preview(QUOTES_INSIDE_THE_BANG, "127.0.0.1"), "127.0.0.1")


def test_a_plain_block_entry_beside_a_quoted_one_still_blocks(preview):
    assert_refused(preview(QUOTES_INSIDE_THE_BANG, "127.0.0.2"), "127.0.0.2")


def test_entries_left_empty_make_no_rule(preview):
    for host in ("localhost", "127.0.0.1", "127.0.0.2"):
        assert_fetched(preview(QUOTE_ONLY_ENTRIES, host), host)
