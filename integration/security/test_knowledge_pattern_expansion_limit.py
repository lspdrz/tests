"""Regression: a knowledge-search pattern must not be able to blow up before it runs.

open-webui 0.11.1 fix `d5b66533e` (PR #28284): the knowledge search tools handed any pattern the
model wrote straight to the `regex` module. That engine materialises counted quantifiers at
compile time, so cost grows with the product of the counts, not with the length of the pattern:
`(a{4000}){4000}` measured ~2.3s and ~4.1 GB of resident memory to compile, and nesting further
takes the box down.

dev compiles with RE2, whose parser refuses a count over 1000, or nested counts whose product
passes 1000, as `invalid repetition size`; the model gets that back as the tool's error and can
fix its own pattern. The scripted model calls `grep_knowledge_files` on a file of the caller's;
an alternation makes the tool read a pattern as a regex. The bomb here is small enough that the
`regex` module compiles it in a moment, so a regression costs the shared instance nothing.

Discriminates: passes on dev ef67cc3fa; with RE2 swapped back for the `regex` module every
oversized pattern compiles and searches instead of coming back as an error.
"""

from __future__ import annotations

import json

import pytest

from harness.tool_calls import run_tool

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

# 500 * 500 = 250,000: the expansion bomb from the original advisory.
EXPANSION_BOMB = "(a{500}){500}|zzz"

LINES = ["aaa found", "aa short", "42 answers", "nothing here", "config retries{50000} here"]


@pytest.fixture
def grep(make_user, upstream):
    """`grep(pattern, **options)` has the model grep one file of a fresh account's."""
    person = make_user()
    client = person.client()
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": ("lines.txt", "\n".join(LINES).encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    file_id = uploaded.json()["id"]

    def search(pattern: str, **options) -> str:
        arguments = {"pattern": pattern, "file_id": file_id, **options}
        return run_tool(client, upstream, "grep_knowledge_files", arguments)

    yield search
    client.close()


def _error(result: str) -> str | None:
    try:
        return json.loads(result).get("error")
    except ValueError:
        return None


def _matched_lines(result: str) -> list[str]:
    if result.startswith("No matches"):
        return []
    return [hit.split(": ", 1)[1] for hit in result.splitlines()]


# narrow: the expansion bomb is refused instead of compiled


def test_expansion_bomb_is_refused_instead_of_compiled(grep):
    error = _error(grep(EXPANSION_BOMB))

    assert error, (
        f"{EXPANSION_BOMB!r} was compiled, so a model can nest the counts a little further and "
        "spend the worker's memory before a single line is searched (#28284)"
    )
    assert "repetition" in error.lower(), (
        f"the refusal reads {error!r}, which does not tell the model the counts are the "
        "problem, so it cannot fix its own pattern (#28284)"
    )


# broad: every count that multiplies past the cap


@pytest.mark.parametrize(
    "pattern",
    [
        "a{50000}|zzz",  # one oversized count
        "a{2001}|zzz",  # just over the cap
        "(a{1000}){101}|zzz",  # product 101,000
        "((a{50}){50}){50}|zzz",  # three nested levels, 125,000
    ],
)
def test_counts_that_multiply_past_the_cap_are_refused(grep, pattern):
    assert _error(grep(pattern)), f"{pattern!r} was compiled into a search (#28284)"


# nearby: patterns that cost nothing to compile still work


@pytest.mark.parametrize(
    "pattern",
    [
        r"\d{2,4}-\w{1,8}",  # the shape real searches use
        r"version \{3000\}|zzz",  # escaped braces are literal text, not a quantifier
        "(a{30}){30}|zzz",  # nested, but the product stays under the cap
        "a{1000}b{1000}|zzz",  # side by side, the counts add rather than multiply
    ],
)
def test_ordinary_quantifier_use_is_accepted(grep, pattern):
    result = grep(pattern)

    assert _error(result) is None, (
        f"{pattern!r} was rejected ({result!r}), so the bound is tight enough to break "
        "searches that cost nothing to compile (#28284)"
    )


def test_literal_search_is_not_subject_to_the_limit(grep):
    """Braces in a search that is not a regex are just characters."""
    assert _matched_lines(grep("retries{50000}")) == ["config retries{50000} here"]


@pytest.mark.parametrize(
    "pattern, case_insensitive, expected",
    [
        ("a{3}|zzz", False, ["aaa found"]),
        (r"\d{2}", False, ["42 answers", "config retries{50000} here"]),
        ("A{2}|zzz", True, ["aaa found", "aa short"]),
    ],
)
def test_quantified_patterns_still_match(grep, pattern, case_insensitive, expected):
    result = grep(pattern, case_insensitive=case_insensitive)

    assert _matched_lines(result) == expected, result
