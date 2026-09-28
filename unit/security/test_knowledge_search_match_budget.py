"""Regression: time a search command spends between its searches must not drain its budget.

open-webui 0.11.0 fix `3ab202626` (PR #27471) gave each `kb_exec` command one matching budget
(`MATCH_BUDGET_SECONDS`, entered with `match_budget()`), charged only for the time spent inside
searches. The catastrophic pattern, the budget shared by a pipeline and the fresh budget of the
next command are pinned over HTTP in integration/security/test_knowledge_search_match_budget.py.

What stays here is the part no request can reach: database round-trips and other coroutines run
between the searches of one command, and that wait must cost nothing. No request can stretch
that gap past the two second budget, so the budget is driven with a scripted clock.

Discriminates: passes on dev `ef67cc3fa`; with the clock started when the matcher is built the
waiting test fails.
"""

import pytest

pytestmark = pytest.mark.regression

# Each clock reading moves this far, so every search costs a fraction of the budget.
CLOCK_STEP_SECONDS = 0.3


class SteppingClock:
    """Stands in for the `time` module: every reading is a step later than the last."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        self.now += CLOCK_STEP_SECONDS
        return self.now


@pytest.fixture(scope="session")
def knowledge_fs(owui_module):
    return owui_module("open_webui.tools.knowledge_fs")


@pytest.fixture
def clock(knowledge_fs, monkeypatch) -> SteppingClock:
    stepping = SteppingClock()
    monkeypatch.setattr(knowledge_fs, "time", stepping)
    return stepping


def test_waiting_between_searches_does_not_drain_the_budget(knowledge_fs, clock):
    """Database round-trips and other coroutines run between searches and cost nothing."""
    with knowledge_fs.match_budget():
        matcher, _ = knowledge_fs.build_matcher("a", use_regex=True)
        clock.now += 10 * knowledge_fs.MATCH_BUDGET_SECONDS

        assert matcher("a line") is True, (
            "time spent outside search() was charged to the budget, so a slow database "
            "makes legitimate searches fail (#27471)"
        )
