"""Guard: a failing filter is logged without the request it failed on.

An error path that renders the failed request costs a whole message history per failure, so a
burst of filter errors becomes a burst of allocations on the event loop, and the log fills with
every user's prompts. A global filter raises in one stage (inlet, stream or outlet) on an
instance logging at DEBUG, where the inlet failure is written; the prompt and the scripted reply
carry a canary, and the log records that report the failure must name the filter and not the
canary. Other DEBUG lines render the request on purpose, so only the failure records (the line
and its traceback) are read.

Twin of unit/footprint/test_filter_error_path_ignores_body.py.

Unpinned: read on upstream dev at v0.11.3 (a253bf0c3), where the path logs only the filter
type and id. Unmarked: nothing to pin.
Discriminates: passes on dev ef67cc3fa, fails for every stage once the two failure log calls in
a copy of it also render `form_data`.
"""

from __future__ import annotations

import re
import time
import uuid

import pytest

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ask
from harness.plugins import installed_function

pytestmark = [pytest.mark.slow, pytest.mark.api, pytest.mark.requires_source]

DEBUG_LOGGING = {"GLOBAL_LOG_LEVEL": "DEBUG"}
RECORD_START = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

FAILING_FILTER = """
class Filter:
    async def {stage}(self, {argument}):
        raise ValueError("the filter broke")
"""


def _failure_records(log: str, filter_id: str) -> list[str]:
    """Each log record naming the filter, with the traceback lines that belong to it."""
    records: list[str] = []
    for line in log.splitlines():
        if RECORD_START.match(line):
            records.append(line)
        elif records:
            records[-1] += "\n" + line
    return [record for record in records if f"filter {filter_id}" in record.splitlines()[0]]


def _logged_failures(instance, offset: int, filter_id: str) -> list[str]:
    """The outlet stage runs after the reply is stored, so its record may land a moment later."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        failures = _failure_records(instance.log_since(offset), filter_id)
        if failures:
            return failures
        time.sleep(0.1)
    return []


@pytest.mark.parametrize(
    "stage, argument", [("inlet", "body"), ("stream", "event"), ("outlet", "body")]
)
def test_a_failing_filter_is_logged_without_the_request(instance_with, stage, argument):
    debug_instance = instance_with(DEBUG_LOGGING)
    admin = admin_of(debug_instance)
    canary = f"canary-{uuid.uuid4().hex}"
    source = FAILING_FILTER.format(stage=stage, argument=argument)
    with installed_function(admin, source, is_global=True) as filter_id:
        debug_instance.upstream.queue(reply.text([f"reply {canary} ", "continues"]))
        offset = debug_instance.log_size()
        with admin.client() as client:
            ask(client, f"prompt {canary}")
        failures = _logged_failures(debug_instance, offset, filter_id)

    assert failures, f"the {stage} failure was not logged, so nothing could have leaked"
    leaking = [record for record in failures if canary in record]
    assert not leaking, f"the {stage} filter failure rendered the request: {leaking[0][:400]}"
