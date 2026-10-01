"""Dependency smoke: the server log and the audit log, both written by loguru.

Once the server has started, loguru writes every line of its log: the lines Open WebUI logs
through the standard library are handed over to it, each still named after the module, function
and line that logged it. By default a line reads `time | LEVEL | module:function:line - message`
with any traceback below; with `LOG_FORMAT=json` it is one JSON object carrying `ts`, `level`,
`msg`, `caller` and, for a logged exception, an `error` with its type, message and traceback.
`GLOBAL_LOG_LEVEL` drops everything below it. With `AUDIT_LOG_LEVEL` set, the audited requests
go to `audit.log` in the data directory and nothing else does; past
`AUDIT_LOG_FILE_ROTATION_SIZE` the file is rotated into a zip archive and started afresh.

Discriminates: passes on dev ef67cc3fa; in a backend copy, handing standard-library records to
loguru at depth 0 names the log handler as the caller of the filter failure in both formats, a
JSON sink that leaves out the exception fails the JSON error
test, adding the sink at `DEBUG` whatever the setting lets the info lines through, dropping the
audit file's filter puts ordinary log lines into the audit log and dropping `compression="zip"`
leaves the rotated audit logs unarchived.
"""

from __future__ import annotations

import json
import re
import time
import uuid
import zipfile

import pytest

from harness.actors import admin_of
from harness.chat import ask
from harness.instance import without_colour
from harness.plugins import installed_function

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source]

WAIT_SECONDS = 10.0
FILTER_FAILURE = "the outlet filter broke"
FAILING_OUTLET = f"""
class Filter:
    def outlet(self, body):
        raise RuntimeError("{FILTER_FAILURE}")
"""
# where Open WebUI logs a failing filter, through the standard library
FILTER_MODULE = "open_webui.utils.filter"
FILTER_REPORT = "Error in outlet filter"
# a line as the default format writes it
TEXT_LINE = re.compile(
    r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3} \| (?P<level>[A-Z]+) +\| "
    r"(?P<module>[\w.]+):(?P<function>[\w<>]+):(?P<line>\d+) - (?P<message>.*)$",
    re.MULTILINE,
)
LOGURU_CALLER = re.compile(r"^open_webui\.[\w.]+:\w+:\d+$")
JSON_AUDITED = {
    "LOG_FORMAT": "json",
    "GLOBAL_LOG_LEVEL": "WARNING",
    "AUDIT_LOG_LEVEL": "METADATA",
    "AUDIT_LOG_FILE_ROTATION_SIZE": "1 KB",
}


@pytest.fixture
def json_audited(instance_with):
    return instance_with(JSON_AUDITED)


def _eventually(read, timeout: float = WAIT_SECONDS):
    """The first truthy `read()` within the timeout, else its last value."""
    deadline = time.monotonic() + timeout
    value = read()
    while not value and time.monotonic() < deadline:
        time.sleep(0.1)
        value = read()
    return value


def _filter_failure_log(target) -> str:
    """The log a chat writes when a global outlet filter raises."""
    offset = target.log_size()
    admin = admin_of(target)
    with installed_function(admin, FAILING_OUTLET, is_global=True), admin.client() as client:
        ask(client, "hello?")
        _eventually(lambda: FILTER_FAILURE in target.log_since(offset))
    return without_colour(target.log_since(offset))


def _json_entries(log: str) -> list[dict]:
    """Every JSON line of the log; libraries still print warnings of their own in between."""
    return [json.loads(line) for line in log.splitlines() if line.startswith("{")]


# ---------------------------------------------------------------- the default format


def test_each_line_names_its_level_and_where_it_was_logged(instance):
    started = [
        line
        for line in TEXT_LINE.finditer(without_colour(instance.log_since(0)))
        if line["message"].startswith("GLOBAL_LOG_LEVEL:")
    ]

    assert started, "no line in loguru's format reports the log level at start"
    assert started[-1]["level"] == "INFO"
    assert (started[-1]["module"], started[-1]["function"]) == (
        "open_webui.utils.logger",
        "start_logger",
    )


def test_a_logged_exception_names_its_module_and_prints_the_traceback(instance):
    logged = _filter_failure_log(instance)

    failures = [line for line in TEXT_LINE.finditer(logged) if line["level"] == "ERROR"]
    assert failures, f"no error line in loguru's format:\n{logged[-2000:]}"
    [reported] = [line for line in failures if line["message"].startswith(FILTER_REPORT)]
    assert reported["module"] == FILTER_MODULE, reported.group(0)
    traceback = logged[reported.end() :]
    assert "Traceback" in traceback and FILTER_FAILURE in traceback


# ---------------------------------------------------------------- JSON, a level and the audit log


def test_a_json_line_carries_the_caller_and_the_exception(json_audited):
    entries = _json_entries(_filter_failure_log(json_audited))

    failures = [entry for entry in entries if isinstance(entry.get("error"), dict)]
    assert failures, f"no JSON line carries the exception: {entries[-5:]}"
    [failure] = [entry for entry in failures if entry["error"]["message"] == FILTER_FAILURE]
    assert failure["level"] == "error"
    assert failure["msg"].startswith(FILTER_REPORT)
    assert LOGURU_CALLER.match(failure["caller"]), failure["caller"]
    assert failure["caller"].split(":")[0] == FILTER_MODULE, failure["caller"]
    assert failure["error"]["type"] == "RuntimeError"
    assert "Traceback" in failure["error"]["stacktrace"]
    assert re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}[+-]\d\d:\d\d$", failure["ts"])


def test_nothing_below_the_global_level_is_written(json_audited):
    entries = _json_entries(json_audited.log_since(0) + _filter_failure_log(json_audited))

    assert {entry["level"] for entry in entries if LOGURU_CALLER.match(entry["caller"])} == {
        "error"
    }
    assert not [entry for entry in entries if entry["msg"].startswith("GLOBAL_LOG_LEVEL:")]


def _create_notes(target, count: int) -> None:
    with admin_of(target).client() as client:
        for index in range(count):
            created = client.post(
                "/api/v1/notes/create",
                json={"title": f"audited {index} {uuid.uuid4().hex}", "data": {"content": {}}},
            )
            assert created.status_code == 200, created.text


def _archives(target) -> list:
    return sorted(target.data_dir.glob("audit.*.log.zip"))


def _audit_entries(target) -> list[dict]:
    """Every entry of the audit log, the rotated archives first."""
    texts = []
    for path in _archives(target):
        with zipfile.ZipFile(path) as archive:
            texts += [archive.read(member).decode() for member in archive.namelist()]
    current = target.data_dir / "audit.log"
    if current.exists():  # gone for a moment while loguru rotates it
        texts.append(current.read_text())
    return [json.loads(line) for text in texts for line in text.splitlines() if line.strip()]


def test_the_audit_log_rotates_into_zip_archives(json_audited):
    before = len(_archives(json_audited))
    _create_notes(json_audited, 10)

    archives = _eventually(lambda: _archives(json_audited)[before:])

    assert archives, "the audit log was never rotated into an archive"
    with zipfile.ZipFile(archives[-1]) as archive:
        [member] = archive.namelist()
        audited = [json.loads(line) for line in archive.read(member).decode().splitlines()]
    assert audited and all(entry["verb"] == "POST" for entry in audited), audited


def test_only_audited_requests_reach_the_audit_log(json_audited):
    _filter_failure_log(json_audited)
    before = len(_audit_entries(json_audited))
    _create_notes(json_audited, 1)

    # the entry is written once the response has gone out
    assert _eventually(lambda: _audit_entries(json_audited)[before:]), "the request was not audited"
    entries = _audit_entries(json_audited)
    # an ordinary log line written there would be an entry without a request
    assert all(entry["verb"] and entry["request_uri"] for entry in entries), [
        entry for entry in entries if not entry["verb"]
    ]
