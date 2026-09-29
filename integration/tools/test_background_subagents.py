"""Journey: a background sub-agent runs on after the reply ends and reports back into its chat.

The model's delegation call with `background` set returns a dispatch handle at once, so the
reply ends while the sub-agent works on, registered as a running task of its own chat. When it
finishes, its report is stored in the parent chat as an internal user message that follows the
reply which dispatched it, and the model continues from it without the person sending anything.
A report that arrives while the reply is still written waits for it. The admin's background
limit refuses a delegation over the cap at once and frees its slot when a sub-agent ends, the
background switch off turns the call into an ordinary one, and neither the concurrent limit nor
a second chat holds a background sub-agent back or mixes the reports. Stopping a sub-agent
through its task lists it as interrupted and a failing one as failed.

A report that comes while the answer to a later question is still written is attached to the
answer before it and hides that question (open-webui/open-webui#31507, fix PR #31557); that
test stays red until the fix merges.

Discriminates: passes on dev 176d31d1d apart from the test named above, which fails there. In
backend copies each test turns red with its edit: the dispatch made to wait for the sub-agent,
the sub-agent left out of the running tasks, the report not stored, stored without its metadata
or under the wrong reply, the report never continuing the chat, a pending report dropped, the
cap ignored, never released or applied to the unlimited setting, the switch ignored, the stop not
reaching the task, the failure not reported, the concurrent limit applied to background work
and the parent's id mixed up between chats.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask, send_message, wait_for_reply

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
SLOW_SECONDS = 3.0
CONCURRENT_ENV = {
    "ENABLE_SUBAGENTS": "true",
    "SUBAGENTS_BACKGROUND_ENABLED": "true",
    "SUBAGENTS_MAX_CONCURRENT": "1",
}


@pytest.fixture
def background_on(admin, preserve):
    """`background_on(SUBAGENTS_MAX_ASYNC=1)` turns background sub-agents on with a limit."""
    preserve(SUBAGENTS)

    def switch_on(**settings) -> None:
        with admin.client() as client:
            current = client.get(SUBAGENTS[0]).json()
            body = {
                **current,
                "ENABLE_SUBAGENTS": True,
                "SUBAGENTS_BACKGROUND_ENABLED": True,
                **settings,
            }
            client.post(SUBAGENTS[1], json=body).raise_for_status()

    return switch_on


def unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:8]}"


def dispatch(task: str, prompt: str, **arguments) -> reply.Reply:
    return reply.tool_call(
        "delegate_task",
        {"task": task, "background": True, **arguments},
        match=reply.answering(prompt),
    )


def report_of(task: str):
    """A `match` for the request that carries the finished sub-agent's report back to the model."""

    def matches(body: dict) -> bool:
        users = [entry for entry in body.get("messages", []) if entry.get("role") == "user"]
        content = str(users[-1].get("content")) if users else ""
        return "[ASYNC SUBAGENT COMPLETE" in content and task in content

    return matches


def history_of(client, chat_id: str) -> dict:
    stored = client.get(f"/api/v1/chats/{chat_id}")
    stored.raise_for_status()
    return stored.json()["chat"]["history"]


def wait_for_message(client, chat_id: str, condition, timeout: float = 60.0) -> dict:
    """The first stored message that satisfies `condition`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for message in history_of(client, chat_id)["messages"].values():
            if condition(message):
                return message
        time.sleep(0.1)
    raise AssertionError(f"no stored message of chat {chat_id} ever satisfied the condition")


def finished_reply(text: str):
    return lambda message: (
        message["role"] == "assistant" and message.get("done") and message["content"] == text
    )


def dispatch_handles(message: dict) -> list[dict]:
    outputs = [item for item in message["output"] if item["type"] == "function_call_output"]
    return [json.loads(item["output"][0]["text"]) for item in outputs]


def two_dispatches(tasks: list[str], prompt: str) -> reply.Reply:
    calls = [
        {
            "id": f"call_{number}",
            "type": "function",
            "function": {
                "name": "delegate_task",
                "arguments": json.dumps({"task": task, "background": True}),
            },
        }
        for number, task in enumerate(tasks)
    ]
    return reply.Reply(tool_calls=calls, match=reply.answering(prompt))


def reports_in(client, chat_id: str) -> list[dict]:
    return [
        message
        for message in history_of(client, chat_id)["messages"].values()
        if (message.get("meta") or {}).get("type") == "subagent"
    ]


def test_the_reply_ends_with_a_dispatch_handle_while_the_subagent_runs_on(
    background_on, make_user, upstream
):
    background_on()
    prompt, task = unique("hand this over"), unique("audit the archive")
    upstream.queue(
        dispatch(task, prompt),
        reply.text("done auditing", delay=SLOW_SECONDS, match=reply.answering(task)),
        reply.text("Handed over.", match=reply.answering(prompt)),
        reply.text("The audit is in.", match=report_of(task)),
    )
    with make_user().client() as client:
        turn, stored = ask(client, prompt)

        [handle] = dispatch_handles(stored)
        assert (handle["status"], handle["mode"], handle["task"]) == (
            "dispatched",
            "background",
            task,
        )
        assert stored["content"] == "Handed over."
        sub_id = handle["subagent_chat_id"]
        assert client.get(f"/api/tasks/chat/{sub_id}").json()["task_ids"] != []
        assert reports_in(client, turn.chat_id) == []
        assert [chat["id"] for chat in client.get("/api/v1/chats/").json()] == [turn.chat_id]

        wait_for_message(client, turn.chat_id, finished_reply("The audit is in."))
        assert client.get(f"/api/tasks/chat/{sub_id}").json()["task_ids"] == []


def test_the_finished_report_follows_the_reply_that_dispatched_it_and_the_model_continues(
    background_on, make_user, upstream
):
    background_on()
    prompt, task = unique("hand this over"), unique("read the ledger")
    answer = unique("the ledger is in March")
    upstream.queue(
        dispatch(task, prompt, context="look at the year 2024"),
        reply.text(answer, match=reply.answering(task)),
        reply.text("Handed over.", match=reply.answering(prompt)),
        reply.text("March it is.", match=report_of(task)),
    )
    with make_user().client() as client:
        turn, stored = ask(client, prompt)
        [handle] = dispatch_handles(stored)
        wait_for_message(client, turn.chat_id, finished_reply("March it is."))
        [report] = reports_in(client, turn.chat_id)
        continuation = [r for r in upstream.chat_requests() if report_of(task)(r)][0]

    assert report["role"] == "user"
    assert report["parentId"] == turn.assistant_message_id
    assert report["meta"] == {
        "internal": True,
        "type": "subagent",
        "delegation_id": handle["delegation_id"],
        "subagent_chat_id": handle["subagent_chat_id"],
    }
    for line in (
        f"[ASYNC SUBAGENT COMPLETE - {handle['delegation_id']}]",
        f"Original task: {task}",
        "Context provided: look at the year 2024",
        f"Subagent chat: {handle['subagent_chat_id']}",
        "Status: completed",
        answer,
    ):
        assert line in report["content"]
    sent = continuation["messages"]
    assert [entry["role"] for entry in sent][-3:] == ["tool", "assistant", "user"]
    assert sent[-1]["content"] == report["content"]


def test_a_report_that_arrives_while_the_reply_is_written_waits_for_it(
    background_on, make_user, upstream
):
    background_on()
    prompt, task = unique("hand this over"), unique("look it up")
    upstream.queue(
        dispatch(task, prompt),
        reply.text("found it", match=reply.answering(task)),
        reply.text("Handed over.", delay=SLOW_SECONDS, match=reply.answering(prompt)),
        reply.text("Looked up.", match=report_of(task)),
    )
    with make_user().client() as client:
        turn, stored = ask(client, prompt)
        wait_for_message(client, turn.chat_id, finished_reply("Looked up."))
        [report] = reports_in(client, turn.chat_id)

    assert stored["content"] == "Handed over."
    assert report["parentId"] == turn.assistant_message_id
    assert len([r for r in upstream.chat_requests() if report_of(task)(r)]) == 1


def test_a_report_that_arrives_before_a_later_answer_ends_follows_the_answer_that_dispatched_it(
    background_on, make_user, upstream
):
    """open-webui/open-webui#31507: the report hangs under the earlier answer and hides the turn."""
    background_on()
    first, prompt, task = unique("first question"), unique("hand this over"), unique("look it up")
    upstream.queue(
        reply.text("First answer.", match=reply.answering(first)),
        dispatch(task, prompt),
        reply.text("found it", match=reply.answering(task)),
        reply.text("Handed over.", delay=SLOW_SECONDS, match=reply.answering(prompt)),
        reply.text("Looked up.", match=report_of(task)),
    )
    with make_user().client() as client:
        opened, _ = ask(client, first)
        second = send_message(
            client,
            prompt,
            chat_id=opened.chat_id,
            parent_id=opened.assistant_message_id,
            history=[
                {"role": "user", "content": first},
                {"role": "assistant", "content": "First answer."},
            ],
        )
        wait_for_reply(client, second)
        wait_for_message(client, opened.chat_id, finished_reply("Looked up."))
        history = history_of(client, opened.chat_id)

    [report] = [m for m in history["messages"].values() if m.get("meta")]
    assert report["parentId"] == second.assistant_message_id, (
        "the report was attached to the earlier answer, so the chat now shows a branch without "
        "the second question"
    )
    shown, cursor = [], history["currentId"]
    while cursor:
        shown.append(history["messages"][cursor]["content"])
        cursor = history["messages"][cursor]["parentId"]
    assert prompt in shown


def tool_outputs(message: dict) -> list[str]:
    outputs = [item for item in message["output"] if item["type"] == "function_call_output"]
    return [item["output"][0]["text"] for item in outputs]


def test_a_background_limit_of_one_refuses_the_next_delegation_and_frees_the_slot_afterwards(
    background_on, make_user, upstream
):
    background_on(SUBAGENTS_MAX_ASYNC=1)
    prompt, later = unique("split this up"), unique("one more thing")
    tasks = [unique("first errand"), unique("second errand"), unique("third errand")]
    upstream.queue(
        two_dispatches(tasks[:2], prompt),
        *[reply.text("done", match=reply.answering(task)) for task in tasks],
        reply.text("Two were asked for.", match=reply.answering(prompt)),
        reply.text("One came back.", match=lambda body: any(report_of(t)(body) for t in tasks[:2])),
        dispatch(tasks[2], later),
        reply.text("Third sent off.", match=reply.answering(later)),
        reply.text("Third came back.", match=report_of(tasks[2])),
    )
    with make_user().client() as client:
        turn, stored = ask(client, prompt)
        outputs = tool_outputs(stored)
        assert len(outputs) == 2
        refused = [text for text in outputs if text.startswith("Error:")]
        assert refused == [
            "Error: Async subagent capacity reached (1 running). "
            "Wait for one to finish or increase subagents.max_async."
        ]
        assert len([text for text in outputs if '"status": "dispatched"' in text]) == 1
        wait_for_message(client, turn.chat_id, finished_reply("One came back."))

        again = send_message(
            client,
            later,
            chat_id=turn.chat_id,
            parent_id=history_of(client, turn.chat_id)["currentId"],
        )
        [handle] = dispatch_handles(wait_for_reply(client, again))
        assert handle["status"] == "dispatched"
        wait_for_message(client, turn.chat_id, finished_reply("Third came back."))


def test_an_unlimited_background_setting_lets_every_delegation_through(
    background_on, make_user, upstream
):
    background_on(SUBAGENTS_MAX_ASYNC=-1)
    prompt = unique("split this up")
    tasks = [unique("first errand"), unique("second errand"), unique("third errand")]
    calls = [
        {
            "id": f"call_{number}",
            "type": "function",
            "function": {
                "name": "delegate_task",
                "arguments": json.dumps({"task": task, "background": True}),
            },
        }
        for number, task in enumerate(tasks)
    ]
    upstream.queue(
        reply.Reply(tool_calls=calls, match=reply.answering(prompt)),
        *[reply.text("done", delay=SLOW_SECONDS, match=reply.answering(task)) for task in tasks],
        reply.text("Three sent off.", match=reply.answering(prompt)),
    )
    with make_user().client() as client:
        _, stored = ask(client, prompt)

    assert [handle["status"] for handle in dispatch_handles(stored)] == ["dispatched"] * 3


def test_a_background_call_while_background_sub_agents_are_off_runs_as_an_ordinary_delegation(
    background_on, make_user, upstream
):
    background_on(SUBAGENTS_BACKGROUND_ENABLED=False)
    prompt, task = unique("hand this over"), unique("look it up")
    answer = unique("found it")
    upstream.queue(
        dispatch(task, prompt),
        reply.text(answer, match=reply.answering(task)),
        reply.text("It is back.", match=reply.answering(prompt)),
    )
    with make_user().client() as client:
        _, stored = ask(client, prompt)

    assert tool_outputs(stored) == [answer]


def test_a_stopped_background_subagent_is_reported_as_interrupted_and_the_chat_continues(
    background_on, make_user, upstream
):
    background_on()
    prompt, task = unique("hand this over"), unique("watch the queue")
    upstream.queue(
        dispatch(task, prompt),
        reply.text("never used", delay=SLOW_SECONDS * 2, match=reply.answering(task)),
        reply.text("Handed over.", match=reply.answering(prompt)),
        reply.text("It was stopped.", match=report_of(task)),
    )
    with make_user().client() as client:
        turn, stored = ask(client, prompt)
        [handle] = dispatch_handles(stored)
        sub_id = handle["subagent_chat_id"]

        stopped = client.post(f"/api/tasks/chat/{sub_id}/stop")
        assert stopped.status_code == 200, stopped.text
        wait_for_message(client, turn.chat_id, finished_reply("It was stopped."))
        [report] = reports_in(client, turn.chat_id)
        sub_reply = wait_for_message(client, sub_id, lambda m: m["role"] == "assistant")

    assert "Status: interrupted" in report["content"]
    assert "The subagent was interrupted before completing." in report["content"]
    assert "never used" not in report["content"]
    assert sub_reply["done"] is True
    assert sub_reply["error"] == {"content": "Sub-agent cancelled."}


def test_a_failing_background_subagent_is_reported_with_its_error(
    background_on, make_user, upstream
):
    background_on()
    prompt, task = unique("hand this over"), unique("call the broken model")
    upstream.queue(
        dispatch(task, prompt),
        reply.error(500, "the helper model is down", match=reply.answering(task)),
        reply.text("Handed over.", match=reply.answering(prompt)),
        reply.text("The helper failed.", match=report_of(task)),
    )
    with make_user().client() as client:
        turn, _ = ask(client, prompt)
        wait_for_message(client, turn.chat_id, finished_reply("The helper failed."))
        [report] = reports_in(client, turn.chat_id)

    assert "Status: error" in report["content"]
    assert "The subagent did not complete successfully." in report["content"]
    assert "the helper model is down" in report["content"]


def test_two_chats_of_one_account_each_get_only_their_own_report(
    background_on, make_user, upstream
):
    background_on()
    prompts = [unique("hand over the north"), unique("hand over the south")]
    tasks = [unique("count north doors"), unique("count south doors")]
    answers = [unique("north has three"), unique("south has nine")]
    upstream.queue(
        *[dispatch(task, prompt) for task, prompt in zip(tasks, prompts)],
        *[
            reply.text(answer, delay=SLOW_SECONDS, match=reply.answering(task))
            for task, answer in zip(tasks, answers)
        ],
        *[reply.text("Sent off.", match=reply.answering(prompt)) for prompt in prompts],
        *[reply.text(f"Read {task}.", match=report_of(task)) for task in tasks],
    )
    with make_user().client() as client:
        turns = [send_message(client, prompt) for prompt in prompts]
        for turn in turns:
            wait_for_reply(client, turn)
        for turn, task in zip(turns, tasks):
            wait_for_message(client, turn.chat_id, finished_reply(f"Read {task}."))

        for turn, task, answer in zip(turns, tasks, answers):
            [report] = reports_in(client, turn.chat_id)
            other = [a for a in answers if a != answer][0]
            assert task in report["content"] and answer in report["content"]
            assert other not in report["content"]


@pytest.mark.slow
def test_a_concurrent_limit_of_one_does_not_hold_background_subagents_back(instance_with):
    from harness.actors import create_user

    launched = instance_with(CONCURRENT_ENV)
    upstream = launched.upstream
    prompt = unique("split the work")
    tasks = [unique("first errand"), unique("second errand")]
    upstream.queue(
        two_dispatches(tasks, prompt),
        *[reply.text("Done.", delay=SLOW_SECONDS, match=reply.answering(t)) for t in tasks],
        reply.text("Both sent off.", match=reply.answering(prompt)),
    )
    with create_user(launched).client() as client:
        turn = send_message(client, prompt)
        deadline = time.monotonic() + 30
        started: list[dict] = []
        while time.monotonic() < deadline and len(started) < 2:
            started = [
                r
                for r in upstream.chat_requests()
                if any(reply.answering(t)(r) and not report_of(t)(r) for t in tasks)
            ]
            time.sleep(0.05)
        wait_for_reply(client, turn)

    assert len(started) == 2
