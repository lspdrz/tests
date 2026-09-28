"""Guard: in-process registries shrink again once their entries are dead.

Each test feeds one of the server's in-memory registries many distinct keys through a real route
and asks the probe of `harness.process_probe` whether the server lets go or never kept them:
sign-in attempts once their rate-limit window has passed, rejected model avatars, finished
replies. A failure here is memory only a restart reclaims; each entry is a few bytes, far below
what the process size can show, so the probe reads the containers themselves. Deleted plugins
giving their source back is measured on the process size in
`test_process_memory_stays_bounded.py`.

Twin of unit/footprint/test_unbounded_process_state.py (its rate limiter, finished task and
profile image cases). Tasks filed under an empty item id stay a unit test: no route creates one.

Unpinned: read on upstream dev at v0.11.3 (a253bf0c3); upstream has since fixed two of the
cases (#29977, #29971), so they assert the fixed behaviour. Unmarked because no issue is filed.
Discriminates: passes on dev ef67cc3fa except the per-chat lock test, which is red there
(`_parent_locks` in `utils/subagents.py` keeps one lock per chat that ever got a reply). In a
copy of dev, never pruning the limiter's expired buckets fails the sign-in test, keeping every
rejected avatar URL in a module-level set (#29971) fails the avatar test, and skipping
`cleanup_task` for a finished reply fails the finished task test.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask, wait_for_reply
from harness.inflight import start_slow_reply
from harness.process_probe import grown, probing

pytestmark = [pytest.mark.api, pytest.mark.requires_source]

ATTEMPTS = 20
RATE_LIMIT_BUCKET_SECONDS = 60
# a bucket is dropped once it is older than the three-minute sign-in window
BUCKETS_UNTIL_DROPPED = 4
SVG_AVATAR = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciLz4="
PNG_AVATAR = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _sign_in_attempts(instance, domain: str) -> None:
    with instance.client() as client:
        for attempt in range(ATTEMPTS):
            refused = client.post(
                "/api/v1/auths/signin",
                json={"email": f"nobody{attempt}-{uuid.uuid4().hex[:6]}@{domain}", "password": "x"},
            )
            assert refused.status_code == 400, refused.text


def _current_bucket() -> int:
    return int(time.time()) // RATE_LIMIT_BUCKET_SECONDS


@pytest.mark.slow
def test_sign_in_attempts_are_forgotten_once_their_window_has_passed(instance, admin):
    """Without Redis the limiter counts attempts in memory, by the email the request names."""
    if instance.redis_url:
        pytest.skip("with Redis the limiter keeps its counts there, with an expiry of their own")
    early_domain, late_domain = (f"{uuid.uuid4().hex[:8]}.example.com" for _ in range(2))

    with probing(admin) as probe:
        _sign_in_attempts(instance, early_domain)
        last_early_bucket = _current_bucket()
        retained_in_window = probe.strings_containing(early_domain)
        while _current_bucket() < last_early_bucket + BUCKETS_UNTIL_DROPPED:
            time.sleep(1)
        _sign_in_attempts(instance, late_domain)
        retained_after_window = probe.strings_containing(early_domain)
        retained_late = probe.strings_containing(late_domain)

    assert retained_in_window >= ATTEMPTS, "the probe cannot see the limiter's keys"
    assert retained_late >= ATTEMPTS, "the probe cannot see the limiter's keys"
    assert retained_after_window == 0, (
        f"{retained_after_window} sign-in emails are still held after their window passed; "
        "every email anyone ever tried is kept until a restart"
    )


def test_rejected_avatar_urls_are_not_kept(admin, make_user):
    """A model's form is checked before the workspace permission, so any account reaches it."""
    person = make_user()
    with probing(admin) as probe, person.client() as client:
        before = probe.containers()
        for attempt in range(ATTEMPTS):
            model_id = f"avatar-{uuid.uuid4().hex[:8]}"
            refused = client.post(
                "/api/v1/models/create",
                json={
                    "id": model_id,
                    "name": model_id,
                    "meta": {"profile_image_url": f"{SVG_AVATAR}{attempt}"},
                    "params": {},
                },
            )
            assert refused.status_code in (401, 403), refused.text
        after = probe.containers()

    retained = grown(before, after, by=ATTEMPTS)
    assert not retained, f"every rejected avatar URL was kept, in {retained} (#29971)"


@pytest.mark.parametrize(
    "avatar, stored", [(SVG_AVATAR, None), (PNG_AVATAR, PNG_AVATAR)], ids=["svg", "png"]
)
def test_an_svg_avatar_is_cleared_and_a_png_is_kept(admin, avatar, stored):
    model_id = f"avatar-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "name": model_id,
                "meta": {"profile_image_url": avatar},
                "params": {},
            },
        )
        assert created.status_code == 200, created.text
        try:
            model = client.get("/api/v1/models/model", params={"id": model_id}).json()
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})

    assert model["meta"]["profile_image_url"] == stored


def test_a_finished_reply_leaves_its_chat_task_list(make_user, upstream):
    with make_user().client() as client:
        turn = start_slow_reply(client, upstream, chunk_delay=0.05)
        assert len(_running_tasks(client, turn.chat_id)) == 1, (
            "the reply is not listed while running"
        )
        wait_for_reply(client, turn)
        deadline = time.monotonic() + 5
        while _running_tasks(client, turn.chat_id) and time.monotonic() < deadline:
            time.sleep(0.05)
        still_listed = _running_tasks(client, turn.chat_id)

    assert still_listed == [], "a finished reply is still filed under its chat"


def _running_tasks(client, chat_id: str) -> list[str]:
    return client.get(f"/api/tasks/chat/{chat_id}").json()["task_ids"]


def test_finished_chats_leave_nothing_behind_per_chat(admin, make_user, upstream):
    """Every reply ends by looking for queued subagent results, taking a lock keyed by the chat;
    the lock is never dropped, so the server holds one for every chat ever answered."""
    person = make_user()
    with probing(admin) as probe, person.client() as client:
        before = probe.containers()
        for attempt in range(ATTEMPTS):
            prompt = f"chat number {attempt}"
            upstream.queue(reply.text("fine", match=reply.answering(prompt)))
            ask(client, prompt)
        after = probe.containers()

    retained = grown(before, after, by=ATTEMPTS)
    assert not retained, (
        f"{ATTEMPTS} finished chats left an entry each in {retained}: the server keeps "
        "something for every chat ever answered until a restart"
    )
