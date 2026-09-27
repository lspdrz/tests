"""The leaderboard activity chart was empty for model ids containing a slash, #30455.

Fix commit `7e2bea825` (open-webui/open-webui#30456). The chart reads
`GET /api/v1/evaluations/leaderboard/{model_id}/history`, with the id URL-encoded by the web
client. The server decodes `%2F` before routing, so an id such as `hf.co/org/model:Q4` split into
extra path segments and matched no API route, so the chart stayed empty for Hugging Face models
served through Ollama. The route now takes the id as a path parameter.

Discriminates: passes on dev efe63bd34; with 7e2bea825 reverted no route matches the slashed ids
(the request falls through to the web app's page, or answers 404 without a frontend), so the
slashed cases fail (the admin-only check among them) and the plain id passes.
"""

from __future__ import annotations

import uuid
from urllib.parse import quote

import pytest

from harness.actors import Actor

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def _rate(actor: Actor, model_id: str, rating: int) -> None:
    with actor.client() as client:
        created = client.post(
            "/api/v1/evaluations/feedback",
            json={"type": "rating", "data": {"rating": rating, "model_id": model_id}},
        )
    assert created.status_code == 200, created.text


def _history(admin: Actor, model_id: str) -> dict:
    # encoded the way the leaderboard's chart requests it
    with admin.client() as client:
        response = client.get(
            f"/api/v1/evaluations/leaderboard/{quote(model_id, safe='')}/history",
            params={"days": 7},
        )
    is_history = response.status_code == 200 and response.headers["content-type"].startswith(
        "application/json"
    )
    # an unrouted API path falls through to the web app's HTML page
    assert is_history, f"no history route matched {model_id!r} (#30455): {response.text[:80]}"
    return response.json()


def _totals(history: dict) -> tuple[int, int]:
    return (
        sum(day["won"] for day in history["history"]),
        sum(day["lost"] for day in history["history"]),
    )


@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param("hf.co/unsloth/model-{tag}:Q4_K_M", id="ollama-hugging-face"),
        pytest.param("org/model-{tag}", id="one-slash"),
    ],
)
def test_the_history_of_a_model_id_with_a_slash_is_served(admin, make_user, model_id):
    model_id = model_id.format(tag=uuid.uuid4().hex[:8])
    rater = make_user()
    _rate(rater, model_id, 1)
    _rate(rater, model_id, 1)
    _rate(rater, model_id, -1)

    history = _history(admin, model_id)

    assert history["model_id"] == model_id
    assert _totals(history) == (2, 1)


# ---------------------------------------------------------------- nearby


def test_a_plain_model_id_still_has_its_own_history(admin, make_user):
    model_id = f"plain-{uuid.uuid4().hex[:8]}"
    rater = make_user()
    _rate(rater, model_id, -1)
    _rate(rater, f"{model_id}/other", 1)

    history = _history(admin, model_id)

    assert history["model_id"] == model_id
    assert _totals(history) == (0, 1)
    assert len(history["history"]) == 7


def test_the_history_stays_admin_only_for_slashed_ids(user):
    with user.client() as client:
        response = client.get(
            f"/api/v1/evaluations/leaderboard/{quote('org/model', safe='')}/history"
        )
    assert response.status_code in (401, 403), response.text[:80]
