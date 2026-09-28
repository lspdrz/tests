"""Journey: banners are set by admins and readable by every signed-in account.

An admin saves a list of banners and reads it back unchanged, in order; a regular user can read
the list and cannot change it.

Discriminates: passes on dev 176d31d1d; in a backend copy, with the save route open to any
verified user the refusal test fails (the user's save is accepted).
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

BANNERS = "/api/v1/configs/banners"

FIRST = {
    "id": "banner-api-first",
    "type": "warning",
    "title": "First",
    "content": "first notice",
    "dismissible": False,
    "timestamp": 1767225600,
}
SECOND = {
    **FIRST,
    "id": "banner-api-second",
    "type": "success",
    "content": "second notice",
    "dismissible": True,
}


@pytest.fixture
def admin_client(admin):
    """The admin's client; the instance's own banners are put back afterwards."""
    with admin.client() as client:
        before = client.get(BANNERS).json()
        yield client
        client.post(BANNERS, json={"banners": before}).raise_for_status()


def test_saved_banners_read_back_in_order_with_their_fields(admin_client):
    saved = admin_client.post(BANNERS, json={"banners": [FIRST, SECOND]})
    assert saved.status_code == 200, saved.text

    listed = admin_client.get(BANNERS)

    assert [{key: banner[key] for key in FIRST} for banner in listed.json()] == [FIRST, SECOND]


def test_a_regular_user_reads_the_banners(admin_client, make_user):
    admin_client.post(BANNERS, json={"banners": [FIRST]}).raise_for_status()

    with make_user().client() as client:
        listed = client.get(BANNERS)

    assert listed.status_code == 200, listed.text
    assert [banner["id"] for banner in listed.json()] == [FIRST["id"]]


def test_a_regular_user_cannot_change_the_banners(admin_client, make_user):
    admin_client.post(BANNERS, json={"banners": [FIRST]}).raise_for_status()

    with make_user().client() as client:
        refused = client.post(BANNERS, json={"banners": []})

    assert refused.status_code in (401, 403), refused.text
    assert [banner["id"] for banner in admin_client.get(BANNERS).json()] == [FIRST["id"]]
