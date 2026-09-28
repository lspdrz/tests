"""Regression: a YouTube link whose transcript YouTube refuses names the reason.

open-webui issue #28361, PR #28362 (commit 121f2404e): the transcript loader caught the
transcript API's exception and returned no documents, so attaching a YouTube link failed with a
generic error and the reason, and the proxy hint that fixes a blocked server, never reached the
user. `YoutubeTranscriptError` now carries a reason per cause, and the attach route answers it.

YouTube is played by `harness.youtube`, set as the admin's Youtube Proxy URL on an instance that
trusts its certificate authority, so every page the transcript API reads comes from the test.

Twin of the YouTube part of unit/retrieval/test_web_loaders.py.

Discriminates: passes on dev ef67cc3fa; with the loader returning `[]` when YouTube refuses the
transcript list (121f2404e reverted) every refusal but the proof-of-origin one (refused later, at
the transcript itself) fails with the generic "Could not read content" error, and with
`_transcript_error_message` answering one fixed message every refusal fails but the generic one;
the transcript and language cases pass on both.
"""

from __future__ import annotations

import pytest

from harness import youtube
from harness.web_retrieval import LOCAL_WEB_FETCH, save_web_settings, web_settings_restored

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

PROXY_HINT = "Youtube Proxy URL"
# video id: (what YouTube answers, what the error must say)
REFUSALS = {
    "botcheck001": (youtube.unplayable("LOGIN_REQUIRED", youtube.BOT_CHECK), PROXY_HINT),
    "recaptcha01": (youtube.blocked_page(), PROXY_HINT),
    "ratelimit01": (youtube.rate_limited(), PROXY_HINT),
    "nocaptions1": (youtube.without_captions(), "Transcripts are disabled"),
    "agerestrict": (youtube.unplayable("LOGIN_REQUIRED", youtube.AGE_CHECK), "age restricted"),
    "unavailable": (youtube.unplayable("ERROR", youtube.UNAVAILABLE), "is unavailable"),
    "privatevid1": (youtube.unplayable("UNPLAYABLE", "Video unavailable"), "is unavailable"),
    "pottoken001": (youtube.needs_verification(), "additional verification"),
}


@pytest.fixture(scope="module")
def fake_youtube():
    with youtube.serving_youtube() as fake:
        yield fake


@pytest.fixture(scope="module")
def youtube_instance(instance_with, fake_youtube):
    return instance_with({**LOCAL_WEB_FETCH, **youtube.youtube_env(fake_youtube)})


@pytest.fixture
def admin_client(youtube_instance, fake_youtube):
    with youtube_instance.client() as client, web_settings_restored(client):
        save_web_settings(
            client, YOUTUBE_LOADER_PROXY_URL=fake_youtube.proxy_url, YOUTUBE_LOADER_LANGUAGE=["en"]
        )
        yield client


def attach(client, url: str):
    return client.post("/api/v1/retrieval/process/web?process=false", json={"url": url})


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


@pytest.mark.parametrize("video_id", REFUSALS)
def test_a_refused_transcript_names_its_reason(admin_client, fake_youtube, video_id):
    answer, reason = REFUSALS[video_id]
    fake_youtube.videos[video_id] = answer

    refused = attach(admin_client, watch_url(video_id))

    assert refused.status_code == 400, refused.text
    detail = refused.json()["detail"]
    assert reason in detail, f"the reason never reached the user (#28361): {detail}"
    assert video_id in detail


def test_a_link_without_a_video_id_is_unavailable(admin_client, fake_youtube):
    link = "https://www.youtube.com/watch?v=short"
    fake_youtube.videos[link] = youtube.unplayable("ERROR", youtube.UNAVAILABLE)

    refused = attach(admin_client, link)

    assert refused.status_code == 400, refused.text
    assert "is unavailable" in refused.json()["detail"]


def test_youtube_failing_outright_is_a_generic_transcript_error(admin_client, fake_youtube):
    fake_youtube.videos["servererror"] = youtube.Video(status=500)

    refused = attach(admin_client, watch_url("servererror"))

    assert refused.status_code == 400, refused.text
    assert "Could not retrieve a transcript" in refused.json()["detail"]
    assert "servererror" in refused.json()["detail"]


def test_a_missing_language_names_the_languages_tried(admin_client, fake_youtube):
    fake_youtube.videos["frenchonly1"] = youtube.with_transcript("bonjour", language_code="fr")
    save_web_settings(admin_client, YOUTUBE_LOADER_LANGUAGE=["de"])

    refused = attach(admin_client, watch_url("frenchonly1"))

    assert refused.status_code == 400, refused.text
    assert "de, en" in refused.json()["detail"]


def test_a_transcript_is_attached(admin_client, fake_youtube):
    fake_youtube.videos["dQw4w9WgXcQ"] = youtube.with_transcript("never gonna", "give you up")

    attached = attach(admin_client, watch_url("dQw4w9WgXcQ"))

    assert attached.status_code == 200, attached.text
    assert attached.json()["content"] == "never gonna give you up"
    assert fake_youtube.requested("/api/timedtext"), "the transcript was never fetched"


def test_a_short_link_is_attached_too(admin_client, fake_youtube):
    fake_youtube.videos["shortlink01"] = youtube.with_transcript("from a short link")

    attached = attach(admin_client, "https://youtu.be/shortlink01")

    assert attached.status_code == 200, attached.text
    assert attached.json()["content"] == "from a short link"
