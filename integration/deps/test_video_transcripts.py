"""Dependency smoke: a YouTube link attached to a chat, read through youtube-transcript-api.

The YouTube loader opens `YouTubeTranscriptApi` with a `GenericProxyConfig` for the admin's
Youtube Proxy URL, lists the video's caption tracks and walks the admin's languages in order,
English last: the first language with a track wins, a manual track over one YouTube generated,
and a track that turns out empty passes the turn to the next language. The snippets it fetches
are joined with single spaces. YouTube is played by `harness/youtube.py`, which offers several
tracks per video (twin of unit/deps/test_youtube_transcript_api.py).

Discriminates: passes on dev ef67cc3fa. In a backend copy that asks for the generated track and
skips the manual lookup, the manual test fails (the library's own lookup already puts manual
tracks first, and the loader asks again when it gets a generated one); walking the languages in
reverse fails the order test; stopping at an empty track fails the empty track test; and joining
the snippets without stripping them fails the joining test.
"""

from __future__ import annotations

import pytest

from harness import youtube
from harness.web_retrieval import LOCAL_WEB_FETCH, save_web_settings, web_settings_restored
from harness.youtube import Track

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]


@pytest.fixture(scope="module")
def fake_youtube():
    with youtube.serving_youtube() as fake:
        yield fake


@pytest.fixture(scope="module")
def youtube_instance(instance_with, fake_youtube):
    return instance_with({**LOCAL_WEB_FETCH, **youtube.youtube_env(fake_youtube)})


@pytest.fixture
def attach(youtube_instance, fake_youtube):
    """`attach(video_id, languages)` attaches the video with those transcript languages."""
    with youtube_instance.client() as client, web_settings_restored(client):

        def attach_video(video_id: str, languages: list[str]) -> str:
            save_web_settings(
                client,
                YOUTUBE_LOADER_PROXY_URL=fake_youtube.proxy_url,
                YOUTUBE_LOADER_LANGUAGE=languages,
            )
            attached = client.post(
                "/api/v1/retrieval/process/web?process=false",
                json={"url": f"https://www.youtube.com/watch?v={video_id}"},
            )
            assert attached.status_code == 200, attached.text
            return attached.json()["content"]

        yield attach_video


def test_a_manual_transcript_is_preferred_over_a_generated_one(attach, fake_youtube):
    fake_youtube.videos["manualfirst"] = youtube.with_tracks(
        Track("en", ("generated words",), generated=True),
        Track("en", ("written by hand",)),
    )

    assert attach("manualfirst", ["en"]) == "written by hand"


def test_a_generated_transcript_is_used_when_nobody_wrote_one(attach, fake_youtube):
    fake_youtube.videos["generated01"] = youtube.with_tracks(
        Track("en", ("only the machine heard this",), generated=True),
    )

    assert attach("generated01", ["en"]) == "only the machine heard this"


def test_the_languages_are_tried_in_the_admins_order_then_english(attach, fake_youtube):
    fake_youtube.videos["twolanguage"] = youtube.with_tracks(
        Track("en", ("good morning",)),
        Track("de", ("guten morgen",)),
    )

    assert attach("twolanguage", ["de"]) == "guten morgen"
    assert attach("twolanguage", ["fr", "de"]) == "guten morgen"
    assert attach("twolanguage", ["fr"]) == "good morning"


def test_an_empty_track_passes_the_turn_to_the_next_language(attach, fake_youtube):
    fake_youtube.videos["emptygerman"] = youtube.with_tracks(
        Track("de", ()),
        Track("en", ("the english track",)),
    )

    assert attach("emptygerman", ["de"]) == "the english track"


def test_the_snippets_are_joined_with_single_spaces(attach, fake_youtube):
    fake_youtube.videos["spacedlines"] = youtube.with_tracks(
        Track("en", ("  the keeper ", "lights the lamp  ", " at dusk")),
    )

    assert attach("spacedlines", ["en"]) == "the keeper lights the lamp at dusk"
