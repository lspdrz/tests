"""Journey: image engines reached by host name, with the threaded and the c-ares resolver.

`AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` decides which resolver the shared aiohttp pool and the
SSRF-safe connector use, and every call to an image engine an admin can point at a local service
goes through one of them. Each test drives one engine by a host name (`localhost`, a hosts-file
name for `::1` alone and a hosts-file name for another local address; ComfyUI answers on
`localhost` only, where the fake binds one address) and expects the same outcome under both
resolvers: Automatic1111's checkpoint switch, model list, connection check and generation, OpenAI's
generation (a picture sent inline and one fetched from a link on the engine) and edit, Gemini's two
generation methods and edit, ComfyUI's connection check, model list, generation and edit, and an
edit of a picture given by link. An engine whose name does not resolve fails each of those calls
with the same status and message, the resolver's own wording aside.

Discriminates: on dev 176d31d1d, a backend copy whose `env.py` installs a resolver that fails every
lookup when the flag is on turns every c-ares run of a by-name test red and leaves every threaded
run green; the same resolver installed for the flag off does the reverse. A failure test stays green
under a failing resolver by design, and a resolver that takes 20 seconds to refuse an unknown name
turns it red. Its c-ares runs skip, naming the reason, on a machine whose DNS server keeps c-ares
from refusing an unknown name at once (a cached reply with a stale EDNS cookie, c-ares issues 1081
and 1271).
"""

from __future__ import annotations

import base64

import pytest

from harness.actors import admin_of
from harness.comfyui import CHECKPOINTS, PNG, FakeComfyUI, comfyui_settings, serving, workflow
from harness.host_names import (
    FAILS_WITHIN,
    RESOLVERS,
    UNRESOLVABLE,
    by_name,
    name_forms,
    serving_by_name,
    timed,
)
from harness.image_engines import (
    CHECKPOINT,
    GEMINI_API_KEY,
    GEMINI_IMAGE_MODEL,
    IMAGEN_MODEL,
    IMAGES_CONFIG,
    PNG_BASE64,
    checkpoint_switches,
    gemini_calls,
    save_image_settings,
    serve_automatic1111,
    serve_gemini,
)
from harness.listener import json_answer
from harness.web_retrieval import LOCAL_WEB_FETCH

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

GENERATED_PNG = base64.b64decode(PNG_BASE64)
SOURCE_DATA_URL = f"data:image/png;base64,{PNG_BASE64}"
OTHER_CHECKPOINT = "switched.safetensors"
UNRESOLVABLE_URL = f"http://{UNRESOLVABLE}:8000"
NO_DETAIL = "Something went wrong :/"
INVALID_URL = "The URL you provided is invalid. Please double-check and try again."
ENGINES = ["automatic1111", "openai", "gemini", "comfyui"]
# what the admin reads when the engine's name does not resolve, the same under both resolvers
GENERATION_FAILURES = {
    "automatic1111": "[ERROR: Failed to connect to the image generation engine]",
    "openai": NO_DETAIL,
    "gemini": NO_DETAIL,
    "comfyui": NO_DETAIL,
}


@pytest.fixture
def fetching_instance(resolver, package_instance_with):
    """The resolver's instance, allowed to fetch a picture a local engine links to."""
    return package_instance_with({**RESOLVERS[resolver], **LOCAL_WEB_FETCH})


@pytest.fixture
def fetching_admin(fetching_instance):
    return admin_of(fetching_instance)


@pytest.fixture
def image_settings(fetching_instance, fetching_admin, preserve):
    """`image_settings(**settings)` points the admin's image settings at an engine."""
    preserve(IMAGES_CONFIG, on=fetching_instance)

    def configure(**settings) -> None:
        with fetching_admin.client() as client:
            save_image_settings(client, **settings)

    return configure


def _automatic1111_settings(base_url: str) -> dict:
    return {
        "ENABLE_IMAGE_GENERATION": True,
        "IMAGE_GENERATION_ENGINE": "automatic1111",
        "AUTOMATIC1111_BASE_URL": base_url,
        "IMAGE_GENERATION_MODEL": CHECKPOINT,
        "IMAGE_SIZE": "512x512",
        "IMAGE_STEPS": 20,
    }


def _openai_settings(base_url: str) -> dict:
    return {
        "ENABLE_IMAGE_GENERATION": True,
        "IMAGE_GENERATION_ENGINE": "openai",
        "IMAGES_OPENAI_API_BASE_URL": base_url,
        "IMAGES_OPENAI_API_KEY": "sk-images",
        "IMAGE_GENERATION_MODEL": "dall-e-2",
        "IMAGE_SIZE": "512x512",
        "ENABLE_IMAGE_EDIT": True,
        "IMAGE_EDIT_ENGINE": "openai",
        "IMAGE_EDIT_MODEL": "gpt-image-1",
        "IMAGES_EDIT_OPENAI_API_BASE_URL": base_url,
        "IMAGES_EDIT_OPENAI_API_KEY": "sk-images",
    }


def _gemini_settings(base_url: str) -> dict:
    return {
        "ENABLE_IMAGE_GENERATION": True,
        "IMAGE_GENERATION_ENGINE": "gemini",
        "IMAGE_GENERATION_MODEL": IMAGEN_MODEL,
        "IMAGES_GEMINI_API_BASE_URL": base_url,
        "IMAGES_GEMINI_API_KEY": GEMINI_API_KEY,
        "IMAGES_GEMINI_ENDPOINT_METHOD": "predict",
        "ENABLE_IMAGE_EDIT": True,
        "IMAGE_EDIT_ENGINE": "gemini",
        "IMAGE_EDIT_MODEL": GEMINI_IMAGE_MODEL,
        "IMAGES_EDIT_GEMINI_API_BASE_URL": base_url,
        "IMAGES_EDIT_GEMINI_API_KEY": GEMINI_API_KEY,
    }


def _serve_openai(listener) -> dict:
    """Answer as OpenAI's image API, one picture inline and one behind a link on the engine."""
    linked = {"url": f"{listener.base_url}/pictures/linked.png"}
    listener.route(
        "POST", "/images/generations", json_answer({"data": [{"b64_json": PNG_BASE64}, linked]})
    )
    listener.route("POST", "/images/edits", json_answer({"data": [{"b64_json": PNG_BASE64}]}))
    listener.route(
        "GET", "/pictures/linked.png", (200, {"Content-Type": "image/png"}, GENERATED_PNG)
    )
    return _openai_settings(listener.base_url)


def _stored(client, image: dict) -> bytes:
    fetched = client.get(image["url"])
    assert fetched.status_code == 200, fetched.text
    return fetched.content


@pytest.mark.parametrize("name_form", name_forms())
def test_automatic1111_answers_every_call_by_name(
    resolver, name_form, image_settings, fetching_admin
):
    with serving_by_name(name_form) as listener, fetching_admin.client() as client:
        serve_automatic1111(listener)
        listener.route(
            "GET",
            "/sdapi/v1/sd-models",
            json_answer([{"title": CHECKPOINT, "model_name": "configured"}]),
        )
        image_settings(**_automatic1111_settings(listener.base_url))
        verified = client.post(
            "/api/v1/images/verify",
            json={"engine": "automatic1111", "url": listener.base_url, "key": ""},
        )
        models = client.get("/api/v1/images/models")
        generated = client.post(
            "/api/v1/images/generations", json={"prompt": "a lighthouse", "model": OTHER_CHECKPOINT}
        )
        images = generated.json()
        stored = _stored(client, images[0]) if generated.status_code == 200 else None

    assert verified.status_code == 200, verified.text
    assert models.status_code == 200, models.text
    assert models.json() == [{"id": CHECKPOINT, "name": "configured"}]
    assert generated.status_code == 200, generated.text
    assert stored == GENERATED_PNG
    assert [switch["sd_model_checkpoint"] for switch in checkpoint_switches(listener)] == [
        OTHER_CHECKPOINT
    ]
    assert len(listener.requests_to("/sdapi/v1/txt2img")) == 1


@pytest.mark.parametrize("name_form", name_forms())
def test_openai_generates_and_edits_by_name(resolver, name_form, image_settings, fetching_admin):
    with serving_by_name(name_form) as listener, fetching_admin.client() as client:
        image_settings(**_serve_openai(listener))
        generated = client.post("/api/v1/images/generations", json={"prompt": "a kite", "n": 2})
        edited = client.post(
            "/api/v1/images/edit", json={"image": SOURCE_DATA_URL, "prompt": "make it blue"}
        )
        pictures = [_stored(client, image) for image in [*generated.json(), *edited.json()]]

    assert generated.status_code == 200, generated.text
    assert edited.status_code == 200, edited.text
    assert pictures == [GENERATED_PNG] * 3
    assert len(listener.requests_to("/images/generations")) == 1
    assert len(listener.requests_to("/images/edits")) == 1
    assert len(listener.requests_to("/pictures/linked.png")) == 1


@pytest.mark.parametrize("name_form", name_forms())
def test_gemini_generates_and_edits_by_name(resolver, name_form, image_settings, fetching_admin):
    with serving_by_name(name_form) as listener, fetching_admin.client() as client:
        serve_gemini(listener)
        image_settings(**_gemini_settings(listener.base_url))
        predicted = client.post("/api/v1/images/generations", json={"prompt": "a lighthouse"})
        image_settings(
            IMAGE_GENERATION_MODEL=GEMINI_IMAGE_MODEL,
            IMAGES_GEMINI_ENDPOINT_METHOD="generateContent",
        )
        generated = client.post("/api/v1/images/generations", json={"prompt": "a red kite"})
        edited = client.post(
            "/api/v1/images/edit", json={"image": SOURCE_DATA_URL, "prompt": "make it blue"}
        )
        pictures = [_stored(client, image) for image in [*predicted.json(), *generated.json()]]
        pictures += [_stored(client, image) for image in edited.json()]

    for answered in (predicted, generated, edited):
        assert answered.status_code == 200, answered.text
    assert pictures == [GENERATED_PNG] * 3
    assert len(gemini_calls(listener, IMAGEN_MODEL, "predict")) == 1
    assert len(gemini_calls(listener, GEMINI_IMAGE_MODEL, "generateContent")) == 2


def test_comfyui_answers_every_call_by_name(resolver, image_settings, fetching_admin):
    fake = FakeComfyUI()
    with serving(fake) as base_url, fetching_admin.client() as client:
        named = by_name(base_url, "localhost")
        image_settings(**comfyui_settings(named, workflow()))
        verified = client.post(
            "/api/v1/images/verify", json={"engine": "comfyui", "url": named, "key": ""}
        )
        models = client.get("/api/v1/images/models")
        generated = client.post("/api/v1/images/generations", json={"prompt": "a lighthouse"})
        edited = client.post(
            "/api/v1/images/edit", json={"image": SOURCE_DATA_URL, "prompt": "add fog"}
        )
        pictures = [_stored(client, image) for image in [*generated.json(), *edited.json()]]

    assert verified.status_code == 200, verified.text
    assert models.status_code == 200, models.text
    assert [model["id"] for model in models.json()] == CHECKPOINTS
    assert generated.status_code == 200, generated.text
    assert edited.status_code == 200, edited.text
    assert pictures == [PNG, PNG]
    assert len(fake.queued) == 2


@pytest.mark.parametrize("name_form", name_forms())
def test_a_picture_given_by_link_is_fetched_and_edited_by_name(
    resolver, name_form, image_settings, fetching_admin
):
    with serving_by_name(name_form) as listener, fetching_admin.client() as client:
        serve_gemini(listener)
        listener.route("GET", "/source.png", (200, {"Content-Type": "image/png"}, GENERATED_PNG))
        image_settings(**_gemini_settings(listener.base_url))
        edited = client.post(
            "/api/v1/images/edit",
            json={"image": f"{listener.base_url}/source.png", "prompt": "make it blue"},
        )

    assert edited.status_code == 200, edited.text
    assert len(listener.requests_to("/source.png")) == 1
    [call] = gemini_calls(listener, GEMINI_IMAGE_MODEL, "generateContent")
    assert call.json()["contents"][0]["parts"][1]["inline_data"]["data"] == PNG_BASE64


def _unresolvable_engine(engine: str) -> dict:
    if engine == "comfyui":
        return comfyui_settings(UNRESOLVABLE_URL, workflow())
    build = {
        "automatic1111": _automatic1111_settings,
        "openai": _openai_settings,
        "gemini": _gemini_settings,
    }[engine]
    return build(UNRESOLVABLE_URL)


@pytest.mark.usefixtures("refuses_unknown_names")
@pytest.mark.parametrize("engine", ENGINES)
def test_an_unresolvable_engine_fails_generation_the_same_way(
    resolver, engine, image_settings, fetching_admin
):
    with fetching_admin.client() as client:
        image_settings(**_unresolvable_engine(engine))
        generated, seconds = timed(
            client.post, "/api/v1/images/generations", json={"prompt": "a lighthouse"}
        )

    assert generated.status_code == 400, generated.text
    assert generated.json()["detail"] == GENERATION_FAILURES[engine]
    assert seconds < FAILS_WITHIN, f"the lookup failed only after {seconds:.1f}s"


@pytest.mark.usefixtures("refuses_unknown_names")
@pytest.mark.parametrize("engine", ["openai", "gemini", "comfyui"])
def test_an_unresolvable_engine_fails_an_edit_the_same_way(
    resolver, engine, image_settings, fetching_admin
):
    with fetching_admin.client() as client:
        image_settings(**_unresolvable_engine(engine))
        edited, seconds = timed(
            client.post,
            "/api/v1/images/edit",
            json={"image": SOURCE_DATA_URL, "prompt": "make it blue"},
        )

    assert edited.status_code == 400, edited.text
    assert edited.json()["detail"] == GENERATION_FAILURES[engine]
    assert seconds < FAILS_WITHIN, f"the lookup failed only after {seconds:.1f}s"


@pytest.mark.usefixtures("refuses_unknown_names")
@pytest.mark.parametrize("engine", ["automatic1111", "comfyui"])
def test_an_unresolvable_engine_fails_verification_and_listing_the_same_way(
    resolver, engine, image_settings, fetching_admin
):
    settings = _unresolvable_engine(engine)
    with fetching_admin.client() as client:
        image_settings(**settings)
        verified, verify_seconds = timed(
            client.post,
            "/api/v1/images/verify",
            json={"engine": engine, "url": UNRESOLVABLE_URL, "key": ""},
        )
        models, model_seconds = timed(client.get, "/api/v1/images/models")

    assert verified.status_code == 400, verified.text
    assert verified.json()["detail"] == INVALID_URL
    assert models.status_code == 400, models.text
    assert models.json()["detail"] == "[ERROR: Failed to retrieve image generation models]"
    assert max(verify_seconds, model_seconds) < FAILS_WITHIN, "the lookup failed slowly"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_an_unresolvable_link_fails_an_edit_the_same_way(resolver, image_settings, fetching_admin):
    with fetching_admin.client() as client:
        image_settings(**_gemini_settings(UNRESOLVABLE_URL))
        edited, seconds = timed(
            client.post,
            "/api/v1/images/edit",
            json={"image": f"{UNRESOLVABLE_URL}/source.png", "prompt": "make it blue"},
        )

    assert edited.status_code == 400, edited.text
    assert edited.json()["detail"] == "[ERROR: Error loading image]"
    assert seconds < FAILS_WITHIN, f"the lookup failed only after {seconds:.1f}s"
