"""Regression: a browser reporting a bare or regional language code got the English interface.

Fix `d603b5799` (open-webui/open-webui#30377): Firefox reports German, Dutch and Polish as bare
codes (`de`, `nl`, `pl`), both Chrome and Firefox report Japanese as `ja`, and Chrome in Latin
America reports `es-419`. No translation bundle is keyed by any of them. Since 0.11.4 the
language detector cached the raw code before anything matched it to a bundle, so these users got
English and kept it on every later visit. The detector now maps a reported code onto a bundle
(`de` and `de-AT` to `de-DE`, `es-419` to `es-ES`, `fr` to `fr-FR`, `ja` to `ja-JP`); a code
with a bundle of its own stays as it is.

A signed-out browser opens the sign-in page with that language as its only preference.

Discriminates: passes on the efe63bd34 build; on a build with d603b5799 reverted every mapped
case fails (the raw code is stored as the locale and, but for `fr`, the page shows English).
The unchanged cases pass on both.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Browser, expect

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

SIGN_IN_LABEL = {
    "de-DE": "Anmelden",
    "nl-NL": "Inloggen",
    "pl-PL": "Zaloguj się",
    "ja-JP": "サインイン",
    "es-ES": "Iniciar Sesión",
    "fr-FR": "Connexion",
    "fr-CA": "Connexion",
    "pt-BR": "Entrar",
}


@pytest.fixture
def open_sign_in(browser: Browser, config):
    """`open_sign_in(reported)` opens the sign-in page in a browser that reports `reported`."""
    contexts = []

    def open_page(reported: str):
        context = browser.new_context(locale=reported, base_url=config.base_url)
        context.set_default_timeout(config.default_timeout)
        contexts.append(context)
        page = context.new_page()
        page.goto("/auth")
        return page

    yield open_page
    for context in contexts:
        context.close()


@pytest.mark.parametrize(
    ("reported", "bundle"),
    [
        ("de", "de-DE"),
        ("de-AT", "de-DE"),
        ("nl", "nl-NL"),
        ("pl", "pl-PL"),
        ("ja", "ja-JP"),
        ("es-419", "es-ES"),
        ("fr", "fr-FR"),
    ],
)
def test_a_reported_language_gets_its_translation(open_sign_in, reported, bundle):
    page = open_sign_in(reported)

    expect(page.get_by_role("button", name=SIGN_IN_LABEL[bundle], exact=True)).to_be_visible()
    assert page.evaluate("localStorage.locale") == bundle, (
        f"a browser reporting {reported!r} was not matched to the {bundle} translation"
    )


@pytest.mark.parametrize(
    ("reported", "label"),
    [("fr-CA", "Connexion"), ("pt-BR", "Entrar"), ("en-US", "Sign in")],
)
def test_a_language_with_its_own_bundle_keeps_it(open_sign_in, reported, label):
    page = open_sign_in(reported)

    expect(page.get_by_role("button", name=label, exact=True)).to_be_visible()
    assert page.evaluate("localStorage.locale") == reported


def test_a_regional_code_of_another_script_is_not_moved_to_a_sibling(open_sign_in):
    page = open_sign_in("zh-HK")

    expect(page.get_by_role("button", name="Sign in", exact=True)).to_be_visible()
    assert page.evaluate("localStorage.locale") == "zh-HK", (
        "Traditional Chinese from Hong Kong was sent to the Simplified zh-CN translation"
    )
