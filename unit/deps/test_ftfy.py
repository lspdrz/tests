"""Dependency contract: ftfy, the property no Open WebUI request relies on.

Open WebUI runs every loaded document through ``ftfy.fix_text(page_content, unescape_html=False)``
in ``retrieval/loaders/main.py``; that is driven from outside in
integration/deps/test_document_extraction.py (mojibake and smart quotes repaired, control
characters and terminal colours removed, a literal ``&amp;`` kept, a blank PDF page read, clean
and non-Latin text passed through).

Kept as a unit contract: that fixing already-fixed text changes nothing. Open WebUI fixes each
document once, so no request runs ``fix_text`` over its own output. Uses the ``depcheck`` fixture
from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "ftfy"


def test_behaviour_idempotent(depcheck):
    """Re-fixing already-fixed text must be a no-op. The loader could plausibly
    process re-ingested content; a non-idempotent fix would corrupt it. Pin
    fix_text(fix_text(x)) == fix_text(x)."""
    mod = depcheck.load(IMPORT_NAME)
    for sample in (
        "plain ascii",
        "The Mona Lisa doesnâ€™t have eyebrows.",
        "café résumé naïve",
        "He said â€œhelloâ€.",
    ):
        once = mod.fix_text(sample)
        twice = mod.fix_text(once)
        assert twice == once, f"fix_text not idempotent for {sample!r}"
