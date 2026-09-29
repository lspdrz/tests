"""Two instances on one database, one per JSON codec, and the text people really put into them.

`ENABLE_ORJSON` swaps the codec behind request bodies, JSON responses, stored JSON columns,
socket payloads and provider traffic from stdlib `json` to orjson. `codec_pair(instance_with)`
boots an instance with it off and one with it on, joined on one database (SQLite by default,
Postgres under `OWUI_TEST_DATABASE=postgres`), so a test writes through either and reads through
either, the way a deployment flipping the switch or rolling it out worker by worker does.
Both share the signing key, so an account's token works on both, and both answer from one
scripted provider (the stdlib instance's), since the shared database holds one connection list.

`MIXED_TEXT` is what users, providers and tools write: accents, CJK, right-to-left scripts,
emoji sequences, a line separator pasted from a PDF, Windows paths, markdown and HTML. `nested()`
is metadata-shaped: numbers, booleans, nulls, empty values and timestamps at several depths.
`stored_text` reads a JSON column the way the database holds it, which is where the two codecs
visibly differ (stdlib escapes non-ASCII and spaces its separators, orjson writes raw UTF-8).
"""

from __future__ import annotations

import dataclasses
from typing import Callable

from harness import backends
from harness.instance import LaunchedInstance

CODECS = ("stdlib", "orjson")
CODEC_ENV = {
    "stdlib": {"ENABLE_ORJSON": "false"},
    "orjson": {"ENABLE_ORJSON": "true"},
}
# (writer, reader): every way data can cross the switch
CROSSINGS = [(writer, reader) for writer in CODECS for reader in CODECS]

MIXED_TEXT = {
    "german": "Grüße aus München: Straße, Äpfel, Öl und Übermaß",
    "french": "Crème brûlée à l'œuf, déjà vu, naïve façade",
    "chinese": "请总结这份报告的要点，并列出三个行动项。",
    "japanese": "東京の天気はどうですか？カタカナとひらがな",
    "korean": "오늘 회의록을 요약해 주세요",
    "arabic": "مرحبا بالعالم، كيف حالك اليوم؟",
    "hebrew": "שלום עולם, מה שלומך?",
    "russian": "Отчёт о продажах за третий квартал",
    "hindi": "नमस्ते, आप कैसे हैं?",
    "emoji": "Launch 🚀 done ✅ family 👨‍👩‍👧‍👦 flag 🇦🇹 thumbs 👍🏽 heart ❤️",
    "math": "∑ x² ≤ ∞, α→β, ½ + ¼ = ¾, 10 °C ± 0.5",
    "pdf_paste": "First line of the page\u2028continued after a PDF line break\u2029next paragraph",
    "windows_path": 'Saved to C:\\Users\\Jörg\\Documents\\report "final".docx',
    "markdown": "# Title\n\n- item *one*\n- `code`\n\n```python\nprint('héllo')\n```\n",
    "html": '<b>bold</b> &amp; <a href="https://example.com/?q=a&b=ç">link</a>',
    "whitespace": "tab\there, two spaces  and trailing newline\n",
}
ALL_MIXED = "\n".join(MIXED_TEXT.values())
# a pasted document: long enough to cross any buffer or preview size
LONG_TEXT = "\n\n".join(f"Abschnitt {index}: {ALL_MIXED}" for index in range(400))


def nested() -> dict:
    """Metadata as clients store it: every JSON type at a few depths, nothing exotic."""
    return {
        "title": MIXED_TEXT["german"],
        "rtl": MIXED_TEXT["arabic"],
        "count": 42,
        "negative": -7,
        "ratio": 0.75,
        "temperature": 0.7,
        "enabled": True,
        "archived": False,
        "missing": None,
        "empty_text": "",
        "empty_list": [],
        "empty_object": {},
        "created_at": 1767225600,
        "updated_at_ms": 1767225600123,
        "iso_date": "2026-01-01T00:00:00Z",
        "tags": [MIXED_TEXT["chinese"], "plain", MIXED_TEXT["emoji"]],
        "levels": {
            "one": {"two": {"three": [1, 2.5, None, "drei", {"vier": MIXED_TEXT["hebrew"]}]}}
        },
    }


def codec_pair(
    instance_with: Callable[[dict[str, str]], LaunchedInstance], extra: dict[str, str] | None = None
) -> dict[str, LaunchedInstance]:
    """An instance per codec on the stdlib one's database and provider; `extra` goes to both."""
    extra = extra or {}
    stdlib = instance_with({**extra, **CODEC_ENV["stdlib"]})
    provider = stdlib.upstream.base_url
    joined = {
        **extra,
        **CODEC_ENV["orjson"],
        "DATABASE_URL": stdlib.database_url,
        "OPENAI_API_BASE_URL": provider,
        "OPENAI_API_BASE_URLS": provider,
        "RAG_OPENAI_API_BASE_URL": provider,
    }
    orjson = dataclasses.replace(instance_with(joined), upstream=stdlib.upstream)
    return {"stdlib": stdlib, "orjson": orjson}


def stored_text(instance: LaunchedInstance, table: str, column: str, row_id: str) -> str:
    """A JSON column's text as the database holds it."""
    rows = backends.read_rows(
        instance,
        f'SELECT CAST("{column}" AS TEXT) AS stored FROM "{table}" WHERE id = :row_id',
        {"row_id": row_id},
    )
    assert rows, f"no {table} row {row_id}"
    return rows[0]["stored"]
