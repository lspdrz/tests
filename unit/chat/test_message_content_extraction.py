"""Extraction and sanitizer paths no HTTP request reaches.

Regressions from 0.11.0 in how the backend pulls text out of a stored assistant turn:

1. `reconcile_tool_pairs` (#26799) keeps an assistant turn whose tool calls were all orphaned
   when its text is in `output`; the replay path only feeds it messages rebuilt from `output`,
   which carry their text in `content`, so only a direct call shows it.
2. The sanitizer still cleans a structure JSON cannot serialize; every request body is JSON, so
   no request can hand it one.

The memory review, tool-image, surrogate and non-text sanitizer cases are pinned over HTTP in
integration/chat/test_message_content_extraction.py.

Discriminates: passes on dev bbfa876af; reverting the `output` fallback in
`get_content_from_message` fails the orphan-tool test. The sanitizer test passes on both.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression

NUL = chr(0)


def structured_reply(text: str) -> dict:
    """An assistant turn whose text lives only in `output`."""
    output = [{"type": "message", "content": [{"type": "output_text", "text": text}]}]
    return {"role": "assistant", "content": None, "output": output}


def test_an_orphaned_structured_turn_keeps_its_text(misc_module):
    orphaned = {
        **structured_reply("Here is what I found."),
        "tool_calls": [{"id": "call_orphan", "type": "function", "function": {"name": "s"}}],
    }

    reconciled = misc_module.reconcile_tool_pairs(
        messages=[{"role": "user", "content": "hi"}, orphaned]
    )

    assistants = [message for message in reconciled if message["role"] == "assistant"]
    assert len(assistants) == 1, "the structured-output turn was dropped as empty"
    assert "tool_calls" not in assistants[0]
    assert misc_module.get_content_from_message(assistants[0]) == "Here is what I found."


def test_a_structure_json_cannot_serialize_is_still_cleaned(misc_module):
    cleaned = misc_module.sanitize_data_for_db({"fn": object, "text": f"x{NUL}y"})

    assert cleaned["text"] == "xy"
