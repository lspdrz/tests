"""The orjson codec's options, fixed in v0.11.1 (commit 78ed5a0235).

`ORJSONCodec.dumps` and `loads` swallowed their options, so `indent`, `sort_keys`, `ensure_ascii`,
`separators` and `object_hook` had no effect. Any option outside orjson's own defaults now falls
back to engineio's stdlib-backed codec.

The integration twin pins what users see: a note's indented JSON (`indent`), a tag search over
data stored by stdlib json (`ensure_ascii`), the stream line reader's limit settings. This file
keeps the options no route passes to the codec (`sort_keys`, other `separators`, `object_hook`)
and the fallback for what orjson rejects. The codec picks its implementation at import, so it is
probed in a child interpreter booted with `ENABLE_ORJSON=true` rather than by re-executing the
module here.

Discriminates: passes on bbfa876af; dropping the option fallback from `ORJSONCodec` fails the
option and `object_hook` tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.regression

PAYLOAD = {"b": [1, {"z": "é", "a": None}], "a": True}
OPTIONS = {
    "sort_keys": {"sort_keys": True},
    "separators": {"separators": (" | ", " -> ")},
    "indent_and_sort_keys": {"indent": 2, "sort_keys": True},
}

CODEC_PROBE = """
import ast, json, sys

sys.path.insert(0, sys.argv[1])
from open_webui.utils.json_codec import JSONCodec

payload, options = ast.literal_eval(sys.argv[2])
print(json.dumps({
    "codec": JSONCodec.__name__,
    "with_options": {name: JSONCodec.dumps(payload, **kwargs) for name, kwargs in options.items()},
    "compact": JSONCodec.dumps(payload),
    "hooked": JSONCodec.loads('{"a": 1}', object_hook=lambda item: {**item, "hooked": True}),
    "int_keys": JSONCodec.loads(JSONCodec.dumps({1: "int key"})),
    "with_default": JSONCodec.dumps({"x": {2, 1}}, default=sorted),
}))
"""


@pytest.fixture(scope="module")
def orjson_codec(open_webui_backend) -> dict:
    """What the checkout's codec returns for each probe, with `ENABLE_ORJSON=true`."""
    probed = subprocess.run(
        [sys.executable, "-c", CODEC_PROBE, str(open_webui_backend), repr((PAYLOAD, OPTIONS))],
        env={**os.environ, "ENABLE_ORJSON": "true"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert probed.returncode == 0, f"the codec probe failed:\n{probed.stderr[-3000:]}"
    results = json.loads(probed.stdout.splitlines()[-1])
    assert results["codec"] == "ORJSONCodec", (
        f"ENABLE_ORJSON=true selected {results['codec']}; retarget the probe at the orjson codec"
    )
    return results


@pytest.mark.parametrize("name", sorted(OPTIONS))
def test_dumps_with_an_option_matches_stdlib(orjson_codec, name):
    expected = json.dumps(PAYLOAD, **OPTIONS[name])

    assert orjson_codec["with_options"][name] == expected, (
        f"ORJSONCodec.dumps ignored {OPTIONS[name]} and wrote orjson's compact output"
    )


def test_loads_honours_object_hook(orjson_codec):
    assert orjson_codec["hooked"] == {"a": 1, "hooked": True}


def test_dumps_without_options_stays_compact_raw_utf8(orjson_codec):
    assert orjson_codec["compact"] == json.dumps(PAYLOAD, separators=(",", ":"), ensure_ascii=False)


def test_what_orjson_rejects_still_falls_back_to_stdlib(orjson_codec):
    assert orjson_codec["int_keys"] == {"1": "int key"}
    assert orjson_codec["with_default"] == '{"x": [1, 2]}'
