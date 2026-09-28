"""Regression: the frontend's scripts and WebAssembly were served as text/plain on Windows hosts.

Commit `d8133c905` (PR #29139, issue #29133, open-webui 0.11.2). `main.py` only registered
`text/javascript` for `.js`, and only when a frontend build existed. Windows hosts carry registry
entries that `mimetypes` reads at init, so `.js`, `.mjs` and `.wasm` came back as whatever the
host said (commonly `text/plain`), and the browser refuses a module script or a WebAssembly
stream of that type: the code interpreter never loaded Pyodide, and the app's own scripts were
at stake the same way. The fix registers all three at import, over whatever the host supplied.

The instance here boots on a host table that says `text/plain` for all three
(`harness.host_mime_types`) and serves the built frontend; every script, module and WebAssembly
file in the build is fetched from it.

The browser twin, e2e/config/test_static_asset_mime_types.py, loads the app on the same kind of
instance.

Discriminates: passes on bbfa876af, fails with the three `mimetypes.add_type` calls removed from
`main.py` (the host's text/plain is what gets served). The control, the ordinary asset and the
cross-origin tests pass on both.
"""

from __future__ import annotations

import pytest

from harness.host_mime_types import CONTROL_EXTENSION, CONTROL_TYPE, mistyped_host_env
from harness.instance import resolve_backend, resolve_frontend_build

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

BROWSER_SAFE_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".wasm": "application/wasm",
}
ORDINARY_TYPES = {".css": "text/css", ".png": "image/png", ".json": "application/json"}


@pytest.fixture(scope="module")
def build():
    backend = resolve_backend()
    found = resolve_frontend_build(backend) if backend else None
    if found is None:
        pytest.skip(
            "no built frontend (run `npm run build` in the checkout or set OPEN_WEBUI_BUILD_DIR)"
        )
    return found


@pytest.fixture(scope="module")
def mistyped_host(instance_with, tmp_path_factory, build):
    return instance_with(mistyped_host_env(tmp_path_factory.mktemp("mistyped-host")))


def _assets(build, suffix: str) -> list[str]:
    found = sorted(path for path in build.rglob(f"*{suffix}") if path.is_file())
    assert found, f"the build holds no {suffix} file; retarget this test"
    return ["/" + path.relative_to(build).as_posix() for path in found]


def _served_type(instance, path: str) -> str:
    with instance.client() as client:
        served = client.get(path)
    assert served.status_code == 200, f"{path}: HTTP {served.status_code}"
    return served.headers.get("content-type", "").split(";")[0].strip()


def test_the_host_table_is_live_on_this_instance(mistyped_host):
    """Control: without this the tests below would prove nothing."""
    probe = mistyped_host.data_dir.parent / "static" / f"probe{CONTROL_EXTENSION}"
    probe.write_bytes(b"x")

    assert _served_type(mistyped_host, f"/static/{probe.name}") == CONTROL_TYPE


@pytest.mark.parametrize("suffix", sorted(BROWSER_SAFE_TYPES))
def test_scripts_and_webassembly_are_served_with_the_browser_safe_type(
    mistyped_host, build, suffix
):
    wrong = {
        path: served
        for path in _assets(build, suffix)
        if (served := _served_type(mistyped_host, path)) != BROWSER_SAFE_TYPES[suffix]
    }

    assert wrong == {}, (
        f"{suffix} files were served with the host registry's type, which the browser refuses, "
        f"so the app's modules and the code interpreter never load (#29133): {wrong}"
    )


@pytest.mark.parametrize("suffix", sorted(ORDINARY_TYPES))
def test_ordinary_assets_keep_their_type(mistyped_host, build, suffix):
    path = _assets(build, suffix)[0]

    assert _served_type(mistyped_host, path) == ORDINARY_TYPES[suffix]


def test_pyodide_assets_still_allow_cross_origin_loading(mistyped_host, build):
    pyodide_files = sorted(path for path in (build / "pyodide").glob("*") if path.is_file())
    if not pyodide_files:
        pytest.skip("the build carries no pyodide/ directory")

    with mistyped_host.client() as client:
        served = client.get(f"/pyodide/{pyodide_files[0].name}")

    assert served.status_code == 200
    assert served.headers.get("access-control-allow-origin") == "*"
