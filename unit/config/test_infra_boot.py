"""Boot-time and deployment-shape regressions fixed between v0.11.0 and v0.11.1: the static part.

* 51 (PR27838, 3dbb4078b, retrieval/vector/dbs/opengauss.py): `SRC_LOG_LEVELS['RAG']` was read
  at import from what is now an empty legacy dict, so any openGauss deployment hit a KeyError.
  This sweep over the source also catches the next module that indexes it.
* 55 (PR28061, 2207876ae, start_windows.bat, issue #28060): key generation read from a file named
  by `%RANDOM%` and `%KEY_FILE%` was unquoted. These run only where cmd.exe exists.
* 120 (0480ca9653 + 4d5084025 / PR28866, Dockerfile, issue #27651): the model caches baked into
  the image were root-only, so runAsNonRoot deployments could not start. The Dockerfile is parsed
  the way docker reads it.

The resolver pin (14), openGauss in use (51), the ColBERT startup record (52) and RDS IAM token
auth for the pgvector store (167) are pinned from outside in integration/config/test_infra_boot.py.

Discriminates: passes on bbfa876af; indexing `SRC_LOG_LEVELS` in opengauss.py and dropping the
cache chmod or the data chown from the Dockerfile each fail their tests. The start_windows.bat
tests need cmd.exe and did not run on the Linux runner this was proven on.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from unit.config.container_files import read_dockerfile, run_commands

pytestmark = pytest.mark.regression


@pytest.fixture(scope="session")
def env_module(owui_module):
    return owui_module("open_webui.env")


@pytest.fixture(scope="session")
def repo_root(open_webui_backend: Path) -> Path:
    return open_webui_backend.parent


# ─────────────────────────────────────────────────────────────────────────────
# 51 — openGauss import-time KeyError
# ─────────────────────────────────────────────────────────────────────────────


def test_no_backend_module_indexes_src_log_levels(open_webui_backend: Path, env_module) -> None:
    """SRC_LOG_LEVELS is an empty legacy dict, so any subscript of it is a KeyError."""
    assert env_module.SRC_LOG_LEVELS == {}

    offenders = []
    for path in (open_webui_backend / "open_webui").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            indexed = node.value if isinstance(node, ast.Subscript) else None
            name = getattr(indexed, "id", None) or getattr(indexed, "attr", None)
            if name == "SRC_LOG_LEVELS":
                offenders.append(f"{path.relative_to(open_webui_backend)}:{node.lineno}")
    assert offenders == []


# ─────────────────────────────────────────────────────────────────────────────
# 55 — start_windows.bat secret key generation
# ─────────────────────────────────────────────────────────────────────────────

_KEY_ALPHABET = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_MISSING_FILE_ERROR = "The system cannot find the file"
_STRIPPED_KEY_ENV = (
    "WEBUI_SECRET_KEY",
    "WEBUI_JWT_SECRET_KEY",
    "WEB_LOADER_ENGINE",
    "WEBUI_SECRET_KEY_FILE",
)


def _run_start_windows_bat(open_webui_backend: Path, workdir: Path, key_file: Path | None):
    """Run the script up to the uvicorn launch, in its own scratch directory."""
    if os.name != "nt":
        pytest.skip("start_windows.bat needs cmd.exe")
    cmd = shutil.which("cmd.exe")
    if cmd is None:
        pytest.skip("cmd.exe not found")

    source_path = open_webui_backend / "start_windows.bat"
    if not source_path.is_file():
        pytest.skip(f"start_windows.bat not found at {source_path}")

    source = source_path.read_text(encoding="utf-8")
    launch_marker = ":: Execute uvicorn"
    if launch_marker not in source:
        pytest.skip("uvicorn launch marker gone from start_windows.bat; update this test")

    script = workdir / "start_windows.bat"
    script.write_text(source[: source.index(launch_marker)], encoding="utf-8", newline="")

    env = {key: value for key, value in os.environ.items() if key not in _STRIPPED_KEY_ENV}
    if key_file is not None:
        env["WEBUI_SECRET_KEY_FILE"] = str(key_file)

    return subprocess.run(
        [cmd, "/c", str(script)],
        capture_output=True,
        text=True,
        cwd=str(workdir),
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )


def test_start_windows_bat_generates_a_usable_secret_key(
    open_webui_backend: Path, tmp_path: Path
) -> None:
    """Narrow: pre-fix this read from a file named by %RANDOM% and wrote nothing."""
    result = _run_start_windows_bat(open_webui_backend, tmp_path, None)

    key_file = tmp_path / ".webui_secret_key"
    assert _MISSING_FILE_ERROR not in result.stderr, result.stderr.strip()[:400]
    assert result.returncode == 0, result.stderr.strip()[:400]
    assert key_file.is_file(), "no secret key file was written"

    key = key_file.read_text(encoding="utf-8").strip()
    assert len(key) == 24
    assert set(key) <= _KEY_ALPHABET


def test_start_windows_bat_handles_a_key_path_with_spaces(
    open_webui_backend: Path, tmp_path: Path
) -> None:
    """Narrow: %KEY_FILE% was unquoted, so any install path with a space broke."""
    key_dir = tmp_path / "Open WebUI"
    key_dir.mkdir()
    key_file = key_dir / "secret key"

    result = _run_start_windows_bat(open_webui_backend, tmp_path, key_file)

    assert _MISSING_FILE_ERROR not in result.stderr, result.stderr.strip()[:400]
    assert result.returncode == 0, result.stderr.strip()[:400]
    assert key_file.is_file(), "no secret key file was written to the spaced path"
    assert set(key_file.read_text(encoding="utf-8").strip()) <= _KEY_ALPHABET


def test_start_windows_bat_reuses_an_existing_key(open_webui_backend: Path, tmp_path: Path) -> None:
    """Nearby: an existing key file is loaded, never regenerated."""
    key_file = tmp_path / ".webui_secret_key"
    key_file.write_text("preexisting-key", encoding="utf-8")

    result = _run_start_windows_bat(open_webui_backend, tmp_path, None)

    assert result.returncode == 0, result.stderr.strip()[:400]
    assert key_file.read_text(encoding="utf-8") == "preexisting-key"
    assert "Generating WEBUI_SECRET_KEY" not in result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# 120 — non-root container startup
# ─────────────────────────────────────────────────────────────────────────────


def _grants_others_read(mode: str) -> bool:
    if mode.isdigit():
        return int(mode[-1]) & 4 == 4
    return any(
        clause[0] in "ao" and "+" in clause and "r" in clause.split("+", 1)[1]
        for clause in mode.split(",")
    )


def _covers(target: str, path: str) -> bool:
    target_path, covered_path = PurePosixPath(target.rstrip("/")), PurePosixPath(path)
    return covered_path == target_path or target_path in covered_path.parents


def test_every_baked_model_cache_is_readable_by_any_user(repo_root: Path) -> None:
    """Narrow: the caches were root-only, so a runAsNonRoot pod could not load its models."""
    runtime = read_dockerfile(repo_root).runtime
    caches = sorted(value for value in runtime.env().values() if "/cache/" in value)
    assert caches, "the runtime stage sets no model cache directory; retarget this guard"

    readable_trees = [
        target
        for command in run_commands(runtime)
        if command[0] == "chmod"
        and "-R" in command
        and _grants_others_read(next(word for word in command[1:] if not word.startswith("-")))
        for target in command[2:]
        if target.startswith("/")
    ]
    unreadable = [
        cache for cache in caches if not any(_covers(tree, cache) for tree in readable_trees)
    ]
    assert not unreadable, (
        f"model caches only root can read, so runAsNonRoot deployments cannot start (#27651): "
        f"{unreadable}"
    )


def test_the_data_directory_is_still_handed_to_the_runtime_user(repo_root: Path) -> None:
    runtime = read_dockerfile(repo_root).runtime
    chowned = [
        target.rstrip("/")
        for command in run_commands(runtime)
        if command[0] == "chown" and "-R" in command
        for target in command[3:]
    ]
    assert "/app/backend/data" in chowned, chowned
