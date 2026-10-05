"""Multi-factor sign-in: an authenticator app, accounts that enrolled one, and the admin's switches.

`Authenticator(secret)` is the app a person scans the setup QR code into. `code()` reads the
current six-digit code the way a person types it, never one it handed out before: the server
accepts each code once, and the steps either side of the current one, so a quick second sign-in
gets the next step's code instead of waiting for the clock. `wrong_code()` is a code no step of
that window shows.

`sign_in(instance, email, password)` presses "Sign in" without a session and returns the answer:
a session, or the next step (`enroll`, `verify`, `recover` or `pending`). `enroll(...)` walks a
first sign-in through setup and returns an `MfaAccount` with the session, the authenticator and
the recovery codes shown once; `verified(...)` signs an enrolled account in again with a code.
Every request without a session comes from an address of its own (`X-Forwarded-For`, which the
instance trusts from loopback), so the per-address limit on the sign-in routes never trips a
later test.

`require_mfa(instance, **switches)` saves the admin's three MFA switches as Authentication
settings do. Any change signs everyone out, the admin included, so it signs the admin back in
(enrolling the first time) and points `instance.client()` at the new session. `add_account`
adds an account the way the admin panel does, which yields no session while MFA is on.
`operator_reset(instance, email, reason)` runs `open-webui mfa reset` on the instance's database,
as an operator with shell access does for someone who lost their authenticator.
"""

from __future__ import annotations

import secrets
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field

import httpx
import pyotp

from harness.instance import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    WEBUI_SECRET_KEY,
    LaunchedInstance,
    isolated_env,
    resolve_backend,
)

ADMIN_CONFIG = "/api/v1/auths/admin/config"
MFA = "/api/v1/auths/mfa"
STEP_SECONDS = 30


class Authenticator:
    """An authenticator app holding one account's secret."""

    def __init__(self, secret: str):
        self.secret = secret
        self.totp = pyotp.TOTP(secret)
        self.last_step = -1

    def code(self) -> str:
        """A code for the earliest step the server still accepts and this app has not shown."""
        if time.time() % STEP_SECONDS > STEP_SECONDS - 3:
            # a step about to fall out of the window could expire on the way to the server
            time.sleep(STEP_SECONDS - time.time() % STEP_SECONDS + 0.5)
        current = int(time.time()) // STEP_SECONDS
        step = max(self.last_step + 1, current - 1)
        while step > current + 1:
            time.sleep(STEP_SECONDS - time.time() % STEP_SECONDS + 0.5)
            current = int(time.time()) // STEP_SECONDS
        self.last_step = step
        return self.totp.at(step * STEP_SECONDS)

    def wrong_code(self) -> str:
        current = int(time.time()) // STEP_SECONDS
        valid = {self.totp.at((current + offset) * STEP_SECONDS) for offset in range(-2, 3)}
        return next(code for code in ("000000", "111111", "222222", "333333") if code not in valid)


@dataclass
class MfaAccount:
    id: str
    email: str
    password: str
    token: str
    authenticator: Authenticator
    recovery_codes: list[str] = field(default_factory=list)
    address: str = field(default_factory=lambda: new_address())
    base_url: str = ""
    sign_in_headers: dict[str, str] = field(default_factory=dict)  # a proxy's, when one signs in

    def client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.token}", "X-Forwarded-For": self.address},
            timeout=120.0,
        )


def new_address() -> str:
    return f"10.{secrets.randbelow(250)}.{secrets.randbelow(250)}.{secrets.randbelow(250) + 1}"


def anonymous(
    instance: LaunchedInstance, address: str | None = None, headers: dict | None = None
) -> httpx.Client:
    """A browser without a session, from an address of its own."""
    return httpx.Client(
        base_url=instance.base_url,
        headers={"X-Forwarded-For": address or new_address(), **(headers or {})},
        timeout=60.0,
    )


def sign_in(
    instance: LaunchedInstance,
    email: str,
    password: str,
    address: str | None = None,
    headers: dict | None = None,
) -> httpx.Response:
    with anonymous(instance, address, headers) as browser:
        return browser.post("/api/v1/auths/signin", json={"email": email, "password": password})


def mfa_step(
    instance: LaunchedInstance, step: str, body: dict, address: str | None = None
) -> httpx.Response:
    with anonymous(instance, address) as browser:
        return browser.post(f"{MFA}/{step}", json=body)


def finish_enrollment(
    instance: LaunchedInstance, challenge_token: str, address: str | None = None
) -> tuple[Authenticator, dict]:
    """Scan the setup key and confirm it; returns the app and the confirmation's answer."""
    setup = mfa_step(instance, "enroll/start", {"challenge_token": challenge_token}, address)
    assert setup.status_code == 200, f"starting setup failed: {setup.text}"
    authenticator = Authenticator(setup.json()["manual_key"])
    confirmed = mfa_step(
        instance,
        "enroll/confirm",
        {"challenge_token": challenge_token, "code": authenticator.code()},
        address,
    )
    assert confirmed.status_code == 200, f"confirming setup failed: {confirmed.text}"
    return authenticator, confirmed.json()


def enroll(instance: LaunchedInstance, email: str, password: str) -> MfaAccount:
    """A first sign-in while MFA is on: password, authenticator setup, recovery codes."""
    address = new_address()
    answer = sign_in(instance, email, password, address)
    assert answer.status_code == 200, f"signing {email} in failed: {answer.text}"
    assert answer.json().get("next_step") == "enroll", answer.json()
    authenticator, session = finish_enrollment(instance, answer.json()["challenge_token"], address)
    return MfaAccount(
        id=session["id"],
        email=email,
        password=password,
        token=session["token"],
        authenticator=authenticator,
        recovery_codes=session["recovery_codes"],
        address=address,
        base_url=instance.base_url,
    )


def verified(instance: LaunchedInstance, account: MfaAccount) -> str:
    """Sign an enrolled account in again with a fresh code; returns and keeps the session."""
    answer = sign_in(
        instance, account.email, account.password, account.address, account.sign_in_headers
    )
    assert answer.status_code == 200, f"signing {account.email} in failed: {answer.text}"
    assert answer.json().get("next_step") == "verify", answer.json()
    session = mfa_step(
        instance,
        "verify",
        {"challenge_token": answer.json()["challenge_token"], "code": account.authenticator.code()},
        account.address,
    )
    assert session.status_code == 200, f"verifying {account.email} failed: {session.text}"
    account.token = session.json()["token"]
    return account.token


def add_account(instance: LaunchedInstance, role: str = "user") -> tuple[str, str, dict]:
    """An account added as the admin panel adds one; returns its email, password and the answer."""
    email = f"mfa-{uuid.uuid4().hex[:12]}@example.com"
    password = "mfa-password-123"
    with anonymous(instance) as browser:
        added = browser.post(
            "/api/v1/auths/add",
            json={"name": email.split("@")[0], "email": email, "password": password, "role": role},
            headers={"Authorization": f"Bearer {instance.admin_token}"},
        )
    assert added.status_code == 200, f"adding {email} failed: {added.text}"
    return email, password, added.json()


def enrolled_account(instance: LaunchedInstance, role: str = "user") -> MfaAccount:
    email, password, _ = add_account(instance, role)
    return enroll(instance, email, password)


_admins: dict[str, MfaAccount] = {}


def require_mfa(
    instance: LaunchedInstance, sign_in_headers: dict | None = None, **switches: bool
) -> dict:
    """Save the MFA switches with the admin's session and sign the admin back in.

    Returns what the save answered (`sessions_revoked` says whether everyone was signed out).
    `sign_in_headers` go with the admin's sign-in, for an instance behind a signing-in proxy.
    """
    with instance.client() as client:
        current = client.get(ADMIN_CONFIG)
        current.raise_for_status()
        saved = client.post(ADMIN_CONFIG, json={**current.json(), **switches})
    assert saved.status_code == 200, f"saving the MFA switches failed: {saved.text}"
    if saved.json()["sessions_revoked"]:
        instance.admin_token = admin_session(instance, sign_in_headers or {})
    return saved.json()


def admin_session(instance: LaunchedInstance, sign_in_headers: dict) -> str:
    """A fresh admin session, through whatever second step the switches now ask for."""
    admin = _admins.get(instance.base_url)
    answer = sign_in(instance, ADMIN_EMAIL, ADMIN_PASSWORD, headers=sign_in_headers)
    assert answer.status_code == 200, f"the admin's sign-in failed: {answer.text}"
    step = answer.json().get("next_step")
    if step is None:
        return answer.json()["token"]
    if step == "enroll":
        authenticator, session = finish_enrollment(instance, answer.json()["challenge_token"])
        _admins[instance.base_url] = MfaAccount(
            id=session["id"],
            email=ADMIN_EMAIL,
            password=ADMIN_PASSWORD,
            token=session["token"],
            authenticator=authenticator,
            recovery_codes=session["recovery_codes"],
            base_url=instance.base_url,
            sign_in_headers=sign_in_headers,
        )
        return session["token"]
    assert step == "verify" and admin is not None, answer.json()
    return verified(instance, admin)


def admin_account(instance: LaunchedInstance) -> MfaAccount:
    """The admin's enrolled account on an instance `require_mfa` switched MFA on for."""
    return _admins[instance.base_url]


def operator_reset(
    instance: LaunchedInstance, email: str, reason: str
) -> subprocess.CompletedProcess:
    """`open-webui mfa reset`, run against the instance's database as the docs show."""
    backend = resolve_backend()
    scratch = instance.data_dir.parent / "operator"
    (scratch / "static").mkdir(parents=True, exist_ok=True)
    settings = {
        "DATA_DIR": str(instance.data_dir),
        "DATABASE_URL": instance.database_url,
        "STATIC_DIR": str(scratch / "static"),
        "WEBUI_SECRET_KEY": WEBUI_SECRET_KEY,
        "PYTHONPATH": str(backend),
    }
    command = [sys.executable, "-c", "from open_webui import app; app()", "mfa", "reset", email]
    return subprocess.run(
        [*command, "--reason", reason],
        cwd=scratch,
        env=isolated_env(settings),
        capture_output=True,
        text=True,
        timeout=300,
    )
