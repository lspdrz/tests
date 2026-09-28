"""Regression: an SSO sign-in whose sub only resembled a number landed on someone else's account.

`17cc56670` (open-webui 0.11.3). Releases before 0.11.1 stored a numeric sub as a JSON number, so
on SQLite the sub lookup also compares with `int(sub)`. It did that for any sub passing
`str.isdecimal()`: a sub whose integer form is not its own text ('007', or a non-ASCII decimal
digit) signed in as a DIFFERENT person whose stored sub is that number, and a sub above 2**63 - 1
was handed to the SQLite driver as an integer it cannot bind, so that sign-in errored. The fix
widens only when `str(int(sub)) == sub` and the value fits in a signed 64 bit integer.

The accounts an older release linked are written into the instance's database with the sub as
a JSON number, and people then sign in through the OIDC stand-in. Each test removes the accounts
it made, so a sub one test stores cannot answer another test's sign-in.

Twin of unit/security/test_oauth_sub_precise_match.py.

Discriminates: passes on dev ef67cc3fa; with 17cc56670 reverted in a copy the zero padded and
non-ASCII decimal subs sign in as the account storing the plain number and the sub beyond the
64 bit range fails to sign in (OverflowError in the lookup).
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Iterator

import pytest

from harness.actors import Actor, create_user
from harness.instance import LaunchedInstance
from harness.oidc_provider import (
    OidcProvider,
    session_user,
    shared_provider,
    sign_in,
    sso_env,
    store_oauth_entry,
)

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

INT64_MAX = 2**63 - 1
ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
EXTENDED_ARABIC_INDIC = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sso(instance_with, idp):
    return instance_with(sso_env(idp))


@dataclass
class Accounts:
    """Accounts an older release linked, and the ones sign-ins make; all removed afterwards."""

    sso: LaunchedInstance
    idp: OidcProvider
    made: list[str] = field(default_factory=list)

    def linked(self, oauth: dict | None) -> Actor:
        account = create_user(self.sso)
        self.made.append(account.id)
        if oauth is not None:
            store_oauth_entry(self.sso, account.id, oauth)
        return account

    def signed_in(self, sub: str, email: str | None = None) -> str | None:
        """Sign in with `sub` (and `email`, else a fresh one); the account's id, None if refused."""
        self.idp.sign_in_as(sub=sub, **({"email": email} if email else {}))
        result = sign_in(self.sso)
        if not result.token:
            return None
        account_id = session_user(self.sso, result.token)["id"]
        self.made.append(account_id)
        return account_id


@pytest.fixture
def accounts(sso, idp) -> Iterator[Accounts]:
    made = Accounts(sso, idp)
    yield made
    with sso.client() as admin:
        for account_id in set(made.made):
            admin.delete(f"/api/v1/users/{account_id}")


def a_number() -> int:
    return 10_000_000 + secrets.randbelow(10**8)


def spellings(number: int) -> dict[str, str]:
    """Texts `str.isdecimal()` accepts whose integer form is `number` but which are not its text."""
    text = str(number)
    return {
        "zero-padded": f"00{text}",
        "one-zero": f"0{text}",
        "many-zeros": f"00000{text}",
        "arabic-indic": text.translate(ARABIC_INDIC),
        "extended-arabic-indic": text.translate(EXTENDED_ARABIC_INDIC),
    }


# --------------------------------------------------------------------------- narrow


@pytest.mark.parametrize("spelling", ["zero-padded", "arabic-indic"])
def test_a_sub_that_only_coerces_to_a_number_does_not_sign_in_as_it(accounts, spelling):
    """Narrow: '0012345678' or its Arabic-Indic digits never reach the account with the number."""
    number = a_number()
    numbered = accounts.linked({"oidc": {"sub": number}})

    arrived_as = accounts.signed_in(spellings(number)[spelling])

    assert arrived_as is not None, "the sign-in was refused"
    assert arrived_as != numbered.id, (
        f"a sub spelled {spellings(number)[spelling]!r} signed in as the account whose stored "
        f"sub is the number {number}"
    )


@pytest.mark.parametrize("spelling", ["zero-padded", "arabic-indic"])
def test_a_sub_that_only_coerces_to_a_number_finds_its_own_account(accounts, spelling):
    """Narrow: with both present, the sub gets its own account; the numeric one scans first."""
    number = a_number()
    sub = spellings(number)[spelling]
    accounts.linked({"oidc": {"sub": number}})
    own = accounts.linked({"oidc": {"sub": sub}})

    assert accounts.signed_in(sub, email=own.email) == own.id


# --------------------------------------------------------------------------- broad


@pytest.mark.parametrize(
    "spelling", ["zero-padded", "one-zero", "many-zeros", "arabic-indic", "extended-arabic-indic"]
)
def test_no_other_spelling_of_a_number_reaches_its_account(accounts, spelling):
    """Broad: neither the account storing the number nor the one storing its text is reached."""
    number = a_number()
    as_number = accounts.linked({"oidc": {"sub": number}})
    as_text = accounts.linked({"oidc": {"sub": str(number)}})

    arrived_as = accounts.signed_in(spellings(number)[spelling])

    assert arrived_as not in (as_number.id, as_text.id, None)


@pytest.mark.parametrize(
    "stored",
    ["number", "text", "int64-max-number", "int64-max-text"],
)
def test_a_plain_numeric_sub_signs_in_as_either_stored_form(accounts, stored):
    """Broad: a plain decimal sub, up to the 64 bit boundary, finds the number or its text."""
    number = INT64_MAX if stored.startswith("int64") else a_number()
    stored_sub = number if stored.endswith("number") else str(number)
    target = accounts.linked({"oidc": {"sub": stored_sub}})

    assert accounts.signed_in(str(number), email=target.email) == target.id


@pytest.mark.parametrize(
    "sub",
    [str(INT64_MAX + 1 + secrets.randbelow(10**6)), str(INT64_MAX)],
    ids=["beyond-int64", "int64-boundary"],
)
def test_a_large_numeric_sub_signs_in_to_the_same_account_twice(accounts, sub):
    """Narrow (beyond-int64) and nearby (the boundary itself): both keep one account."""
    email = f"large-{secrets.token_hex(4)}@example.com"
    first = accounts.signed_in(sub, email=email)
    assert first is not None, "the sign-in failed"
    assert accounts.signed_in(sub, email=email) == first


# --------------------------------------------------------------------------- nearby


def test_an_opaque_sub_signs_in_as_exactly_its_account(accounts):
    """Nearby: an opaque sub matches its own account, not the one a character away."""
    tag = secrets.token_hex(4)
    alice = accounts.linked({"oidc": {"sub": f"abc{tag}5"}})
    accounts.linked({"oidc": {"sub": f"abc{tag}6"}})

    assert accounts.signed_in(f"abc{tag}5", email=alice.email) == alice.id


def test_a_sub_linked_under_another_provider_is_not_used(accounts):
    """Nearby: the same sub stored for another provider, or no link at all, is a new account."""
    sub = f"shared-{secrets.token_hex(4)}"
    elsewhere = accounts.linked({"github": {"sub": sub}})
    unlinked = accounts.linked(None)

    arrived_as = accounts.signed_in(sub)

    assert arrived_as not in (elsewhere.id, unlinked.id, None)


def test_an_empty_sub_is_refused(sso, idp):
    """Nearby: a provider sending no sub signs nobody in."""
    idp.sign_in_as(sub="")
    result = sign_in(sso)
    assert result.token is None and result.error
