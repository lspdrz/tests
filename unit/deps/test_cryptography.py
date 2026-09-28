"""Dependency contract: cryptography, for the enterprise license check.

Open WebUI uses `cryptography` for two jobs. Fernet encrypts the stored OAuth sessions and the
OAuth client info; that path is driven from outside in integration/deps/test_auth_stack.py (a
session token encrypted at rest, decrypted when forwarded, refused under another key, and a key
Fernet cannot use stopping the boot). The other job is the license check, which this module
pins:

  * **AES-GCM AEAD** (`cryptography.hazmat.primitives.ciphers.aead.AESGCM`) decrypts the license
    blob in `utils/auth.py` with a SHA-256 key, a 12-byte nonce and ``associated_data=None``.
  * **Ed25519 signature verification**: the license public key is loaded with
    ``serialization.load_pem_public_key(...)`` (`env.py`) and its ``.verify(signature, data)``
    checks the license signature (`utils/auth.py`).

Kept as a unit contract: the license check only runs on a license blob signed by the key in
`LICENSE_PUBLIC_KEY`, and driving it from outside means minting a working license with a key of
the test's own, which a public suite should not ship. Uses the `depcheck` fixture from
unit/deps/conftest.py.
"""

from __future__ import annotations

import hashlib
import os

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "cryptography"
DIST_NAME = "cryptography"

# Every dotted symbol the Open WebUI backend resolves on `cryptography`.
# Paths are relative to the top-level `cryptography` package object.
USED_SYMBOLS = [
    # AES-GCM AEAD — license blob decryption.
    "hazmat.primitives.ciphers.aead.AESGCM",
    # PEM public-key loading — license public key.
    "hazmat.primitives.serialization",
    "hazmat.primitives.serialization.load_pem_public_key",
    # Ed25519 — license signature verification.
    "hazmat.primitives.asymmetric.ed25519",
    "hazmat.primitives.asymmetric.ed25519.Ed25519PublicKey",
    "hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey",
]

# Submodules the backend imports directly (must be importable as modules).
USED_SUBMODULES = [
    "cryptography.hazmat.primitives.serialization",
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    "cryptography.hazmat.primitives.ciphers.aead",
    "cryptography.exceptions",
]


# --------------------------------------------------------------------------- #
# Import + API surface
# --------------------------------------------------------------------------- #
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "cryptography"


def test_version_reported(depcheck):
    """Sanity: the installed distribution version is resolvable (so bump
    tooling and this suite agree on what's under test)."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    """Every cryptography symbol the codebase references must still exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_used_submodules_importable(depcheck):
    """The exact submodules the backend `from`-imports must import cleanly.

    These are nested hazmat paths that have moved between major versions in the
    past; pin them as importable modules, not just attribute lookups.
    """
    depcheck.load(IMPORT_NAME)  # skip cleanly if cryptography is absent
    for name in USED_SUBMODULES:
        mod = depcheck.try_load(name)
        assert mod is not None, f"submodule {name!r} no longer importable"


def test_top_level_exception_classes_exist(depcheck):
    """`cryptography.exceptions.{InvalidSignature,InvalidTag}` are the failure
    types raised by Ed25519 verify and AES-GCM decrypt; the backend relies on
    those raising (caught by broad `except`)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(
        mod,
        ["exceptions.InvalidSignature", "exceptions.InvalidTag"],
    )


# --------------------------------------------------------------------------- #
# serialization.load_pem_public_key — env.py
# --------------------------------------------------------------------------- #
def test_load_pem_public_key_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "hazmat.primitives.serialization.load_pem_public_key")


def test_load_pem_public_key_accepts_data_param(depcheck):
    """env.py calls ``serialization.load_pem_public_key(<pem bytes>)`` — a
    single positional `data` argument. Pin that the first parameter is `data`
    (the historic `backend=` arg is deprecated but still accepted)."""
    serialization = depcheck.load("cryptography.hazmat.primitives.serialization")
    depcheck.assert_params(serialization.load_pem_public_key, ["data"])


def test_serialization_pem_enums_exist(depcheck):
    """The Encoding/PublicFormat enums the contract uses to *produce* a PEM for
    these tests (and that the ecosystem relies on) must still be present."""
    serialization = depcheck.load("cryptography.hazmat.primitives.serialization")
    names = set(dir(serialization))
    for attr in ("Encoding", "PublicFormat", "PrivateFormat", "NoEncryption"):
        assert attr in names, f"serialization.{attr} missing"
    assert hasattr(serialization.Encoding, "PEM")
    assert hasattr(serialization.PublicFormat, "SubjectPublicKeyInfo")
    assert hasattr(serialization.PublicFormat, "Raw")
    assert hasattr(serialization.PrivateFormat, "Raw")


def test_load_pem_public_key_parses_env_style_armor(depcheck):
    """Reproduce env.py exactly: wrap a base64 body in PEM armor and parse it.

    env.py builds the PEM as f-string armor around ``LICENSE_PUBLIC_KEY`` (the
    raw base64 of a DER SubjectPublicKeyInfo) and encodes to bytes. Generate a
    real Ed25519 key, emit its SPKI body, rebuild that armor, and confirm it
    round-trips back to a usable public key.
    """
    serialization = depcheck.load("cryptography.hazmat.primitives.serialization")
    ed25519 = depcheck.load("cryptography.hazmat.primitives.asymmetric.ed25519")

    priv = ed25519.Ed25519PrivateKey.generate()
    spki_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    # Strip armor to the raw base64 body, mimicking the configured env value.
    body = b"".join(
        line for line in spki_pem.splitlines() if line and not line.startswith(b"-----")
    ).decode()

    armored = f"""
-----BEGIN PUBLIC KEY-----
{body}
-----END PUBLIC KEY-----
""".encode()

    loaded = serialization.load_pem_public_key(armored)
    assert isinstance(loaded, ed25519.Ed25519PublicKey)


def test_load_pem_public_key_rejects_garbage(depcheck):
    """Malformed PEM must raise, not silently return None (env-config safety)."""
    serialization = depcheck.load("cryptography.hazmat.primitives.serialization")
    with pytest.raises(Exception):
        serialization.load_pem_public_key(b"-----BEGIN PUBLIC KEY-----\nnope\n")


# --------------------------------------------------------------------------- #
# Ed25519 — license signature verification (utils/auth.py: pk.verify(...))
# --------------------------------------------------------------------------- #
def test_ed25519_class_surface(depcheck):
    """Pin the Ed25519 public/private class methods the license flow needs:
    private `generate`/`public_key`/`sign`, public `verify` (+ the raw-bytes
    constructors used to build keys without PEM)."""
    ed25519 = depcheck.load("cryptography.hazmat.primitives.asymmetric.ed25519")
    priv_names = set(dir(ed25519.Ed25519PrivateKey))
    for attr in ("generate", "from_private_bytes", "public_key", "sign"):
        assert attr in priv_names, f"Ed25519PrivateKey.{attr} missing"
    pub_names = set(dir(ed25519.Ed25519PublicKey))
    for attr in ("from_public_bytes", "verify", "public_bytes"):
        assert attr in pub_names, f"Ed25519PublicKey.{attr} missing"


def test_ed25519_generate_returns_private_key(depcheck):
    ed25519 = depcheck.load("cryptography.hazmat.primitives.asymmetric.ed25519")
    priv = ed25519.Ed25519PrivateKey.generate()
    assert isinstance(priv, ed25519.Ed25519PrivateKey)
    assert isinstance(priv.public_key(), ed25519.Ed25519PublicKey)


def test_ed25519_verify_param_order(depcheck):
    """utils/auth.py calls ``pk.verify(signature, data)`` — verify takes
    (signature, data) in that order. Pin the parameter names/order."""
    ed25519 = depcheck.load("cryptography.hazmat.primitives.asymmetric.ed25519")
    pub = ed25519.Ed25519PrivateKey.generate().public_key()
    depcheck.assert_params(pub.verify, ["signature", "data"])


def test_ed25519_sign_verify_roundtrip(depcheck):
    """Exercise the real license-verify primitive: a signature from the
    matching private key must verify with no exception (verify returns None)."""
    ed25519 = depcheck.load("cryptography.hazmat.primitives.asymmetric.ed25519")
    priv = ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key()
    message = b"open-webui-license-payload"
    sig = priv.sign(message)
    assert pub.verify(sig, message) is None  # no raise == valid


def test_ed25519_verify_rejects_tampered_message(depcheck):
    """A signature over different bytes must raise InvalidSignature — this is
    the security property the license check relies on."""
    crypto = depcheck.load(IMPORT_NAME)
    ed25519 = depcheck.load("cryptography.hazmat.primitives.asymmetric.ed25519")
    InvalidSignature = crypto.exceptions.InvalidSignature

    priv = ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key()
    sig = priv.sign(b"genuine-payload")
    with pytest.raises(InvalidSignature):
        pub.verify(sig, b"forged-payload")


def test_ed25519_verify_rejects_wrong_key(depcheck):
    """A signature verified against a *different* public key must raise — a
    forged license signed by an attacker key must not pass."""
    crypto = depcheck.load(IMPORT_NAME)
    ed25519 = depcheck.load("cryptography.hazmat.primitives.asymmetric.ed25519")
    InvalidSignature = crypto.exceptions.InvalidSignature

    signer = ed25519.Ed25519PrivateKey.generate()
    other_pub = ed25519.Ed25519PrivateKey.generate().public_key()
    sig = signer.sign(b"payload")
    with pytest.raises(InvalidSignature):
        other_pub.verify(sig, b"payload")


def test_ed25519_verify_via_pem_loaded_key(depcheck):
    """End-to-end mirror of env.py + auth.py: load the public key from PEM
    (as env.py does), then verify a signature with it (as auth.py does)."""
    serialization = depcheck.load("cryptography.hazmat.primitives.serialization")
    ed25519 = depcheck.load("cryptography.hazmat.primitives.asymmetric.ed25519")

    priv = ed25519.Ed25519PrivateKey.generate()
    pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    pub = serialization.load_pem_public_key(pem)
    sig = priv.sign(b"data")
    assert pub.verify(sig, b"data") is None


# --------------------------------------------------------------------------- #
# AESGCM — license blob decryption (utils/auth.py)
# --------------------------------------------------------------------------- #
def test_aesgcm_class_surface(depcheck):
    """Pin AESGCM's class methods: instances are constructed ``AESGCM(key)``
    and used via ``.encrypt`` / ``.decrypt`` / ``.generate_key``."""
    aead = depcheck.load("cryptography.hazmat.primitives.ciphers.aead")
    names = set(dir(aead.AESGCM))
    for attr in ("encrypt", "decrypt", "generate_key"):
        assert attr in names, f"AESGCM.{attr} missing"
    assert callable(aead.AESGCM)


def test_aesgcm_constructible_with_sha256_key(depcheck):
    """auth.py builds the key as ``sha256(...).digest()`` (32 bytes) and passes
    it positionally to ``AESGCM(kb)``. Confirm a 32-byte key is accepted.

    NOTE: AESGCM is a Rust-backed class whose ``__init__`` signature isn't
    introspectable (shows ``*args, **kwargs``), so this is checked behaviourally
    rather than via signature inspection.
    """
    aead = depcheck.load("cryptography.hazmat.primitives.ciphers.aead")
    key = hashlib.sha256(b"some-license-key").digest()
    assert len(key) == 32
    inst = aead.AESGCM(key)
    assert inst is not None


def test_aesgcm_encrypt_decrypt_roundtrip(depcheck):
    """Mirror auth.py: 12-byte nonce, ``associated_data=None``, decrypt back to
    the original plaintext bytes."""
    aead = depcheck.load("cryptography.hazmat.primitives.ciphers.aead")
    key = hashlib.sha256(b"license-secret").digest()
    aesgcm = aead.AESGCM(key)
    nonce = os.urandom(12)  # auth.py uses a 12-byte (nl=12) nonce prefix
    plaintext = b'{"exp":"2099-01-01","name":"Org"}'

    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    assert ciphertext != plaintext
    assert aesgcm.decrypt(nonce, ciphertext, None) == plaintext


def test_aesgcm_decrypt_signature_positional_args(depcheck):
    """auth.py calls ``aesgcm.decrypt(ln, lt, None)`` — three positionals
    (nonce, data, associated_data). Verify that exact arity works by call,
    since the signature isn't introspectable on the Rust class."""
    aead = depcheck.load("cryptography.hazmat.primitives.ciphers.aead")
    aesgcm = aead.AESGCM(aead.AESGCM.generate_key(bit_length=256))
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, b"payload", None)
    # Positional (nonce, data, associated_data) — the call shape auth.py uses.
    assert aesgcm.decrypt(nonce, ct, None) == b"payload"


def test_aesgcm_decrypt_wrong_key_raises(depcheck):
    """Decrypting with the wrong key (wrong license key) must raise InvalidTag,
    not return garbage — auth.py wraps the whole decrypt in try/except and
    treats any raise as an invalid license."""
    crypto = depcheck.load(IMPORT_NAME)
    aead = depcheck.load("cryptography.hazmat.primitives.ciphers.aead")
    InvalidTag = crypto.exceptions.InvalidTag

    nonce = os.urandom(12)
    ct = aead.AESGCM(hashlib.sha256(b"right").digest()).encrypt(nonce, b"x", None)
    wrong = aead.AESGCM(hashlib.sha256(b"wrong").digest())
    with pytest.raises(InvalidTag):
        wrong.decrypt(nonce, ct, None)


def test_aesgcm_decrypt_tampered_ciphertext_raises(depcheck):
    """Flipping a ciphertext byte must fail the GCM tag check (InvalidTag)."""
    crypto = depcheck.load(IMPORT_NAME)
    aead = depcheck.load("cryptography.hazmat.primitives.ciphers.aead")
    InvalidTag = crypto.exceptions.InvalidTag

    key = hashlib.sha256(b"k").digest()
    aesgcm = aead.AESGCM(key)
    nonce = os.urandom(12)
    ct = bytearray(aesgcm.encrypt(nonce, b"sensitive", None))
    ct[0] ^= 0xFF
    with pytest.raises(InvalidTag):
        aesgcm.decrypt(nonce, bytes(ct), None)


def test_aesgcm_invalid_tag_is_exception_subclass(depcheck):
    """auth.py relies on a bad decrypt being caught by a broad ``except
    Exception``. Pin that InvalidTag subclasses Exception."""
    crypto = depcheck.load(IMPORT_NAME)
    assert issubclass(crypto.exceptions.InvalidTag, Exception)


def test_aesgcm_generate_key_bit_length(depcheck):
    """``AESGCM.generate_key(bit_length=...)`` keyword must keep working."""
    aead = depcheck.load("cryptography.hazmat.primitives.ciphers.aead")
    key = aead.AESGCM.generate_key(bit_length=256)
    assert isinstance(key, (bytes, bytearray))
    assert len(key) == 32
