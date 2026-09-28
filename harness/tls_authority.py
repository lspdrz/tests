"""A certificate authority of a test's own, and a certificate it signs for the hosts a fake plays.

A fake that stands in for an HTTPS site behind a proxy answers the TLS handshake for that site's
name. `issue_certificate(directory, authority_name, hosts)` writes a fresh authority and a leaf
certificate for `hosts` into `directory` and returns their paths; an instance that is told to
trust the authority's file takes the fake for the real site.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


@dataclass
class IssuedCertificate:
    authority: Path
    certificate: Path
    key: Path


def _key_usage_for_signing() -> x509.KeyUsage:
    return x509.KeyUsage(
        digital_signature=True,
        content_commitment=False,
        key_encipherment=False,
        data_encipherment=False,
        key_agreement=False,
        key_cert_sign=True,
        crl_sign=True,
        encipher_only=False,
        decipher_only=False,
    )


def issue_certificate(directory: Path, authority_name: str, hosts: list[str]) -> IssuedCertificate:
    now = datetime.datetime.now(datetime.timezone.utc)
    valid = {"not_valid_before": now - datetime.timedelta(days=1)}
    valid["not_valid_after"] = now + datetime.timedelta(days=7)
    authority_key = ec.generate_private_key(ec.SECP256R1())
    authority_subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, authority_name)])
    authority = (
        x509.CertificateBuilder(**valid)
        .subject_name(authority_subject)
        .issuer_name(authority_subject)
        .public_key(authority_key.public_key())
        .serial_number(x509.random_serial_number())
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(_key_usage_for_signing(), critical=True)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(authority_key.public_key()), critical=False
        )
        .sign(authority_key, hashes.SHA256())
    )
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder(**valid)
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hosts[0])]))
        .issuer_name(authority_subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(host) for host in hosts]), critical=False
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(authority_key.public_key()),
            critical=False,
        )
        .sign(authority_key, hashes.SHA256())
    )
    issued = IssuedCertificate(
        authority=directory / "ca.pem",
        certificate=directory / "leaf.pem",
        key=directory / "leaf-key.pem",
    )
    issued.authority.write_bytes(authority.public_bytes(serialization.Encoding.PEM))
    issued.certificate.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    issued.key.write_bytes(
        leaf_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return issued
