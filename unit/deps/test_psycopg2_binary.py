"""Dependency contract: what psycopg2 must read from a TLS database URL, and openGauss's dialect.

psycopg2 is the synchronous Postgres driver: it migrates and loads the settings at boot and
serves the pgvector store, all driven over HTTP in integration/deps/test_database_stack.py and
test_vector_stores.py. Open WebUI strips the SSL parameters from `DATABASE_URL` and reattaches
them in libpq's own spelling (`sslmode`, `sslrootcert`, `sslcert`, `sslkey`) for it, and the
bare `ssl=` form is exercised there too. A connection that actually uses TLS needs a server
built with SSL, which the embedded Postgres is not, so that libpq reads those keys back from the
URL stays here, parsed without connecting. The openGauss vector store subclasses SQLAlchemy's
psycopg2 dialect, and no openGauss server runs locally either.

Discriminates: a psycopg2 whose DSN parser drops or renames any of the four SSL keys, or a
SQLAlchemy without the psycopg2 dialect class, fails here.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck


def test_the_reattached_ssl_parameters_are_read_back(depcheck):
    psycopg2 = depcheck.load("psycopg2")
    parsed = psycopg2.extensions.parse_dsn(
        "postgresql://u@h/db?sslmode=verify-full&sslrootcert=/etc/ssl/root.crt"
        "&sslcert=/etc/ssl/client.crt&sslkey=/etc/ssl/client.key"
    )
    assert parsed["sslmode"] == "verify-full"
    assert parsed["sslrootcert"] == "/etc/ssl/root.crt"
    assert parsed["sslcert"] == "/etc/ssl/client.crt"
    assert parsed["sslkey"] == "/etc/ssl/client.key"


def test_the_opengauss_store_finds_the_psycopg2_dialect_to_extend(depcheck):
    depcheck.load("psycopg2")
    dialects = depcheck.load("sqlalchemy.dialects.postgresql.psycopg2")
    assert inspect.isclass(dialects.PGDialect_psycopg2)
