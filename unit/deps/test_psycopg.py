"""Dependency contract: what psycopg 3 must read from a TLS database URL.

psycopg 3 is the async Postgres driver every request runs on, driven over HTTP in
integration/deps/test_database_stack.py. Open WebUI passes `DATABASE_URL` to it with the SSL
parameters in libpq's spelling (`sslmode`, `sslrootcert`, `sslcert`, `sslkey`) and leaves the
rest to libpq. A connection that actually uses TLS needs a server built with SSL, which the
embedded Postgres is not, so that psycopg reads those keys back from the URL stays here, parsed
without connecting.

Discriminates: a psycopg whose conninfo parser drops or renames any of the four SSL keys fails
here.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck


def test_the_ssl_parameters_of_the_url_are_read_back(depcheck):
    conninfo = depcheck.load("psycopg.conninfo")
    parsed = conninfo.conninfo_to_dict(
        "postgresql://u@h:5432/db?sslmode=verify-full&sslrootcert=/etc/ssl/root.crt"
        "&sslcert=/etc/ssl/client.crt&sslkey=/etc/ssl/client.key"
    )
    assert parsed["sslmode"] == "verify-full"
    assert parsed["sslrootcert"] == "/etc/ssl/root.crt"
    assert parsed["sslcert"] == "/etc/ssl/client.crt"
    assert parsed["sslkey"] == "/etc/ssl/client.key"
