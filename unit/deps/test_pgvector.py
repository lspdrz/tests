"""Dependency contract: pgvector's half-precision column, the one no local Postgres can hold.

Open WebUI's pgvector store declares its embedding column as
``VECTOR_TYPE_FACTORY(dim=VECTOR_LENGTH)``, where the factory is pgvector's ``HALFVEC`` when
``PGVECTOR_USE_HALFVEC`` is on (embeddings over 2000 dimensions) and ``Vector`` otherwise, and
ranks chunks with ``column.cosine_distance(...)``, pgvector's ``<=>``. The ``Vector`` path is
driven end to end in integration/deps/test_vector_stores.py on the embedded Postgres. That
server ships the vector extension at 0.6.2, and ``halfvec`` only arrived in 0.7.0, so the
``HALFVEC`` path stays here: the type imports, takes ``dim`` and compiles to ``HALFVEC(n)`` DDL
and to ``<=>`` in the ordering, against SQLAlchemy's PostgreSQL dialect without a connection.

Discriminates: with ``HALFVEC`` renamed away, its ``dim`` parameter renamed or its compiler
emitting ``VECTOR`` the matching test fails.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck


def _halfvec(depcheck):
    depcheck.load("pgvector")
    return depcheck.load("pgvector.sqlalchemy").HALFVEC


def _postgres_dialect(depcheck):
    depcheck.load("sqlalchemy")
    from sqlalchemy.dialects import postgresql

    return postgresql.dialect()


def test_halfvec_is_a_column_type_taking_its_dimension(depcheck):
    from sqlalchemy.types import TypeEngine

    halfvec = _halfvec(depcheck)
    assert issubclass(halfvec, TypeEngine)
    depcheck.assert_params(halfvec.__init__, ["dim"])


def test_a_halfvec_column_is_created_as_halfvec(depcheck):
    column_type = _halfvec(depcheck)(dim=4096)
    ddl = str(column_type.compile(dialect=_postgres_dialect(depcheck))).upper()
    assert ddl == "HALFVEC(4096)"


def test_a_halfvec_column_ranks_by_cosine_distance(depcheck):
    import sqlalchemy as sa

    table = sa.Table(
        "document_chunk",
        sa.MetaData(),
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("vector", _halfvec(depcheck)(dim=3)),
    )
    distance = table.c.vector.cosine_distance([0.1, 0.2, 0.3]).label("distance")
    query = sa.select(table.c.id, distance).order_by(distance)
    compiled = str(query.compile(dialect=_postgres_dialect(depcheck)))
    assert "document_chunk.vector <=>" in compiled
    assert "ORDER BY distance" in compiled
