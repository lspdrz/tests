"""Dependency contract: docx2txt, the call shapes Open WebUI does not use.

Open WebUI reads a .docx upload with ``docx2txt.process(Path(file_path))`` in
``retrieval/loaders/local.py``; that is driven from outside in
integration/deps/test_document_extraction.py (every paragraph in order, non-Latin text intact,
an empty document reported as empty).

Kept as a unit contract: `process` also takes a file-like object or a plain string path, which
no Open WebUI request passes it. The .docx packages are built in memory (a .docx is a zip of
OOXML parts). Uses ``depcheck`` from conftest.py.
"""

from __future__ import annotations

import zipfile
from io import BytesIO

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "docx2txt"
DIST_NAME = "docx2txt"


# ---------------------------------------------------------------------------
# Helpers: build a minimal valid .docx (OOXML zip) in memory.
# ---------------------------------------------------------------------------

_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" '
    'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    "</Types>"
)

_ROOT_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="word/document.xml"/>'
    "</Relationships>"
)


def _make_docx(paragraphs: list[str]) -> BytesIO:
    """Assemble a valid single-section .docx containing the given paragraphs."""
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _ROOT_RELS)
        z.writestr("word/document.xml", document_xml)
    buf.seek(0)
    return buf


def test_behaviour_accepts_file_like_object(depcheck):
    """The loader can hand process() a file-like (BytesIO); zip-based reading
    must work on an in-memory stream, not only a path."""
    mod = depcheck.load(IMPORT_NAME)
    bio = _make_docx(["Stream based content"])
    text = mod.process(bio)
    assert "Stream based content" in text


def test_behaviour_accepts_path(depcheck, tmp_path):
    """Docx2txtLoader(file_path) passes a filesystem path; process() must read
    a .docx from disk. (tmp_path is pytest's per-test temp dir — local, no net.)"""
    mod = depcheck.load(IMPORT_NAME)
    p = tmp_path / "doc.docx"
    p.write_bytes(_make_docx(["Path based content"]).getvalue())
    text = mod.process(str(p))
    assert "Path based content" in text
