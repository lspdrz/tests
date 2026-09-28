"""Dependency contract: the unstructured element data the loader keeps for .rst uploads.

``UnstructuredLoader`` in ``retrieval/loaders/local.py`` imports the partitioner for a format by
name and calls it on the uploaded file. In its ``elements`` mode it keeps one document per element,
built from the element's data model::

    Document(page_content=str(element),
             metadata={**element.metadata.to_dict(), 'category': element.category,
                       'element_id': element.id})

Every format the loader sends to unstructured is uploaded end to end in
integration/deps/test_document_extraction.py (legacy Word and PowerPoint, Office files under a
legacy name, Outlook messages, workbooks, XML) and integration/retrieval/test_document_ingestion.py
(a mail saved as .msg). Only .rst uses the ``elements`` mode, and unstructured converts .rst with a
pandoc binary, so where pandoc is missing no upload reaches that data model; it is pinned here on
an XML partition, which needs no binary.

Discriminates: with ``partition_xml`` returning no elements (a pytest plugin patching the
installed library), the test goes red.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.depcheck

TEXT = "The harbour lighthouse budget was approved."


def test_partition_xml_returns_elements_the_loader_can_read(depcheck, tmp_path):
    depcheck.load("unstructured")
    # a partition module that no longer imports is breakage, not an absent package
    partition_xml = importlib.import_module("unstructured.partition.xml").partition_xml
    path = tmp_path / "notes.xml"
    path.write_text(f"<?xml version='1.0'?><notes><note>{TEXT}</note></notes>")

    elements = partition_xml(filename=str(path))

    assert TEXT in "\n\n".join(map(str, elements))
    for element in elements:
        assert isinstance(element.metadata.to_dict(), dict)
        assert isinstance(element.category, str)
        assert isinstance(element.id, str)
