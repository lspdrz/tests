"""Dependency smoke: local embedding and reranking models, run by sentence-transformers.

With a blank embedding engine Open WebUI loads its embedding model as a `SentenceTransformer`
from a path, and embeds with `encode(texts, batch_size=, prompt=prefix)`; the question's prefix
is `RAG_EMBEDDING_QUERY_PREFIX`. The "token_transformers" splitter measures chunks with the
model's own `tokenizer` when no tokenizer model is set. With hybrid search on, a reranking model
is loaded as a `CrossEncoder(path, trust_remote_code=, activation_fn=Sigmoid)` and scores each
(question, chunk) pair with `predict(pairs, batch_size=)`; those scores order the result.

The models are built on disk without a download (`harness.local_embedding`): the embedding model
gives each keyword a dimension of its own, the reranker's model code (loaded with
`trust_remote_code`) scores a pair by the number of keywords both sides contain. The query prefix
here names the orchard three times, so a question about the harbour ends up nearer the orchard.

Discriminates: passes on dev ef67cc3fa. In a backend copy, dropping the `prompt` from the local
`encode` call fails the prefix test, measuring chunks by characters fails the tokenizer test and
building the `CrossEncoder` without its `activation_fn` fails the reranker test (the model's own
identity activation leaves the raw keyword counts, as an MS MARCO reranker's would).
"""

from __future__ import annotations

import httpx
import pytest

from harness.actors import admin_of
from harness.instance import LaunchedInstance
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.local_embedding import (
    local_embedding_env,
    save_keyword_model,
    save_keyword_reranker,
)

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
EMBEDDING_KEYWORDS = ["orchard", "harbour", "lighthouse", "keeper"]
RERANKER_KEYWORDS = ["lighthouse", "keeper", "ferry", "harbour"]
QUERY_PREFIX = "orchard orchard orchard "


@pytest.fixture(scope="module")
def models(tmp_path_factory) -> dict[str, str]:
    embedding = save_keyword_model(tmp_path_factory.mktemp("embedding"), EMBEDDING_KEYWORDS)
    reranker = save_keyword_reranker(tmp_path_factory.mktemp("reranker"), RERANKER_KEYWORDS)
    return {
        **local_embedding_env(embedding),
        "RAG_EMBEDDING_QUERY_PREFIX": QUERY_PREFIX,
        "ENABLE_RAG_HYBRID_SEARCH": "true",
        "RAG_RERANKING_MODEL": str(reranker),
        "RAG_RERANKING_MODEL_TRUST_REMOTE_CODE": "true",
    }


@pytest.fixture
def local(instance_with, models) -> LaunchedInstance:
    return instance_with(models)


def _query(client: httpx.Client, collection_name: str, query: str, **options) -> dict:
    answered = client.post(
        "/api/v1/retrieval/query/doc",
        json={"collection_name": collection_name, "query": query, **options},
    )
    assert answered.status_code == 200, answered.text
    return answered.json()


def test_a_question_is_embedded_with_the_query_prefix(local):
    orchard = "The orchard is north of the house."
    harbour = "The harbour is south of the house."
    with admin_of(local).client() as client, knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "apples.txt", orchard)
        add_text_file(client, knowledge_id, "boats.txt", harbour)
        found = _query(client, knowledge_id, "where is the harbour?", k=2, hybrid=False)

    assert found["documents"] == [[orchard, harbour]], "the question lost its orchard prefix"


def test_chunks_are_measured_with_the_local_models_tokenizer(local, preserve):
    preserve(RETRIEVAL_CONFIG, on=local)
    words = " ".join(["harbour lighthouse keeper orchard"] * 10)
    with admin_of(local).client() as client:
        saved = client.post(
            RETRIEVAL_CONFIG[1],
            json={
                "TEXT_SPLITTER": "token_transformers",
                "RAG_TOKENIZER_MODEL": "",
                "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER": False,
                "CHUNK_MIN_SIZE_TARGET": 0,
                "CHUNK_SIZE": 5,
                "CHUNK_OVERLAP": 0,
            },
        )
        assert saved.status_code == 200, saved.text
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "true", "process_in_background": "false"},
            files={"file": ("words.txt", words.encode(), "text/plain")},
        )
        assert uploaded.status_code == 200, uploaded.text
        file_id = uploaded.json()["id"]
        chunks = _query(client, f"file-{file_id}", "harbour", k=100, hybrid=False)["documents"][0]

    # five tokens a chunk: the word-level tokenizer counts each word once
    assert sorted(len(chunk.split()) for chunk in chunks) == [5] * 8, chunks


def test_the_local_reranker_orders_and_scores_the_chunks(local):
    keeper = "The lighthouse keeper rows out at dawn."
    lighthouse = "The lighthouse stands on the rocks."
    ferry = "The ferry leaves at noon."
    with admin_of(local).client() as client, knowledge_base(client) as knowledge_id:
        for name, text in (("ferry.txt", ferry), ("rocks.txt", lighthouse), ("dawn.txt", keeper)):
            add_text_file(client, knowledge_id, name, text)
        found = _query(
            client, knowledge_id, "Who is the lighthouse keeper?", k=3, k_reranker=3, hybrid=True
        )

    assert found["documents"] == [[keeper, lighthouse, ferry]]
    # the sigmoid of two, one and no shared keywords
    assert found["distances"][0] == pytest.approx([0.8808, 0.7311, 0.5], abs=1e-3)
