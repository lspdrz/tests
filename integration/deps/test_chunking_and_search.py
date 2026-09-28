"""Dependency smoke: chunking and hybrid search, driven through the retrieval API.

`save_docs_to_vector_db` in `routers/retrieval.py` cuts every processed file with a
langchain-text-splitters splitter chosen by the admin's `TEXT_SPLITTER`: the recursive character
splitter (""), the tiktoken-measured `TokenTextSplitter` ("token") and, before either, the
`MarkdownHeaderTextSplitter` when `ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER` is on, whose sections are
merged up to `CHUNK_MIN_SIZE_TARGET` measured by the splitter's own count: tokens from tiktoken's
`get_encoding(...).encode` for "token", with special-token markers allowed as plain text. The
configured encoding is loaded by name first, so an unknown one fails the upload naming it (twin of
unit/deps/test_tiktoken.py). The "token_transformers" splitter measures with the tokenizer
`RAG_TOKENIZER_MODEL` names, here a SentencePiece `spiece.model` whose pieces transformers reads
through sentencepiece's model proto, or a tokenizer transformers' `AutoTokenizer` loads from a
directory (twin of the tokenizer part of unit/deps/test_transformers.py). With
`ENABLE_RAG_HYBRID_SEARCH`, `/api/v1/retrieval/query/doc` ranks the chunks with rank_bm25's
`BM25Okapi` inside langchain-classic's ensemble and compression retrievers: the ensemble fuses the
keyword and the vector retriever (a langchain-core retriever answering asynchronously) and drops a
chunk both found twice, and the compressor scores what is left against the query's embedding. A bump
that breaks a splitter leaves a file as one chunk; one that breaks BM25 fails or misranks the query:
a word every chunk shares counts for next to nothing next to a rare one, and a chunk holding more of
the query's rare words ranks above one holding fewer. The provider embeds every text as the same
vector, so only the keyword ranking can pick a chunk, and every chunk scores a cosine of 1 against
the query.

Discriminates: passes on dev bbfa876af. A backend copy without the character splitter's
`split_documents` fails only the character test; one without the token splitter's `split_documents`,
the markdown `split_text` and `BM25Okapi` fails only the other four. On dev ef67cc3fa, an ensemble
that keeps both copies of a chunk (patched in at import) fails the fusion test, and so does a
compressor that keeps no score. A `BM25Okapi` that weighs every word alike (its IDF set to one,
patched in at import) ranks the chunk that repeats "harbour" first and fails the rare term test.
Measuring the transformers splitter's chunks by characters, or a sentencepiece model proto that
reads nothing (patched in at import), fails the SentencePiece test. A token measure that encodes
without `disallowed_special` fails the merge test on its marker, and one that counts characters
leaves its three sections apart; the transformers splitter measuring characters in place of the
tokenizer's `encode` fails its test.
"""

from __future__ import annotations

import os

import httpx
import pytest

from harness.actors import admin_of
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.local_embedding import save_sentencepiece_tokenizer

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source]

RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
DEAD_PROXY = "http://127.0.0.1:9"

PARAGRAPHS = [
    f"Paragraph {index} of the harbour log records the tides, the ferries and the weather."
    for index in range(12)
]


@pytest.fixture
def retrieval_settings(preserve, admin):
    """`update(**settings)` changes the document settings for this test."""
    preserve(RETRIEVAL_CONFIG)
    client = admin.client()

    def update(**settings) -> None:
        updated = client.post(RETRIEVAL_CONFIG[1], json=settings)
        assert updated.status_code == 200, updated.text

    yield update
    client.close()


def _upload(client: httpx.Client, filename: str, text: str) -> str:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (filename, text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    file_id = uploaded.json()["id"]
    stored = client.get(f"/api/v1/files/{file_id}").json()
    assert stored["data"].get("status") == "completed", stored["data"].get("error")
    return file_id


def _query(client: httpx.Client, file_id: str, query: str, **options) -> list[str]:
    """The chunks `/query/doc` returns for the file, best first."""
    answered = client.post(
        "/api/v1/retrieval/query/doc",
        json={"collection_name": f"file-{file_id}", "query": query, **options},
    )
    assert answered.status_code == 200, answered.text
    return answered.json()["documents"][0]


def _all_chunks(client: httpx.Client, file_id: str) -> list[str]:
    return _query(client, file_id, "harbour", k=100)


def _tiktoken_loads_offline(encoding_name: str) -> bool:
    """Whether the instance can build the token splitter without downloading its BPE file."""
    import tiktoken

    with pytest.MonkeyPatch.context() as patch:
        for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            patch.setenv(name, DEAD_PROXY)
        for name in ("NO_PROXY", "no_proxy"):
            patch.delenv(name, raising=False)
        try:
            tiktoken.get_encoding(encoding_name)
        except OSError:  # the download, refused by the dead proxy
            return False
    return True


# ---------------------------------------------------------------- splitters


def test_the_character_splitter_cuts_by_length(retrieval_settings, make_user):
    retrieval_settings(
        TEXT_SPLITTER="",
        ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=False,
        CHUNK_SIZE=120,
        CHUNK_OVERLAP=0,
    )
    with make_user().client() as client:
        chunks = _all_chunks(client, _upload(client, "log.txt", "\n\n".join(PARAGRAPHS)))

    assert len(chunks) > 1, "the document was stored as one chunk"
    assert max(map(len, chunks)) <= 120, [len(chunk) for chunk in chunks]


def test_the_token_splitter_cuts_by_tokens(retrieval_settings, make_user):
    _require_default_encoding()
    retrieval_settings(
        TEXT_SPLITTER="token",
        ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=False,
        CHUNK_SIZE=30,
        CHUNK_OVERLAP=0,
    )
    with make_user().client() as client:
        chunks = _all_chunks(client, _upload(client, "log.txt", "\n\n".join(PARAGRAPHS)))

    assert len(chunks) > 1, "the document was stored as one chunk"
    # 30 tokens of English run far past 30 characters.
    assert max(map(len, chunks)) > 60, [len(chunk) for chunk in chunks]


def test_the_transformers_splitter_counts_sentencepiece_pieces(
    retrieval_settings, make_user, tmp_path
):
    words = ["harbour", "lighthouse", "keeper", "ferry", "tides", "dawn"]
    tokenizer = save_sentencepiece_tokenizer(tmp_path / "sentencepiece", words)
    retrieval_settings(
        TEXT_SPLITTER="token_transformers",
        RAG_TOKENIZER_MODEL=str(tokenizer),
        ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=False,
        CHUNK_MIN_SIZE_TARGET=0,
        CHUNK_SIZE=10,
        CHUNK_OVERLAP=0,
    )
    with make_user().client() as client:
        chunks = _all_chunks(client, _upload(client, "log.txt", " ".join(words * 9)))

    # a word is one piece; the splitter measures each word and space alone, each with a closing
    # piece, so three words and their two spaces fill ten
    assert [len(chunk.split()) for chunk in chunks] == [3] * 18, chunks


def _save_word_tokenizer(directory) -> None:
    """A tokenizer that counts each word and each punctuation mark as one token."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = Tokenizer(WordLevel(vocab={"[UNK]": 0, "harbour": 1}, unk_token="[UNK]"))
    words.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=words, unk_token="[UNK]").save_pretrained(directory)


def test_the_transformers_splitter_counts_with_the_named_tokenizer(
    retrieval_settings, make_user, tmp_path
):
    pytest.importorskip("transformers", reason="the Token (Transformers) splitter needs it")
    from transformers import AutoTokenizer

    _save_word_tokenizer(tmp_path)
    retrieval_settings(
        TEXT_SPLITTER="token_transformers",
        RAG_TOKENIZER_MODEL=str(tmp_path),
        ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=False,
        CHUNK_SIZE=20,
        CHUNK_OVERLAP=0,
    )
    with make_user().client() as client:
        chunks = _all_chunks(client, _upload(client, "log.txt", "\n\n".join(PARAGRAPHS)))

    tokenizer = AutoTokenizer.from_pretrained(str(tmp_path))
    assert len(chunks) > 1, "the document was stored as one chunk"
    assert max(len(tokenizer.encode(chunk)) for chunk in chunks) <= 20, chunks
    # 20 words and marks run far past 20 characters
    assert min(map(len, chunks)) > 40, [len(chunk) for chunk in chunks]


def test_the_markdown_splitter_cuts_at_headers(retrieval_settings, make_user):
    retrieval_settings(
        TEXT_SPLITTER="",
        ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=True,
        CHUNK_MIN_SIZE_TARGET=0,
        CHUNK_SIZE=1000,
        CHUNK_OVERLAP=0,
    )
    sections = ["# Tides", "## Ferries", "## Weather"]
    markdown = "\n\n".join(f"{header}\n\n{PARAGRAPHS[0]}" for header in sections)
    with make_user().client() as client:
        chunks = _all_chunks(client, _upload(client, "log.md", markdown))

    # The whole file fits one character chunk, so only the header split can cut it.
    assert sorted(chunk.splitlines()[0].strip() for chunk in chunks) == sorted(sections), chunks


def _require_default_encoding() -> None:
    # env-only; the scratch instance inherits this process's environment
    encoding_name = os.environ.get("TIKTOKEN_ENCODING_NAME", "cl100k_base")
    if not _tiktoken_loads_offline(encoding_name):
        pytest.skip(f"the {encoding_name} BPE file is not cached and may not be downloaded")


def test_small_sections_are_merged_up_to_a_size_counted_in_tokens(retrieval_settings, make_user):
    _require_default_encoding()
    retrieval_settings(
        TEXT_SPLITTER="token",
        ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=True,
        CHUNK_MIN_SIZE_TARGET=30,
        CHUNK_SIZE=200,
        CHUNK_OVERLAP=0,
    )
    # each section runs past 30 characters but stays well under 30 tokens
    sections = [
        "# Tides\n\nThe tide turns at noon today.",
        "## Ferries\n\nThe ferry <|endoftext|> leaves hourly.",
        "## Weather\n\nFog is expected by the evening.",
    ]
    with make_user().client() as client:
        chunks = _all_chunks(client, _upload(client, "log.md", "\n\n".join(sections)))

    assert len(chunks) == 1, chunks
    assert all(section.splitlines()[0] in chunks[0] for section in sections), chunks


@pytest.mark.slow
def test_an_unknown_token_encoding_fails_the_upload_by_name(instance_with):
    # the encoding name is only read from the environment
    unknown = instance_with({"TIKTOKEN_ENCODING_NAME": "harbour_base"})
    with admin_of(unknown).client() as client:
        saved = client.post(
            RETRIEVAL_CONFIG[1],
            json={"TEXT_SPLITTER": "token", "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER": False},
        )
        assert saved.status_code == 200, saved.text
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "true", "process_in_background": "false"},
            files={"file": ("log.txt", "\n\n".join(PARAGRAPHS).encode(), "text/plain")},
        )
        stored = client.get(f"/api/v1/files/{uploaded.json()['id']}").json()

    assert stored["data"].get("status") == "failed", stored["data"]
    assert "harbour_base" in stored["data"].get("error", ""), stored["data"]


# ---------------------------------------------------------------- hybrid search


@pytest.mark.parametrize("term", ["zephyrquartz", "obsidianwharf"])
def test_bm25_finds_the_one_chunk_with_the_term(retrieval_settings, make_user, term):
    retrieval_settings(
        ENABLE_RAG_HYBRID_SEARCH=True,
        TEXT_SPLITTER="",
        ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=False,
        CHUNK_SIZE=120,
        CHUNK_OVERLAP=0,
    )
    paragraphs = list(PARAGRAPHS)
    paragraphs[4] += " zephyrquartz"
    paragraphs[9] += " obsidianwharf"
    with make_user().client() as client:
        file_id = _upload(client, "log.txt", "\n\n".join(paragraphs))
        best = _query(client, file_id, term, k=1, k_reranker=1, hybrid_bm25_weight=1)

    assert len(best) == 1 and term in best[0], best


def test_the_ensemble_fuses_both_retrievers_once_per_chunk(retrieval_settings, make_user):
    retrieval_settings(
        ENABLE_RAG_HYBRID_SEARCH=True,
        TEXT_SPLITTER="",
        ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=False,
        CHUNK_SIZE=120,
        CHUNK_OVERLAP=0,
    )
    paragraphs = list(PARAGRAPHS)
    paragraphs[7] += " zephyrquartz"
    with make_user().client() as client:
        file_id = _upload(client, "log.txt", "\n\n".join(paragraphs))
        stored = len(_all_chunks(client, file_id))
        answered = client.post(
            "/api/v1/retrieval/query/doc",
            json={
                "collection_name": f"file-{file_id}",
                "query": "zephyrquartz",
                "k": stored,
                "k_reranker": stored,
                "hybrid_bm25_weight": 0.5,
            },
        )

    assert answered.status_code == 200, answered.text
    [documents], [distances] = answered.json()["documents"], answered.json()["distances"]
    # both retrievers return every chunk; fused, each comes back once
    assert len(documents) == len(set(documents)) == stored, documents
    assert any("zephyrquartz" in document for document in documents)
    assert distances == pytest.approx([1.0] * stored), distances


# every line says "harbour"; one says nothing else, two carry the rare words
HARBOUR_LINES = [
    "Gulls circle the harbour at dawn",
    "Nets dry on the harbour wall",
    "harbour harbour harbour harbour",
    "One grey zephyrquartz glints in the harbour sand below the old pier wall",
    "Crabs hide in the harbour mud",
    "An obsidianwharf and a zephyrquartz wash up in the harbour after storms",
]


def test_bm25_weighs_a_rare_term_above_a_common_one(retrieval_settings, admin):
    retrieval_settings(ENABLE_RAG_HYBRID_SEARCH=True)
    with admin.client() as client, knowledge_base(client) as knowledge_id:
        for index, line in enumerate(HARBOUR_LINES):
            add_text_file(client, knowledge_id, f"line-{index}.txt", line)

        def best_two(query: str) -> list[str]:
            answered = client.post(
                "/api/v1/retrieval/query/collection",
                json={
                    "collection_names": [knowledge_id],
                    "query": query,
                    "k": 2,
                    "k_reranker": 2,
                    "hybrid_bm25_weight": 1,
                },
            )
            assert answered.status_code == 200, answered.text
            return [chunk.strip() for chunk in answered.json()["documents"][0]]

        rare_over_common = best_two("harbour zephyrquartz")
        both_rare_words = best_two("zephyrquartz obsidianwharf")

    assert sorted(rare_over_common) == sorted([HARBOUR_LINES[3], HARBOUR_LINES[5]])
    assert both_rare_words == [HARBOUR_LINES[5], HARBOUR_LINES[3]]
