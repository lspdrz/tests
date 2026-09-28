"""A local SentenceTransformer model on disk, for an instance that embeds without any service.

The local engine (a blank `RAG_EMBEDDING_ENGINE`) loads a SentenceTransformer by name or path.
`save_keyword_model(directory, keywords)` writes a tiny static-embedding model there: each
keyword owns one dimension and every other word a small shared one, so a chunk and a question
point the same way exactly when they share a keyword, and nothing is downloaded.
`local_embedding_env(directory)` is the environment of an instance booted on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

UNKNOWN = "[UNK]"
BACKGROUND = 0.05


def save_keyword_model(directory: Path, keywords: list[str]) -> Path:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding

    vocabulary = {UNKNOWN: 0, **{keyword: index + 1 for index, keyword in enumerate(keywords)}}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocabulary, unk_token=UNKNOWN))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    weights = numpy.zeros((len(vocabulary), len(keywords) + 1), dtype=numpy.float32)
    weights[:, -1] = BACKGROUND
    for index in range(1, len(vocabulary)):
        weights[index, index - 1] = 1.0

    model = SentenceTransformer(modules=[StaticEmbedding(tokenizer, embedding_weights=weights)])
    model.save(str(directory))
    return directory


def local_embedding_env(directory: Path) -> dict[str, str]:
    return {"RAG_EMBEDDING_ENGINE": "", "RAG_EMBEDDING_MODEL": str(directory)}
