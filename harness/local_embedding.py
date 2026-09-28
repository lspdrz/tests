"""A local SentenceTransformer model on disk, for an instance that embeds without any service.

The local engine (a blank `RAG_EMBEDDING_ENGINE`) loads a SentenceTransformer by name or path.
`save_keyword_model(directory, keywords)` writes a tiny static-embedding model there: each
keyword owns one dimension and every other word a small shared one, so a chunk and a question
point the same way exactly when they share a keyword, and nothing is downloaded.
`local_embedding_env(directory)` is the environment of an instance booted on it.
`save_keyword_reranker(directory, keywords)` writes a cross-encoder for local reranking that
scores a question and a chunk by the keywords they share, and `save_sentencepiece_tokenizer`
a SentencePiece tokenizer in which each given word is one piece.
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


RERANKER_SOURCE = """
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class KeywordRerankerConfig(PretrainedConfig):
    model_type = "keyword_reranker"

    def __init__(self, keyword_ids=(), **kwargs):
        self.keyword_ids = list(keyword_ids)
        super().__init__(**kwargs)


class KeywordReranker(PreTrainedModel):
    config_class = KeywordRerankerConfig

    def __init__(self, config):
        super().__init__(config)
        self.offset = torch.nn.Parameter(torch.zeros(1))
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        keywords = set(self.config.keyword_ids)
        scores = []
        for ids, segments in zip(input_ids.tolist(), token_type_ids.tolist()):
            question = {token for token, segment in zip(ids, segments) if segment == 0}
            passage = {token for token, segment in zip(ids, segments) if segment == 1}
            scores.append([float(len(question & passage & keywords))])
        return SequenceClassifierOutput(logits=torch.tensor(scores) + self.offset)
"""
SPECIAL_TOKENS = ["[PAD]", UNKNOWN, "[CLS]", "[SEP]"]


def save_keyword_reranker(directory: Path, keywords: list[str]) -> Path:
    """A cross-encoder whose score for a pair is the number of keywords both sides contain.

    Its model class ships as code in the directory, loaded with `trust_remote_code`. Like the
    MS MARCO rerankers its config names the identity as its activation, so a caller that sets
    none reads the raw count.
    """
    import importlib.util
    import sys

    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "keyword_reranker.py").write_text(RERANKER_SOURCE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "keyword_reranker", directory / "keyword_reranker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # transformers looks a model's class up by module name while saving it
    sys.modules.setdefault("keyword_reranker", module)

    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS + keywords)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocabulary, unk_token=UNKNOWN))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocabulary["[CLS]"]), ("[SEP]", vocabulary["[SEP]"])],
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNKNOWN,
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_input_names=["input_ids", "token_type_ids", "attention_mask"],
    ).save_pretrained(str(directory))

    config = module.KeywordRerankerConfig(
        keyword_ids=[vocabulary[keyword] for keyword in keywords],
        num_labels=1,
        architectures=["KeywordReranker"],
        sbert_ce_default_activation_function="torch.nn.modules.linear.Identity",
        auto_map={
            "AutoConfig": "keyword_reranker.KeywordRerankerConfig",
            "AutoModelForSequenceClassification": "keyword_reranker.KeywordReranker",
        },
    )
    module.KeywordReranker(config).save_pretrained(str(directory))
    return directory


def save_sentencepiece_tokenizer(directory: Path, words: list[str]) -> Path:
    """A SentencePiece tokenizer (`spiece.model`) trained so that each of `words` is one piece."""
    import io
    import json
    import random

    import sentencepiece

    shuffler = random.Random(0)
    corpus = [" ".join(shuffler.sample(words, len(words))) for _ in range(200)]
    model = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(corpus),
        model_writer=model,
        vocab_size=40,
        model_type="unigram",
        character_coverage=1.0,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "spiece.model").write_bytes(model.getvalue())
    tokenizer_config = {"tokenizer_class": "T5Tokenizer"}
    (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    return directory
