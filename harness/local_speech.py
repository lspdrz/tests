"""A tiny SpeechT5 in a Hugging Face cache of its own, for local text-to-speech offline.

The "transformers" speech engine loads `microsoft/speecht5_tts` (with the
`microsoft/speecht5_hifigan` vocoder) and the speaker voices of the `Matthijs/cmu-arctic-xvectors`
dataset by name, from the Hub or the cache under `HF_HOME`. `save_tiny_speecht5(home)` writes
all three into `home` without downloading anything: a one-layer SpeechT5 and vocoder with random
weights, a character tokenizer, and one voice, `SPEAKER`. Whatever it is asked to say, it speaks
a short 16 kHz noise. Boot an instance with `HF_HOME` set to `home` to use it.
"""

from __future__ import annotations

import glob
import io
import json
import shutil
import tempfile
from pathlib import Path

SPEAKER = "cmu_us_slt_arctic-wav-arctic_a0508"
SAMPLE_RATE = 16000
COMMIT = "0" * 40


def _snapshot(home: Path, repo: str) -> Path:
    """The cached snapshot directory the Hub client resolves `repo` to when offline."""
    root = home / "hub" / f"models--{repo.replace('/', '--')}"
    (root / "refs").mkdir(parents=True, exist_ok=True)
    (root / "refs" / "main").write_text(COMMIT)
    snapshot = root / "snapshots" / COMMIT
    snapshot.mkdir(parents=True, exist_ok=True)
    return snapshot


def _save_models(home: Path) -> None:
    import sentencepiece
    import torch
    from transformers import (
        SpeechT5Config,
        SpeechT5FeatureExtractor,
        SpeechT5ForTextToSpeech,
        SpeechT5HifiGan,
        SpeechT5HifiGanConfig,
        SpeechT5Processor,
        SpeechT5Tokenizer,
    )

    torch.manual_seed(0)
    speech = _snapshot(home, "microsoft/speecht5_tts")
    characters = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(["the harbour lighthouse keeper"] * 50),
        model_writer=characters,
        vocab_size=30,
        model_type="char",
        character_coverage=1.0,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    (speech / "spm_char.model").write_bytes(characters.getvalue())
    tokenizer = SpeechT5Tokenizer(str(speech / "spm_char.model"))
    SpeechT5Processor(
        feature_extractor=SpeechT5FeatureExtractor(), tokenizer=tokenizer
    ).save_pretrained(str(speech))
    layers = {"hidden_size": 16, "encoder_layers": 1, "decoder_layers": 1}
    heads = {"encoder_attention_heads": 2, "decoder_attention_heads": 2}
    widths = {"encoder_ffn_dim": 16, "decoder_ffn_dim": 16, "speech_decoder_prenet_units": 16}
    postnet = {"speech_decoder_postnet_units": 16, "speech_decoder_postnet_layers": 1}
    config = SpeechT5Config(vocab_size=len(tokenizer), **layers, **heads, **widths, **postnet)
    SpeechT5ForTextToSpeech(config).save_pretrained(str(speech))

    vocoder = SpeechT5HifiGanConfig(
        upsample_initial_channel=16,
        upsample_rates=[4, 4],
        upsample_kernel_sizes=[8, 8],
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3]],
    )
    SpeechT5HifiGan(vocoder).save_pretrained(str(_snapshot(home, "microsoft/speecht5_hifigan")))


def _save_voices(home: Path) -> None:
    """The voices dataset, prepared in the datasets cache the way a first download leaves it."""
    import datasets
    import pyarrow
    import pyarrow.parquet

    work = Path(tempfile.mkdtemp())
    try:
        voices = pyarrow.table({"filename": [SPEAKER], "xvector": [[0.01] * 512]})
        pyarrow.parquet.write_table(voices, work / "validation.parquet")
        files = {"validation": str(work / "validation.parquet")}
        datasets.load_dataset("parquet", data_files=files, cache_dir=str(work / "cache"))
        found = glob.glob(str(work / "cache" / "parquet" / "*" / "*" / "*"))
        [prepared] = [Path(path) for path in found if Path(path).is_dir()]
        cached = home / "datasets" / "Matthijs___cmu-arctic-xvectors" / "default" / "0.0.0" / "0"
        cached.mkdir(parents=True)
        info = json.loads((prepared / "dataset_info.json").read_text(encoding="utf-8"))
        info.update(config_name="default", dataset_name="cmu-arctic-xvectors")
        (cached / "dataset_info.json").write_text(json.dumps(info), encoding="utf-8")
        for arrow in prepared.glob("*.arrow"):
            renamed = arrow.name.replace("parquet-", "cmu-arctic-xvectors-")
            shutil.copy(arrow, cached / renamed)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def save_tiny_speecht5(home: Path) -> Path:
    _save_models(home)
    _save_voices(home)
    return home
