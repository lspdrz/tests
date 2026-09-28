"""Many tiny SpeechT5 voices in a Hugging Face cache, for telling local TTS speakers apart.

The `transformers` text-to-speech engine asks the Hub for `microsoft/speecht5_tts` (and the
pipeline for its vocoder, `microsoft/speecht5_hifigan`) and for the speaker embeddings in the
`Matthijs/cmu-arctic-xvectors` dataset. `save_tiny_speech(directory)` fills a Hugging Face home
there with all three, so nothing is downloaded: a one-layer SpeechT5 with random weights and a
character tokenizer, a small HiFi-GAN and `SPEAKERS` embeddings, each speaker named by `speaker`
and pointing its own way. The model is built never to predict the end of speech, so every
sentence is spoken for the longest time the model allows for its length, and the same sentence
in the same voice always comes out the same.
`harness.local_speech` has a single voice; this one has one per speaker.
`local_voices_env(directory)` is the environment of an instance booted on that home, offline.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

MODEL = "microsoft/speecht5_tts"
VOCODER = "microsoft/speecht5_hifigan"
DATASET = "Matthijs/cmu-arctic-xvectors"
REVISION = "0" * 40
SPEAKERS = 6800  # the engine falls back to speaker 6799
EMBEDDING_SIZE = 8
SAMPLING_RATE = 16000
ALPHABET = "abcdefghijklmnopqrstuvwxyz .,"


def speaker(index: int) -> str:
    """The dataset's name for the speaker at `index`, the name an admin sets as the TTS model."""
    return f"cmu_us_harbour_arctic-wav-arctic_a{index:04d}"


def _snapshot(hub: Path, repo_id: str) -> Path:
    """A cached copy of a model repository, the way `huggingface_hub` lays one out."""
    folder = hub / f"models--{repo_id.replace('/', '--')}"
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text(REVISION)
    snapshot = folder / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    return snapshot


def _save_model(hub: Path, scratch: Path) -> None:
    import sentencepiece
    import torch
    from transformers import SpeechT5Config, SpeechT5ForTextToSpeech, SpeechT5Tokenizer

    snapshot = _snapshot(hub, MODEL)
    corpus = scratch / "corpus.txt"
    corpus.write_text((ALPHABET + "\n") * 20)
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(snapshot / "spm_char"),
        model_type="char",
        vocab_size=len(set(ALPHABET)) + 4,
        bos_id=0,
        pad_id=1,
        eos_id=2,
        unk_id=3,
        minloglevel=2,
    )
    tokenizer = SpeechT5Tokenizer(str(snapshot / "spm_char.model"))
    tokenizer.save_pretrained(snapshot)
    config = SpeechT5Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        speaker_embedding_dim=EMBEDDING_SIZE,
        num_mel_bins=8,
        speech_decoder_prenet_units=16,
        speech_decoder_prenet_dropout=0.0,  # SpeechT5 drops out even when speaking
        speech_decoder_postnet_units=16,
        speech_decoder_postnet_layers=2,
        speech_decoder_postnet_kernel=3,
        reduction_factor=2,
        max_speech_positions=2000,
        max_text_positions=200,
    )
    torch.manual_seed(0)
    model = SpeechT5ForTextToSpeech(config)
    with torch.no_grad():
        # the stop probability stays near zero, so speech runs to the length limit
        model.speech_decoder_postnet.prob_out.weight.zero_()
        model.speech_decoder_postnet.prob_out.bias.fill_(-10.0)
    model.save_pretrained(snapshot)


def _save_vocoder(hub: Path) -> None:
    import torch
    from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig

    config = SpeechT5HifiGanConfig(
        model_in_dim=8,
        sampling_rate=SAMPLING_RATE,
        upsample_initial_channel=16,
        upsample_rates=[4, 4],
        upsample_kernel_sizes=[8, 8],
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3]],
    )
    torch.manual_seed(0)
    vocoder = SpeechT5HifiGan(config)
    with torch.no_grad():
        # random weights whisper at a millionth of full scale, below what MP3 keeps
        vocoder.conv_post.weight.mul_(200_000)
    vocoder.save_pretrained(_snapshot(hub, VOCODER))


def _save_speakers(datasets_cache: Path, scratch: Path) -> None:
    """The dataset as `datasets` keeps one it prepared, which it reads back offline."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets.data_files import DataFilesDict
    from datasets.packaged_modules.parquet.parquet import Parquet

    table = pa.table(
        {
            "filename": [speaker(index) for index in range(SPEAKERS)],
            "xvector": [
                [((index + dimension) % 5 - 2) / 2 for dimension in range(EMBEDDING_SIZE)]
                for index in range(SPEAKERS)
            ],
        }
    )
    parquet = scratch / "validation.parquet"
    pq.write_table(table, parquet)
    builder = Parquet(
        cache_dir=str(scratch / "prepared"),
        dataset_name=DATASET.split("/")[1],
        repo_id=DATASET,
        data_files=DataFilesDict.from_patterns({"validation": [str(parquet)]}),
    )
    builder.download_and_prepare()
    # a Hub download keeps it under default/<version>/<revision>, where the offline lookup reads
    prepared = Path(builder.cache_dir)
    target = datasets_cache / DATASET.replace("/", "___") / "default" / prepared.name / REVISION
    target.parent.mkdir(parents=True)
    shutil.move(str(prepared), str(target))


def save_tiny_speech(directory: Path) -> Path:
    """Fill a Hugging Face home under `directory` with the voice; returns the home."""
    for package in ("torch", "transformers", "sentencepiece", "datasets", "soundfile"):
        pytest.importorskip(package, reason=f"local text-to-speech needs {package}")
    home = directory / "huggingface"
    scratch = directory / "scratch"
    scratch.mkdir(parents=True)
    _save_model(home / "hub", scratch)
    _save_vocoder(home / "hub")
    _save_speakers(home / "datasets", scratch)
    shutil.rmtree(scratch)
    return home


def local_voices_env(home: Path) -> dict[str, str]:
    """The environment of an instance whose Hugging Face home is `home`, kept offline."""
    return {
        "HF_HOME": str(home),
        "HF_HUB_CACHE": str(home / "hub"),
        "HF_DATASETS_CACHE": str(home / "datasets"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
