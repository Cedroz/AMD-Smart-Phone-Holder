"""Resemblyzer CPU embedding wrapper; replace embed() to swap encoders."""

import time

import numpy as np

import config

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder

        _encoder = VoiceEncoder(device="cpu")
    return _encoder


def embed(audio):
    """Return a unit-norm speaker embedding from 16 kHz mono float audio."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    vector = np.asarray(_get_encoder().embed_utterance(audio), dtype=np.float32)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("The speaker encoder returned an invalid embedding")
    return vector / norm


def benchmark(repeats=config.EMBED_BENCHMARK_REPEATS):
    audio = np.zeros(int(config.WINDOW_SECONDS * config.SAMPLE_RATE), dtype=np.float32)
    embed(audio)
    started = time.perf_counter()
    for _ in range(repeats):
        embed(audio)
    elapsed_ms = (time.perf_counter() - started) * 1000 / repeats
    print(f"CPU embedding latency: {elapsed_ms:.1f} ms per {config.WINDOW_SECONDS:.1f} s window ({repeats} runs)")
    return elapsed_ms


if __name__ == "__main__":
    benchmark()