"""Small shared helpers for loading and slicing mono audio."""

import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import config


def to_mono_16k(audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != config.SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), config.SAMPLE_RATE)
        audio = resample_poly(
            audio,
            config.SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)
    return audio.reshape(-1)


def load_audio(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    return to_mono_16k(audio, sample_rate)


def windows(audio):
    window_samples = int(config.WINDOW_SECONDS * config.SAMPLE_RATE)
    hop_samples = int(config.HOP_SECONDS * config.SAMPLE_RATE)
    for start in range(0, max(0, len(audio) - window_samples + 1), hop_samples):
        yield start / config.SAMPLE_RATE, audio[start : start + window_samples]