"""Evaluate labeled test WAVs after mixing in noise at configured SNRs."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

import config
from audio_utils import load_audio
from evaluate import classify_audio
from voiceprints import load_voiceprints


def mix_at_snr(signal, noise, snr_db):
    if not len(noise):
        raise ValueError("Noise WAV is empty")
    tiled_noise = np.resize(noise, len(signal))
    signal_rms = np.sqrt(np.mean(np.square(signal)))
    noise_rms = np.sqrt(np.mean(np.square(tiled_noise)))
    if noise_rms == 0:
        raise ValueError("Noise WAV is silent")
    scale = signal_rms / (10 ** (snr_db / 20) * noise_rms)
    return np.clip(signal + tiled_noise * scale, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_folder", type=Path)
    parser.add_argument("noise_wav", type=Path)
    parser.add_argument("--voiceprints", type=Path, default=config.VOICEPRINTS_PATH)
    args = parser.parse_args()
    files = sorted(args.test_folder.rglob("*.wav"))
    if not files:
        raise SystemExit(f"No labeled WAV files found under {args.test_folder}")
    noise = load_audio(args.noise_wav)
    voiceprints = load_voiceprints(args.voiceprints)
    for snr_db in config.NOISE_SNR_LEVELS_DB:
        truth = []
        predicted = []
        for path in files:
            label = path.parent.name
            results = classify_audio(mix_at_snr(load_audio(path), noise, snr_db), voiceprints)
            truth.extend([label] * len(results))
            predicted.extend(speaker for _, speaker, _ in results)
        print(f"SNR {snr_db:+} dB: accuracy {accuracy_score(truth, predicted):.3f} ({len(truth)} windows)")


if __name__ == "__main__":
    main()