"""Offline labeled-WAV evaluation and optional turn-switch latency check."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

import config
from audio_utils import load_audio, windows
from decide import SpeakerSmoother, decide
from embed import embed
from vad import VoiceActivityDetector
from voiceprints import load_voiceprints


def classify_audio(audio, voiceprints, threshold=config.UNKNOWN_THRESHOLD):
    """Run the live VAD, embed, score, and smoothing path over one clip."""
    detector = VoiceActivityDetector()
    smoother = SpeakerSmoother()
    results = []
    for timestamp, window in windows(audio):
        if not detector.is_speech(window):
            smoother.reset()
            results.append((timestamp, "silence", 0.0))
            continue
        candidate, confidence = decide(voiceprints, embed(window), threshold)
        speaker, confidence = smoother.update(candidate, confidence)
        results.append((timestamp, speaker, confidence))
    return results


def evaluate_files(files, voiceprints):
    truth = []
    predicted = []
    for path in files:
        label = path.parent.name
        if label not in voiceprints:
            raise ValueError(f"Test label {label!r} has no voiceprint")
        results = classify_audio(load_audio(path), voiceprints)
        truth.extend([label] * len(results))
        predicted.extend(speaker for _, speaker, _ in results)
    return truth, predicted


def print_metrics(truth, predicted, speaker_names):
    labels = [*speaker_names, "unknown", "silence"]
    print(f"Accuracy: {accuracy_score(truth, predicted):.3f} ({len(truth)} windows)")
    print("Confusion matrix (rows=true, columns=predicted):")
    print("labels:", " ".join(labels))
    print(confusion_matrix(truth, predicted, labels=labels))


def report_turn_switch(path, voiceprints, turn_seconds=config.TURN_SECONDS):
    results = classify_audio(load_audio(path), voiceprints)
    names = list(voiceprints)
    for turn_index, start in enumerate(np.arange(turn_seconds, len(load_audio(path)) / config.SAMPLE_RATE, turn_seconds), start=1):
        expected = names[turn_index % len(names)]
        detection = next(
            (timestamp for timestamp, speaker, _ in results if timestamp >= start and speaker == expected),
            None,
        )
        if detection is None:
            print(f"Expected {expected} at {start:.2f}s: no switch detected")
        else:
            print(f"Expected {expected} at {start:.2f}s: switched after {detection - start:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_folder", type=Path)
    parser.add_argument("--voiceprints", type=Path, default=config.VOICEPRINTS_PATH)
    parser.add_argument("--turn-taking-wav", type=Path)
    parser.add_argument("--turn-seconds", type=float, default=config.TURN_SECONDS)
    args = parser.parse_args()
    voiceprints = load_voiceprints(args.voiceprints)
    files = sorted(args.test_folder.rglob("*.wav"))
    if not files:
        raise SystemExit(f"No labeled WAV files found under {args.test_folder}")
    truth, predicted = evaluate_files(files, voiceprints)
    print_metrics(truth, predicted, list(voiceprints))
    if args.turn_taking_wav:
        report_turn_switch(args.turn_taking_wav, voiceprints, args.turn_seconds)


if __name__ == "__main__":
    main()