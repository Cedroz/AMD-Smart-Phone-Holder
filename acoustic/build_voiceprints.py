"""Build normalized average embeddings from each person's enrollment WAVs."""

import numpy as np
import config
from audio_utils import load_audio
from embed import embed
from vad import VoiceActivityDetector
from voiceprints import save_voiceprints


def main():
    detector = VoiceActivityDetector()
    voiceprints = {}
    window_samples = int(config.WINDOW_SECONDS * config.SAMPLE_RATE)
    for speaker in config.SPEAKERS:
        files = sorted((config.ENROLL_DIR / speaker).rglob("*.wav"))
        if not files:
            print(f"{speaker}: no enrollment audio yet; skipping")
            continue
        embeddings = []
        for path in files:
            audio = load_audio(path)
            for start in range(0, max(0, len(audio) - window_samples + 1), window_samples):
                chunk = audio[start : start + window_samples]
                if detector.is_speech(chunk):
                    embeddings.append(embed(chunk))
        if not embeddings:
            raise SystemExit(f"No speech chunks detected for {speaker}; check their enrollment WAVs.")
        average = np.mean(embeddings, axis=0)
        voiceprints[speaker] = average / np.linalg.norm(average)
        print(f"{speaker}: embedded {len(embeddings)} speech chunks from {len(files)} WAV files")
    save_voiceprints(voiceprints)
    print(f"Saved voiceprints to {config.VOICEPRINTS_PATH}")


if __name__ == "__main__":
    main()