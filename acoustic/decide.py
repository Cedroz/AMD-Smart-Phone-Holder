"""Pure speaker scoring and consecutive-window switch smoothing."""

import numpy as np

import config


def decide(voiceprints, window_embedding, threshold=config.UNKNOWN_THRESHOLD):
    """Return (best speaker or 'unknown', cosine similarity confidence)."""
    if not voiceprints:
        raise ValueError("No voiceprints were supplied")
    embedding = np.asarray(window_embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Window embedding must have a finite nonzero norm")
    embedding = embedding / norm
    scores = {}
    for name, voiceprint in voiceprints.items():
        print_vector = np.asarray(voiceprint, dtype=np.float32).reshape(-1)
        print_norm = np.linalg.norm(print_vector)
        if print_norm == 0 or not np.isfinite(print_norm) or print_vector.shape != embedding.shape:
            raise ValueError(f"Invalid voiceprint for {name}")
        scores[name] = float(np.dot(embedding, print_vector / print_norm))
    speaker = max(scores, key=scores.get)
    confidence = scores[speaker]
    return (speaker if confidence >= threshold else "unknown"), confidence


class SpeakerSmoother:
    def __init__(self, count=config.SMOOTHING_COUNT):
        self.count = count
        self.current = "unknown"
        self.current_confidence = 0.0
        self.pending = None
        self.pending_count = 0

    def reset(self):
        self.current = "unknown"
        self.current_confidence = 0.0
        self.pending = None
        self.pending_count = 0

    def update(self, speaker, confidence):
        if speaker == self.current:
            self.current_confidence = confidence
            self.pending = None
            self.pending_count = 0
        else:
            if speaker == self.pending:
                self.pending_count += 1
            else:
                self.pending = speaker
                self.pending_count = 1
            if self.pending_count >= self.count:
                self.current = speaker
                self.current_confidence = confidence
                self.pending = None
                self.pending_count = 0
        return self.current, self.current_confidence