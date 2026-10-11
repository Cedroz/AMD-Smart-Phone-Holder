"""WebRTC voice activity detection for float32 mono audio at 16 kHz."""

import numpy as np
import webrtcvad

import config


class SpeechHysteresis:
    def __init__(self, silence_hops=config.VAD_SILENCE_HOPS):
        self.silence_hops = silence_hops
        self.active = False
        self.quiet_hops = 0

    def update(self, detected):
        if detected:
            self.active = True
            self.quiet_hops = 0
        elif self.active:
            self.quiet_hops += 1
            if self.quiet_hops >= self.silence_hops:
                self.active = False
                self.quiet_hops = 0
        return self.active


class VoiceActivityDetector:
    def __init__(self, mode=config.VAD_MODE):
        self.detector = webrtcvad.Vad(mode)
        self.frame_samples = int(config.SAMPLE_RATE * config.VAD_FRAME_MS / 1000)
        self.hysteresis = SpeechHysteresis()

    def speech_ratio(self, audio):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        frame_bytes = self.frame_samples * 2
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        decisions = [
            self.detector.is_speech(pcm[start : start + frame_bytes], config.SAMPLE_RATE)
            for start in range(0, len(pcm) - frame_bytes + 1, frame_bytes)
        ]
        return sum(decisions) / len(decisions) if decisions else 0.0

    def is_speech(self, audio):
        detected = self.speech_ratio(audio) >= config.MIN_SPEECH_RATIO
        return self.hysteresis.update(detected)