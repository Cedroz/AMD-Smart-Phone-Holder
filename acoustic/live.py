"""Real-time speaker recognition and an importable speaker_stream generator."""

import argparse
import time

from capture import AudioCapture, TCPAudioCapture
from decide import SpeakerSmoother, decide
from embed import embed
from vad import VoiceActivityDetector
from voiceprints import load_voiceprints
import config


def speaker_stream(device=config.MIC_DEVICE_INDEX, voiceprints_path=config.VOICEPRINTS_PATH, local_mic=None):
    """Yield (speaker, confidence, Unix timestamp) for each live audio hop."""
    if local_mic is None:
        local_mic = config.MIC_INPUT_MODE == "local"
    voiceprints = load_voiceprints(voiceprints_path)
    detector = VoiceActivityDetector()
    smoother = SpeakerSmoother()
    capture_type = AudioCapture if local_mic else TCPAudioCapture
    capture_args = {"device": device} if local_mic else {}
    with capture_type(**capture_args) as capture:
        for audio in capture.windows():
            timestamp = time.time()
            if not detector.is_speech(audio):
                smoother.reset()
                yield "silence", 0.0, timestamp
                continue
            candidate, confidence = decide(voiceprints, embed(audio))
            speaker, confidence = smoother.update(candidate, confidence)
            yield speaker, confidence, timestamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=config.MIC_DEVICE_INDEX)
    parser.add_argument("--voiceprints", default=str(config.VOICEPRINTS_PATH))
    mic_group = parser.add_mutually_exclusive_group()
    mic_group.add_argument("--local-mic", dest="local_mic", action="store_true", help="capture from the current host")
    mic_group.add_argument("--forwarded-mic", dest="local_mic", action="store_false", help="capture from the forwarded laptop mic")
    parser.set_defaults(local_mic=config.MIC_INPUT_MODE == "local")
    args = parser.parse_args()
    try:
        for speaker, confidence, timestamp in speaker_stream(args.device, args.voiceprints, args.local_mic):
            print(f"\r{speaker:<12} confidence={confidence:.3f}  t={timestamp:.1f}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print()


if __name__ == "__main__":
    main()