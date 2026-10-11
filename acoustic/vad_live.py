"""Print live microphone speech/silence state at each hop."""

import argparse

import config
from capture import AudioCapture, TCPAudioCapture
from vad import VoiceActivityDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int)
    mic_group = parser.add_mutually_exclusive_group()
    mic_group.add_argument("--local-mic", dest="local_mic", action="store_true", help="capture from the current host")
    mic_group.add_argument("--forwarded-mic", dest="local_mic", action="store_false", help="capture from the forwarded laptop mic")
    parser.set_defaults(local_mic=config.MIC_INPUT_MODE == "local")
    args = parser.parse_args()
    detector = VoiceActivityDetector()
    capture_type = AudioCapture if args.local_mic else TCPAudioCapture
    capture_args = {"device": args.device} if args.local_mic else {}
    with capture_type(**capture_args) as capture:
        try:
            for audio in capture.windows():
                ratio = detector.speech_ratio(audio)
                state = detector.hysteresis.update(ratio >= config.MIN_SPEECH_RATIO)
                label = "speech" if state else "silence"
                print(f"\r{label:<8} speech-ratio={ratio:.2f}", end="", flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            print()


if __name__ == "__main__":
    main()