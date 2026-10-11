"""Record varied enrollment speech for each configured speaker."""

import argparse

import soundfile as sf

import config
from capture import AudioCapture, TCPAudioCapture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=config.MIC_DEVICE_INDEX)
    parser.add_argument("--speaker", action="append", choices=config.SPEAKERS)
    mic_group = parser.add_mutually_exclusive_group()
    mic_group.add_argument("--local-mic", dest="local_mic", action="store_true", help="record from a microphone attached to this host")
    mic_group.add_argument("--forwarded-mic", dest="local_mic", action="store_false", help="record from the forwarded laptop microphone")
    parser.set_defaults(local_mic=config.MIC_INPUT_MODE == "local")
    args = parser.parse_args()

    phase_count = len(config.ENROLL_PROMPTS)
    phase_seconds = config.ENROLL_SECONDS // phase_count
    if phase_seconds != config.ENROLL_PHASE_SECONDS:
        raise SystemExit("ENROLL_SECONDS must be evenly split across ENROLL_PROMPTS in config.py")
    capture_type = AudioCapture if args.local_mic else TCPAudioCapture
    capture_args = {"device": args.device} if args.local_mic else {}
    with capture_type(**capture_args) as capture:
        for speaker in args.speaker or config.SPEAKERS:
            person_dir = config.ENROLL_DIR / speaker
            person_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nSpeaker: {speaker}. Total recording: {config.ENROLL_SECONDS} seconds.")
            for phase, prompt in enumerate(config.ENROLL_PROMPTS, start=1):
                input(f"\nPart {phase}/{phase_count}: {prompt}\nStart the Mac mic sender, then press Enter.")
                print("Recording now...")
                audio = capture.record(phase_seconds)
                output = person_dir / f"enrollment_{phase}.wav"
                sf.write(output, audio, config.SAMPLE_RATE)
                print(f"Saved {output}")
    print(f"Enrollment WAVs saved under {config.ENROLL_DIR}")


if __name__ == "__main__":
    main()