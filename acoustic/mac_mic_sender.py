"""Run on the Mac laptop: stream its selected microphone through an SSH tunnel."""

import argparse
import socket
import wave
from pathlib import Path

import sounddevice as sd

import config


def record_wav(seconds, output_path, device):
    audio = sd.rec(
        int(seconds * config.SAMPLE_RATE),
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=device,
    )
    sd.wait()
    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(config.SAMPLE_RATE)
        output.writeframes(audio.tobytes())
    print(f"Saved recording to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="local SSH-forwarded port host")
    parser.add_argument("--port", type=int, default=config.MIC_TCP_FORWARD_PORT)
    parser.add_argument("--device", type=int, default=None, help="Mac input device; defaults to the system default")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--test-seconds", type=float, default=0)
    parser.add_argument("--test-output", default="test.wav")
    parser.add_argument("--enroll-speaker", choices=config.SPEAKERS)
    parser.add_argument("--enroll-dir", default="enroll")
    args = parser.parse_args()

    devices = sd.query_devices()
    inputs = [(index, item) for index, item in enumerate(devices) if item["max_input_channels"] > 0]
    if args.list_devices:
        if not inputs:
            raise SystemExit("No audio input device found on this Mac.")
        print("Mac audio input devices:")
        for index, item in inputs:
            print(f"  {index}: {item['name']} (up to {item['max_input_channels']} channels)")
        return
    if not inputs:
        raise SystemExit("No audio input device found on this Mac.")
    if args.device is not None and args.device not in {index for index, _ in inputs}:
        raise SystemExit(f"Device {args.device} is not an available Mac audio input.")

    device = args.device
    label = "system default" if device is None else f"{device}: {sd.query_devices(device)['name']}"
    print(f"Using Mac microphone {label} at {config.SAMPLE_RATE} Hz mono.")
    if args.test_seconds > 0:
        record_wav(args.test_seconds, args.test_output, device)
        return

    if args.enroll_speaker:
        speaker_dir = Path(args.enroll_dir) / args.enroll_speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        for phase, prompt in enumerate(config.ENROLL_PROMPTS, start=1):
            input(f"Part {phase}/{len(config.ENROLL_PROMPTS)}: {prompt}\nPress Enter to start recording.")
            output_path = speaker_dir / f"enrollment_{phase}.wav"
            record_wav(config.ENROLL_PHASE_SECONDS, output_path, device)
        print(f"Saved enrollment clips under {speaker_dir}")
        return

    try:
        connection = socket.create_connection((args.host, args.port), timeout=config.MIC_CONNECT_TIMEOUT_SECONDS)
    except OSError as error:
        raise SystemExit(f"Could not reach SSH-forwarded audio receiver at {args.host}:{args.port}: {error}") from error

    try:
        with connection:
            with sd.InputStream(
                device=device,
                samplerate=config.SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=config.TCP_AUDIO_BLOCK_FRAMES,
            ) as stream:
                print("Streaming Mac microphone to remote speaker-ID process. Press Ctrl+C to stop.")
                while True:
                    audio, overflowed = stream.read(config.TCP_AUDIO_BLOCK_FRAMES)
                    if overflowed:
                        print("Warning: Mac microphone input overflowed.")
                    connection.sendall(audio.tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        print("Mac microphone stream stopped.")


if __name__ == "__main__":
    main()