"""List microphone devices and record a short WAV test clip."""

import argparse

import sounddevice as sd
import soundfile as sf

import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=config.MIC_DEVICE_INDEX)
    parser.add_argument("--output", default=str(config.PROJECT_DIR / "test.wav"))
    args = parser.parse_args()

    devices = sd.query_devices()
    inputs = [(index, device) for index, device in enumerate(devices) if device["max_input_channels"] > 0]
    if not inputs:
        raise SystemExit("No audio input device found. Connect a microphone or check host audio access.")
    print("Audio input devices:")
    for index, device in inputs:
        print(f"  {index}: {device['name']} (up to {device['max_input_channels']} channels)")

    device_index = args.device
    if device_index is None:
        device_index = int(input("Enter input device index to record from: "))
    if device_index not in {index for index, _ in inputs}:
        raise SystemExit(f"Device {device_index} is not an available audio input.")

    print(f"Recording {config.TEST_RECORD_SECONDS} seconds from device {device_index}...")
    audio = sd.rec(
        int(config.TEST_RECORD_SECONDS * config.SAMPLE_RATE),
        samplerate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
        dtype="float32",
        device=device_index,
    )
    sd.wait()
    sf.write(args.output, audio, config.SAMPLE_RATE)
    print(f"Saved microphone test recording to {args.output}")


if __name__ == "__main__":
    main()