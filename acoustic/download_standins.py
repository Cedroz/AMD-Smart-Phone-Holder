"""Stream three LibriSpeech speakers into placeholder enrollment folders."""

import argparse

import numpy as np
import soundfile as sf

import config
from audio_utils import to_mono_16k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=config.ENROLL_SECONDS)
    args = parser.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("Install the pinned requirements first; datasets is needed for LibriSpeech.") from error

    dataset = load_dataset(
        "openslr/librispeech_asr",
        config.LIBRISPEECH_CONFIG,
        split=config.LIBRISPEECH_SPLIT,
        streaming=True,
    )
    total_seconds = args.seconds + config.STANDIN_TEST_SECONDS
    seconds = {name: 0.0 for name in config.STANDIN_SPEAKERS}
    audio_parts = {name: [] for name in config.STANDIN_SPEAKERS}
    source_to_name = {}
    print("Streaming LibriSpeech validation audio. Placeholder mapping follows first three speaker IDs encountered.")
    for row in dataset:
        source_id = str(row["speaker_id"])
        if source_id not in source_to_name and len(source_to_name) < len(config.STANDIN_SPEAKERS):
            source_to_name[source_id] = config.STANDIN_SPEAKERS[len(source_to_name)]
            print(f"LibriSpeech speaker {source_id} -> placeholder {source_to_name[source_id]}")
        name = source_to_name.get(source_id)
        if name is None or seconds[name] >= total_seconds:
            if all(value + 1e-3 >= total_seconds for value in seconds.values()):
                break
            continue
        samples = to_mono_16k(row["audio"]["array"], row["audio"]["sampling_rate"])
        remaining = int((args.seconds + config.STANDIN_TEST_SECONDS - seconds[name]) * config.SAMPLE_RATE)
        samples = samples[:remaining]
        if len(samples):
            audio_parts[name].append(samples)
            seconds[name] += len(samples) / config.SAMPLE_RATE

    for name in config.STANDIN_SPEAKERS:
        if seconds[name] + 1e-3 < total_seconds:
            raise SystemExit(f"Only collected {seconds[name]:.1f}s for {name}; dataset stream ended early.")
        output_dir = config.ENROLL_DIR / name / "standin_librispeech"
        output_dir.mkdir(parents=True, exist_ok=True)
        samples = np.concatenate(audio_parts[name])
        enrollment_samples = int(args.seconds * config.SAMPLE_RATE)
        enrollment_path = output_dir / "placeholder.wav"
        sf.write(enrollment_path, samples[:enrollment_samples], config.SAMPLE_RATE)
        test_dir = config.TEST_WAVS_DIR / name
        test_dir.mkdir(parents=True, exist_ok=True)
        test_path = test_dir / "librispeech_placeholder.wav"
        sf.write(test_path, samples[enrollment_samples:], config.SAMPLE_RATE)
        print(f"Saved {args.seconds}s enrollment to {enrollment_path}")
        print(f"Saved {config.STANDIN_TEST_SECONDS}s labeled test audio to {test_path}")
    print("These are plumbing-test placeholders only; remove each standin_librispeech folder before replacing with real enrollment.")


if __name__ == "__main__":
    main()