"""Shared tunables for the CPU-only speaker identification project."""

from pathlib import Path

SPEAKERS = ("Arnav", "Edwin", "Jaisree", "Matthew")
SAMPLE_RATE = 16_000
CHANNELS = 1
WINDOW_SECONDS = 1.5
HOP_SECONDS = 0.5
UNKNOWN_THRESHOLD = 0.45  # lower = names people more readily; the placeholder voices keep strangers out
SMOOTHING_COUNT = 2  # windows in a row before switching speaker (2 = about 1 s)
MIC_DEVICE_INDEX = None
MIC_INPUT_MODE = "tcp"
MIC_TCP_HOST = "127.0.0.1"
MIC_TCP_PORT = 8765
MIC_TCP_FORWARD_PORT = 18765
TCP_AUDIO_BLOCK_FRAMES = 480
MIC_CONNECT_TIMEOUT_SECONDS = 30
RING_BUFFER_SECONDS = 5.0

VAD_MODE = 1
VAD_FRAME_MS = 30
MIN_SPEECH_RATIO = 0.3
VAD_SILENCE_HOPS = 2

TEST_RECORD_SECONDS = 5
ENROLL_SECONDS = 75
ENROLL_PHASE_SECONDS = 25
ENROLL_PROMPTS = (
    "Read at a normal volume, about an arm's length from the microphone.",
    "Speak more quietly, then more loudly; vary your distance from the microphone.",
    "Talk freely about any topic, changing your pace and emphasis.",
)

EMBED_BENCHMARK_REPEATS = 10
EMBED_THREADS = 2  # CPU threads for the speaker encoder (PyTorch would otherwise use every core)
LIBRISPEECH_CONFIG = "clean"
LIBRISPEECH_SPLIT = "validation"
LIBRISPEECH_PLACEHOLDER_IDS = ("1272", "1284", "1320")
STANDIN_SPEAKERS = SPEAKERS[1:]
STANDIN_TEST_SECONDS = 15

TURN_SECONDS = 10.0
NOISE_SNR_LEVELS_DB = (-5, 0, 5, 10, 20)

PROJECT_DIR = Path(__file__).resolve().parent
ENROLL_DIR = PROJECT_DIR / "enroll"
VOICEPRINTS_PATH = PROJECT_DIR / "voiceprints.npz"
TEST_WAVS_DIR = PROJECT_DIR / "test_wavs"