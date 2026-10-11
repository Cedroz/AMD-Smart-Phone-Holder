"""Live 16 kHz mono capture and a thread-safe rolling audio buffer."""

import threading
import time
import socket

import numpy as np

import config


class AudioRingBuffer:
    def __init__(self, seconds=config.RING_BUFFER_SECONDS):
        self.capacity = int(seconds * config.SAMPLE_RATE)
        self._data = np.zeros(self.capacity, dtype=np.float32)
        self._write_index = 0
        self._filled = 0
        self._lock = threading.Lock()
        self.updated = threading.Event()

    def append(self, samples):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        with self._lock:
            if len(samples) >= self.capacity:
                self._data[:] = samples[-self.capacity :]
                self._write_index = 0
                self._filled = self.capacity
            else:
                end = self._write_index + len(samples)
                first_count = min(len(samples), self.capacity - self._write_index)
                self._data[self._write_index : self._write_index + first_count] = samples[:first_count]
                remaining = len(samples) - first_count
                if remaining:
                    self._data[:remaining] = samples[first_count:]
                self._write_index = end % self.capacity
                self._filled = min(self.capacity, self._filled + len(samples))
        self.updated.set()

    def latest(self, sample_count):
        with self._lock:
            if self._filled < sample_count:
                return None
            start = (self._write_index - sample_count) % self.capacity
            end = start + sample_count
            if end <= self.capacity:
                return self._data[start:end].copy()
            return np.concatenate((self._data[start:], self._data[: end % self.capacity])).copy()


class AudioCapture:
    def __init__(self, device=config.MIC_DEVICE_INDEX):
        self.buffer = AudioRingBuffer()
        self.device = device
        self.stream = None
        self._record_lock = threading.Lock()
        self._recorded = []
        self._record_remaining = 0
        self._record_done = threading.Event()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"Microphone warning: {status}", flush=True)
        self.append(indata[:, 0])

    def append(self, samples):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.buffer.append(samples)
        with self._record_lock:
            if self._record_remaining:
                kept = samples[: self._record_remaining].copy()
                self._recorded.append(kept)
                self._record_remaining -= len(kept)
                if self._record_remaining == 0:
                    self._record_done.set()

    def record(self, seconds, timeout=config.MIC_CONNECT_TIMEOUT_SECONDS):
        sample_count = int(seconds * config.SAMPLE_RATE)
        with self._record_lock:
            self._recorded = []
            self._record_remaining = sample_count
            self._record_done.clear()
        if not self._record_done.wait(timeout):
            with self._record_lock:
                self._record_remaining = 0
                self._recorded = []
            raise TimeoutError("No microphone audio arrived in time; check the laptop mic sender and SSH tunnel.")
        with self._record_lock:
            return np.concatenate(self._recorded)

    def start(self):
        import sounddevice as sd

        self.stream = sd.InputStream(
            device=self.device,
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()
        return self

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def windows(self, window_seconds=config.WINDOW_SECONDS, hop_seconds=config.HOP_SECONDS):
        window_samples = int(window_seconds * config.SAMPLE_RATE)
        next_time = None
        while True:
            if next_time is not None:
                time.sleep(max(0.0, next_time - time.monotonic()))
            audio = self.buffer.latest(window_samples)
            if audio is None:
                self.buffer.updated.wait(hop_seconds)
                self.buffer.updated.clear()
                continue
            yield audio
            now = time.monotonic()
            next_time = now + hop_seconds if next_time is None or next_time <= now else next_time + hop_seconds

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


class TCPAudioCapture(AudioCapture):
    """Receive 16-bit mono PCM forwarded from the laptop through an SSH tunnel."""

    def __init__(self, host=config.MIC_TCP_HOST, port=config.MIC_TCP_PORT):
        super().__init__(device=None)
        self.host = host
        self.port = port
        self.server = None
        self.client = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        self.server.settimeout(1.0)
        self._thread = threading.Thread(target=self._receive, daemon=True)
        self._thread.start()
        print(f"Waiting for forwarded laptop microphone on {self.host}:{self.port}...")
        return self

    def _receive(self):
        while not self._stop_event.is_set():
            try:
                client, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.client = client
            client.settimeout(1.0)
            print("Laptop microphone connected.", flush=True)
            pending = b""
            try:
                while not self._stop_event.is_set():
                    try:
                        block = client.recv(8192)
                    except socket.timeout:
                        continue
                    if not block:
                        break
                    pending += block
                    usable_bytes = len(pending) - (len(pending) % 2)
                    if usable_bytes:
                        pcm = np.frombuffer(pending[:usable_bytes], dtype="<i2")
                        self.append(pcm.astype(np.float32) / 32768.0)
                        pending = pending[usable_bytes:]
            except OSError as error:
                if not self._stop_event.is_set():
                    print(f"Laptop microphone connection ended: {error}", flush=True)
            finally:
                client.close()
                self.client = None

    def stop(self):
        self._stop_event.set()
        for connection in (self.client, self.server):
            if connection is not None:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)