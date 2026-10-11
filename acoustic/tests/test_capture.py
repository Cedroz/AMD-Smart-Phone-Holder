import socket
import unittest

import numpy as np

from capture import AudioRingBuffer, TCPAudioCapture


class AudioRingBufferTests(unittest.TestCase):
    def test_latest_returns_ordered_recent_audio_after_wrap(self):
        buffer = AudioRingBuffer(seconds=5 / 16_000)
        buffer.append(np.array([1, 2, 3], dtype=np.float32))
        np.testing.assert_array_equal(buffer.latest(3), np.array([1, 2, 3], dtype=np.float32))
        buffer.append(np.array([4, 5, 6], dtype=np.float32))
        np.testing.assert_array_equal(buffer.latest(4), np.array([3, 4, 5, 6], dtype=np.float32))


class TCPAudioCaptureTests(unittest.TestCase):
    def test_receives_16_bit_pcm_into_audio_buffer(self):
        capture = TCPAudioCapture(port=0).start()
        client = socket.create_connection(("127.0.0.1", capture.server.getsockname()[1]))
        try:
            pcm = np.full(2400, 16384, dtype="<i2")
            client.sendall(pcm.tobytes())
            self.assertTrue(capture.buffer.updated.wait(timeout=2))
            audio = capture.buffer.latest(2400)
            self.assertIsNotNone(audio)
            np.testing.assert_allclose(audio, np.full(2400, 0.5, dtype=np.float32))
        finally:
            client.close()
            capture.stop()


if __name__ == "__main__":
    unittest.main()