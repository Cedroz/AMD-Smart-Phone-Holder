import unittest

from vad import SpeechHysteresis


class SpeechHysteresisTests(unittest.TestCase):
    def test_requires_two_quiet_hops_to_leave_speech(self):
        smoother = SpeechHysteresis(silence_hops=2)
        self.assertTrue(smoother.update(True))
        self.assertTrue(smoother.update(False))
        self.assertFalse(smoother.update(False))

    def test_speech_resets_quiet_hop_count(self):
        smoother = SpeechHysteresis(silence_hops=2)
        smoother.update(True)
        smoother.update(False)
        self.assertTrue(smoother.update(True))
        self.assertTrue(smoother.update(False))
        self.assertFalse(smoother.update(False))

    def test_starts_in_silence(self):
        smoother = SpeechHysteresis(silence_hops=2)
        self.assertFalse(smoother.update(False))


if __name__ == "__main__":
    unittest.main()