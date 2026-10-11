import unittest

import numpy as np

from decide import SpeakerSmoother, decide


class DecideTests(unittest.TestCase):
    def setUp(self):
        self.voiceprints = {
            "Arnav": np.array([1.0, 0.0]),
            "Edwin": np.array([0.0, 1.0]),
        }

    def test_returns_best_match(self):
        speaker, confidence = decide(self.voiceprints, np.array([0.9, 0.1]), threshold=0.5)
        self.assertEqual(speaker, "Arnav")
        self.assertGreater(confidence, 0.9)

    def test_low_score_is_unknown(self):
        speaker, _ = decide(self.voiceprints, np.array([1.0, 0.0]), threshold=1.01)
        self.assertEqual(speaker, "unknown")

    def test_switch_requires_consecutive_wins(self):
        smoother = SpeakerSmoother(count=3)
        for _ in range(3):
            current, _ = smoother.update("Arnav", 0.9)
        self.assertEqual(current, "Arnav")
        self.assertEqual(smoother.update("Edwin", 0.95)[0], "Arnav")
        self.assertEqual(smoother.update("Arnav", 0.92)[0], "Arnav")
        self.assertEqual(smoother.update("Edwin", 0.95)[0], "Arnav")
        self.assertEqual(smoother.update("Edwin", 0.95)[0], "Arnav")
        self.assertEqual(smoother.update("Edwin", 0.95)[0], "Edwin")

    def test_unknown_requires_consecutive_wins_to_switch(self):
        smoother = SpeakerSmoother(count=2)
        smoother.update("Arnav", 0.9)
        smoother.update("Arnav", 0.9)
        self.assertEqual(smoother.update("unknown", 0.3)[0], "Arnav")
        self.assertEqual(smoother.update("unknown", 0.3)[0], "unknown")


if __name__ == "__main__":
    unittest.main()