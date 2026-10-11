import importlib.util
from pathlib import Path
import unittest

from followme.gestures import (FREEZE, LOWER, LANDSCAPE, PARTY, PORTRAIT, RESUME,
                               GestureInterpreter, GesturePipeline, GestureZone, HandObservation, LSignDetector,
                               PointDetector, RAISE, TwoPalmsDetector, hand_shape, l_shape, pointing)
from followme.phase5.replay import Clock

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'models/gestures/gesture_recognizer.task'
MANIFEST = ROOT/'models/gestures/gesture_manifest.json'
FIXTURES = ROOT/'test/fixtures/gestures'
FACE = (.5, .4)


def palm(x=.5, y=.4, score=.9): return HandObservation('Open_Palm', score, x, y)
def thumb(x=.5, y=.4, score=.9): return HandObservation('Thumb_Up', score, x, y)


class InterpreterTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.interpreter = GestureInterpreter()

    def feed(self, hands, seconds, face=FACE):
        """Feeds the same hands at 20 Hz for `seconds`; returns every command emitted."""
        commands = []
        for _ in range(round(seconds / .05)):
            self.clock.advance()
            command = self.interpreter.update(hands, self.clock(), face)
            if command:
                commands.append(command)
        return commands

    def test_held_palm_freezes_once(self):
        self.assertEqual(self.feed([palm()], .6), [])
        self.assertEqual(self.feed([palm()], .3), [FREEZE])
        self.assertEqual(self.feed([palm()], 5), [])

    def test_thumb_resumes_after_cooldown(self):
        self.assertEqual(self.feed([palm()], 1), [FREEZE])
        self.assertEqual(self.feed([thumb()], 2), [RESUME])

    def test_brief_gesture_is_ignored(self):
        self.assertEqual(self.feed([palm()], .5), [])
        self.assertEqual(self.feed([], .5), [])
        self.assertEqual(self.feed([palm()], .5), [])

    def test_single_dropped_frame_keeps_hold(self):
        self.feed([palm()], .4)
        self.feed([], .1)
        self.assertEqual(self.feed([palm()], .4), [FREEZE])

    def test_switching_gesture_restarts_hold(self):
        self.feed([palm()], .5)
        self.assertEqual(self.feed([thumb()], .5), [])
        self.assertEqual(self.feed([thumb()], .3), [RESUME])

    def test_cooldown_blocks_and_does_not_bank_time(self):
        self.feed([palm()], 1)
        self.assertEqual(self.feed([thumb()], .5), [])  # inside the 1 s cooldown
        self.assertEqual(self.feed([thumb()], .6), [])  # hold only counts from cooldown end
        self.assertEqual(self.feed([thumb()], .5), [RESUME])

    def test_same_gesture_needs_release_to_fire_again(self):
        self.feed([palm()], 1)
        self.assertEqual(self.feed([palm()], 3), [])
        self.feed([], .5)
        self.assertEqual(self.feed([palm()], 1), [FREEZE])

    def test_hand_outside_zone_is_ignored(self):
        self.assertEqual(self.feed([palm(y=.9)], 2), [])  # down at the counter
        self.assertEqual(self.feed([palm(x=.95)], 2), [])  # far to the side
        self.assertEqual(self.feed([palm(y=.6)], 1), [FREEZE])  # shoulder height

    def test_zone_without_face_uses_frame_height(self):
        self.assertEqual(self.feed([palm(y=.7)], 2, face=None), [])
        self.assertEqual(self.feed([palm(y=.5)], 1, face=None), [FREEZE])

    def test_low_score_and_unbound_gestures_are_ignored(self):
        self.assertEqual(self.feed([palm(score=.4)], 2), [])
        self.assertEqual(self.feed([HandObservation('Victory', .95, .5, .4)], 2), [])

    def test_best_scoring_hand_in_zone_wins(self):
        hands = [thumb(score=.99, y=.95), palm(score=.7)]
        self.assertEqual(self.feed(hands, 1), [FREEZE])

    def test_pending_reports_progress(self):
        self.assertIsNone(self.interpreter.pending(self.clock()))
        self.feed([palm()], .4)
        command, fraction = self.interpreter.pending(self.clock())
        self.assertEqual(command, FREEZE)
        self.assertAlmostEqual(fraction, .35 / .7, places=2)

    def test_custom_bindings_and_zone(self):
        self.interpreter = GestureInterpreter({'Victory': 'flip'}, hold_s=.2, zone=GestureZone(below_face=.6))
        self.assertEqual(self.feed([HandObservation('Victory', .9, .5, .95)], .5), ['flip'])



def flat_hand(fingers_up=True, curled=False):
    """21 synthetic landmarks: wrist at the bottom, fingers pointing up (or down)."""
    sign = -1 if fingers_up else 1
    points = [(.5, .5)] + [(.5, .5)] * 20
    for finger, x in enumerate((.44, .48, .52, .56)):
        base = 5 + finger * 4
        points[base] = (x, .5 + sign * .1)
        for joint in range(1, 4):
            reach = .1 + (.02 * joint if curled else .035 * joint)
            points[base + joint] = (x, .5 + sign * (reach if not curled or joint < 3 else .06))
    return points


class HandShapeTests(unittest.TestCase):
    def test_open_and_curled(self):
        open_hand, fingers = hand_shape(flat_hand())
        curled, _ = hand_shape(flat_hand(curled=True))
        self.assertEqual(fingers, 'up')
        self.assertGreater(open_hand, 1.6)
        self.assertLess(curled, 1.3)

    def test_fingers_down(self):
        self.assertEqual(hand_shape(flat_hand(fingers_up=False))[1], 'down')

    def test_degenerate_hand(self):
        self.assertEqual(hand_shape([(.5, .5)] * 21), (None, None))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()

    def test_pipeline_still_freezes_and_resumes(self):
        pipeline = GesturePipeline()
        frames = [[palm()]] * 20 + [[]] * 10 + [[thumb()]] * 30
        commands = [c for c in (pipeline.update(h, self._tick(), FACE) for h in frames) if c]
        self.assertEqual(commands, [FREEZE, RESUME])

    def _tick(self):
        self.clock.advance()
        return self.clock()


class RecognizerLandmarkTests(unittest.TestCase):
    def test_unclassified_hands_keep_their_motion_landmarks(self):
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import Mock
        from followme.gestures.recognizer import MediaPipeGestureRecognizer
        recognizer = MediaPipeGestureRecognizer.__new__(MediaPipeGestureRecognizer)
        recognizer._mp = SimpleNamespace(Image=Mock(), ImageFormat=SimpleNamespace(SRGB=0))
        recognizer._recognizer = Mock()
        landmarks = [SimpleNamespace(x=x, y=y) for x, y in flat_hand()]
        for categories in ([[]], []):
            with self.subTest(categories=categories):
                recognizer._recognizer.recognize_for_video.return_value = SimpleNamespace(
                    gestures=categories, hand_landmarks=[landmarks])
                hands = recognizer.recognize(np.zeros((480, 640, 3), np.uint8), 0)
                self.assertEqual(len(hands), 1)
                self.assertEqual(hands[0].label, 'None')
                self.assertEqual(hands[0].fingers, 'up')
                self.assertGreater(hands[0].openness, 1.6)


def l_hand(thumb_dx=-.04, curl_others=True, index_up=True):
    """21 synthetic landmarks: index finger up, thumb out sideways by `thumb_dx` per joint, middle,
    ring and little fingers curled (or straight with curl_others=False)."""
    points = flat_hand()
    for joint in range(1, 5):  # thumb, from the base of the palm outwards
        points[joint] = (.46 + thumb_dx * joint, .48 - .01 * joint)
    if not index_up:
        points[6], points[7], points[8] = (.44, .42), (.45, .45), (.46, .47)
    if curl_others:
        for base, x in ((9, .48), (13, .52), (17, .56)):
            points[base + 1], points[base + 2], points[base + 3] = (x, .37), (x, .41), (x, .43)
    return points


def sign(hand, l_shape=True, x=.5, y=.4):
    return HandObservation('None', .5, x, y, 1.2, 'up', hand, l_shape)


class LShapeTests(unittest.TestCase):
    def test_l_shape(self):
        self.assertTrue(l_shape(l_hand()))
        self.assertTrue(l_shape(l_hand(thumb_dx=.04)))  # thumb out the other way: the other hand

    def test_open_hand_is_not_an_l(self):
        self.assertFalse(l_shape(flat_hand()))
        self.assertFalse(l_shape(l_hand(curl_others=False)))

    def test_tucked_thumb_is_not_an_l(self):
        self.assertFalse(l_shape(l_hand(thumb_dx=-.005)))

    def test_curled_index_is_not_an_l(self):
        self.assertFalse(l_shape(l_hand(index_up=False)))

    def test_degenerate_hand(self):
        self.assertFalse(l_shape([(.5, .5)] * 21))


class LSignTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.detector = LSignDetector()

    def feed(self, hands, seconds):
        commands = []
        for _ in range(round(seconds / .05)):
            self.clock.advance()
            command = self.detector.update(hands, self.clock(), FACE)
            if command:
                commands.append(command)
        return commands

    def test_left_l_rotates_to_landscape_once(self):
        self.assertEqual(self.feed([sign('Left')], .4), [])
        self.assertEqual(self.feed([sign('Left')], 2), [LANDSCAPE])

    def test_right_l_rotates_back_to_portrait(self):
        self.assertEqual(self.feed([sign('Left')], 1) + self.feed([], 1.5) + self.feed([sign('Right')], 1),
                         [LANDSCAPE, PORTRAIT])

    def test_brief_l_is_ignored(self):
        self.assertEqual(self.feed([sign('Left')], .3) + self.feed([], .5), [])

    def test_not_an_l_or_unknown_hand_is_ignored(self):
        self.assertEqual(self.feed([sign('Left', l_shape=False)], 2), [])
        self.assertEqual(self.feed([sign(None)], 2), [])

    def test_hand_outside_zone_is_ignored(self):
        self.assertEqual(self.feed([sign('Left', y=.9)], 2), [])

    def test_pipeline(self):
        pipeline, commands = GesturePipeline(), []
        for hands in [[sign('Left')]] * 20 + [[]] * 30 + [[sign('Right')]] * 20:
            self.clock.advance()
            command = pipeline.update(hands, self.clock(), FACE)
            if command:
                commands.append(command)
        self.assertEqual(commands, [LANDSCAPE, PORTRAIT])

    def test_recognizer_reports_the_persons_own_hand(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import numpy as np
        from followme.gestures.recognizer import MediaPipeGestureRecognizer
        recognizer = MediaPipeGestureRecognizer.__new__(MediaPipeGestureRecognizer)
        recognizer._mp = SimpleNamespace(Image=Mock(), ImageFormat=SimpleNamespace(SRGB=0))
        recognizer._recognizer = Mock()
        landmarks = [SimpleNamespace(x=x, y=y) for x, y in l_hand()]
        # MediaPipe assumes a mirrored selfie; the arm's camera isn't mirrored, so 'Right' is the left hand.
        recognizer._recognizer.recognize_for_video.return_value = SimpleNamespace(
            gestures=[[]], hand_landmarks=[landmarks], handedness=[[SimpleNamespace(category_name='Right')]])
        hand = recognizer.recognize(np.zeros((480, 640, 3), np.uint8), 0)[0]
        self.assertEqual((hand.hand, hand.l_shape), ('Left', True))


def point_hand(up=True):
    """Index finger out (up, or flipped to point down), thumb tucked, other fingers curled."""
    points = l_hand(thumb_dx=-.005)
    return points if up else [(x, 1 - y) for x, y in points]


def pointer(direction, x=.5, y=.4):
    return HandObservation('Pointing_Up' if direction == 'up' else 'None', .8, x, y, 1.2, direction,
                           'Left', False, direction)


class PointingTests(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(pointing(point_hand()), 'up')
        self.assertEqual(pointing(point_hand(up=False)), 'down')

    def test_l_and_open_hand_are_not_pointing(self):
        self.assertIsNone(pointing(l_hand()))
        self.assertIsNone(pointing(flat_hand()))

    def test_pointing_sideways_is_ignored(self):
        sideways = [(.5 + (y - .5), .5 + (x - .5)) for x, y in point_hand()]  # rotated 90 degrees
        self.assertIsNone(pointing(sideways))


class PointDetectorTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.detector = PointDetector()

    def feed(self, hands, seconds):
        commands = []
        for _ in range(round(seconds / .05)):
            self.clock.advance()
            command = self.detector.update(hands, self.clock(), FACE)
            if command:
                commands.append(command)
        return commands

    def test_point_up_raises_and_repeats_while_held(self):
        self.assertEqual(self.feed([pointer('up')], .4), [])
        self.assertEqual(self.feed([pointer('up')], 1.6), [RAISE] * 15)  # ten steps a second after the hold
        self.assertEqual(self.feed([], 1), [])

    def test_point_down_lowers(self):
        self.assertEqual(self.feed([pointer('down')], 1), [LOWER] * 5)

    def test_brief_point_is_ignored(self):
        self.assertEqual(self.feed([pointer('up')], .3) + self.feed([], .5), [])

    def test_switching_direction_restarts_hold(self):
        self.assertEqual(self.feed([pointer('up')], .7) + self.feed([pointer('down')], .3), [RAISE] * 2)

    def test_hand_outside_zone_is_ignored(self):
        self.assertEqual(self.feed([pointer('up', y=.9)], 2), [])

    def test_pipeline_point_does_not_freeze(self):
        pipeline, commands = GesturePipeline(), []
        for hands in [[pointer('up')]] * 20:
            self.clock.advance()
            command = pipeline.update(hands, self.clock(), FACE)
            if command:
                commands.append(command)
        self.assertEqual(set(commands), {RAISE})


def two_palms(spread=.25):
    return [palm(x=.5 - spread), palm(x=.5 + spread)]


class TwoPalmsTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()

    def feed(self, detector, hands, seconds):
        commands = []
        for _ in range(round(seconds / .05)):
            self.clock.advance()
            command = detector.update(hands, self.clock(), FACE)
            if command:
                commands.append(command)
        return commands

    def test_two_palms_for_two_and_a_half_seconds_party_once(self):
        detector = TwoPalmsDetector()
        self.assertEqual(self.feed(detector, two_palms(), 2.3), [])
        self.assertEqual(self.feed(detector, two_palms(), 3), [PARTY])

    def test_lowering_hands_too_soon_does_nothing(self):
        detector = TwoPalmsDetector()
        self.assertEqual(self.feed(detector, two_palms(), 2) + self.feed(detector, [], 1)
                         + self.feed(detector, two_palms(), 2), [])

    def test_open_hands_count_even_if_mediapipe_is_unsure(self):
        unsure = [HandObservation('None', .3, x, .4, 2., 'up') for x in (.3, .7)]
        self.assertEqual(self.feed(TwoPalmsDetector(), unsure, 3), [PARTY])

    def test_fists_do_not_count(self):
        fists = [HandObservation('Closed_Fist', .9, x, .4, 1., 'up') for x in (.3, .7)]
        self.assertEqual(self.feed(TwoPalmsDetector(), fists, 3), [])

    def test_progress_for_the_countdown(self):
        detector = TwoPalmsDetector()
        self.feed(detector, two_palms(), 1.25)
        self.assertAlmostEqual(detector.progress(self.clock()), .5, delta=.05)
        self.feed(detector, two_palms(), 2)
        self.assertIsNone(detector.progress(self.clock()))  # fired

    def test_one_palm_is_not_party(self):
        self.assertEqual(self.feed(TwoPalmsDetector(), [palm()], 4), [])

    def test_hands_wider_apart_than_one_hand_zone_still_count(self):
        self.assertEqual(self.feed(TwoPalmsDetector(), two_palms(spread=.5), 3), [PARTY])

    def test_raise_again_after_lowering(self):
        detector = TwoPalmsDetector()
        self.assertEqual(self.feed(detector, two_palms(), 3) + self.feed(detector, [], 4)
                         + self.feed(detector, two_palms(), 3), [PARTY, PARTY])

    def test_pipeline_two_palms_party_without_freezing(self):
        pipeline, commands = GesturePipeline(), []
        for hands in [two_palms()] * 60:
            self.clock.advance()
            command = pipeline.update(hands, self.clock(), FACE)
            if command:
                commands.append(command)
        self.assertEqual(commands, [PARTY])

    def test_pipeline_single_palm_still_freezes(self):
        pipeline, commands = GesturePipeline(), []
        for hands in [[palm()]] * 30:
            self.clock.advance()
            command = pipeline.update(hands, self.clock(), FACE)
            if command:
                commands.append(command)
        self.assertEqual(commands, [FREEZE])


@unittest.skipUnless(importlib.util.find_spec('mediapipe') and MODEL.is_file(), 'needs the vision environment and model')
class RecognizerTests(unittest.TestCase):
    def setUp(self):
        from followme.gestures.recognizer import MediaPipeGestureRecognizer
        self.recognizer = MediaPipeGestureRecognizer(MODEL, MANIFEST)
        self.addCleanup(self.recognizer.close)

    def frames(self, name):
        import cv2
        return cv2.imread(str(FIXTURES/name))

    def test_blank_frame_has_no_hands(self):
        import numpy as np
        self.assertEqual(self.recognizer.recognize(np.zeros((480, 640, 3), np.uint8), 0), [])

    def test_photo_labels(self):
        photos = [('open_palm.jpg', 'Open_Palm'), ('thumbs_up.jpg', 'Thumb_Up'), ('victory.jpg', 'Victory')]
        for index, (name, label) in enumerate(photos):
            # Video mode tracks hands between frames, so the first frame after a scene cut can come back empty.
            for repeat in range(3):
                hands = self.recognizer.recognize(self.frames(name), (index * 3 + repeat) * 50)
            hands = [hand for hand in hands if hand.label != 'None']
            self.assertEqual([hand.label for hand in hands], [label], name)
            self.assertTrue(0 <= hands[0].x <= 1 and 0 <= hands[0].y <= 1)
            if label != 'Thumb_Up':  # a thumbs-up hand is sideways
                self.assertEqual(hands[0].fingers, 'up', name)

    def test_held_photos_freeze_then_resume(self):
        interpreter, commands = GestureInterpreter(), []
        for index in range(80):
            frame = self.frames('open_palm.jpg' if index < 30 else 'thumbs_up.jpg')
            hands = self.recognizer.recognize(frame, index * 50)
            face = next(((hand.x, hand.y) for hand in hands if hand.label != 'None'), None)  # stand-in face position
            commands.append(interpreter.update(hands, index * .05, face))
        self.assertEqual([command for command in commands if command], [FREEZE, RESUME])


if __name__ == '__main__':
    unittest.main()
