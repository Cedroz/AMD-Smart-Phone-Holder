import importlib.util
from pathlib import Path
import unittest

from followme.gestures import (CLOSER, FARTHER, FREEZE, LANDSCAPE, PORTRAIT, RESUME, BeckonDetector,
                               GestureInterpreter, GesturePipeline, GestureZone, HandObservation, TwistDetector,
                               hand_pose, hand_shape)
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



def curl(openness, fingers='up', x=.5, y=.4):
    label = 'Open_Palm' if openness > 1.6 else 'Closed_Fist' if openness < 1. else 'None'
    return HandObservation(label, .9, x, y, openness, fingers)


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


class BeckonTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.detector = BeckonDetector()

    def run_frames(self, frames, detector=None):
        detector, commands = detector or self.detector, []
        for hands in frames:
            self.clock.advance()
            command = detector.update(hands, self.clock(), FACE)
            if command:
                commands.append(command)
        return commands

    @staticmethod
    def beckon(cycles, fingers='up', open_frames=4, closed_frames=4, **hand):
        """One cycle = open for open_frames, curled for closed_frames, at 20 Hz."""
        frames = []
        for _ in range(cycles):
            frames += [[curl(2., fingers, **hand)]] * open_frames + [[curl(1., fingers, **hand)]] * closed_frames
        return frames + [[curl(2., fingers, **hand)]] * open_frames

    def test_beckon_steps_closer_after_second_cycle(self):
        self.assertEqual(self.run_frames(self.beckon(1)), [])
        self.setUp()
        self.assertEqual(self.run_frames(self.beckon(4)), [CLOSER, CLOSER, CLOSER])

    def test_fingers_down_flick_steps_farther(self):
        self.assertEqual(self.run_frames(self.beckon(3, fingers='down')), [FARTHER, FARTHER])

    def test_static_open_hand_does_nothing(self):
        self.assertEqual(self.run_frames([[curl(2.)]] * 60), [])

    def test_slow_curls_are_ignored(self):
        self.assertEqual(self.run_frames(self.beckon(4, closed_frames=20)), [])

    def test_long_pause_restarts_count(self):
        frames = self.beckon(1) + [[curl(2.)]] * 40 + self.beckon(1)
        self.assertEqual(self.run_frames(frames), [])

    def test_hand_outside_zone_is_ignored(self):
        self.assertEqual(self.run_frames(self.beckon(4, y=.9)), [])

    def test_sideways_hand_is_ignored(self):
        self.assertEqual(self.run_frames(self.beckon(4, fingers=None)), [])

    def test_switching_direction_restarts_count(self):
        frames = self.beckon(1) + self.beckon(1, fingers='down')
        self.assertEqual(self.run_frames(frames), [])

    def test_lost_hand_resets(self):
        frames = self.beckon(1) + [[]] * 10 + self.beckon(1)
        self.assertEqual(self.run_frames(frames), [])

    def test_pipeline_beckon_does_not_freeze(self):
        pipeline = GesturePipeline()
        commands = [c for c in (pipeline.update(h, self._tick(), FACE) for h in self.beckon(8)) if c]
        self.assertEqual(set(commands), {CLOSER})

    def test_pipeline_still_freezes_and_resumes(self):
        pipeline = GesturePipeline()
        frames = [[palm()]] * 20 + [[]] * 10 + [[thumb()]] * 30
        commands = [c for c in (pipeline.update(h, self._tick(), FACE) for h in frames) if c]
        self.assertEqual(commands, [FREEZE, RESUME])

    def _tick(self):
        self.clock.advance()
        return self.clock()


def blade(angle, spread=.2, openness=2., label='Open_Palm', x=.5, y=.4):
    """A flat hand seen edge-on (spread small) pointing at `angle` degrees (0 up, 90 right)."""
    return HandObservation(label, .9, x, y, openness, None, spread, angle)


def rotate_points(points, degrees, pivot=(.5, .5)):
    import math
    a = math.radians(degrees)
    return [(pivot[0] + (x - pivot[0]) * math.cos(a) - (y - pivot[1]) * math.sin(a),
             pivot[1] + (x - pivot[0]) * math.sin(a) + (y - pivot[1]) * math.cos(a)) for x, y in points]


class HandPoseTests(unittest.TestCase):
    def test_palm_facing_camera_is_wide_and_upright(self):
        spread, angle = hand_pose(flat_hand())
        self.assertGreater(spread, .5)
        self.assertAlmostEqual(angle, 0, delta=1)

    def test_edge_on_hand_is_narrow(self):
        edge_on = [(.5 + (x - .5) * .2, y) for x, y in flat_hand()]  # knuckles nearly overlap
        self.assertLess(hand_pose(edge_on)[0], .5)

    def test_angle_follows_the_fingers(self):
        self.assertAlmostEqual(hand_pose(rotate_points(flat_hand(), 90))[1], 90, delta=1)
        self.assertAlmostEqual(hand_pose(rotate_points(flat_hand(), -90))[1], -90, delta=1)
        self.assertAlmostEqual(abs(hand_pose(flat_hand(fingers_up=False))[1]), 180, delta=1)

    def test_degenerate_hand(self):
        self.assertEqual(hand_pose([(.5, .5)] * 21), (None, None))


class TwistTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.detector = TwistDetector()

    def run_frames(self, frames, detector=None):
        detector, commands = detector or self.detector, []
        for hands in frames:
            self.clock.advance()
            command = detector.update(hands, self.clock(), FACE)
            if command:
                commands.append(command)
        return commands

    @staticmethod
    def turn(start, end, hold_frames=10, turn_frames=6, **hand):
        """Hold at `start` degrees, sweep to `end`, hold there (20 Hz)."""
        sweep = [start + (end - start) * i / turn_frames for i in range(1, turn_frames)]
        return ([[blade(start, **hand)]] * hold_frames + [[blade(a, **hand)] for a in sweep]
                + [[blade(end, **hand)]] * hold_frames)

    def test_upright_to_sideways_rotates_to_landscape(self):
        self.assertEqual(self.run_frames(self.turn(0, 90)), [LANDSCAPE])

    def test_turning_back_rotates_to_portrait(self):
        self.assertEqual(self.run_frames(self.turn(0, -90) + self.turn(-90, 0)), [LANDSCAPE, PORTRAIT])

    def test_holding_one_pose_does_nothing(self):
        self.assertEqual(self.run_frames([[blade(0)]] * 80), [])
        self.assertEqual(self.run_frames([[blade(90)]] * 80, TwistDetector()), [])

    def test_palm_facing_camera_does_not_count(self):
        self.assertEqual(self.run_frames(self.turn(0, 90, spread=.85)), [])

    def test_fist_does_not_count(self):
        self.assertEqual(self.run_frames(self.turn(0, 90, openness=1.)), [])

    def test_slow_steady_turn_still_counts(self):
        self.assertEqual(self.run_frames(self.turn(0, 90, turn_frames=40)), [LANDSCAPE])

    def test_pausing_half_way_is_ignored(self):
        frames = [[blade(0)]] * 10 + [[blade(45)]] * 40 + [[blade(90)]] * 10
        self.assertEqual(self.run_frames(frames), [])

    def test_brief_flicker_is_ignored(self):
        frames = [[blade(0)]] * 10 + [[blade(90)]] * 3 + [[blade(0)]] * 10
        self.assertEqual(self.run_frames(frames), [])

    def test_hand_outside_zone_is_ignored(self):
        self.assertEqual(self.run_frames(self.turn(0, 90, y=.9)), [])

    def test_lost_hand_resets(self):
        frames = [[blade(0)]] * 10 + [[]] * 20 + [[blade(90)]] * 10
        self.assertEqual(self.run_frames(frames), [])

    def test_pipeline_twist_does_not_freeze(self):
        pipeline, commands = GesturePipeline(), []
        for hands in self.turn(0, 90, hold_frames=30) + self.turn(90, 0, hold_frames=30):
            self.clock.advance()
            command = pipeline.update(hands, self.clock(), FACE)
            if command:
                commands.append(command)
        self.assertEqual(commands, [LANDSCAPE, PORTRAIT])


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
