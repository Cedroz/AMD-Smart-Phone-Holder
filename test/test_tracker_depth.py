"""Depth following checks using simulated perception, IK, and motors only."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from tracker import track


class DepthTests(unittest.TestCase):
    def args(self, *extra):
        argv = ['track.py', '--distance', '1', '--distance-reference', '1',
                '--distance-face-width', '100', '--smoothing', '0', '--squeeze', '0', *extra]
        with patch('sys.argv', argv), patch.object(track, 'run') as run:
            track.main()
        return run.call_args.args[0]

    def test_direction_tolerance_rate_and_travel_bounds(self):
        depth = track.DepthFollower(self.args())
        depth.update((100, 100, 50, 50, 'face'), 640)  # 2 m: extend
        self.assertEqual(depth.distance, 2)
        self.assertAlmostEqual(depth.propose(.3, .3, .05), .301)
        self.assertAlmostEqual(depth.propose(.3, .3, 10), .302)  # stall capped
        self.assertAlmostEqual(depth.propose(.35, .3, .05), .35)
        depth.update((100, 100, 200, 200, 'face'), 640)  # 0.5 m: retract
        self.assertAlmostEqual(depth.propose(.3, .3, .05), .299)
        self.assertAlmostEqual(depth.propose(.25, .3, .05), .25)
        depth.update((100, 100, 100, 100, 'face:Alice 0.90'), 640)
        self.assertEqual(depth.distance, 1)
        self.assertEqual(depth.propose(.3, .3, .05), .3)

    def test_unavailable_depth_holds_and_clears_filter(self):
        depth = track.DepthFollower(self.args('--smoothing', '0.5'))
        for target in (None, (100, 100, 50, 50, 'person'),
                       (100, 100, 50, 50, 'profile'), (0, 100, 50, 50, 'face'),
                       (600, 100, 50, 50, 'face'), (100, 100, 0, 50, 'face')):
            depth.update((100, 100, 50, 50, 'face'), 640)
            depth.update(target, 640)
            self.assertIsNone(depth.distance)
            self.assertEqual(depth.propose(.3, .3, .05), .3)
        depth.update((100, 100, 100, 100, 'face'), 640)
        self.assertEqual(depth.distance, 1)
        depth.update((100, 100, 50, 50, 'face'), 640)
        self.assertEqual(depth.distance, 1.5)

    def test_ik_changes_forward_axis_preserving_height_and_home(self):
        hik = track.HeightIK.__new__(track.HeightIK)
        hik.home = np.eye(4)
        hik.home[:3, 3] = [.3, 0, .2]
        hik.q = np.zeros(5)
        hik.kin = Mock()
        hik.kin.inverse_kinematics.return_value = hik.q.copy()
        hik.kin.forward_kinematics.return_value = hik.home.copy()
        hik.solve(.25, .35)
        pose = hik.kin.inverse_kinematics.call_args.args[1]
        np.testing.assert_allclose(pose[:3, 3], [.35, 0, .25])
        np.testing.assert_allclose(hik.home[:3, 3], [.3, 0, .2])
        hik.solve(.25)
        pose = hik.kin.inverse_kinematics.call_args.args[1]
        self.assertEqual(pose[0, 3], .3)  # existing height-only callers

    def test_loop_keeps_pan_and_only_commits_valid_ik(self):
        for error, jump, accepted, height in ((0, 1, True, False), (.02, 1, False, False),
                                             (0, 10, False, False), (0, 1, True, True)):
            with self.subTest(error=error, jump=jump, height=height):
                args = self.args('--tilt', '--preview-port', '0', '--port', 'fake')
                args.height = height
                robot = Mock()
                hik = SimpleNamespace(z0=.2, reach0=.3, q=np.zeros(5))
                solutions = []
                def solve(z, reach):
                    solutions.append((z, reach))
                    q = hik.q.copy()
                    q[1] += jump
                    return q, error, 0
                hik.solve = solve
                callbacks = []
                camera = Mock()
                frames = iter([np.zeros((480, 640, 3), dtype=np.uint8)] * 2)
                def read(seq):
                    frame = next(frames, None)
                    if frame is None:
                        callbacks[0]()
                    return seq + 1, frame
                camera.read.side_effect = read
                detector = Mock()
                detector.detect.return_value = [(400, 300, 50, 50, 'face')]
                with patch.object(track, 'Camera', return_value=camera), \
                     patch.object(track, 'connect_arm', return_value=robot), \
                     patch.object(track, 'read_joints', return_value={j: 0 for j in [*track.ARM_JOINTS, track.GRIP]}), \
                     patch.object(track, 'HeightIK', return_value=hik), \
                     patch.object(track, 'FaceAndBodyDetector', return_value=detector), \
                     patch.object(track.signal, 'signal', side_effect=lambda _, cb: callbacks.append(cb)), \
                     patch.object(track.time, 'monotonic', side_effect=iter(np.arange(20) * .05)):
                    track.run(args)
                self.assertEqual(len(solutions), 2)
                self.assertAlmostEqual(solutions[0][1], .301)
                self.assertAlmostEqual(solutions[1][1], .302 if accepted else .301)
                if height:
                    self.assertEqual(solutions[0][0], .2)  # wrist threshold not reached yet
                    self.assertLess(solutions[1][0], solutions[0][0])
                else:
                    self.assertTrue(all(z == .2 for z, _ in solutions))
                actions = [call.args[0] for call in robot.send_action.call_args_list]
                self.assertTrue(all('shoulder_pan.pos' in action for action in actions))
                self.assertTrue(all(action['wrist_flex.pos'] > 0 for action in actions))
                self.assertEqual(actions[-1]['shoulder_lift.pos'], 2 if accepted else 0)
                robot.disconnect.assert_called_once()

    def test_invalid_configuration_rejected_before_run(self):
        for extra in (['--distance-reference', 'nan'], ['--distance-face-width', '0'],
                      ['--depth-rate', '-1'], ['--depth-forward', '-1'],
                      ['--smoothing', '1'], ['--detector', 'yolo']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    self.args(*extra)
                self.assertEqual(raised.exception.code, 2)
        with patch('sys.argv', ['track.py', '--distance', '1']), \
             patch.object(track, 'run') as run, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                track.main()
            run.assert_not_called()


class CommandTests(unittest.TestCase):
    def args(self, *extra):
        with patch('sys.argv', ['track.py', '--preview-port', '0', '--squeeze', '0', *extra]), patch.object(track, 'run') as run:
            track.main()
        return run.call_args.args[0]

    def test_depth_commands_bounds_and_freeze_cancel(self):
        control = track.TrackerControl(self.args())
        control.reach0 = control.reach = control.requested_reach = .3
        for _ in range(10):
            control.apply('closer')
        self.assertAlmostEqual(control.requested_reach, .35)
        self.assertAlmostEqual(control.propose_reach(.05), .301)
        control.apply('freeze')
        self.assertEqual(control.requested_reach, .3)
        self.assertTrue(control.apply('farther').startswith('ignored'))
        control.apply('resume')
        self.assertTrue(control.apply('closer', held=False).startswith('ignored'))
        for _ in range(10):
            control.apply('farther')
        self.assertAlmostEqual(control.requested_reach, .25)
        self.assertAlmostEqual(control.propose_reach(.05), .299)

    def test_distance_commands_change_the_following_target(self):
        control = track.TrackerControl(DepthTests().args())
        control.apply('closer')
        self.assertAlmostEqual(control.args.distance, .98)
        control.apply('farther')
        self.assertAlmostEqual(control.args.distance, 1)
        control.args.distance = .01
        self.assertTrue(control.apply('closer').startswith('ignored'))
        self.assertEqual(control.args.distance, .01)

    def test_api_enqueues_validates_and_reports_state(self):
        from fastapi.testclient import TestClient
        control = track.TrackerControl(self.args('--dry-run'))
        with TestClient(track.create_api(control)) as client:
            response = client.post('/commands', json={'command': 'freeze'})
            self.assertEqual(response.status_code, 202)
            self.assertFalse(client.get('/status').json()['frozen'])  # acceptance is not application
            control.apply(control.commands.get_nowait())
            control.publish()
            self.assertTrue(client.get('/status').json()['frozen'])
            for body in ({'command': 'invalid'}, {}, {'command': 1}):
                self.assertEqual(client.post('/commands', json=body).status_code, 422)
            self.assertTrue(control.commands.empty())
            self.assertEqual(client.post('/grip').status_code, 202)
            self.assertEqual(control.commands.get_nowait(), 'grip')
            for _ in range(32):
                control.submit('closer')
            self.assertEqual(client.post('/commands', json={'command': 'freeze'}).status_code, 503)
            self.assertEqual(client.get('/stream').status_code, 503)
            self.assertIn('Closer', client.get('/').text)

    def test_gestures_freeze_resume_and_depth_use_same_loop(self):
        args = self.args('--gestures', '--tilt', '--port', 'fake', '--smoothing', '0')
        control = track.TrackerControl(args)
        camera, robot, worker = Mock(), Mock(), Mock()
        hik = SimpleNamespace(z0=.2, reach0=.3, q=np.zeros(5))
        hik.solve = Mock(return_value=(np.ones(5), 0, 0))
        worker.infer.side_effect = [{'command': c, 'error': None}
                                   for c in ('freeze', None, 'resume', 'closer', None)]
        callbacks, positions = [], []
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frames = iter([frame] * 5)
        def read(seq):
            frame = next(frames, None)
            if frame is None:
                callbacks[0]()
            return seq + 1, frame
        camera.read.side_effect = read
        robot.send_action.side_effect = lambda action: positions.append((control.frozen, action))
        detector = Mock()
        detector.detect.return_value = [(400, 300, 50, 50, 'face')]
        with patch.object(track, 'TrackerControl', return_value=control), \
             patch.object(track, 'Camera', return_value=camera), \
             patch.object(track, 'connect_arm', return_value=robot), \
             patch.object(track, 'read_joints', return_value={j: 0 for j in [*track.ARM_JOINTS, track.GRIP]}), \
             patch.object(track, 'HeightIK', return_value=hik), \
             patch.object(track, 'FaceAndBodyDetector', return_value=detector), \
             patch.object(track, 'start_gestures', return_value=worker), \
             patch.object(track.signal, 'signal', side_effect=lambda _, cb: callbacks.append(cb)), \
             patch.object(track.time, 'monotonic', side_effect=iter(np.arange(20) * .05)):
            track.run(args)
        self.assertEqual(len(positions), 3)  # only resumed frames issue actions
        self.assertTrue(all(not frozen for frozen, _ in positions))
        self.assertGreater(control.reach, .3)
        self.assertEqual(control.requested_reach, .32)
        self.assertEqual(control.snapshot()['mode'], 'stopped')
        worker.close.assert_called_once()
        robot.disconnect.assert_called_once()

    def test_freeze_processed_without_camera_frame(self):
        args = self.args('--dry-run')
        control = track.TrackerControl(args)
        control.submit('freeze')
        callback = []
        camera = Mock()
        def read(seq):
            callback[0]()
            return seq, None
        camera.read.side_effect = read
        with patch.object(track, 'TrackerControl', return_value=control), \
             patch.object(track, 'Camera', return_value=camera), \
             patch.object(track, 'FaceAndBodyDetector'), \
             patch.object(track.signal, 'signal', side_effect=lambda _, cb: callback.append(cb)), \
             patch.object(track, 'connect_arm') as connect:
            track.run(args)
        connect.assert_not_called()
        self.assertTrue(control.snapshot()['frozen'])
        self.assertEqual(control.snapshot()['last_command']['command'], 'freeze')
        camera.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
