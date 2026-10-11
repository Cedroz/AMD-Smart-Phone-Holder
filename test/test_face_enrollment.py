"""Enrollment API and camera-loop tests; no hardware or downloaded models needed."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient
from tracker import faces, track


def args(*extra):
    with patch('sys.argv', ['track.py', '--preview-port', '0', '--squeeze', '0', *extra]), \
            patch.object(track, 'run') as run:
        track.main()
    return run.call_args.args[0]


class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db = Path(self.directory.name) / 'faces.npz'
        self.now = 0.0
        # Exercise real database writes while simulating face detection and features.
        self.recognizer = faces.FaceRecognizer.__new__(faces.FaceRecognizer)
        self.recognizer.database = self.db
        self.recognizer.names = np.empty(0, dtype=str)
        self.recognizer.features = np.empty((0, 128), dtype=np.float32)
        self.recognizer.faces = Mock(return_value=[object()])
        self.recognizer.feature = Mock(return_value=np.ones(128, dtype=np.float32) / np.sqrt(128))
        self.enrollment = faces.FaceEnrollment(self.db, recognizer=self.recognizer,
            samples=3, seconds=5, clock=lambda: self.now)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_api_start_progress_completion_and_replacement(self):
        control = track.TrackerControl(args('--dry-run'))
        control.enrollment = self.enrollment
        with TestClient(track.create_api(control)) as client:
            page = client.get('/').text
            self.assertIn('Cancel enrollment', page)
            self.assertIn('face-progress', page)
            response = client.post('/faces/enrollment', json={'name': ' Alice '})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()['state'], 'starting')
            self.assertFalse(self.db.exists())
            self.assertEqual(client.post('/faces/enrollment', json={'name': 'Bob'}).status_code, 409)
            self.enrollment.update(self.frame)
            state = client.get('/status').json()['enrollment']
            self.assertEqual((state['state'], state['samples']), ('running', 1))
            self.enrollment.update(self.frame)  # same instant: no duplicate sample
            self.assertEqual(self.enrollment.snapshot()['samples'], 1)
            for self.now in (.5, 1.0):
                self.enrollment.update(self.frame)
            state = client.get('/status').json()['enrollment']
            self.assertEqual(state['state'], 'completed')
            self.assertIn('Saved Alice', state['message'])
            with np.load(self.db) as data:
                self.assertEqual(list(data['names']), ['Alice'] * 3)
            # A second person's enrollment retains the first person.
            self.enrollment.start('Bob')
            for self.now in (1.0, 1.5, 2.0):
                self.enrollment.update(self.frame)
            self.enrollment.start('Alice')
            for self.now in (2.0, 2.5, 3.0):
                self.enrollment.update(self.frame)
            with np.load(self.db) as data:
                self.assertEqual(sorted(data['names']), ['Alice'] * 3 + ['Bob'] * 3)

    def test_cancel_timeout_and_multiple_faces_preserve_existing_database(self):
        self.recognizer.enroll('Existing', [self.recognizer.feature.return_value])
        previous = self.db.read_bytes()
        self.enrollment.start('Alice')
        self.enrollment.update(self.frame)
        self.enrollment.cancel()
        self.enrollment.update(self.frame)
        self.assertEqual(self.enrollment.snapshot()['state'], 'cancelled')
        self.assertEqual(self.db.read_bytes(), previous)
        self.enrollment.start('Alice')
        self.recognizer.faces.return_value = [object(), object()]
        self.enrollment.update(self.frame)
        self.assertEqual(self.enrollment.snapshot()['samples'], 0)
        self.assertIn('Multiple faces', self.enrollment.snapshot()['message'])
        self.now = 6
        self.enrollment.update(None)
        self.assertEqual(self.enrollment.snapshot()['state'], 'error')
        self.assertEqual(self.db.read_bytes(), previous)

    def test_bad_names_and_model_failure_are_reported(self):
        control = track.TrackerControl(args('--dry-run'))
        control.enrollment = self.enrollment
        with TestClient(track.create_api(control)) as client:
            for body in ({}, {'name': ''}, {'name': 'unknown'}, {'name': 'x' * 81}, {'name': '\n'}):
                self.assertEqual(client.post('/faces/enrollment', json=body).status_code, 422)
            self.assertEqual(client.post('/faces/enrollment/cancel').json()['state'], 'idle')
        self.enrollment.recognizer = None
        self.enrollment.start('Alice')
        with patch.object(faces, 'FaceRecognizer', side_effect=FileNotFoundError('model missing')):
            self.enrollment.update(self.frame)
        self.assertEqual(self.enrollment.snapshot()['state'], 'error')
        self.assertIn('model missing', self.enrollment.snapshot()['message'])

    def test_overlay_associates_names_with_faces_and_rejects_unknown_or_ambiguous_boxes(self):
        alice, bob, stranger = np.eye(128, dtype=np.float32)[:3]
        self.recognizer.names = np.array(['Alice Smith', 'Bob'])
        self.recognizer.features = np.stack([alice, bob])
        self.recognizer.threshold = .45
        self.recognizer.faces.return_value = [(10, 10, 40, 40), (100, 10, 40, 40)]
        self.recognizer.feature.side_effect = lambda frame, face: alice if face[0] == 10 else bob
        boxes = [(10, 10, 40, 40, 'face'), (100, 10, 40, 40, 'face'), (0, 0, 200, 200, 'person')]
        self.assertEqual(self.enrollment.labels(self.frame, boxes), {0: 'Alice Smith', 1: 'Bob'})
        self.recognizer.feature.side_effect = lambda frame, face: alice if face[0] == 10 else stranger
        self.assertEqual(self.enrollment.labels(self.frame, boxes), {0: 'Alice Smith'})
        # Two faces inside a single body box cannot identify one person unambiguously.
        self.assertEqual(self.enrollment.labels(self.frame, [boxes[2]]), {})
        self.recognizer.faces.return_value = [(10, 10, 40, 40)]
        self.assertEqual(self.enrollment.labels(self.frame, [boxes[2]]), {0: 'Alice Smith'})
        self.assertEqual(boxes[0][4], 'face')  # Keep targeting/depth/gesture semantics intact.

    def test_saved_database_is_loaded_for_overlay_and_new_enrollment_updates_it(self):
        self.recognizer.enroll('Existing', [self.recognizer.feature.return_value])
        enrollment = faces.FaceEnrollment(self.db, samples=1, clock=lambda: self.now)
        self.recognizer.recognize = Mock(side_effect=lambda frame: [
            ((10, 10, 40, 40), str(self.recognizer.names[-1]), .9)])
        box = [(10, 10, 40, 40, 'face')]
        with patch.object(faces, 'FaceRecognizer', return_value=self.recognizer) as load:
            self.assertEqual(enrollment.labels(self.frame, box), {0: 'Existing'})
            self.assertEqual(enrollment.labels(self.frame, box), {0: 'Existing'})
            load.assert_called_once()
        enrollment.start('New Name')
        enrollment.update(self.frame)
        self.assertEqual(enrollment.snapshot()['state'], 'completed')
        self.assertEqual(enrollment.labels(self.frame, box), {0: 'New Name'})

    def test_overlay_renders_name_and_keeps_selected_box_green(self):
        boxes = [(10, 5, 40, 40, 'face'), (100, 10, 40, 40, 'face')]
        with patch.object(track.cv2, 'putText') as text, patch.object(track.cv2, 'rectangle') as rectangle:
            track.annotate(self.frame, boxes, boxes[0], None, .1, 'DRY RUN', {0: 'Alice Smith'})
        self.assertEqual(text.call_args_list[0].args[1], 'Alice Smith')
        self.assertEqual(text.call_args_list[0].args[2], (10, 15))
        self.assertEqual(text.call_args_list[1].args[1], 'face 40px')
        self.assertEqual(rectangle.call_args_list[0].args[3], (0, 255, 0))

    def test_tracking_uses_single_camera_and_blocks_motion_while_enrolling(self):
        options = args('--port', 'fake')
        control = track.TrackerControl(options)
        control.enrollment = self.enrollment
        self.enrollment.start('Alice')
        control.submit('test_motion')
        camera, robot, detector = Mock(), Mock(), Mock()
        detector.detect.return_value = [(400, 300, 50, 50, 'face')]
        callbacks = []
        frames = iter([self.frame, self.frame])
        def read(seq):
            frame = next(frames, None)
            if frame is None:
                callbacks[0]()
            return seq + 1, frame
        camera.read.side_effect = read
        with patch.object(track, 'TrackerControl', return_value=control), \
             patch.object(track, 'Camera', return_value=camera) as open_camera, \
             patch.object(track, 'connect_arm', return_value=robot), \
             patch.object(track, 'read_joints', return_value={j: 0 for j in [*track.ARM_JOINTS, track.GRIP]}), \
             patch.object(track, 'FaceAndBodyDetector', return_value=detector), \
             patch.object(track.signal, 'signal', side_effect=lambda _, cb: callbacks.append(cb)):
            track.run(options)
        open_camera.assert_called_once()
        detector.detect.assert_not_called()
        robot.send_action.assert_not_called()
        self.assertEqual(control.snapshot()['motion_blocker'], 'camera unavailable')
        self.assertIn('enrollment', control.last_command['result'])
        camera.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
