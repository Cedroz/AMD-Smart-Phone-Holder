"""Web follow-mode policy and API tests; no camera, motors, or audio model needed."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient
from tracker import faces, track


def args():
    with patch('sys.argv', ['track.py', '--dry-run', '--preview-port', '0']), \
            patch.object(track, 'run') as run:
        track.main()
    return run.call_args.args[0]


class FollowModesTests(unittest.TestCase):
    def setUp(self):
        self.control = track.TrackerControl(args())
        self.control.face_names = ['Alice', 'Bob']
        self.control.voice_names = ['Alice', 'Bob', 'MissingFace']

    def test_manual_ignores_voice_and_speaker_overrides_selected_person(self):
        c = self.control
        c.find('Alice')
        c.hear('Bob', .9)
        self.assertEqual(c.following(), 'Alice')
        c.set_follow_mode('speaker')
        self.assertEqual(c.following(), 'Alice')  # Selected person is initial target.
        c.hear('Bob', .9)
        self.assertEqual(c.following(), 'Bob')
        self.assertEqual(c.snapshot()['active_target'], 'Bob')
        self.assertEqual(c.find_name, 'Alice')  # Mobile selection stays intact.
        c.set_follow_mode('manual')
        self.assertEqual(c.following(), 'Alice')

    def test_silence_unknown_low_confidence_and_missing_face_do_not_switch(self):
        c = self.control
        c.set_follow_mode('speaker')
        with patch.object(track.time, 'monotonic', return_value=100):
            c.hear('Alice', .9)
        for speaker, confidence in [('silence', 0), ('unknown', .9), ('Bob', .4), ('MissingFace', .95)]:
            with patch.object(track.time, 'monotonic', return_value=102):
                c.hear(speaker, confidence)
                self.assertEqual(c.following(), 'Alice')
        with patch.object(track.time, 'monotonic', return_value=105):
            self.assertIsNone(c.following())  # Expired audio must not select a stranger.

    def test_mobile_tap_takes_effect_until_next_accepted_speaker(self):
        c = self.control
        c.set_follow_mode('speaker')
        c.hear('Bob', .9)
        c.find('Alice')
        self.assertEqual(c.following(), 'Alice')
        c.hear('silence', 0)
        self.assertEqual(c.following(), 'Alice')
        c.hear('Bob', .9)
        self.assertEqual(c.following(), 'Bob')

    def test_api_status_controls_and_invalid_inputs(self):
        with TestClient(track.create_api(self.control)) as client:
            page = client.get('/').text
            self.assertIn('Manual / Go to', page)
            self.assertIn('Follow speaker', page)
            self.assertIn('id="follow-people"', page)
            self.assertIn("fetch('/find'", page)
            self.assertIn('Laptop microphone setup', page)
            self.assertEqual(client.post('/find', json={'name': 'Alice'}).status_code, 202)
            status = client.post('/follow', json={'mode': 'speaker'}).json()
            self.assertEqual(status['follow_mode'], 'speaker')
            self.assertEqual(status['active_target'], 'Alice')
            client.post('/voice', json={'speaker': 'Bob', 'confidence': .9})
            self.assertEqual(client.get('/status').json()['active_target'], 'Bob')
            for body in ({}, {'mode': 'invalid'}):
                self.assertEqual(client.post('/follow', json=body).status_code, 422)
            self.assertEqual(client.post('/voice', json={'speaker': 'Bob', 'confidence': 1.5}).status_code, 422)
            self.control.voice_names = []
            self.assertEqual(client.post('/follow', json={'mode': 'speaker'}).status_code, 409)

    def test_listener_connection_and_error_status(self):
        c = self.control
        c.voice_listener_state = 'running'
        self.assertEqual(c.snapshot()['voice_listener'], 'waiting for audio')
        c.set_follow_mode('speaker')
        self.assertIn('laptop sender', c.snapshot()['voice_feedback'])
        c.hear('silence', 0)
        self.assertEqual(c.snapshot()['voice_listener'], 'connected')
        self.assertIn('waiting for speech', c.snapshot()['voice_feedback'])
        c.voice_listener_state = 'error: microphone failed'
        self.assertIn('error', c.snapshot()['voice_listener'])
        c.set_follow_mode('speaker')
        self.assertEqual(c.voice_listener_state, 'off')

    def test_voice_feedback_explains_rejected_speakers(self):
        c = self.control
        c.set_follow_mode('speaker')
        c.voice_listener_state = 'running'
        for name, score, explanation in [('unknown', .6, 'no enrolled voice'),
                                         ('MissingFace', .9, 'face must be enrolled'),
                                         ('Alice', .4, 'below the 0.5')]:
            c.hear(name, score)
            self.assertIn(explanation, c.snapshot()['voice_feedback'])
            self.assertIsNone(c.following())

    def test_speaker_mode_starts_listener_and_filters_tracking_target(self):
        options = args()
        c = self.control
        c.set_follow_mode('speaker')
        camera, listener, recognizer = Mock(), Mock(), Mock()
        listener.poll.return_value = None
        recognizer.names = np.array(['Alice', 'Bob'])
        recognizer.detect.return_value = [(10, 10, 40, 40, 'face:Alice 0.90'),
                                          (100, 10, 40, 40, 'face:Bob 0.90')]
        callbacks, calls = [], 0
        def read(seq):
            nonlocal calls
            calls += 1
            if calls == 1:
                c.hear('Bob', .9)
                return seq + 1, np.zeros((480, 640, 3), dtype=np.uint8)
            callbacks[0]()
            return seq, None
        camera.read.side_effect = read
        with patch.object(track, 'TrackerControl', return_value=c), \
             patch.object(track, 'Camera', return_value=camera), \
             patch.object(faces, 'FaceRecognizer', return_value=recognizer), \
             patch.object(track, 'FaceAndBodyDetector'), \
             patch.object(track, 'start_voice', return_value=(recognizer, listener)) as start, \
             patch.object(track, 'pick_target', wraps=track.pick_target) as choose, \
             patch.object(track.signal, 'signal', side_effect=lambda _, cb: callbacks.append(cb)):
            track.run(options)
        start.assert_called_once()
        self.assertEqual(choose.call_args.args[0], [(100, 10, 40, 40, 'face:Bob 0.90')])
        listener.terminate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
