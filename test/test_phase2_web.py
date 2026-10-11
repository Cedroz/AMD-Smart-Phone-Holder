"""Camera-only API controls and calibration boundaries."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from followme.phase2.runtime import Runtime
from followme.phase2.web import app_for

ROOT=Path(__file__).resolve().parents[1]


class WebTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        with patch('serial.Serial',side_effect=AssertionError('Serial access')):
            self.runtime=Runtime(ROOT,directory.name)
        self.addCleanup(self.runtime.close)
        self.client=TestClient(app_for(self.runtime,'external'))

    def test_status_and_console_are_camera_only(self):
        self.assertEqual(self.client.get('/').status_code,200)
        status=self.client.get('/api/user-state').json()
        self.assertFalse(status['motion_enabled'])
        for path in ('/api/mode','/api/stop','/api/reset'):
            self.assertEqual(self.client.post(path).status_code,404)

    def test_stale_stream_is_unavailable(self):
        self.assertEqual(self.client.get('/stream').status_code,503)

    def test_cross_origin_calibration_and_reacquisition_rejected(self):
        for path in ('/api/zones','/api/reacquire'):
            response=self.client.post(path,json={},headers={'origin':'https://another.invalid'})
            self.assertEqual(response.status_code,403)

    def test_invalid_calibration_does_not_write_configuration(self):
        before=self.runtime.zone_path.read_bytes()
        self.assertEqual(self.client.post('/api/zones',json={'regions':{},'margin':.03}).status_code,422)
        self.assertEqual(self.runtime.zone_path.read_bytes(),before)

    def test_worker_failure_requires_restart(self):
        self.runtime.worker_error='simulated worker death'
        self.assertEqual(self.client.post('/api/reacquire').status_code,409)


if __name__=='__main__':unittest.main()
