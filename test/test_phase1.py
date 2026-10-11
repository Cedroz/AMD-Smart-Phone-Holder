import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from followme.environment import preflight, verified_model
from followme.protocol import JsonSocket, MAX_MESSAGE
from followme.vision_client import VisionClient

ROOT=Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def pair(self):
        a,b=socket.socketpair()
        self.addCleanup(a.close); self.addCleanup(b.close)
        return JsonSocket(a),b

    def test_fragmented_messages_and_buffered_next_message(self):
        channel,peer=self.pair()
        peer.sendall(b'{"type":')
        threading.Timer(.01,lambda:peer.sendall(b'"one"}\n{"type":"two"}\n')).start()
        self.assertEqual(channel.receive()['type'],'one')
        self.assertEqual(channel.receive()['type'],'two')

    def test_invalid_json_and_non_object(self):
        for payload in (b'bad\n',b'[]\n',b'{"n":NaN}\n'):
            channel,peer=self.pair(); peer.sendall(payload)
            with self.assertRaises(ValueError): channel.receive()

    def test_oversized_send(self):
        channel,peer=self.pair()
        with self.assertRaisesRegex(ValueError,'4 MiB'):
            channel.send({'data':'x'*MAX_MESSAGE})

    def test_oversized_receive_without_delimiter(self):
        channel,peer=self.pair()
        channel.buffer.extend(b'x'*MAX_MESSAGE)
        with self.assertRaisesRegex(ValueError,'4 MiB'):channel.receive()

    def test_timeout_and_worker_exit(self):
        channel,peer=self.pair()
        with self.assertRaises(TimeoutError): channel.receive(.01)
        peer.close()
        with self.assertRaises(EOFError): channel.receive()


class EnvironmentTests(unittest.TestCase):
    def config(self,**changes):
        import yaml
        cfg=yaml.safe_load((ROOT/'config/environment.yaml').read_text())
        cfg.update(changes)
        tmp=tempfile.NamedTemporaryFile(mode='w',suffix='.yaml',delete=False)
        yaml.safe_dump(cfg,tmp);tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return tmp.name

    def test_default_preflight_does_not_open_hardware(self):
        with patch('serial.Serial',side_effect=AssertionError('Serial opened')), patch('cv2.VideoCapture',side_effect=AssertionError('Camera opened')):
            report=preflight(ROOT)
        self.assertFalse(report['hardware_opened'])
        self.assertTrue(report['calibration_unchanged'])

    def test_motion_enabled_config_rejected(self):
        for changes in ({'dry_run':False},{'mode':'follow'},{'autonomous_motion_approved':True}):
            with self.assertRaises(ValueError):preflight(ROOT,self.config(**changes))

    def test_changed_calibration_identity_rejected(self):
        import yaml
        cfg=yaml.safe_load((ROOT/'config/environment.yaml').read_text())
        cfg['arms']['follower']['calibration_id']='different_arm'
        with self.assertRaisesRegex(ValueError,'Calibration ID'):
            preflight(ROOT,self.config(arms=cfg['arms']))

    def test_missing_interpreter_and_model(self):
        with self.assertRaises(FileNotFoundError):preflight(ROOT,self.config(vision_python='missing/python'))
        with self.assertRaises(FileNotFoundError):preflight(ROOT,self.config(model_path='missing.task'))

    def test_corrupt_model_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp)/'model.task';model.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'mismatch'):
                verified_model(model,ROOT/'models/manifest.json')


class SupervisionTests(unittest.TestCase):
    def client(self):
        import yaml
        return VisionClient(ROOT,yaml.safe_load((ROOT/'config/environment.yaml').read_text()),startup_timeout=.3,inference_timeout=.1)

    def fake_popen(self,code):
        original=subprocess.Popen
        def launch(command,**kwargs):
            fd=command[command.index('--fd')+1]
            return original([sys.executable,'-c',code,fd],**kwargs)
        return launch

    def test_start_timeout_reaps_child(self):
        client=self.client()
        with patch('followme.vision_client.subprocess.Popen',self.fake_popen('import time;time.sleep(30)')):
            with self.assertRaises(TimeoutError):client.start()
        self.assertIsNone(client.process);self.assertIsNone(client.channel)

    def test_worker_exit_reaps_child(self):
        client=self.client()
        with patch('followme.vision_client.subprocess.Popen',self.fake_popen('pass')):
            with self.assertRaises(EOFError):client.start()
        self.assertIsNone(client.process)

    def test_handshake_mismatch_rejected(self):
        client=self.client()
        code='import socket,sys;socket.socket(fileno=int(sys.argv[1])).sendall(b\'{"type":"ready","protocol_version":999}\\n\')'
        with patch('followme.vision_client.subprocess.Popen',self.fake_popen(code)):
            with self.assertRaises(ValueError):client.start()
        self.assertIsNone(client.process)

    def test_inference_timeout_and_identity_mismatch(self):
        for mode in ('sleep','mismatch'):
            client=self.client()
            channel,peer=socket.socketpair()
            client.channel=JsonSocket(channel)
            client.process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
            self.addCleanup(client.close);self.addCleanup(peer.close)
            if mode=='mismatch':peer.sendall(b'{"type":"result","request_id":"wrong"}\n')
            with self.assertRaises(TimeoutError if mode=='sleep' else ValueError):
                client.infer('external',1,1,b'jpeg')
            self.assertIsNone(client.process)

    def test_latest_buffers_replace_old_frames_and_busy_request_rejected(self):
        client=self.client()
        client.submit_latest('external',1,1,b'old')
        client.submit_latest('external',2,2,b'new')
        client.submit_latest('wrist',1,1,b'wrist')
        self.assertEqual(len(client.latest),2)
        with patch.object(client,'infer',return_value='result') as infer:
            self.assertEqual(client.infer_latest('external'),'result')
            infer.assert_called_once_with('external',2,2,b'new')
        client.request_lock.acquire()
        try:
            with self.assertRaises(RuntimeError):client.infer('external',1,1,b'')
        finally:client.request_lock.release()


if __name__=='__main__':unittest.main()
