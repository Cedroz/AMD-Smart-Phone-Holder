"""Exercise live-adapter boundaries with a mocked SDK and no serial hardware."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from followme.robotics.adapters import LeRobotAdapter
from followme.robotics.contracts import JOINTS
from followme.robotics.trajectories import fake_library


class MockBus:
    def __init__(self, calibrated=True):
        self.is_connected = False
        self.is_calibrated = calibrated
        self.calls = []
        self.mode = 0
        self.port_handler = SimpleNamespace(setPacketTimeout=lambda n: None, packet_timeout=50.)

    def connect(self):
        self.calls.append('connect')
        self.is_connected = True

    def disconnect(self, disable_torque=True):
        self.calls.append(('disconnect', disable_torque))
        self.is_connected = False

    def read(self, register, joint, **kwargs):
        self.calls.append(('read',register,joint))
        return self.mode if register == 'Operating_Mode' else 0

    def write(self, register, joint, value, **kwargs):
        self.calls.append(('write',register,joint,value))


class SDKTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.calibration = self.root/'calibration.json'
        self.calibration.write_text('{}')
        self.bus = MockBus()
        self.robot = SimpleNamespace(bus=self.bus, calibration_fpath=self.calibration,
            get_observation=lambda: {f'{j}.pos':0. for j in JOINTS},
            connect=lambda **kw: self.fail('Robot.connect must never be called'))
        module = SimpleNamespace(SO101Follower=lambda config: self.robot,
                                 SO101FollowerConfig=lambda **kw: SimpleNamespace(**kw))
        self.patch = patch.dict(sys.modules, {'lerobot.robots.so_follower':module})
        self.patch.start()
        self.process_patch = patch('followme.robotics.adapters.subprocess.run',
                                   return_value=SimpleNamespace(returncode=1, stdout=''))
        self.process_mock = self.process_patch.start()
        self.adapter = LeRobotAdapter(self.root,self.root/'fake-device')
        self.adapter.bind_owner()

    def tearDown(self):
        self.adapter.disconnect()
        self.process_patch.stop()
        self.patch.stop()
        self.directory.cleanup()

    def test_inspect_reads_without_configuration_and_preserves_torque(self):
        self.adapter.connect()
        self.adapter.read_joint_positions()
        self.adapter.disconnect()
        self.assertEqual(self.bus.calls,['connect',('disconnect',False)])
        self.assertEqual(self.adapter.physical_commands,0)
        self.assertFalse(self.adapter.armed)

    def test_busy_device_never_opens_bus(self):
        self.process_mock.return_value = SimpleNamespace(returncode=0,stdout='123')
        with self.assertRaisesRegex(RuntimeError,'busy_device'):
            self.adapter.connect()
        self.assertEqual(self.bus.calls,[])
        self.assertIsNone(self.adapter.lock)

    def test_calibration_mismatch_closes_preserving_torque(self):
        self.bus.is_calibrated = False
        with self.assertRaisesRegex(RuntimeError,'calibration_mismatch'):
            self.adapter.connect()
        self.assertEqual(self.bus.calls,['connect',('disconnect',False)])
        self.assertIsNone(self.adapter.lock)

    def test_denied_arm_leaves_torque_unchanged(self):
        self.adapter.connect()
        with patch('followme.robotics.adapters.verify_live_gate',side_effect=ValueError('physical_gates_pending')):
            with self.assertRaisesRegex(ValueError,'physical_gates_pending'):
                self.adapter.arm(fake_library())
        self.assertEqual(self.bus.calls,['connect'])
        self.assertFalse(self.adapter.armed)

    def test_read_only_send_rejected(self):
        self.adapter.connect()
        with self.assertRaisesRegex(RuntimeError,'not_armed'):
            self.adapter.send({})
        self.assertEqual(self.adapter.physical_commands,0)

    def test_arming_seeds_all_goals_before_torque_and_never_changes_gripper(self):
        self.adapter.connect()
        with patch('followme.robotics.adapters.verify_live_gate',return_value='mock-validation'):
            self.adapter.arm(fake_library())
        writes = [call for call in self.bus.calls if isinstance(call,tuple) and call[0]=='write']
        self.assertEqual(len(writes),10)
        self.assertTrue(all(call[1]=='Goal_Position' for call in writes[:5]))
        self.assertTrue(all(call[1]=='Torque_Enable' for call in writes[5:]))
        self.assertTrue(all(call[2]!='gripper' for call in writes))
        self.assertTrue(self.adapter.armed)

    def test_arming_refuses_unsupported_mode_without_writes(self):
        self.adapter.connect()
        self.bus.mode = 1
        with patch('followme.robotics.adapters.verify_live_gate',return_value='mock-validation'):
            with self.assertRaisesRegex(ValueError,'unsupported_operating_mode'):
                self.adapter.arm(fake_library())
        self.assertFalse(any(isinstance(call,tuple) and call[0]=='write' for call in self.bus.calls))

    def test_arming_rechecks_calibration_before_any_write(self):
        self.adapter.connect()
        self.calibration.write_text('{"changed":true}')
        with patch('followme.robotics.adapters.verify_live_gate',return_value='mock-validation'):
            with self.assertRaisesRegex(ValueError,'calibration_mismatch'):
                self.adapter.arm(fake_library())
        self.assertFalse(any(isinstance(call,tuple) and call[0]=='write' for call in self.bus.calls))


if __name__ == '__main__':
    unittest.main()
