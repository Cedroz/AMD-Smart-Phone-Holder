"""Verify the inspection boundary without importing LeRobot or opening hardware."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'inspect_phase0', Path(__file__).resolve().parents[1] / 'scripts/inspect_phase0.py')
inspection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspection)


class FakeBus:
    def __init__(self, fail=False):
        self.is_connected = False
        self.fail = fail
        self.closed_with_torque_disabled = None

    def connect(self):
        self.is_connected = True
        if self.fail:
            raise RuntimeError('Simulated handshake failure')

    def read_calibration(self):
        return {}

    @property
    def is_calibrated(self):
        return True

    def sync_read(self, register, **kwargs):
        return {}

    def disconnect(self, disable_torque=True):
        self.closed_with_torque_disabled = disable_torque
        self.is_connected = False


class InspectionTests(unittest.TestCase):
    def test_success_preserves_torque_and_blocks_writes(self):
        bus = FakeBus()
        report = inspection.read_bus(bus)
        self.assertTrue(report['matches_saved_calibration'])
        self.assertFalse(bus.closed_with_torque_disabled)
        for write in (bus.write, bus.sync_write):
            with self.assertRaisesRegex(RuntimeError, 'prohibited'):
                write('Goal_Position', 100)

    def test_handshake_failure_still_closes_without_disabling_torque(self):
        bus = FakeBus(fail=True)
        with self.assertRaisesRegex(RuntimeError, 'handshake'):
            inspection.read_bus(bus)
        self.assertFalse(bus.is_connected)
        self.assertFalse(bus.closed_with_torque_disabled)


if __name__ == '__main__':
    unittest.main()
