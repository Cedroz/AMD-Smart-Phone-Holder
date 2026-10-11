"""Phase 3 acceptance: controlled time, fake hardware, no physical I/O."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from followme.robotics.adapters import FakeAdapter, LeRobotAdapter, verify_live_gate
from followme.robotics.contracts import ARM_JOINTS, JointSnapshot, Limits, TargetRequest
from followme.robotics.controller import Controller
from followme.robotics.trajectories import PoseLibrary, fake_library


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now

    def advance(self, seconds=.05):
        self.now += seconds


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.adapter = FakeAdapter(self.clock)
        self.library = fake_library()
        self.controller = Controller(self.adapter, self.library, clock=self.clock)
        self.controller.tick()

    def tearDown(self):
        self.controller.close()

    def follow(self):
        self.controller.heartbeat()
        self.controller.update_tracking(self.clock())
        self.controller.set_mode('follow')
        self.controller.tick()

    def move(self, identity='test'):
        self.follow()
        result = self.controller.request_safe_pose(TargetRequest(identity, 'station_b', 'test', self.clock()))
        self.assertTrue(result.accepted, result.reason)
        self.controller.tick()

    def step(self, dt=.05, tracking=True, heartbeat=True):
        self.clock.advance(dt)
        if tracking:
            self.controller.update_tracking(self.clock())
        if heartbeat:
            self.controller.heartbeat()
        self.controller.tick()

    def test_default_hold_never_constructs_live_adapter(self):
        with patch.object(LeRobotAdapter, '__init__', side_effect=AssertionError('live construction')):
            self.step()
        self.assertEqual(self.controller.status().mode, 'hold')
        self.assertEqual(self.adapter.physical_commands, 0)
        self.assertEqual(self.adapter.simulated_commands, 0)

    def test_success_and_gripper_unchanged(self):
        self.adapter.positions['gripper'] = 37.
        self.move()
        for _ in range(80):
            self.step()
        self.assertEqual(self.controller.status().current_pose_name, 'station_b')
        self.assertEqual(self.adapter.positions['gripper'], 37.)
        commands = [event for event in self.controller.log.events if event['event'] == 'command']
        self.assertTrue(commands)
        self.assertTrue(all('gripper' not in e['requested'] for e in commands))
        self.assertEqual(self.adapter.physical_commands, 0)

    def test_concurrent_requests_one_slot(self):
        self.follow()
        results = []
        threads = [threading.Thread(target=lambda i=i: results.append(self.controller.request_safe_pose(
            TargetRequest(str(i), 'station_b', 'concurrent', self.clock())))) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sum(result.accepted for result in results), 1)

    def test_wrong_io_owner(self):
        errors = []
        def other():
            try:
                self.adapter.read_joint_positions()
            except RuntimeError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=other)
        thread.start()
        thread.join()
        self.assertEqual(errors, ['wrong_io_owner'])

    def test_stale_future_and_duplicate_requests(self):
        self.follow()
        for timestamp in (self.clock()-1, self.clock()+1, float('nan')):
            self.assertFalse(self.controller.request_safe_pose(TargetRequest('old','station_b','test',timestamp)).accepted)
        self.assertTrue(self.controller.request_safe_pose(TargetRequest('id','station_b','test',self.clock())).accepted)
        self.assertEqual(self.controller.request_safe_pose(TargetRequest('id','station_b','test',self.clock())).reason,
                         'duplicate_request')

    def test_pending_expired_before_execution(self):
        self.follow()
        self.controller.request_safe_pose(TargetRequest('expired','station_b','test',self.clock()))
        self.step(.6)
        self.assertEqual(self.adapter.simulated_commands, 0)
        self.assertTrue(any(e.get('reason') == 'expired_request' for e in self.controller.log.events))

    def test_hold_cancels_and_has_no_backlog(self):
        self.move()
        for _ in range(20):
            self.step()
        self.controller.request_hold()
        self.step()
        self.assertEqual(self.controller.status().mode, 'hold')
        count = self.adapter.simulated_commands
        self.step()
        self.assertEqual(self.adapter.simulated_commands, count)
        self.assertIsNone(self.controller.status().current_pose_name)

    def test_stop_has_priority_over_follow_and_reset(self):
        self.move()
        self.controller.reset()
        self.controller.set_mode('follow')
        self.controller.stop_motion()
        self.step()
        self.assertEqual(self.controller.status().state, 'STOPPED')
        self.step()
        self.assertEqual(self.controller.status().state, 'STOPPED')

    def test_stop_arriving_during_read(self):
        self.move()
        original = self.adapter.read_joint_positions
        def interrupted_read():
            snapshot = original()
            self.controller.stop_motion()
            return snapshot
        self.adapter.read_joint_positions = interrupted_read
        self.step()
        self.assertEqual(self.controller.status().state, 'STOPPED')
        self.adapter.read_joint_positions = original

    def test_stop_arriving_before_dispatch(self):
        self.move()
        original = self.library.validate_target
        def interrupt(target):
            original(target)
            self.controller.stop_motion()
        self.library.validate_target = interrupt
        self.step()
        self.assertEqual(self.controller.status().state, 'STOPPED')
        self.library.validate_target = original

    def test_stale_tracking_cancels(self):
        self.move()
        for _ in range(12):
            self.step(tracking=False)
        self.assertEqual(self.controller.status().mode, 'hold')

    def test_ambiguous_tracking_cancels(self):
        self.move()
        self.controller.update_tracking(self.clock(), ambiguous=True)
        self.controller.tick()
        self.assertEqual(self.controller.status().mode, 'hold')

    def test_heartbeat_loss_cancels(self):
        self.move()
        for _ in range(65):
            self.step(heartbeat=False)
        self.assertEqual(self.controller.status().mode, 'hold')

    def test_lateness_faults_without_catchup(self):
        self.move()
        self.step(.2)
        self.assertEqual(self.controller.status().state, 'FAULT')
        self.assertEqual(self.controller.status().fault_reason, 'loop_lateness')

    def test_invalid_feedback_faults(self):
        self.move()
        self.adapter.invalid = True
        self.step()
        self.assertEqual(self.controller.status().state, 'FAULT')

    def test_stuck_joint_faults_without_claiming_arrival(self):
        self.move()
        self.adapter.stuck = True
        for _ in range(80):
            self.step()
        self.assertEqual(self.controller.status().state, 'FAULT')
        self.assertFalse(any(e['event'] == 'motion_arrived' for e in self.controller.log.events))

    def test_write_failure_leaves_hold_unconfirmed(self):
        self.move()
        self.adapter.fail_write = True
        self.step()
        self.assertEqual(self.controller.status().state, 'FAULT')
        self.assertFalse(self.controller.status().hold_confirmed)
        self.assertIn('hold_unconfirmed',self.controller.status().fault_reason)

    def test_slow_read_inhibits_dispatch(self):
        self.move()
        original = self.adapter.read_joint_positions
        def slow_read():
            self.clock.advance(.06)
            return original()
        self.adapter.read_joint_positions = slow_read
        self.step()
        self.assertEqual(self.controller.status().state,'FAULT')
        self.assertEqual(self.controller.status().fault_reason,'io_budget_exceeded')
        self.adapter.read_joint_positions = original

    def test_settling_timeout_uses_measured_feedback(self):
        limits = Limits(arrival_deg=.5)
        self.controller.limits = limits
        self.controller.library = fake_library(limits)
        self.move()
        for _ in range(60):
            self.step()
        self.adapter.positions['shoulder_pan'] = 9.
        self.adapter.stuck = True
        for _ in range(45):
            self.step()
        self.assertEqual(self.controller.status().state,'FAULT')
        self.assertEqual(self.controller.status().fault_reason,'settling_timeout')

    def test_waypoints_require_settling_before_next_segment(self):
        data = self.library.data
        waypoint = {j:0. for j in ARM_JOINTS}
        waypoint['shoulder_pan'] = 5.
        data['transitions']['station_a->station_b'].update(waypoints=[waypoint], durations_s=[1.5,1.5])
        self.controller.library = PoseLibrary(data)
        self.move()
        for _ in range(90):
            self.step()
        self.assertEqual(self.controller.status().current_pose_name,'station_b')

    def test_reset_rejects_unresolved_calibration(self):
        self.controller.stop_motion()
        self.step()
        self.adapter.calibration_matches = False
        self.controller.reset()
        self.step()
        self.assertEqual(self.controller.status().state,'FAULT')
        self.assertIn('calibration_mismatch',self.controller.status().fault_reason)

    def test_clipping_faults_and_unconfirmed_hold(self):
        self.move()
        self.adapter.clipped = True
        self.step()
        self.assertEqual(self.controller.status().state, 'FAULT')
        self.assertIn('hold_unconfirmed', self.controller.status().fault_reason)
        self.assertFalse(self.controller.status().hold_confirmed)

    def test_read_failure_reset_requires_resolution(self):
        self.move()
        self.adapter.fail_read = True
        self.step()
        self.controller.reset()
        self.step()
        self.assertEqual(self.controller.status().state, 'FAULT')
        self.adapter.fail_read = False
        self.step()
        self.assertEqual(self.controller.status().state, 'HOLDING')
        self.assertFalse(self.controller.status().motion_enabled)

    def test_reset_nonanchor_does_not_restore_or_resume(self):
        self.move()
        for _ in range(25):
            self.step()
        self.controller.stop_motion()
        self.step()
        self.controller.reset()
        self.step()
        self.assertIsNone(self.controller.status().current_pose_name)
        self.follow()
        self.assertFalse(self.controller.status().motion_enabled)

    def test_unapproved_reverse_path(self):
        self.library.data['transitions'].pop('station_a->station_b')
        self.move()
        self.assertEqual(self.adapter.simulated_commands, 0)
        self.assertTrue(any(e.get('reason') == 'unapproved_path' for e in self.controller.log.events))

    def test_failed_handshake_cleanup(self):
        adapter = FakeAdapter(self.clock)
        adapter.calibration_matches = False
        controller = Controller(adapter, self.library, clock=self.clock)
        controller.tick()
        self.assertEqual(controller.status().state, 'FAULT')
        controller.close()
        self.assertFalse(adapter.connected)
        self.assertEqual(adapter.calls[-1][0], 'disconnect_preserve_torque')


class ValidationTests(unittest.TestCase):
    def test_limits_and_units(self):
        for limits in ({'velocity': 11}, {'relative': 3}, {'hz': 0}, {'freshness_s': 1}):
            with self.assertRaises(ValueError):
                Limits(**limits)
        with self.assertRaises(ValueError):
            JointSnapshot(dict.fromkeys(ARM_JOINTS,0.), 0.)
        with self.assertRaises(ValueError):
            JointSnapshot({**dict.fromkeys(ARM_JOINTS,0.),'gripper':0},0.,dict.fromkeys(ARM_JOINTS,'radians'))

    def test_derivative_bounds_and_joint_limits(self):
        data = fake_library().data
        for duration in (.1, float('nan'), -1):
            bad = copy.deepcopy(data)
            bad['transitions']['station_a->station_b']['durations_s'] = [duration]
            library = PoseLibrary(bad)
            with self.assertRaises(ValueError):
                library.path('station_a','station_b')
        bad = copy.deepcopy(data)
        bad['poses']['station_b']['joints']['wrist_roll'] = 31.
        with self.assertRaises(ValueError):
            PoseLibrary(bad)

    def test_fake_library_cannot_be_live(self):
        with self.assertRaisesRegex(ValueError, 'backend'):
            PoseLibrary(fake_library().data, backend='live')

    def test_production_library_empty_and_physical_gate_pending(self):
        root = Path(__file__).resolve().parents[1]
        library = PoseLibrary.load(root/'config/safe_poses.yaml')
        self.assertEqual(library.poses,{})
        with self.assertRaises(ValueError):
            verify_live_gate(root, library)

    def test_library_configuration_copied(self):
        data = fake_library().data
        library = PoseLibrary(data)
        data['poses']['station_a']['joints']['shoulder_pan'] = 99
        self.assertEqual(library.poses['station_a']['joints']['shoulder_pan'], 0)


if __name__ == '__main__':
    unittest.main()
