"""Owned adapters. Live inspection uses bus reads, never Robot.connect()."""
import fcntl
import hashlib
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from .contracts import ARM_JOINTS, JOINTS, JointSnapshot, validate_joints


class OwnedAdapter:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.owner = None
        self.connected = False
        self.armed = False
        self.physical_commands = 0

    def bind_owner(self):
        if self.owner is not None and self.owner != threading.get_ident():
            raise RuntimeError('wrong_io_owner')
        self.owner = threading.get_ident()

    def check_owner(self):
        if self.owner != threading.get_ident():
            raise RuntimeError('wrong_io_owner')


class FakeAdapter(OwnedAdapter):
    backend = 'fake'

    def __init__(self, clock=time.monotonic, delay_s=0.):
        super().__init__(clock)
        self.positions = dict.fromkeys(JOINTS, 0.)
        self.calls = []
        self.simulated_commands = 0
        self.delay_s = delay_s
        self.stuck = False
        self.invalid = False
        self.fail_read = False
        self.fail_write = False
        self.clipped = False
        self.calibration_matches = True

    def call(self, name):
        self.check_owner()
        self.calls.append((name, threading.get_ident(), self.clock()))
        if self.delay_s:
            time.sleep(self.delay_s)

    def connect(self, read_only=True):
        self.call('connect')
        self.connected = True
        if not self.calibration_matches:
            raise RuntimeError('calibration_mismatch')
        self.armed = True  # Simulation only; no serial object is constructed.

    def read_joint_positions(self):
        self.call('read')
        if self.fail_read:
            raise RuntimeError('transport_failure')
        values = dict(self.positions)
        if self.invalid:
            values['wrist_roll'] = float('nan')
        return JointSnapshot(values, self.clock())

    def send(self, targets):
        self.call('write')
        validate_joints(targets, include_gripper=False)
        if self.fail_write:
            raise RuntimeError('transport_failure')
        actual = dict(targets)
        if self.clipped:
            actual['shoulder_pan'] += .1
        self.simulated_commands += 1
        if not self.stuck:
            self.positions.update(actual)
        return actual

    def disconnect(self):
        self.call('disconnect_preserve_torque')
        self.connected = False


def verify_live_gate(root, library):
    """All live approval comes from recorded physical evidence, never a CLI boolean."""
    import yaml
    path = Path(root)/'artifacts/phase-0/hardware.yaml'
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    if data['assessment']['overall_outcome'] != 'passed':
        raise ValueError('physical_gates_pending')
    if any(data[section].get('status') != 'passed' for section in
           ('payload', 'workspace', 'physical_stop', 'mount_inspection')):
        raise ValueError('physical_gates_pending')
    if any(data['candidate_locations'][name].get('status') != 'passed'
           for name in ('station_a', 'station_b')):
        raise ValueError('physical_gates_pending')
    digest = hashlib.sha256(raw).hexdigest()
    if library.backend != 'live' or library.data.get('hardware_sha256') != digest:
        raise ValueError('changed_setup')
    if not library.poses or not library.data.get('transitions'):
        raise ValueError('unapproved_path')
    for key in library.data['transitions']:
        source, destination = key.split('->')
        library.path(source, destination)
    return digest


class LeRobotAdapter(OwnedAdapter):
    backend = 'live'

    def __init__(self, root, port, identity='follower_arm', clock=time.monotonic):
        super().__init__(clock)
        self.root = Path(root)
        self.port = str(port)
        self.identity = identity
        self.robot = None
        self.lock = None
        self.library = None
        self.calibration_hash = None

    def connect(self, read_only=True):
        self.check_owner()
        if not read_only:
            raise ValueError('explicit_arming_required')
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        lock_id = hashlib.sha256(str(Path(self.port).resolve()).encode()).hexdigest()
        self.lock = open(Path(tempfile.gettempdir())/f'followme-{lock_id}.lock', 'a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = subprocess.run(['fuser', self.port], capture_output=True, text=True, timeout=5)
            if result.returncode not in (0, 1) or result.stdout.strip():
                raise RuntimeError('busy_device')
            self.robot = SO101Follower(SO101FollowerConfig(
                port=self.port, id=self.identity, cameras={}, use_degrees=True,
                disable_torque_on_disconnect=False, max_relative_target=2., num_read_retries=0))
            # Bound SDK packet waits locally; do not modify global SDK behavior.
            bus = self.robot.bus
            bus.default_timeout = 20
            handler = bus.port_handler
            original_timeout = handler.setPacketTimeout
            def bounded_timeout(packet_length):
                original_timeout(packet_length)
                handler.packet_timeout = min(handler.packet_timeout, 20.)
            handler.setPacketTimeout = bounded_timeout
            self.robot.bus.connect()
            serial_port = getattr(handler, 'ser', None)
            if serial_port is not None:
                serial_port.exclusive = True
            # This method pings/reads only. SDK Robot.connect/configure is never called.
            self.connected = True
            if not self.robot.bus.is_calibrated:
                raise RuntimeError('calibration_mismatch')
            self.calibration_hash = hashlib.sha256(self.robot.calibration_fpath.read_bytes()).hexdigest()
            self.read_joint_positions()
        except BaseException:
            self.disconnect()
            raise

    def read_joint_positions(self):
        self.check_owner()
        observation = self.robot.get_observation()
        return JointSnapshot({j: observation[f'{j}.pos'] for j in JOINTS}, self.clock())

    def arm(self, library):
        self.check_owner()
        verify_live_gate(self.root, library)
        if not self.robot.bus.is_calibrated or hashlib.sha256(
                self.robot.calibration_fpath.read_bytes()).hexdigest() != self.calibration_hash:
            raise ValueError('calibration_mismatch')
        snapshot = self.read_joint_positions()
        anchor = library.recognize(snapshot.positions)
        if anchor is None:
            raise ValueError('unknown_start_pose')
        bus = self.robot.bus
        for j in ARM_JOINTS:
            if bus.read('Operating_Mode', j, normalize=False) != 0:
                raise ValueError('unsupported_operating_mode')
            if bus.read('Torque_Enable', j, normalize=False):
                if abs(bus.read('Goal_Position', j)-snapshot.positions[j]) > library.limits.arrival_deg:
                    raise ValueError('nonstationary_goal')
        # Seed every disabled joint before enabling any; gripper is untouched.
        disabled = [j for j in ARM_JOINTS if not bus.read('Torque_Enable', j, normalize=False)]
        for j in disabled:
            bus.write('Goal_Position', j, snapshot.positions[j])
            self.physical_commands += 1
        for j in disabled:
            bus.write('Torque_Enable', j, 1, normalize=False)
        self.library = library
        self.armed = True
        after = self.read_joint_positions()
        if any(abs(after.positions[j]-snapshot.positions[j]) > library.limits.arrival_deg for j in ARM_JOINTS):
            self.armed = False
            raise RuntimeError('arming_deviation')

    def send(self, targets):
        self.check_owner()
        if not self.armed:
            raise RuntimeError('not_armed')
        verify_live_gate(self.root, self.library)
        if hashlib.sha256(self.robot.calibration_fpath.read_bytes()).hexdigest() != self.calibration_hash:
            raise RuntimeError('calibration_mismatch')
        validate_joints(targets, include_gripper=False)
        result = self.robot.send_action({f'{j}.pos': v for j, v in targets.items()})
        self.physical_commands += 1
        return {j: result[f'{j}.pos'] for j in ARM_JOINTS}

    def disconnect(self):
        self.check_owner()
        try:
            if self.robot is not None and self.robot.bus.is_connected:
                self.robot.bus.disconnect(disable_torque=False)
        finally:
            self.connected = False
            self.armed = False
            if self.lock is not None:
                self.lock.close()
                self.lock = None
