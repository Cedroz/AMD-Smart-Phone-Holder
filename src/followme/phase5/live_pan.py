"""Exclusive hardware owner for bounded, supervised pan/tilt following."""
import fcntl
import math
from pathlib import Path
import subprocess
import tempfile

from followme.robotics.adapters import OwnedAdapter
from followme.robotics.contracts import JointSnapshot
from .jog import PORT


class LivePanAdapter(OwnedAdapter):
    backend = 'live'

    def __init__(self, bounds, clock=None, relative_window=None, tilt_window=None, calibrated_bounds=False):
        kwargs = {} if clock is None else {'clock':clock}
        super().__init__(**kwargs)
        self.bounds = bounds
        self.relative_window = relative_window
        self.tilt_window = tilt_window
        self.calibrated_bounds = calibrated_bounds
        self.joint_bounds = {'shoulder_pan':bounds}
        self.bus = None
        self.robot = None
        self.lock_file = None

    def connect(self, read_only=True):
        self.check_owner()
        self.lock_file = open(Path(tempfile.gettempdir())/'followme-follower-commission.lock','w')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owners = subprocess.run(['fuser',PORT],capture_output=True,text=True)
        if owners.returncode not in (0,1) or owners.stdout.strip():
            raise RuntimeError('follower_busy')
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        self.robot = SO101Follower(SO101FollowerConfig(port=PORT,id='follower_arm',use_degrees=True,
                                                      disable_torque_on_disconnect=False))
        self.bus = self.robot.bus
        self.bus.connect()
        self.connected = True
        if not self.bus.is_calibrated:
            raise RuntimeError('calibration_mismatch')
        if self.relative_window is not None:
            origin = self.bus.read('Present_Position','shoulder_pan')
            if not math.isfinite(origin):
                raise RuntimeError('invalid_pan_origin')
            self.bounds[:] = [origin-self.relative_window,origin+self.relative_window]
        if self.tilt_window is not None:
            origin=self.bus.read('Present_Position','wrist_flex')
            if not math.isfinite(origin):
                raise RuntimeError('invalid_tilt_origin')
            self.joint_bounds['wrist_flex']=[origin-self.tilt_window,origin+self.tilt_window]
        if self.calibrated_bounds:
            for joint,bounds in self.joint_bounds.items():
                calibration = self.bus.calibration[joint]
                motor_id = self.bus.motors[joint].id
                endpoints = [self.bus._normalize({motor_id:raw})[motor_id]
                             for raw in (calibration.range_min,calibration.range_max)]
                bounds[:] = [max(bounds[0],min(endpoints)),min(bounds[1],max(endpoints))]
                if not all(math.isfinite(v) for v in bounds) or bounds[0]>=bounds[1]:
                    raise RuntimeError(f'{joint}_invalid_calibrated_range')
        self.read_joint_positions()
        self.armed = self.bus.read('Torque_Enable','shoulder_pan',normalize=False) == 1

    def read_joint_positions(self):
        self.check_owner()
        stamp = self.clock()
        values = self.bus.sync_read('Present_Position')
        now = self.clock()
        if now-stamp > .1:
            raise RuntimeError('motor_feedback_stalled')
        result = JointSnapshot(values,now)
        if not self.bounds[0]-.25 <= values['shoulder_pan'] <= self.bounds[1]+.25:
            raise RuntimeError('restore_pan_to_tested_window')
        for joint,bounds in self.joint_bounds.items():
            if not bounds[0]-.25 <= values[joint] <= bounds[1]+.25:
                raise RuntimeError(f'{joint}_outside_local_window')
            if self.bus.read('Status',joint,normalize=False) != 0:
                raise RuntimeError(f'{joint}_motor_status_fault')
        return result

    def arm(self):
        self.check_owner()
        self.hold()
        for joint in self.joint_bounds:
            self.bus.write('Torque_Limit',joint,400,normalize=False)
            self.bus.enable_torque(joint)
        self.armed = True

    def hold(self):
        self.check_owner()
        for joint in self.joint_bounds:
            raw = self.bus.read('Present_Position',joint,normalize=False)
            self.bus.write('Goal_Position',joint,raw,normalize=False)
            self.physical_commands += 1

    def send(self, targets):
        self.check_owner()
        for joint,bounds in self.joint_bounds.items():
            value=targets[joint]
            if not self.armed or not math.isfinite(value) or not bounds[0]<=value<=bounds[1]:
                raise RuntimeError('pan_target_inhibited')
        for joint in self.joint_bounds:
            motor_id=self.bus.motors[joint].id if hasattr(self.bus,'motors') else 1
            raw=self.bus._unnormalize({motor_id:targets[joint]})[motor_id]
            self.bus.write('Goal_Position',joint,raw,normalize=False)
            if self.bus.read('Goal_Position',joint,normalize=False)!=raw:
                raise RuntimeError('pan_goal_not_acknowledged')
            self.physical_commands += 1
        return dict(targets)

    def disconnect(self):
        self.check_owner()
        try:
            if self.bus and self.bus.is_connected:
                self.bus.disconnect(disable_torque=False)
        finally:
            self.connected = False
            if self.lock_file:
                self.lock_file.close()
