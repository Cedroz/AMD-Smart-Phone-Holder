"""Immutable controller contracts and validated software limits."""
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping

ARM_JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
JOINTS = ARM_JOINTS + ('gripper',)


def frozen_values(values):
    return MappingProxyType(dict(values))


def validate_joints(values, include_gripper=True):
    expected = JOINTS if include_gripper else ARM_JOINTS
    if set(values) != set(expected):
        raise ValueError('invalid_joint_set')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
           for v in values.values()):
        raise ValueError('invalid_joint_value')


@dataclass(frozen=True)
class JointSnapshot:
    positions: Mapping[str, float]
    timestamp_s: float
    units: Mapping[str, str] = field(default_factory=lambda: {
        **{j: 'degrees' for j in ARM_JOINTS}, 'gripper': 'percent'})

    def __post_init__(self):
        validate_joints(self.positions)
        if not 0 <= self.positions['gripper'] <= 100:
            raise ValueError('invalid_gripper_percent')
        if not isfinite(self.timestamp_s) or self.units != {
                **{j: 'degrees' for j in ARM_JOINTS}, 'gripper': 'percent'}:
            raise ValueError('invalid_units_or_timestamp')
        object.__setattr__(self, 'positions', frozen_values(self.positions))
        object.__setattr__(self, 'units', frozen_values(self.units))


@dataclass(frozen=True)
class TargetRequest:
    request_id: str
    target_pose_name: str
    reason: str
    timestamp_s: float


@dataclass(frozen=True)
class RequestResult:
    accepted: bool
    reason: str
    request_id: str = ''


@dataclass(frozen=True)
class RobotStatus:
    backend: str
    mode: str = 'hold'
    state: str = 'HOLDING'
    measured: JointSnapshot | None = None
    current_pose_name: str | None = None
    target_pose_name: str | None = None
    fault_reason: str | None = None
    motion_enabled: bool = False
    hold_confirmed: bool = False
    worker_heartbeat_s: float = 0.0


@dataclass(frozen=True)
class Limits:
    velocity: float = 10.0
    acceleration: float = 20.0
    relative: float = 2.0
    hz: float = 20.0
    freshness_s: float = .5
    heartbeat_s: float = 3.0
    deviation_deg: float = 3.0
    deviation_s: float = .3
    arrival_deg: float = 2.0
    arrival_s: float = .25
    settling_s: float = 2.0

    def __post_init__(self):
        for value in vars(self).values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError('invalid_limits')
        if self.velocity > 10 or self.acceleration > 20 or self.relative > 2 or self.hz > 20:
            raise ValueError('software_ceiling_exceeded')
        if self.freshness_s > .5 or self.heartbeat_s > 3:
            raise ValueError('watchdog_ceiling_exceeded')


def fresh(timestamp, now, budget):
    return isinstance(timestamp, (int, float)) and isfinite(timestamp) and 0 <= now - timestamp <= budget
