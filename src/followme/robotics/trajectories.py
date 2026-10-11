"""Directed approved paths and continuous quintic derivative checks."""
from dataclasses import dataclass
import hashlib
import json
from math import ceil, sqrt, isfinite
from pathlib import Path
from .contracts import ARM_JOINTS, Limits, frozen_values, validate_joints


@dataclass(frozen=True)
class Segment:
    start: object
    end: object
    duration_s: float

    def sample(self, elapsed):
        u = max(0., min(1., elapsed / self.duration_s))
        s = 10*u**3 - 15*u**4 + 6*u**5
        return {j: self.start[j] + s*(self.end[j]-self.start[j]) for j in ARM_JOINTS}


class PoseLibrary:
    def __init__(self, data, limits=Limits(), backend='fake'):
        # Copy caller-owned configuration: later mutation must not change approved paths.
        self.data = json.loads(json.dumps(data))
        self.limits = limits
        self.backend = backend
        self.source = self.data.get('backend')
        self.poses = self.data.get('poses', {})
        self.bounds = self.data.get('joint_limits', {})
        if self.source != backend:
            raise ValueError('library_backend_mismatch')
        if self.poses:
            if self.data.get('units') != 'degrees' or set(self.bounds) != set(ARM_JOINTS):
                raise ValueError('invalid_units_or_limits')
            for bound in self.bounds.values():
                if len(bound) != 2 or any(not isfinite(v) for v in bound) or bound[0] >= bound[1]:
                    raise ValueError('invalid_joint_limits')
            for pose in self.poses.values():
                self.validate_target(pose['joints'])
                if pose.get('gripper_unchanged') is not True:
                    raise ValueError('gripper_must_remain_fixed')
        self.digest = hashlib.sha256(json.dumps(self.data, sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path, limits=Limits(), backend='live'):
        import yaml
        return cls(yaml.safe_load(Path(path).read_text()), limits, backend)

    def validate_target(self, values):
        validate_joints(values, include_gripper=False)
        for j in ARM_JOINTS:
            lo, hi = self.bounds[j]
            if not lo <= values[j] <= hi:
                raise ValueError('joint_limit')

    def recognize(self, positions):
        matches = [name for name, pose in self.poses.items()
                   if all(abs(positions[j]-pose['joints'][j]) <= self.limits.arrival_deg for j in ARM_JOINTS)]
        return matches[0] if len(matches) == 1 else None

    def path(self, source, destination):
        transition = self.data.get('transitions', {}).get(f'{source}->{destination}')
        if not transition or transition.get('approved') is not True:
            raise ValueError('unapproved_path')
        if self.backend == 'live':
            revision = self.data.get('hardware_sha256')
            if not revision or transition.get('hardware_sha256') != revision:
                raise ValueError('changed_setup')
            for name in (source, destination):
                if self.poses[name].get('approved') is not True or self.poses[name].get('hardware_sha256') != revision:
                    raise ValueError('unapproved_pose')
        points = [self.poses[source]['joints']] + transition.get('waypoints', []) + [self.poses[destination]['joints']]
        durations = transition.get('durations_s', [])
        if len(durations) != len(points)-1:
            raise ValueError('invalid_durations')
        segments = []
        for start, end, duration in zip(points, points[1:], durations):
            self.validate_target(start)
            self.validate_target(end)
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not isfinite(duration) or duration <= 0:
                raise ValueError('invalid_duration')
            for j in ARM_JOINTS:
                delta = abs(end[j]-start[j])
                minimum = max(1.875*delta/self.limits.velocity,
                              sqrt((10*sqrt(3)/3)*delta/self.limits.acceleration))
                if duration + 1e-9 < minimum:
                    raise ValueError('derivative_limit')
            segment = Segment(frozen_values(start), frozen_values(end), duration)
            previous = segment.sample(0)
            for i in range(1, ceil(duration*self.limits.hz)+1):
                sample = segment.sample(i/self.limits.hz)
                self.validate_target(sample)
                if any(abs(sample[j]-previous[j]) > self.limits.relative for j in ARM_JOINTS):
                    raise ValueError('relative_limit')
                previous = sample
            segments.append(segment)
        return tuple(segments)


def fake_library(limits=Limits()):
    a = dict.fromkeys(ARM_JOINTS, 0.)
    b = {**a, 'shoulder_pan': 10.}
    return PoseLibrary({'backend': 'fake', 'units': 'degrees',
        'joint_limits': {j: [-30., 30.] for j in ARM_JOINTS},
        'poses': {name: {'joints': joints, 'gripper_unchanged': True}
                  for name, joints in [('station_a', a), ('station_b', b)]},
        'transitions': {f'{s}->{d}': {'approved': True, 'waypoints': [], 'durations_s': [3.]}
                        for s, d in [('station_a', 'station_b'), ('station_b', 'station_a')]}}, limits)
