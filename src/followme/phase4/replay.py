"""Standalone supervised harness reusing, but never changing, Phase 3."""
import hashlib
import signal
import time
import uuid
from pathlib import Path

from followme.robotics.adapters import FakeAdapter, LeRobotAdapter, verify_live_gate
from followme.robotics.contracts import Limits, TargetRequest
from followme.robotics.controller import Controller, EventLog
from followme.robotics.trajectories import PoseLibrary
from .library import check, project, content_digest


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value


class GripperGuard(LeRobotAdapter):
    def __init__(self, *args, target, tolerance, **kwargs):
        super().__init__(*args, **kwargs)
        self.gripper_target, self.gripper_tolerance = target, tolerance

    def read_joint_positions(self):
        snapshot = super().read_joint_positions()
        if abs(snapshot.positions['gripper']-self.gripper_target) > self.gripper_tolerance:
            raise RuntimeError('fixed_gripper_deviation')
        return snapshot


def replay(data, source, destination, root, live=False, interrupt=None, at=.5,
           log_path=None):
    check(data)
    if not 0 < at < 1:
        raise ValueError('interrupt_fraction_out_of_range')
    if interrupt not in (None, 'hold', 'stop'):
        raise ValueError('invalid_interrupt')
    backend = 'live' if live else 'fake'
    limits = Limits(**data['limits'])
    root = Path(root).resolve()
    hardware_path = root/'artifacts/phase-0/hardware.yaml'
    revision = hashlib.sha256(hardware_path.read_bytes()).hexdigest() if live else 'offline'
    library = PoseLibrary(project(data, backend, revision), limits, backend)
    segments = library.path(source, destination)
    if live:
        if data['setup'].get('hardware_sha256') != revision:
            raise ValueError('setup_changed_since_teaching')
        verify_live_gate(root, library)
        clock = time.monotonic
        adapter = GripperGuard(root, '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128613-if00',
                              clock=clock, target=data['gripper']['target'],
                              tolerance=data['gripper']['tolerance'])
    else:
        clock = Clock()
        adapter = FakeAdapter(clock)
        adapter.positions.update(data['poses'][source]['joints'])
        adapter.positions['gripper'] = data['gripper']['target']
    logger = EventLog(log_path, clock)
    controller = Controller(adapter, library, limits, clock, logger)
    previous_handler = None
    interrupted_at = None
    latency = None
    try:
        controller.start()
        if controller.status().state == 'FAULT':
            raise RuntimeError(controller.status().fault_reason)
        if live and adapter.calibration_hash != data['calibration_sha256']:
            raise ValueError('calibration_changed_since_teaching')
        if library.recognize(controller.status().measured.positions) != source:
            raise ValueError('supervised_restoration_required')
        controller.request_arm()
        controller.heartbeat()
        controller.update_tracking(clock(), True)
        controller.set_mode('follow')
        controller.tick()
        request = TargetRequest(uuid.uuid4().hex, destination, 'supervised_phase4_validation', clock())
        accepted = controller.request_safe_pose(request)
        if not accepted.accepted:
            raise RuntimeError(accepted.reason)
        started = clock()
        duration = sum(s.duration_s for s in segments) + len(segments)*(limits.arrival_s+.1)
        deadline = started+duration+limits.settling_s+2
        if live:
            previous_handler = signal.signal(signal.SIGINT, lambda *_: controller.stop_motion('operator_ctrl_c'))
        while clock() < deadline:
            if live:
                time.sleep(1/limits.hz)
            else:
                clock.value += 1/limits.hz
            controller.heartbeat()
            controller.update_tracking(clock(), True)
            if interrupt and interrupted_at is None and clock()-started >= duration*at:
                interrupted_at = clock()
                controller.request_hold() if interrupt == 'hold' else controller.stop_motion()
            controller.tick()
            status = controller.status()
            if status.state in ('FAULT', 'STOPPED', 'HOLDING'):
                if interrupted_at is not None:
                    latency = clock()-interrupted_at
                passed = interrupted_at is not None and status.hold_confirmed and status.fault_reason is None
                break
            if status.current_pose_name == destination and status.target_pose_name is None:
                passed = interrupt is None
                break
        else:
            passed = False
        return {'backend':backend, 'content_sha256':content_digest(data),
                'direction':f'{source}->{destination}', 'passed':passed,
                'interrupt':interrupt, 'latency_s':latency, 'state':controller.status().state,
                'interrupt_fraction':at if interrupt else None,
                'fault_reason':controller.status().fault_reason,
                'physical_commands':adapter.physical_commands,
                'events':list(logger.events), 'duration_s':clock()-started}
    finally:
        try:
            controller.close()
        finally:
            if previous_handler is not None:
                signal.signal(signal.SIGINT, previous_handler)
