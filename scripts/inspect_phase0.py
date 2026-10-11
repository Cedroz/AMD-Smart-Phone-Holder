#!/usr/bin/env python3
"""Collect Phase 0 evidence without moving motors or changing calibration.

Default: filesystem/process inventory only. --read-motors opts into bus reads.
Existing evidence is never overwritten; each run gets a timestamped directory.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = Path.home() / '.cache/huggingface/lerobot/calibration'
ARMS = {
    'follower': ('5C4C128613', 'follower_arm', 'robots/so_follower'),
    'leader': ('5C4C128105', 'leader_arm', 'teleoperators/so_leader'),
}


def owners(path):
    result = subprocess.run(['fuser', str(path)], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(f'Cannot inspect owners of {path}: {result.stderr}')
    return [int(pid) for pid in result.stdout.split()]


def prohibit_write(*args, **kwargs):
    raise RuntimeError('Motor writes are prohibited during Phase 0 inspection')


def read_bus(bus):
    """Keep write methods blocked and preserve torque even on read failure."""
    bus.write = prohibit_write
    bus.sync_write = prohibit_write
    try:
        bus.connect()
        calibration = bus.read_calibration()
        return {
            'on_device_calibration': {k: asdict(v) for k, v in calibration.items()},
            'matches_saved_calibration': bus.is_calibrated,
            'positions': bus.sync_read('Present_Position'),
            'units': {name: 'percent' if name == 'gripper' else 'degrees'
                      for name in calibration},
            'torque_enabled': bus.sync_read('Torque_Enable', normalize=False),
            'motion_commands_sent': 0,
            'limitations': 'Register match does not establish mechanical accuracy or payload readiness.',
        }
    finally:
        # Close a partially opened port as well as a successfully connected bus.
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


def inspect_arm(role, read_motors):
    serial, identity, category = ARMS[role]
    path = Path(f'/dev/serial/by-id/usb-1a86_USB_Single_Serial_{serial}-if00')
    saved = CALIBRATION / category / f'{identity}.json'
    result = {'path': str(path), 'present': path.exists(), 'resolved_path': str(path.resolve()),
              'calibration_id': identity, 'calibration_path': str(saved), 'physical_identity':
              'Session follower unplug test; physical labels still require operator verification'}
    if saved.exists():
        content = saved.read_bytes()
        result['saved_calibration'] = json.loads(content)
        result['calibration_sha256'] = hashlib.sha256(content).hexdigest()
    else:
        result['error'] = 'Saved calibration missing; no serial inspection attempted'
        return result
    result['owners'] = owners(path) if path.exists() else []
    if read_motors and path.exists():
        if result['owners']:
            result['error'] = 'Serial adapter busy; inspection skipped'
        else:
            try:
                from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
                from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
                cls, config = ((SO101Follower, SO101FollowerConfig) if role == 'follower'
                               else (SO101Leader, SO101LeaderConfig))
                robot = cls(config(port=str(path), id=identity, use_degrees=True))
                # Never call robot.connect(): it configures motors/torque.
                result['bus_read'] = read_bus(robot.bus)
                if not result['bus_read']['matches_saved_calibration']:
                    result['error'] = 'On-device calibration does not match saved calibration; no changes made'
                if hashlib.sha256(saved.read_bytes()).hexdigest() != result['calibration_sha256']:
                    result['error'] = 'Saved calibration file changed during inspection'
            except Exception as exc:
                result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def camera_evidence(path, output):
    import cv2
    cap = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Camera did not provide a frame')
        image = output / f'{path.name}.jpg'
        if not cv2.imwrite(str(image), frame):
            raise RuntimeError('Could not save camera screenshot')
        return {'snapshot': str(image.relative_to(ROOT)), 'shape': list(frame.shape),
                'reported_fps': cap.get(cv2.CAP_PROP_FPS),
                'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode(errors='replace')}
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--read-motors', action='store_true')
    parser.add_argument('--capture-cameras', action='store_true')
    args = parser.parse_args()
    now = datetime.now(ZoneInfo('America/Los_Angeles'))
    output = ROOT / 'artifacts/phase-0/evidence' / now.strftime('%Y%m%dT%H%M%S.%f%z')
    output.mkdir(parents=True, exist_ok=False)
    report = {'recorded_at': now.isoformat(), 'operator': 'automated inspection',
              'read_motors_requested': args.read_motors, 'autonomous_motion_approved': False,
              'arms': {role: inspect_arm(role, args.read_motors) for role in ARMS}, 'cameras': []}
    for path in sorted(Path('/dev/v4l/by-id').glob('*video-index0')):
        camera = {'path': str(path), 'resolved_path': str(path.resolve()),
                  'role': None, 'role_status': 'pending', 'owners': owners(path)}
        if args.capture_cameras:
            if camera['owners']:
                camera['error'] = 'Camera busy; screenshot skipped'
            else:
                try:
                    camera.update(camera_evidence(path, output))
                except Exception as exc:
                    camera['error'] = f'{type(exc).__name__}: {exc}'
        report['cameras'].append(camera)
    report['inventory_complete'] = all(a['present'] for a in report['arms'].values()) and len(report['cameras']) == 2
    for command, filename in [(['lsusb'], 'lsusb.txt'), (['ip', '-brief', 'address'], 'network.txt')]:
        result = subprocess.run(command, capture_output=True, text=True)
        (output / filename).write_text(result.stdout + result.stderr)
    target = output / 'inventory.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(target)
    errors = [a['error'] for a in report['arms'].values() if 'error' in a]
    errors += [c['error'] for c in report['cameras'] if 'error' in c]
    if not report['inventory_complete']:
        errors.append('Expected two serial adapters and two camera capture devices')
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
