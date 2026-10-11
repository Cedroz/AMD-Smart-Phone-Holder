"""Standalone Phase 3 commands; leaves the existing FollowMe CLI unchanged."""
import argparse
import json
from pathlib import Path
import time
from .adapters import FakeAdapter, LeRobotAdapter
from .contracts import Limits, TargetRequest
from .controller import Controller
from .configuration import load_robot_config
from .trajectories import fake_library


def main():
    parser = argparse.ArgumentParser(description='Phase 3 independent robot diagnostics')
    parser.add_argument('command', choices=['simulate', 'inspect'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--live-read', action='store_true', help='Explicit read-only follower inspection')
    parser.add_argument('--dry-run', action='store_true', help='Explicit default: never connect serial')
    args = parser.parse_args()
    config, limits = load_robot_config(args.root)
    if args.live_read and (args.command != 'inspect' or args.dry_run):
        parser.error('--live-read is only allowed for inspect without --dry-run')
    if args.live_read:
        adapter = LeRobotAdapter(args.root, config['follower_port'], config['calibration_id'])
        adapter.bind_owner()
        try:
            adapter.connect()
            snapshot = adapter.read_joint_positions()
            print(json.dumps({'backend': 'live', 'read_only': True, 'calibration_matches': True,
                'positions': dict(snapshot.positions), 'units': dict(snapshot.units),
                'physical_commands': adapter.physical_commands}, indent=2))
        finally:
            adapter.disconnect()
        return
    adapter = FakeAdapter()
    controller = Controller(adapter, fake_library(limits), limits)
    try:
        controller.tick()
        if args.command == 'simulate':
            controller.heartbeat()
            controller.update_tracking(time.monotonic())
            controller.set_mode('follow')
            controller.tick()
            result = controller.request_safe_pose(TargetRequest('demo-1', 'station_b', 'synthetic demo', time.monotonic()))
            if not result.accepted:
                raise RuntimeError(result.reason)
            deadline = time.monotonic()+6
            while time.monotonic() < deadline:
                controller.heartbeat()
                controller.update_tracking(time.monotonic())
                controller.tick()
                if controller.status().current_pose_name == 'station_b':
                    break
                time.sleep(1/controller.limits.hz)
            else:
                raise RuntimeError('simulation_did_not_arrive')
        status = controller.status()
        print(json.dumps({'backend': status.backend, 'state': status.state,
            'pose': status.current_pose_name, 'fault': status.fault_reason,
            'physical_commands': adapter.physical_commands,
            'simulated_commands': adapter.simulated_commands}, indent=2))
    finally:
        controller.close()


if __name__ == '__main__':
    main()
