"""Supervised bounded commissioning jog, separate from autonomous following."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

PORT = '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128613-if00'


def targets(start, direction, degrees=1., steps=40):
    if not math.isfinite(start) or direction not in (-1, 1) or not 0 < degrees <= 40*360/4095:
        raise ValueError('invalid_jog')
    # Ease in and out over two seconds; maximum velocity is 0.94 deg/s.
    for step in range(1, steps+1):
        t = step / steps
        yield start + direction * degrees * (10*t**3 - 15*t**4 + 6*t**5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--direction', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--ticks', type=int, choices=range(1,41), default=11)
    parser.add_argument('--power-cutoff', help='Reachable physical actuator power cutoff, confirmed by operator')
    args = parser.parse_args()
    degrees = args.ticks*360/4095
    duration = max(2.,degrees*1.875)
    steps = math.ceil(duration/.05)
    if not args.execute:
        print(json.dumps({'joint':'shoulder_pan', 'relative_degrees':args.direction*degrees,
                          'duration_s':steps*.05, 'hardware_opened':False,
                          'other_joints':'unchanged', 'autonomous_tracking':False}))
        return
    if not args.power_cutoff or not args.power_cutoff.strip():
        parser.error('--execute requires the confirmed physical --power-cutoff')
    lock = open(Path(tempfile.gettempdir())/'followme-follower-commission.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owners = subprocess.run(['fuser', PORT], capture_output=True, text=True)
    if owners.returncode not in (0,1) or owners.stdout.strip():
        raise RuntimeError('Follower ownership could not be established')
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id='follower_arm', use_degrees=True))
    bus = robot.bus
    armed = False
    start = None
    try:
        bus.connect()  # Avoid Robot.connect(): do not configure or arm all motors.
        if not bus.is_calibrated:
            raise RuntimeError('calibration_mismatch')
        start = bus.read('Present_Position', 'shoulder_pan')
        start_raw = bus.read('Present_Position', 'shoulder_pan', normalize=False)
        if not isinstance(start, (int,float)) or not math.isfinite(start):
            raise RuntimeError('invalid_feedback')
        # Set the existing position before enabling this single motor's torque.
        bus.write('Goal_Position', 'shoulder_pan', start_raw, normalize=False)
        bus.write('Torque_Limit', 'shoulder_pan', 400, normalize=False)
        bus.enable_torque('shoulder_pan')
        armed = True
        begin = time.monotonic()
        for index, target in enumerate(targets(start,args.direction,degrees,steps),1):
            deadline = begin + index*.05
            time.sleep(max(0,deadline-time.monotonic()))
            if time.monotonic()-deadline > .1:
                raise RuntimeError('jog_loop_stalled')
            stamp = time.monotonic()
            measured = bus.read('Present_Position','shoulder_pan')
            if time.monotonic()-stamp > .1 or not math.isfinite(measured):
                raise RuntimeError('feedback_stale_or_invalid')
            if abs(measured-start) > degrees+.5:
                raise RuntimeError('jog_feedback_outside_envelope')
            raw_target = start_raw + round((target-start)*4095/360)
            if not 0 <= raw_target <= 4095:
                raise RuntimeError('raw_target_out_of_range')
            bus.write('Goal_Position','shoulder_pan',raw_target,normalize=False)
            actual_goal = bus.read('Goal_Position','shoulder_pan',normalize=False)
            if actual_goal != raw_target:
                raise RuntimeError('goal_register_mismatch')
            print(json.dumps({'step':index,'requested_ticks':raw_target,
                              'goal_register_ticks':actual_goal,'measured_degrees':measured}),flush=True)
        # Allow the servo to settle while continuing to check the same envelope.
        for _ in range(20):
            time.sleep(.05)
            measured = bus.read('Present_Position','shoulder_pan')
            if not math.isfinite(measured) or abs(measured-start) > degrees+.5:
                raise RuntimeError('jog_feedback_outside_envelope')
        measured = bus.read('Present_Position','shoulder_pan')
        if abs(measured-(start+args.direction*degrees)) > .5:
            raise RuntimeError('jog_did_not_converge')
        print(json.dumps({'result':'completed', 'start_degrees':start,
                          'measured_degrees':measured, 'power_cutoff':args.power_cutoff,
                          'torque_retained':True, 'autonomous_tracking':False}))
    except BaseException:
        if armed:
            try:
                measured = bus.read('Present_Position','shoulder_pan')
                if math.isfinite(measured) and abs(measured-start) <= degrees+.5:
                    bus.write('Goal_Position','shoulder_pan',measured)
            except Exception:
                print('Hold unconfirmed: use physical actuator power cutoff.')
        raise
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        lock.close()


if __name__ == '__main__':
    main()
