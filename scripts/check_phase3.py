#!/usr/bin/env python3
"""Independent fake-hardware acceptance and randomized cancellation benchmark."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import time
import unittest
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from followme.robotics.adapters import FakeAdapter
from followme.robotics.contracts import TargetRequest
from followme.robotics.controller import Controller, EventLog, RobotWorker
from followme.robotics.trajectories import fake_library


def wait_until(predicate, timeout=1.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.001)
    raise RuntimeError('benchmark_wait_timeout')


def cancellation_trials(count, directory):
    generator = random.Random(3101)
    measurements = []
    for index in range(count):
        delay = generator.uniform(0., .002)
        adapter = FakeAdapter(delay_s=delay)
        logger = EventLog(directory/'controller-events.jsonl')
        controller = Controller(adapter, fake_library(), logger=logger)
        worker = RobotWorker(controller)
        worker.start()
        try:
            wait_until(lambda: controller.status().current_pose_name == 'station_a')
            controller.heartbeat()
            controller.update_tracking(time.monotonic())
            controller.set_mode('follow')
            wait_until(lambda: controller.status().motion_enabled)
            result = controller.request_safe_pose(TargetRequest(str(index),'station_b','benchmark',time.monotonic()))
            if not result.accepted:
                raise RuntimeError(result.reason)
            wait_until(lambda: controller.status().state == 'REPOSITIONING')
            time.sleep(generator.uniform(0., .04))
            started = time.monotonic()
            stopped = bool(index % 2)
            if stopped:
                controller.stop_motion('benchmark')
            else:
                controller.request_hold('benchmark')
            wait_until(lambda: controller.status().state == ('STOPPED' if stopped else 'HOLDING'))
            elapsed = time.monotonic()-started
            if not controller.status().hold_confirmed or controller.status().fault_reason:
                raise RuntimeError('benchmark_hold_not_confirmed')
            if adapter.physical_commands != 0 or len({call[1] for call in adapter.calls}) != 1:
                raise RuntimeError('benchmark_ownership_failure')
            measurements.append({'trial': index, 'operation': 'stop' if stopped else 'hold',
                                 'io_delay_s': delay, 'handling_latency_s': elapsed,
                                 'physical_commands': adapter.physical_commands})
        finally:
            worker.close()
            if worker.error:
                raise RuntimeError(worker.error)
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=100)
    args = parser.parse_args()
    if args.trials < 100:
        parser.error('Acceptance requires at least 100 trials')
    directory = ROOT/'artifacts/phase-3/evidence'/datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%dT%H%M%S.%f%z')
    directory.mkdir(parents=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT/'test'), pattern='test_phase3*.py')
    with open(directory/'tests.txt','w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {'unit_tests_run': result.testsRun, 'unit_tests_passed': result.wasSuccessful(),
              'backend': 'fake', 'physical_commands': 0, 'live_movement_tested': False,
              'seed': 3101, 'healthy_io_delay_range_s': [0.,.002], 'cancel_target_s': .25}
    try:
        if not result.wasSuccessful():
            raise RuntimeError('unit_tests_failed')
        measurements = cancellation_trials(args.trials, directory)
        (directory/'cancellation-trials.json').write_text(json.dumps(measurements,indent=2)+'\n')
        report.update(trials=len(measurements), max_handling_latency_s=max(m['handling_latency_s'] for m in measurements))
        report['passed'] = report['max_handling_latency_s'] <= .25
    except Exception as exc:
        report.update(passed=False, error=str(exc))
    (directory/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'evidence_directory':str(directory), **report},indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
