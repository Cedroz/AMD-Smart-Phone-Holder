"""Randomized real-thread integration interruption trials (fake I/O only)."""
import argparse
import json
from pathlib import Path
import random
import time
from followme.robotics.adapters import FakeAdapter
from followme.robotics.controller import Controller, RobotWorker
from followme.robotics.trajectories import fake_library
from .config import load_config
from .planner import Integration
from .runtime import synthetic


def benchmark(config, count=100, seed=5101):
    rng = random.Random(seed)
    trials = []
    for index in range(count):
        delay = rng.uniform(0, .002)
        adapter = FakeAdapter(delay_s=delay)
        controller = Controller(adapter, fake_library())
        worker = RobotWorker(controller)
        integration = Integration(controller, config)
        worker.start()
        try:
            deadline = time.monotonic()+2
            while controller.status().current_pose_name is None:
                if time.monotonic() > deadline: raise RuntimeError('worker_start_timeout')
                time.sleep(.005)
            now = time.monotonic()
            integration.heartbeat('bench')
            integration.tick(synthetic(now, 0, 'station_a', now))
            integration.set_mode('follow', 'bench')
            since, seq, deadline = time.monotonic(), 0, time.monotonic()+3
            while controller.status().state != 'REPOSITIONING':
                now = time.monotonic(); seq += 1
                integration.heartbeat('bench')
                integration.tick(synthetic(now, seq, 'station_b', since))
                if now > deadline: raise RuntimeError(f'motion_not_started:{integration.status()}')
                time.sleep(.01)
            # Randomize cancellation phase within a recognized near-anchor interval.
            until = time.monotonic()+rng.uniform(.01,.25)
            while time.monotonic() < until:
                seq += 1
                integration.heartbeat('bench')
                integration.tick(synthetic(time.monotonic(), seq, 'station_b', since))
                time.sleep(.005)
            trigger = time.monotonic()
            kind = 'hold' if index % 2 == 0 else 'stop'
            integration.hold('benchmark_hold') if kind == 'hold' else integration.stop()
            expected = 'HOLDING' if kind == 'hold' else 'STOPPED'
            while controller.status().state != expected:
                if time.monotonic()-trigger > .25: raise RuntimeError('cancellation_deadline_exceeded')
                time.sleep(.001)
            handled = next(e['timestamp_s'] for e in reversed(controller.log.events) if e['event']=='cancel')
            latency = handled-trigger
            owners = {call[1] for call in adapter.calls}
            trials.append({'trial_id':index+1, 'kind':kind, 'latency_s':latency,
                           'fake_io_delay_s':delay, 'physical_commands':adapter.physical_commands,
                           'single_io_owner':len(owners)==1, 'hold_confirmed':controller.status().hold_confirmed})
        finally:
            worker.close()
    return {'backend':'fake','seed':seed,'trials':trials,'passed':all(
        0 <= t['latency_s'] <= .25 and t['single_io_owner'] and t['hold_confirmed'] and
        t['physical_commands']==0 for t in trials), 'max_latency_s':max(t['latency_s'] for t in trials)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=int,default=100)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.trials < 1: parser.error('positive trial count required')
    root=Path(__file__).resolve().parents[3]
    result=benchmark(load_config(root),args.trials)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
