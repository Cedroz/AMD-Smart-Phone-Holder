"""Controlled-time integration replay; never imports a live adapter."""
import json
from pathlib import Path
from followme.robotics.adapters import FakeAdapter
from followme.robotics.controller import Controller
from followme.robotics.trajectories import fake_library
from .planner import Integration
from .runtime import synthetic, SessionLog


class Clock:
    def __init__(self): self.now = 100.
    def __call__(self): return self.now
    def advance(self): self.now = round(self.now + .05, 9)


def run(config, output, input_path=None):
    output = Path(output)
    if any((output/name).exists() for name in ['integration.jsonl','controller.jsonl','results.json']):
        raise ValueError('replay_output_not_empty_use_new_directory')
    output.mkdir(parents=True, exist_ok=True)
    clock = Clock()
    adapter = FakeAdapter(clock)
    log = SessionLog(output/'integration.jsonl', clock)
    controller = Controller(adapter, fake_library(), clock=clock,
                            logger=SessionLog(output/'controller.jsonl', clock))
    integration = Integration(controller, config, clock, log)
    records = None
    if input_path:
        records = [json.loads(line) for line in Path(input_path).read_text().splitlines() if line.strip()]
        if not records:
            raise ValueError('empty_replay')
        from .planner import number
        times = [r.get('timestamp_s') for r in records]
        if any(not number(t) or t < 0 for t in times) or times != sorted(times):
            raise ValueError('unordered_or_invalid_replay')
        clock.now = times[0]
    controller.tick()
    results = []
    try:
        if input_path:
            # Input is a canonical observation snapshot + original replay event time.
            latest = None
            for record in records:
                t = record['timestamp_s']
                while clock()+.05 < t-1e-9:
                    clock.advance()
                    integration.tick(latest)
                    controller.tick()
                clock.now = t
                if 'snapshot' in record:
                    latest = record['snapshot']
                    integration.tick(latest)
                action = record.get('action')
                if action == 'heartbeat':
                    integration.heartbeat('replay')
                elif action == 'follow':
                    integration.set_mode('follow', 'replay')
                elif action == 'hold':
                    integration.hold('replay_hold')
                elif action == 'stop':
                    integration.stop()
                elif action == 'reset':
                    integration.reset()
                elif action is not None:
                    raise ValueError('unknown_replay_action')
                controller.tick()
                log('replay_input', **record)
        else:
            sequence = 0
            integration.heartbeat('replay')
            integration.tick(synthetic(clock(), sequence, 'station_a', clock()))
            integration.set_mode('follow', 'replay')
            controller.tick()
            for trial in range(10):
                target = 'station_b' if trial % 2 == 0 else 'station_a'
                zone = next(k for k,v in config['zone_to_pose'].items() if v == target)
                since = clock()
                arrived = False
                for step in range(200):
                    sequence += 1
                    integration.heartbeat('replay')
                    integration.tick(synthetic(clock(), sequence, zone, since))
                    controller.tick()
                    if controller.status().current_pose_name == target and controller.status().state == 'TRACKING':
                        # Observe settled arrival to start the cooldown.
                        clock.advance()
                        sequence += 1
                        integration.tick(synthetic(clock(), sequence, zone, since))
                        arrived = True
                        break
                    clock.advance()
                results.append({'trial_id':trial+1, 'target':target, 'arrived':arrived})
                if not arrived:
                    raise RuntimeError(f'trial_{trial+1}_failed:{integration.status()}')
        result = {'backend':'fake', 'synthetic':input_path is None, 'trials':results,
                  'correct_destinations':sum(r['arrived'] for r in results),
                  'decisions':integration.decisions, 'physical_commands':adapter.physical_commands,
                  'single_io_owner':len({call[1] for call in adapter.calls}) == 1,
                  'fault_reason':controller.status().fault_reason}
        (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    finally:
        controller.close()
