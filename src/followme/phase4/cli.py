"""Run via python -m followme.phase4; no changes to other phase entrypoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

from . import library
from .replay import replay


def evidence_file(session, label, content):
    path = session/f'{label}-{uuid.uuid4().hex}.json'
    path.write_text(json.dumps(content, indent=2, allow_nan=False)+'\n')
    return path


def collect(root, duration):
    """Read-only candidate capture. Never arm or call Robot.connect()."""
    from followme.robotics.adapters import LeRobotAdapter
    adapter = LeRobotAdapter(root, '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128613-if00')
    adapter.bind_owner()
    samples = []
    try:
        adapter.connect(read_only=True)
        deadline = time.monotonic()+duration
        while time.monotonic() <= deadline:
            s = adapter.read_joint_positions()
            samples.append({'timestamp_s':s.timestamp_s, 'positions':dict(s.positions)})
            time.sleep(.05)
    finally:
        adapter.disconnect()
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--session', type=Path, help='Phase 4 session directory; never production config')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init')
    p = sub.add_parser('capture'); p.add_argument('--pose', choices=library.STATIONS, required=True)
    p.add_argument('--samples', type=Path); p.add_argument('--read-motors', action='store_true')
    p = sub.add_parser('record'); p.add_argument('--source', choices=library.STATIONS, required=True)
    p.add_argument('--destination', choices=library.STATIONS, required=True)
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--indices', required=True, help='Comma-separated operator-selected sample indices, including endpoints')
    sub.add_parser('check')
    sub.add_parser('evidence-template')
    p = sub.add_parser('replay'); p.add_argument('--source', choices=library.STATIONS, required=True)
    p.add_argument('--destination', choices=library.STATIONS, required=True)
    p.add_argument('--live', action='store_true'); p.add_argument('--dry-run', action='store_true')
    p.add_argument('--confirm-supervised', action='store_true')
    p.add_argument('--interrupt', choices=['hold','stop']); p.add_argument('--at', type=float, default=.5)
    p = sub.add_parser('cycles'); p.add_argument('--live', action='store_true')
    p.add_argument('--confirm-supervised', action='store_true')
    p = sub.add_parser('approve'); p.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == 'init':
        session = args.session or root/'artifacts/phase-4/evidence'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        session.mkdir(parents=True, exist_ok=False)
        raw = (root/'artifacts/phase-0/hardware.yaml').read_bytes()
        import yaml
        hardware = yaml.safe_load(raw)
        calibration = Path(hardware['devices']['arms']['follower']['calibration_path']).read_bytes()
        data = library.blank({'hardware_sha256':hashlib.sha256(raw).hexdigest()}, hashlib.sha256(calibration).hexdigest())
        library.save(session/'safe_poses.yaml', data)
        print(session)
        return 0
    if args.session is None:
        parser.error('--session is required for this command')
    session = args.session.resolve()
    path = session/'safe_poses.yaml'
    data = library.load(path)
    if args.command == 'capture':
        if bool(args.samples) == args.read_motors:
            parser.error('Choose --samples or --read-motors')
        if args.read_motors:
            samples = collect(root, 2.2)
        else:
            samples = library.read_samples(args.samples)
        raw = session/f'capture-{args.pose}-{uuid.uuid4().hex}.jsonl'
        raw.write_text(''.join(json.dumps(s)+'\n' for s in samples))
        library.capture(data, args.pose, samples, raw.name)
        library.save(path, data)
    elif args.command == 'record':
        samples = library.read_samples(args.samples)
        raw = session/f'route-{uuid.uuid4().hex}.jsonl'
        raw.write_text(''.join(json.dumps(s)+'\n' for s in samples))
        library.record(data,args.source,args.destination,samples,[int(i) for i in args.indices.split(',')],raw.name)
        library.save(path,data)
    elif args.command == 'check':
        library.check(data)
        print('Offline trajectory checks passed; physical paths remain unapproved.')
    elif args.command == 'evidence-template':
        template = {'backend':'live','content_sha256':library.content_digest(data),
                    'operator':None,'recorded_at':None,
                    **{k:False for k in ['phase0_passed','controller_passed','limits_measured',
                        'holds_60s_passed','both_paths_clear','fixed_gripper_passed','physical_stop_passed','recovery_passed']},
                    'cycles':[], 'interruptions':[], 'physical_attachments':[]}
        print(evidence_file(session,'acceptance-template',template))
    elif args.command in ('replay','cycles'):
        if args.live and not args.confirm_supervised:
            parser.error('--live requires --confirm-supervised and recorded physical gates')
        if getattr(args, 'dry_run', False) and args.live:
            parser.error('--dry-run and --live are mutually exclusive')
        if args.command == 'replay':
            result = replay(data,args.source,args.destination,root,args.live,args.interrupt,args.at)
            result_path = evidence_file(session,'replay',result)
        else:
            cycles = []
            for trial in range(1,11):
                legs = [replay(data,s,d,root,args.live) for s,d in [('station_a','station_b'),('station_b','station_a')]]
                passed = all(r['passed'] for r in legs)
                result_path = evidence_file(session,f'cycle-{trial}',{'backend':'live' if args.live else 'fake',
                    'content_sha256':library.content_digest(data),'passed':passed,'trial_id':trial,'legs':legs})
                cycles.append({'trial_id':trial,'log':str(result_path.relative_to(root)),
                               'sha256':hashlib.sha256(result_path.read_bytes()).hexdigest()})
                if not passed:
                    break
            result = {'passed':len(cycles)==10 and passed,'backend':'live' if args.live else 'fake','cycles':cycles}
            result_path = evidence_file(session,'cycles-summary',result)
        print(result_path)
        return 0 if result['passed'] else 1
    elif args.command == 'approve':
        evidence = json.loads(args.evidence.read_text())
        library.approve(data,evidence,root)
        # Store approval separately; never replace another phase's production config.
        library.save(session/f'approved-{library.content_digest(data)}.yaml',data)
        print('Approved Phase 4 artifact saved; production configuration unchanged.')
    return 0
