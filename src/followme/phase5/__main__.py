import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Independent Phase 5 integration; defaults to fake hardware.')
    parser.add_argument('command', choices=['inspect', 'replay', 'run'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--input', type=Path, help='Canonical replay JSONL, with timestamp_s and snapshot')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--live', action='store_true')
    mode.add_argument('--dry-run', action='store_true', help='Default; no physical serial access')
    parser.add_argument('--perception', choices=['synthetic', 'native', 'proxy'], default='synthetic')
    parser.add_argument('--host')
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    from .config import load_config, inspect
    config = load_config(args.root, args.config)
    if args.live and args.command != 'run':
        parser.error('--live is only valid for run')
    if args.command == 'inspect':
        print(json.dumps(inspect(args.root, config), indent=2))
    elif args.command == 'replay':
        from .replay import run
        print(json.dumps(run(config, args.output or args.root/'artifacts/phase-5/evidence/replay', args.input), indent=2))
    else:
        import uvicorn
        import signal
        from .runtime import Runtime
        from .web import app_for
        runtime = Runtime(args.root, config, args.live, args.perception, args.output)
        try:
            runtime.start()
            # Uvicorn re-raises captured signals after restoring this handler.
            # Keep SIGTERM graceful so the robot and capture owners can close.
            signal.signal(signal.SIGTERM, lambda *_: None)
            uvicorn.run(app_for(runtime, config), host=args.host or config['host'],
                        port=args.port or config['port'], access_log=False)
        finally:
            runtime.close()


if __name__ == '__main__':
    main()
