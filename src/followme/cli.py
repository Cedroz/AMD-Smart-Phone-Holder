"""Phase 1 commands deliberately perform no camera/serial opens."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='FollowMe environment preflight')
    parser.add_argument('command',choices=['inspect'])
    parser.add_argument('--root',type=Path,default=Path.cwd())
    args=parser.parse_args()
    from followme.environment import preflight
    print(json.dumps(preflight(args.root),indent=2))
