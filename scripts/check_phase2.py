#!/usr/bin/env python3
"""Synthetic Phase 2 worker/IPC smoke; never opens a camera or robot."""
import json
from pathlib import Path
import time

from followme.environment import preflight
from followme.phase2.vision import Phase2VisionClient
from followme.phase2.tracking import validate_settings


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shoulders',action='store_true',help='Load the optional independent Pose Landmarker model')
    args=parser.parse_args()
    import cv2
    import numpy as np
    import yaml
    root=Path(__file__).resolve().parents[1]
    config=preflight(root)['config']
    settings=validate_settings(yaml.safe_load((root/'config/phase2/perception.yaml').read_text()))
    if args.shoulders:
        settings.update(position_source='shoulders',pose_model_path='models/phase2/pose_landmarker_lite.task',pose_model_manifest='models/phase2/pose_manifest.json')
    client=Phase2VisionClient(root,{**config,**settings})
    try:
        ready=client.start()
        ok,jpeg=cv2.imencode('.jpg',np.zeros((480,640,3),np.uint8))
        assert ok
        replies=[]
        for role in ('external','wrist'):
            reply=client.infer(role,1,time.monotonic_ns(),jpeg.tobytes())
            assert reply['error'] is None and reply['face_count']==0
            replies.append(reply)
        print(json.dumps({'passed':True,'ready':ready,'replies':replies,'serial_opened':False,'camera_opened':False},indent=2))
    finally: client.close()


if __name__=='__main__': main()
