#!/usr/bin/env python3
"""Fetch an optional official pose model into Phase 2's own model directory.

Does not switch perception sources, alter Phase 1's face model, or touch locks.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import urllib.request
from zoneinfo import ZoneInfo

from followme.environment import verified_model


def main():
    root=Path(__file__).resolve().parents[1]
    directory=root/'models/phase2'
    directory.mkdir(parents=True,exist_ok=True)
    model=directory/'pose_landmarker_lite.task'
    manifest=directory/'pose_manifest.json'
    if manifest.exists():
        verified_model(model,manifest)
        print('Verified cached optional pose model:',model)
        return
    if model.exists(): raise RuntimeError('Unmanifested model exists; retain and inspect it before proceeding')
    url='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
    with urllib.request.urlopen(url,timeout=30) as response: content=response.read(20*1024*1024+1)
    if not content or len(content)>20*1024*1024: raise ValueError('Invalid model size')
    record={'url':url,'version':'1','sha256':hashlib.sha256(content).hexdigest(),
            'size_bytes':len(content),'retrieved_at':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()}
    temporary=model.with_suffix('.tmp');temporary.write_bytes(content);temporary.replace(model)
    manifest.write_text(json.dumps(record,indent=2)+'\n')
    print('Optional pose model prepared:',model)


if __name__=='__main__': main()
