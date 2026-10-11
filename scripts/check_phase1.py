#!/usr/bin/env python3
"""Check environments and synthetic worker inference without opening hardware."""
import argparse
import base64
from datetime import datetime
from importlib.metadata import version, distributions
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vision-python',default='.venv-vision/bin/python')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    from followme.environment import preflight
    from followme.vision_client import VisionClient
    import yaml
    config=yaml.safe_load((root/'config/environment.yaml').read_text())
    config['vision_python']=str((root/args.vision_python).absolute())
    # Preserve the configuration file: alternate rebuild interpreter is a test override.
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w',suffix='.yaml') as tmp:
        yaml.safe_dump(config,tmp); tmp.flush()
        report=preflight(root,tmp.name)
    if not all(device['present'] for device in report['devices'].values()):
        raise RuntimeError('Expected camera/serial device absent; no hardware opened')
    providers=[d.metadata['Name'] for d in distributions() if d.metadata['Name'].lower().startswith('opencv-')]
    if providers!=['opencv-python-headless']:
        raise RuntimeError(f'Runtime OpenCV providers: {providers}')
    if any(d.metadata['Name'].lower()=='mediapipe' for d in distributions()):
        raise RuntimeError('MediaPipe must remain isolated from runtime')
    if version('lerobot')!='0.6.1':
        raise RuntimeError('LeRobot version changed')
    baseline={line.split('==')[0].lower().replace('_','-'):line.split('==')[1]
              for line in (root/'requirements/baseline.txt').read_text().splitlines() if '==' in line}
    current={d.metadata['Name'].lower().replace('_','-'):d.version for d in distributions()}
    if any(current.get(name)!=value for name,value in baseline.items()):
        raise RuntimeError('Runtime baseline package version changed')
    import torch,torchcodec,av,cv2,fastapi,uvicorn
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    import numpy as np
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    subprocess.run([config['vision_python'],'-m','pip','check'],check=True)
    vision_inventory=json.loads(subprocess.check_output([config['vision_python'],'-I','-c',
        'import json,sys; from importlib.metadata import distributions; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in distributions()}}))'],text=True))
    vp=[n for n in vision_inventory['packages'] if n.lower().startswith('opencv-')]
    if vp!=['opencv-contrib-python'] or any(n.lower()=='lerobot' for n in vision_inventory['packages']):
        raise RuntimeError('Vision environment isolation check failed')
    blank=np.zeros((480,640,3),dtype=np.uint8)
    ok,jpeg=cv2.imencode('.jpg',blank)
    if not ok: raise RuntimeError('Synthetic JPEG encoding failed')
    client=VisionClient(root,config)
    try:
        report['handshake']=client.start()
        start=time.monotonic()
        replies=[]
        for role in ('external','wrist'):
            for seq in (1,2):
                timestamp=time.monotonic_ns()
                response=client.infer(role,seq,timestamp,jpeg.tobytes())
                if response.get('error') or response.get('face_count')!=0:
                    raise RuntimeError(f'Synthetic inference failed: {response}')
                replies.append(response)
        invalid=client.infer('external',3,time.monotonic_ns(),b'not a JPEG')
        if not invalid.get('error'): raise RuntimeError('Invalid JPEG accepted')
        report['worker_smoke']={'inferences':len(replies),'blank_face_counts':[r['face_count'] for r in replies],
                                'invalid_jpeg_rejected':True,'elapsed_s':time.monotonic()-start}
    finally:
        client.close()
    report.update({'recorded_at':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
        'runtime_python':sys.executable,'platform':platform.platform(),
        'torch_runtime':torch.__version__,'gpu_available':torch.cuda.is_available(),
        'torchcodec':torchcodec.__version__,'vision_inventory':vision_inventory,
        'motion_commands_sent':0,'software_checks_passed':True,
        'physical_gate':'pending; environment checks do not approve movement'})
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
