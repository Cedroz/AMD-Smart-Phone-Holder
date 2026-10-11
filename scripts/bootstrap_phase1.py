#!/usr/bin/env python3
"""Install locked candidates; promotion and clean rebuilds are explicit options."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import urllib.request
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]


def run(command,env=None):
    print('+ '+' '.join(map(str,command)),flush=True)
    subprocess.run(list(map(str,command)),cwd=ROOT,env=env,check=True)


def model():
    manifest=json.loads((ROOT/'models/manifest.json').read_text())
    target=ROOT/'models/face_landmarker.task'
    if target.exists():
        content=target.read_bytes()
        if len(content)!=manifest['size_bytes'] or hashlib.sha256(content).hexdigest()!=manifest['sha256']:
            raise ValueError('Cached model digest mismatch; refusing replacement')
        return
    with urllib.request.urlopen(manifest['source_url'],timeout=30) as response:
        content=response.read(32*1024*1024+1)
    if len(content)!=manifest['size_bytes'] or hashlib.sha256(content).hexdigest()!=manifest['sha256']:
        raise ValueError('Downloaded model digest mismatch')
    temporary=target.with_suffix('.tmp')
    temporary.write_bytes(content)
    temporary.replace(target)


def install(directory,kind):
    directory=Path(directory).absolute()
    if directory==ROOT/'.venv':
        raise ValueError('Working environment requires explicit promotion')
    marker=directory/'.followme-managed'
    if directory.exists() and any(directory.iterdir()) and not marker.exists():
        raise ValueError(f'Refusing to modify unmanaged environment: {directory}')
    if not (directory/'bin/python').exists():
        run([sys._base_executable,'-m','venv','--without-pip',directory])
    marker.write_text('FollowMe Phase 1 managed environment\n')
    python=directory/'bin/python'
    run([sys.executable,'-m','pip','--python',python,'install','--require-hashes','-r',ROOT/f'requirements/{kind}-build.lock'])
    run([sys.executable,'-m','pip','--python',python,'install','--require-hashes','--no-build-isolation','-r',ROOT/f'requirements/{kind}.lock'])
    run([sys.executable,'-m','pip','--python',python,'install','--no-deps','--no-build-isolation','-e',ROOT])
    (directory/'.followme-lock-sha256').write_text(hashlib.sha256((ROOT/f'requirements/{kind}.lock').read_bytes()).hexdigest()+'\n')
    return python


def check(runtime,vision,output):
    import yaml
    config=yaml.safe_load((ROOT/'config/environment.yaml').read_text())
    lib=Path(config['ffmpeg_library_dir'])
    if not (lib/'libavutil.so.60').exists():
        raise RuntimeError('FFmpeg 8 shared libraries missing; see native baseline evidence')
    env=os.environ.copy()
    env.pop('PYTHONPATH',None); env.pop('PYTHONHOME',None)
    env['LD_LIBRARY_PATH']=str(lib)+((':'+env['LD_LIBRARY_PATH']) if env.get('LD_LIBRARY_PATH') else '')
    run([runtime,ROOT/'scripts/check_phase1.py','--vision-python',vision,'--output',output],env)


def promote(vision,output):
    baseline={}
    for line in (ROOT/'requirements/baseline.txt').read_text().splitlines():
        if '==' in line:
            name,value=line.split('=='); baseline[name.lower().replace('_','-')]=value
    from importlib.metadata import distributions
    current={d.metadata['Name'].lower().replace('_','-'):d.version for d in distributions()}
    if any(current.get(k)!=v for k,v in baseline.items()):
        raise RuntimeError('Working runtime no longer matches baseline; promotion refused')
    # Compile preserved baseline first; all old versions remain constrained in lock.
    run([sys.executable,'-m','pip','install','--require-hashes','--no-build-isolation','-r',ROOT/'requirements/runtime.lock'])
    run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation','-e',ROOT])
    check(sys.executable,vision,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--promote',action='store_true')
    parser.add_argument('--verify-rebuild',action='store_true')
    args=parser.parse_args()
    if sys.version_info[:2]!=(3,12) or platform.system()!='Linux' or platform.machine()!='x86_64':
        raise RuntimeError('Committed locks require Linux x86_64 and Python 3.12')
    evidence=ROOT/'artifacts/phase-1/evidence'/datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%dT%H%M%S.%f%z')
    evidence.mkdir(parents=True,exist_ok=False)
    (ROOT/'artifacts/phase-1/latest-run.txt').write_text(str(evidence.relative_to(ROOT))+'\n')
    model()
    runtime=install(ROOT/'.venv-runtime-candidate','runtime')
    vision=install(ROOT/'.venv-vision','vision')
    check(runtime,vision,evidence/'candidate.json')
    if args.verify_rebuild:
        with tempfile.TemporaryDirectory(prefix='followme-rebuild-') as tmp:
            rebuilt_runtime=install(Path(tmp)/'runtime','runtime')
            rebuilt_vision=install(Path(tmp)/'vision','vision')
            check(rebuilt_runtime,rebuilt_vision,evidence/'rebuild.json')
    if args.promote:
        promote(vision,evidence/'working-runtime.json')


if __name__=='__main__':
    main()
