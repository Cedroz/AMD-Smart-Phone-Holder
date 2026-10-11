"""Filesystem-only preflight and model verification."""
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def verified_model(model, manifest):
    expected = json.loads(Path(manifest).read_text())
    if Path(model).stat().st_size != expected['size_bytes'] or digest(model) != expected['sha256']:
        raise ValueError('Model hash/size mismatch; expected manifest retained')
    return expected


def preflight(root, config_path=None):
    import yaml
    root = Path(root)
    config = yaml.safe_load(Path(config_path or root/'config/environment.yaml').read_text())
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported environment configuration')
    if config.get('dry_run') is not True or config.get('mode') != 'hold' or config.get('autonomous_motion_approved') is not False:
        raise ValueError('Phase 1 requires dry-run, Hold, and motion disabled')
    if set(config.get('cameras',{}))!={'external','wrist'} or set(config.get('arms',{}))!={'follower','leader'}:
        raise ValueError('Expected external/wrist cameras and follower/leader arms')
    interpreter = resolve(root, config['vision_python'])
    identity = json.loads(subprocess.check_output([str(interpreter), '-I', '-c',
        'import sys,json; print(json.dumps({"executable":sys.executable,"version":list(sys.version_info[:2])}))'],
        text=True, timeout=10))
    if identity['version'] != [3, 12]:
        raise ValueError('Vision interpreter must be Python 3.12')
    if Path(identity['executable']).absolute() != interpreter.absolute():
        raise ValueError('Vision interpreter identity mismatch')
    manifest = verified_model(resolve(root,config['model_path']), resolve(root,config['model_manifest']))
    baseline_dir = root / (root/'artifacts/phase-1/baseline-path.txt').read_text().strip()
    hashes = json.loads((baseline_dir/'calibration-hashes.json').read_text())
    for role,arm in hashes.items():
        if config['arms'][role]['calibration_id']!=Path(arm['path']).stem:
            raise ValueError('Calibration ID differs from verified baseline')
        if digest(arm['path']) != arm['sha256']:
            raise ValueError('Calibration file changed since baseline')
    devices = {}
    for role,path in {**config['cameras'], **{k:v['port'] for k,v in config['arms'].items()}}.items():
        p=Path(path)
        result=subprocess.run(['fuser', str(p)],capture_output=True,text=True,timeout=5) if p.exists() else None
        if result and result.returncode not in (0,1):
            raise RuntimeError('Cannot inspect device ownership')
        devices[role]={'path':str(p),'present':p.exists(),'resolved':str(p.resolve()),
                       'owners':[int(pid) for pid in result.stdout.split()] if result else []}
    return {'config':config,'vision_identity':identity,'model_sha256':manifest['sha256'],
            'devices':devices,'calibration_unchanged':True,'hardware_opened':False}
