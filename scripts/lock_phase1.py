#!/usr/bin/env python3
"""Lock a pip-checked resolved environment to artifacts for this Python/platform.

Use PyPI-published SHA-256 digests instead of downloading every foreign-platform
wheel just to hash it. Source-only packages build with the locked build toolchain.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['runtime','vision'])
    parser.add_argument('python',type=Path)
    args=parser.parse_args()
    identity=json.loads(subprocess.check_output([str(args.python),'-I','-c',
        'import sys,json,platform;print(json.dumps([list(sys.version_info[:2]),platform.system(),platform.machine()]))'],text=True))
    if identity!=[[3,12],platform.system(),platform.machine()] or sys.version_info[:2]!=(3,12):
        raise RuntimeError('Lock generator and target must use Python 3.12 on the same platform')
    subprocess.run([str(args.python),'-m','pip','check'],check=True)
    packages=json.loads(subprocess.check_output([str(args.python),'-I','-c',
        'import json;from importlib.metadata import distributions;print(json.dumps({d.metadata["Name"]:d.version for d in distributions() if d.metadata["Name"]!="followme"}))'],text=True))
    tags={tag:i for i,tag in enumerate(sys_tags())}
    def artifact(item):
        name,version=item
        request=urllib.request.Request(f'https://pypi.org/pypi/{name}/{version}/json',headers={'User-Agent':'FollowMe-phase1-lock'})
        with urllib.request.urlopen(request,timeout=30) as response:
            metadata=json.load(response)
        candidates=[]
        sources=[]
        for file in metadata['urls']:
            if file.get('yanked'):continue
            if file['packagetype']=='sdist':sources.append(file)
            elif file['packagetype']=='bdist_wheel':
                _,_,_,wheel_tags=parse_wheel_filename(file['filename'])
                matches=[tags[tag] for tag in wheel_tags if tag in tags]
                if matches:candidates.append((min(matches),file['filename'],file))
        if candidates:
            selected=min(candidates,key=lambda value:(value[0],value[1]))[2]
        elif len(sources)==1:
            selected=sources[0]
        else:
            raise RuntimeError(f'No unambiguous artifact for {name}=={version}')
        return name,version,selected
    with ThreadPoolExecutor(max_workers=8) as pool:
        artifacts=list(pool.map(artifact,sorted(packages.items(),key=lambda pair:pair[0].lower())))
    header=f'# FollowMe {args.kind}: {platform.system()} {platform.machine()}, Python {sys.version_info.major}.{sys.version_info.minor}\n# Resolved environment validated with pip check; artifacts verified against PyPI digests.\n'
    def line(item):
        name,version,file=item
        return f'{name}=={version} --hash=sha256:{file["digests"]["sha256"]}\n'
    (ROOT/f'requirements/{args.kind}.lock').write_text(header+''.join(map(line,artifacts)))
    build_names={'pip','setuptools','wheel','packaging'}
    (ROOT/f'requirements/{args.kind}-build.lock').write_text(header+''.join(line(item) for item in artifacts if item[0].lower() in build_names))
    (ROOT/f'requirements/{args.kind}-artifacts.json').write_text(json.dumps({
        'python':sys.version,'platform':platform.platform(),'artifacts':[
            {'name':name,'version':version,'url':file['url'],'filename':file['filename'],
             'sha256':file['digests']['sha256'],'kind':file['packagetype']}
            for name,version,file in artifacts]},indent=2)+'\n')
    print(f'Locked {len(artifacts)} {args.kind} packages')


if __name__=='__main__':main()
