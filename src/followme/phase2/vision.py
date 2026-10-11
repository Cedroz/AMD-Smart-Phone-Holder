"""Phase 2 launch extension; Phase 1 client and worker stay unchanged."""
import os
from pathlib import Path
import socket
import subprocess

from followme.environment import resolve, verified_model
from followme.protocol import JsonSocket, VERSION
from followme.vision_client import VisionClient


class Phase2VisionClient(VisionClient):
    def start(self):
        if self.process is not None:
            raise RuntimeError('Worker already started')
        model = resolve(self.root,self.config['model_path'])
        manifest = resolve(self.root,self.config['model_manifest'])
        expected = verified_model(model,manifest)
        interpreter = resolve(self.root,self.config['vision_python'])
        parent,child = socket.socketpair()
        self.channel = JsonSocket(parent)
        env = os.environ.copy()
        for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','LD_LIBRARY_PATH'): env.pop(key,None)
        env['PYTHONNOUSERSITE']='1'
        command=[str(interpreter),'-I','-m','followme.phase2.vision_worker','--fd',str(child.fileno()),
                 '--model',str(model),'--manifest',str(manifest),'--threshold',str(self.config['face_threshold'])]
        if self.config.get('position_source') == 'shoulders':
            command += ['--pose-model',str(resolve(self.root,self.config['pose_model_path'])),
                        '--pose-manifest',str(resolve(self.root,self.config['pose_model_manifest']))]
        try:
            self.process=subprocess.Popen(command,pass_fds=(child.fileno(),),env=env,
                                          stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=None,cwd=self.root)
            child.close()
            ready=self.channel.receive(self.startup_timeout)
            if (ready.get('type')!='ready' or ready.get('protocol_version')!=VERSION
                or ready.get('model_sha256')!=expected['sha256']
                or ready.get('mediapipe_version')!=self.config['mediapipe_version']
                or Path(ready.get('interpreter','')).absolute()!=interpreter.absolute()):
                raise ValueError('Phase 2 worker startup mismatch')
            return ready
        except BaseException:
            child.close()
            self.close()
            raise
