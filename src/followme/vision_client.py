"""Supervise a vision subprocess and exchange a single outstanding request."""
import base64
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import uuid

from followme.environment import resolve, verified_model
from followme.protocol import JsonSocket, VERSION


class VisionClient:
    def __init__(self, root, config, startup_timeout=30, inference_timeout=5):
        self.root=Path(root)
        self.config=config
        self.startup_timeout=startup_timeout
        self.inference_timeout=inference_timeout
        self.process=None
        self.channel=None
        self.request_lock=threading.Lock()
        self.latest_lock=threading.Lock()
        self.latest={}

    def start(self):
        if self.process is not None:
            raise RuntimeError('Worker already started')
        model=resolve(self.root,self.config['model_path'])
        manifest=resolve(self.root,self.config['model_manifest'])
        expected=verified_model(model,manifest)
        interpreter=resolve(self.root,self.config['vision_python'])
        parent,child=socket.socketpair()
        self.channel=JsonSocket(parent)
        env=os.environ.copy()
        for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','LD_LIBRARY_PATH'):
            env.pop(key,None)
        env['PYTHONNOUSERSITE']='1'
        # Run the worker from this checkout's src/, not from whichever checkout the vision environment
        # was pip-installed from (otherwise a second clone silently runs the first clone's gesture code).
        bootstrap=("import runpy,sys;src,module=sys.argv[1],sys.argv[2];del sys.argv[1:3];"
                   "sys.path.insert(0,src);runpy.run_module(module,run_name='__main__',alter_sys=True)")
        try:
            self.process=subprocess.Popen([str(interpreter),'-I','-c',bootstrap,str(Path(self.root)/'src'),
                self.config.get('worker_module','followme.vision_worker'),
                '--fd',str(child.fileno()),'--model',str(model),'--manifest',str(manifest)],
                pass_fds=(child.fileno(),),env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                stderr=None,cwd=self.root)
            child.close()
            ready=self.channel.receive(self.startup_timeout)
            expected_version=self.config['mediapipe_version']
            if ready.get('type')!='ready' or ready.get('protocol_version')!=VERSION or ready.get('model_sha256')!=expected['sha256'] or ready.get('mediapipe_version')!=expected_version or Path(ready.get('interpreter','')).absolute()!=interpreter.absolute():
                raise ValueError('Vision startup handshake mismatch')
            return ready
        except BaseException:
            child.close()
            self.close()
            raise

    def submit_latest(self,role,sequence,capture_monotonic_ns,jpeg):
        if role not in ('external','wrist'):
            raise ValueError('Unknown camera role')
        with self.latest_lock:
            self.latest[role]=(sequence,capture_monotonic_ns,jpeg)

    def infer_latest(self,role):
        with self.latest_lock:
            frame=self.latest.pop(role,None)
        return None if frame is None else self.infer(role,*frame)

    def infer(self,role,sequence,capture_monotonic_ns,jpeg,face_center=None):
        if not self.request_lock.acquire(blocking=False):
            raise RuntimeError('One inference request may be outstanding')
        try:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError('Vision worker is not running')
            request={'type':'frame','request_id':uuid.uuid4().hex,'camera_role':role,'sequence':sequence,
                     'capture_monotonic_ns':capture_monotonic_ns,'jpeg_base64':base64.b64encode(jpeg).decode()}
            if face_center is not None:
                request['face_center']=face_center
            deadline=time.monotonic()+self.inference_timeout
            self.channel.send(request,self.inference_timeout)
            remaining=deadline-time.monotonic()
            if remaining<=0:
                raise TimeoutError('Inference deadline expired')
            reply=self.channel.receive(remaining)
            if reply.get('type')!='result' or any(reply.get(key)!=request[key] for key in ('request_id','camera_role','sequence','capture_monotonic_ns')):
                raise ValueError('Vision response identity mismatch')
            return reply
        except BaseException:
            self.close()
            raise
        finally:
            self.request_lock.release()

    def close(self):
        if self.channel:
            self.channel.close()
            self.channel=None
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            self.process=None
