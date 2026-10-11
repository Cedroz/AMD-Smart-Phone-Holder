"""Read-only reuse of tracker/track.py's face/body detection and target selection."""
import importlib.util
import inspect
import threading
import time
import yaml

from followme.environment import preflight
from followme.phase2.cameras import Capture


class TrackerPerception:
    def __init__(self,root,log_dir=None,preview_sources=False,settings_override=None):
        self.root=root
        self.config=preflight(root)['config']
        self.settings={**yaml.safe_load((root/'config/phase2/perception.yaml').read_text()),**(settings_override or {})}
        spec=importlib.util.spec_from_file_location('followme_reference_tracker',root/'tracker/track.py')
        self.reference=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.reference)  # Does not run its main or open motors.
        self.has_jump_parameter='max_jump' in inspect.signature(self.reference.pick_target).parameters
        self.capture=None
        self.stopped=threading.Event()
        self.lock=threading.Lock()
        self.state={}
        self.jpeg=None
        self.worker_error=None
        self.aim=None
        self.count=0
        self.started=time.monotonic()

    def start(self):
        import cv2
        device=preflight(self.root)['devices']['external']
        if not device['present'] or device['owners']:
            raise RuntimeError('tracking_camera_missing_or_busy')
        r=self.reference
        if (self.root/'tracker/models/yolov8n_320.onnx').is_file() and (self.root/'tracker/models/face_detection_yunet_2023mar.onnx').is_file():
            self.detector=r.FaceAndBodyDetector(r.YOLO_MODEL,r.FACE_MODEL)
        else:
            self.detector=r.PersonDetector()
        self.capture=Capture(self.config['cameras']['external'],'external',self.settings)
        self.thread=threading.Thread(target=self.run,name='phase5-tracker-perception',daemon=True)
        self.thread.start()

    def run(self):
        import cv2
        import numpy as np
        sequence=-1
        last_seen=time.monotonic()
        try:
            while not self.stopped.is_set():
                start=time.monotonic()
                frame=self.capture.poll(time.monotonic_ns())
                if not frame or frame['sequence']==sequence:
                    self.stopped.wait(.02)
                    continue
                sequence=frame['sequence']
                image=cv2.imdecode(np.frombuffer(frame['jpeg'],np.uint8),cv2.IMREAD_COLOR)
                h,w=image.shape[:2]
                boxes=self.detector.detect(image)
                target=self.reference.pick_target(boxes,self.aim,.25*w) if self.has_jump_parameter else self.reference.pick_target(boxes,self.aim)
                if target is not None:
                    point=self.reference.aim_point(target)
                    self.aim=point if self.aim is None else tuple(.5*s+.5*a for s,a in zip(self.aim,point))
                    last_seen=start
                elif start-last_seen>2:
                    self.aim=None
                valid=target is not None and time.monotonic_ns()-frame['capture_monotonic_ns']<=500_000_000
                center=[self.aim[0]/w,self.aim[1]/h] if valid else None
                state={'camera_role':'external','sequence':sequence,'capture_monotonic_ns':frame['capture_monotonic_ns'],
                       'visible':valid,'tracking_valid':valid,'ambiguous':False,'selection_required':False,
                       'track_generation':0,'candidate_zone':None,'stable_zone':None,
                       'face_center_filtered':center if valid and target[4]=='face' else None,
                       'tracking_center':center,'target_kind':target[4] if target else None,
                       'detections':len(boxes),'target_selection':'largest_then_nearest',
                       'invalid_reason':None if valid else 'target_missing_or_stale'}
                annotated=self.reference.annotate(image,boxes,target,self.aim,.10,'FOLLOWME | '+(target[4] if target else 'searching'))
                jpeg=cv2.imencode('.jpg',annotated)[1].tobytes()
                with self.lock:
                    self.state,self.jpeg=state,jpeg
                    self.count+=1
                self.stopped.wait(max(0,.1-(time.monotonic()-start)))
        except Exception as exc:
            self.worker_error=str(exc)

    def status(self):
        with self.lock:
            state=dict(self.state)
            if time.monotonic_ns()-state.get('capture_monotonic_ns',0)>500_000_000:
                state.update(visible=False,tracking_valid=False,invalid_reason='camera_stale')
            return {'state':state,'source_modes':{'external':'native_camera'},'worker_error':self.worker_error,
                    'metrics':{'external':{'inferences':self.count,'error':self.capture.error if self.capture else 'camera_not_started',
                                           'inference_fps':self.count/max(.001,time.monotonic()-self.started)}},
                    'settings':self.settings,'detector':'track.py face/body','mode':'hold'}

    def preview(self,role):
        with self.lock:
            if role!='external' or self.jpeg is None or time.monotonic_ns()-self.state.get('capture_monotonic_ns',0)>500_000_000:
                return None
            return self.state['sequence'],self.jpeg

    def reacquire(self):
        self.aim=None

    def close(self):
        self.stopped.set()
        if hasattr(self,'thread'):self.thread.join(2)
        if self.capture:self.capture.close()
