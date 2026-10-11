"""Camera-only runtime, independent freshness monitoring and inference scheduler."""
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import threading
import time
from zoneinfo import ZoneInfo

from followme.phase2.cameras import Capture
from followme.environment import preflight
from followme.phase2.tracking import Tracker, Zones, finite, validate_settings
from followme.phase2.vision import Phase2VisionClient


class Runtime:
    def __init__(self, root, log_dir=None, preview_sources=False, settings_override=None):
        import yaml
        self.root = Path(root)
        report = preflight(self.root)
        self.config = report['config']
        self.preview_sources = preview_sources
        self.source_modes = {}
        self.settings = validate_settings({**yaml.safe_load((self.root/'config/phase2/perception.yaml').read_text()),**(settings_override or {})})
        self.zone_path = self.root/'config/phase2/zones.yaml'
        zones = Zones(yaml.safe_load(self.zone_path.read_text()),self.config['cameras']['external'],self.settings)
        self.tracker = Tracker(self.settings, zones)
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.frames = {}
        self.replies = {}
        self.captures = {}
        self.inflight = None
        self.worker_error = None
        self.threads = []
        self.stats = {role:{'capture_frames':0,'inferences':0,'ages_ms':[],'durations_ms':[], 'dropped_frames':0,'last_sequence':0} for role in ('external','wrist')}
        self.started = time.monotonic()
        self.log_dir = Path(log_dir) if log_dir else self.root/'artifacts/phase-2/evidence'/datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%dT%H%M%S.%f%z')
        self.log_dir.mkdir(parents=True,exist_ok=True)
        self.log = (self.log_dir/'observations.jsonl').open('a',buffering=1)
        self.last_status = None
        self.samples = self.valid_samples = 0
        self.last_published = -1
        self.client = Phase2VisionClient(self.root,{**self.config,**self.settings})
        self.record({'type':'session','config':self.settings,'zones':zones.config,
                     'model_sha256':report['model_sha256'],'calibration_unchanged':report['calibration_unchanged'],
                     'config_hashes':{name:hashlib.sha256((self.root/'config'/name).read_bytes()).hexdigest() for name in ('environment.yaml','phase2/perception.yaml','phase2/zones.yaml')},
                     'camera_paths':self.config['cameras'],'preview_sources':preview_sources,'motion_enabled':False})

    def record(self, event):
        with self.lock:
            if not self.log.closed:
                self.log.write(json.dumps({'logged_monotonic_ns':time.monotonic_ns(),**event},allow_nan=False)+'\n')

    def start(self):
        # Recheck ownership immediately before capture; never kill another owner.
        report = preflight(self.root)
        for role in ('external','wrist'):
            device = report['devices'][role]
            if device['owners'] and not self.preview_sources:
                raise RuntimeError(f'{role} camera busy: PIDs {device["owners"]}; stop identified legacy previews first')
            if not device['present'] and role == 'external' and not self.preview_sources:
                raise RuntimeError('External camera missing')
        self.client.start()
        try:
            for role in ('external','wrist'):
                if report['devices'][role]['present'] or self.preview_sources:
                    use_proxy = self.preview_sources and (report['devices'][role]['owners'] or not report['devices'][role]['present'])
                    self.source_modes[role] = 'legacy_preview_proxy' if use_proxy else 'native_camera'
                    source = f'http://127.0.0.1:{8000 if role=="external" else 8001}/' if use_proxy else self.config['cameras'][role]
                    self.captures[role] = Capture(source,role,self.settings)
            for function in (self.monitor, self.inference):
                thread = threading.Thread(target=self.worker,args=(function,),daemon=True)
                self.threads.append(thread)
                thread.start()
        except BaseException:
            self.close()
            raise

    def worker(self, function):
        try:
            function()
        except Exception as exc:
            if self.stopped.is_set():
                return
            with self.lock:
                self.worker_error = f'{type(exc).__name__}: {exc}'
                self.tracker.invalidate(time.monotonic_ns(),'worker_failure',clear_selection=True)
                self.record({'type':'worker_failure','error':self.worker_error})

    def monitor(self):
        while not self.stopped.wait(.025):
            now = time.monotonic_ns()
            with self.lock:
                for role, capture in self.captures.items():
                    frame = capture.poll(now)
                    if frame and (role not in self.frames or frame['sequence'] != self.frames[role]['sequence']):
                        self.frames[role] = frame
                        stat = self.stats[role]
                        stat['capture_frames'] += max(0,frame['sequence']-stat['last_sequence'])
                        stat['last_sequence'] = frame['sequence']
                    if role == 'external' and capture.error:
                        self.tracker.invalidate(now,'external_camera_failure',clear_selection=True)
                self.tracker.tick(now)
                snapshot = self.tracker.snapshot(now)
                signature = (snapshot['invalid_reason'],snapshot['track_generation'],snapshot['candidate_zone'],snapshot['stable_zone'],snapshot['selection_required'])
                if signature != self.last_status:
                    self.record({'type':'state_transition','state':snapshot})
                    self.last_status = signature
                if now-self.last_published >= 200_000_000:
                    self.samples += 1
                    self.valid_samples += int(snapshot['tracking_valid'])
                    self.record({'type':'state','state':snapshot})
                    self.last_published = now

    def inference(self):
        due = {'external':0.,'wrist':0.}
        sent = {'external':-1,'wrist':-1}
        while not self.stopped.wait(.005):
            if self.worker_error:
                return
            now = time.monotonic()
            role = None
            with self.lock:
                for candidate in ('external','wrist'):
                    hz = self.settings[f'{candidate}_hz']
                    frame = self.frames.get(candidate)
                    if hz and frame and now>=due[candidate] and frame['sequence']>sent[candidate] and time.monotonic_ns()-frame['capture_monotonic_ns']<=self.tracker.freshness and not self.captures[candidate].error:
                        role = candidate
                        break
                if not role:
                    continue
                frame = self.frames[role]
                sent[role] = frame['sequence']
                due[role] = max(now,due[role]+1/self.settings[f'{role}_hz'])
                self.inflight = role
                if self.settings[f'{role}_hz'] and due[role]<=now:
                    due[role] = now+1/self.settings[f'{role}_hz']
            jpeg = frame['jpeg']
            if self.settings['inference_width'] < self.settings['capture_width']:
                import cv2
                import numpy as np
                image = cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
                w = self.settings['inference_width']
                image = cv2.resize(image,(w,round(image.shape[0]*w/image.shape[1])))
                jpeg = cv2.imencode('.jpg',image)[1].tobytes()
            start = time.monotonic_ns()
            reply = self.client.infer(role,frame['sequence'],frame['capture_monotonic_ns'],jpeg)
            finished = time.monotonic_ns()
            with self.lock:
                stat = self.stats[role]
                stat['inferences'] += 1
                stat['ages_ms'].append((finished-frame['capture_monotonic_ns'])/1e6)
                stat['durations_ms'].append((finished-start)/1e6)
                # Bounded metric reservoirs; total counters retained.
                stat['ages_ms'] = stat['ages_ms'][-3000:]
                stat['durations_ms'] = stat['durations_ms'][-3000:]
                stat['dropped_frames'] = max(0,stat['capture_frames']-stat['inferences'])
                logged_reply=dict(reply)
                # Replay needs bbox extrema, not hundreds of per-face landmarks.
                # Preserve malformed inputs rather than hiding validation failures.
                faces=reply.get('landmarks',[])
                if all(face and all(isinstance(p,dict) and all(finite(p.get(k)) and 0<=p[k]<=1 for k in ('x','y')) for p in face) for face in faces):
                    logged_reply['landmarks']=[[{k:min(p[k] for p in face) for k in ('x','y')},
                                               {k:max(p[k] for p in face) for k in ('x','y')}] for face in faces]
                    logged_reply['landmark_representation']='bounding_box_extrema'
                self.record({'type':'observation','reply':logged_reply,'processed_monotonic_ns':finished})
                self.replies[role] = reply
                if role == 'external' and not self.worker_error and not self.captures[role].error:
                    self.tracker.update(reply,finished)
                if reply.get('error'):
                    raise RuntimeError(reply['error'])
                self.inflight = None

    def status(self):
        with self.lock:
            now = time.monotonic_ns()
            elapsed = max(.001,time.monotonic()-self.started)
            metrics = {}
            for role, stat in self.stats.items():
                ages = sorted(stat['ages_ms'])
                metrics[role] = {k:v for k,v in stat.items() if k not in ('ages_ms','durations_ms')}
                metrics[role].update(capture_fps=stat['capture_frames']/elapsed,inference_fps=stat['inferences']/elapsed,
                                     p50_age_ms=ages[len(ages)//2] if ages else None,
                                     p95_age_ms=ages[min(len(ages)-1,int(.95*len(ages)))] if ages else None,
                                     metadata=self.captures[role].metadata if role in self.captures else {},
                                     error=self.captures[role].error if role in self.captures else 'Camera missing')
            return {'state':self.tracker.snapshot(now),'motion_enabled':False,'mode':'hold',
                    'source_modes':dict(self.source_modes),
                    'timestamp_limitations':'Proxy acquisition timestamps measure delivery, not original camera capture; proxy runs do not pass live freshness acceptance.' if 'legacy_preview_proxy' in self.source_modes.values() else None,
                    'zones':self.tracker.zones.config,'settings':self.settings,'worker_error':self.worker_error,
                    'metrics':metrics,'valid_observation_fraction':self.valid_samples/max(1,self.samples),
                    'log_directory':str(self.log_dir),'elapsed_s':elapsed}

    def reacquire(self):
        with self.lock:
            if self.worker_error:
                raise ValueError('Restart perception to recover the failed worker')
            self.tracker.reacquire()
            self.record({'type':'reacquire'})

    def save_zones(self, regions, margin):
        import yaml
        with self.lock:
            capture = self.captures.get('external')
            if not capture or capture.error or 'external' not in self.frames:
                raise ValueError('A fresh external camera is required')
            if time.monotonic_ns()-self.frames['external']['capture_monotonic_ns'] > self.tracker.freshness:
                raise ValueError('External frame is stale')
            w,h = self.settings['capture_width'],self.settings['capture_height']
            if self.settings['rotation']['external'] in (90,270): w,h=h,w
            config = {'schema_version':1,'status':'calibrated','camera_path':self.config['cameras']['external'],
                      'rotation':self.settings['rotation']['external'],'image_size':[w,h],
                      'position_source':self.settings['position_source'],'margin':margin,'regions':regions,
                      'recorded_at':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),'operator':'local calibration console'}
            zones = Zones(config,self.config['cameras']['external'],self.settings)
            snapshot = self.log_dir/'zone-calibration.jpg'
            snapshot.write_bytes(self.frames['external']['jpeg'])
            config['screenshot'] = str(snapshot)
            config['config_sha256'] = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
            if self.zone_path.exists():
                (self.log_dir/f'zones-before-{time.monotonic_ns()}.yaml').write_bytes(self.zone_path.read_bytes())
            temporary = self.zone_path.with_suffix('.tmp')
            temporary.write_text(yaml.safe_dump(config,sort_keys=False))
            temporary.replace(self.zone_path)
            self.tracker.zones = zones
            self.tracker.reacquire()
            self.record({'type':'zones','config':config})
            return config

    def preview(self, role):
        import cv2
        import numpy as np
        with self.lock:
            frame = self.frames.get(role)
            capture = self.captures.get(role)
            if not frame or not capture or capture.error or time.monotonic_ns()-frame['capture_monotonic_ns'] > self.tracker.freshness:
                return None
            state = asdict(self.tracker.state)
            zones = self.tracker.zones
            reply = self.replies.get(role)
        image = cv2.imdecode(np.frombuffer(frame['jpeg'],np.uint8),cv2.IMREAD_COLOR)
        h,w = image.shape[:2]
        if reply and not reply.get('error') and time.monotonic_ns()-reply['capture_monotonic_ns']<=self.tracker.freshness:
            for face in reply.get('landmarks',[]):
                for point in face[::8]:
                    if all(isinstance(point.get(key),(int,float)) and 0<=point[key]<=1 for key in ('x','y')):
                        cv2.circle(image,(round(point['x']*w),round(point['y']*h)),1,(220,160,80),-1)
        if role == 'external':
            for name,rect in zones.regions.items():
                a,b,c,d=rect
                cv2.rectangle(image,(round(a*w),round(b*h)),(round(c*w),round(d*h)),(0,200,255),2)
                cv2.putText(image,name,(round(a*w),max(20,round(b*h))),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,200,255),1)
            for key,color in [('face_center_raw',(80,80,255)),('face_center_filtered',(80,255,80)),('body_center_filtered',(255,150,80))]:
                point = state.get(key)
                if point: cv2.circle(image,(round(point[0]*w),round(point[1]*h)),5,color,2)
        age = (time.monotonic_ns()-frame['capture_monotonic_ns'])/1e6
        lines=[f'{role} seq={frame["sequence"]} age={age:.0f}ms | CAMERA ONLY',
               f'{state["invalid_reason"] or "tracking"} zone={state["stable_zone"]} candidate={state["candidate_zone"]}' if role=='external' else f'Wrist inference {"on" if self.settings["wrist_hz"] else "off"}; never selects a workstation']
        for i,line in enumerate(lines):
            cv2.putText(image,line,(8,h-35+i*22),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
        return frame['sequence'],cv2.imencode('.jpg',image)[1].tobytes()

    def close(self):
        self.stopped.set()
        for capture in self.captures.values(): capture.close()
        self.client.close()
        for thread in self.threads: thread.join(timeout=2)
        with self.lock:
            if not self.log.closed:
                (self.log_dir/'metrics.json').write_text(json.dumps(self.status(),indent=2,allow_nan=False))
                self.log.close()
