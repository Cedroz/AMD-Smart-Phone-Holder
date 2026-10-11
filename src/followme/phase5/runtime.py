"""Independent supervisor. Defaults to fake robot and synthetic perception."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import threading
import time
from zoneinfo import ZoneInfo

from followme.robotics.adapters import FakeAdapter
from followme.robotics.controller import Controller, RobotWorker, EventLog
from followme.robotics.trajectories import fake_library
from .config import verify_approvals, inspect
from .planner import Integration


def synthetic(now, sequence, zone='station_a', since=None, generation=0):
    return {'state': {'camera_role':'external', 'sequence':sequence, 'track_generation':generation,
            'capture_monotonic_ns':int(now*1e9), 'visible':True, 'tracking_valid':True,
            'ambiguous':False, 'selection_required':False, 'candidate_zone':zone,
            'stable_zone':zone, 'candidate_since_ns':int((now if since is None else since)*1e9)},
            'source_modes':{'external':'synthetic'}, 'worker_error':None}


class SessionLog:
    def __init__(self, path, clock=time.monotonic):
        self.path, self.clock = Path(path), clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def __call__(self, event, **values):
        with self.lock, self.path.open('a') as f:
            f.write(json.dumps({'timestamp_s':self.clock(), 'event':event, **values}, allow_nan=False)+'\n')


class Runtime:
    def __init__(self, root, config, live=False, perception='synthetic', output=None):
        self.root, self.config = Path(root).resolve(), config
        self.readiness = inspect(self.root, config)
        if live and perception != 'native':
            raise ValueError('live_requires_native_perception')
        self.output = Path(output or self.root/'artifacts/phase-5/evidence'/datetime.now(
            ZoneInfo('America/Los_Angeles')).strftime('%Y%m%dT%H%M%S.%f%z'))
        self.output.mkdir(parents=True, exist_ok=True)
        self.log = SessionLog(self.output/'integration.jsonl')
        self.initial_hashes = {}
        self.provider = None
        self.started = False
        self.perception_mode = perception
        guard = None
        if live and config.get('behavior') == 'continuous_centering' and config.get('centering_trial'):
            from types import SimpleNamespace
            from followme.robotics.contracts import Limits
            from .live_pan import LivePanAdapter
            trial = config.get('centering_trial',{})
            bounds = trial.get('pan_bounds_deg')
            relative_window = trial.get('relative_window_deg')
            calibrated_bounds = trial.get('calibrated_travel_bounds',False)
            pan_max = 90 if calibrated_bounds else 2
            if relative_window is not None and (isinstance(relative_window,bool) or not isinstance(relative_window,(int,float)) or not 0 < relative_window <= pan_max):
                raise ValueError('invalid_local_trial_window')
            if not isinstance(bounds,list) or len(bounds)!=2 or not all(isinstance(v,(int,float)) for v in bounds) or not 0 < bounds[1]-bounds[0] <= 2.1:
                raise ValueError('tested_pan_window_required')
            evidence = self.root/trial.get('evidence','')
            if not evidence.is_file() or hashlib.sha256(evidence.read_bytes()).hexdigest()!=trial.get('evidence_sha256'):
                raise ValueError('centering_trial_evidence_required')
            if not trial.get('operator_power_cutoff_confirmed'):
                raise ValueError('operator_power_cutoff_required')
            tilt_window=trial.get('tilt_window_deg') if config.get('tracking_loop')=='track_xy' else None
            tilt_max = 25 if calibrated_bounds else 1
            if tilt_window is not None and (isinstance(tilt_window,bool) or not isinstance(tilt_window,(int,float)) or not 0<tilt_window<=tilt_max):
                raise ValueError('invalid_local_tilt_window')
            adapter = LivePanAdapter(bounds, relative_window=relative_window,tilt_window=tilt_window,
                                     calibrated_bounds=calibrated_bounds)
            library = SimpleNamespace(backend='live',limits=Limits(velocity=1.,acceleration=2.))
            guard = lambda: None  # Trial guard is bounded in the worker and adapter.
        elif live:
            data, library = verify_approvals(self.root, config)
            if config.get('behavior') == 'continuous_centering':
                raise ValueError('centering_requires_validated_live_response_mapping')
            from followme.phase4.replay import GripperGuard
            import yaml
            hardware = yaml.safe_load((self.root/'artifacts/phase-0/hardware.yaml').read_text())
            arm = hardware['devices']['arms']['follower']
            adapter = GripperGuard(self.root, arm['path'], arm['calibration_id'],
                                   target=data['gripper']['target'], tolerance=data['gripper']['tolerance'])
            def guard():
                _, verified = verify_approvals(self.root, config)
                if verified.digest != library.digest:
                    raise ValueError('approved_library_changed')
        else:
            library = fake_library()
            adapter = FakeAdapter()
        controller_type, integration_type = Controller, Integration
        if config.get('behavior') == 'continuous_centering':
            from .centering import CenteringController, CenteringIntegration
            controller_type, integration_type = CenteringController, CenteringIntegration
            if config.get('tracking_loop')=='track_xy':
                from .track_xy import XYController
                controller_type=XYController
        extra = {'bounds':config['centering_trial']['pan_bounds_deg']} if live and config.get('behavior')=='continuous_centering' else {}
        self.controller = controller_type(adapter, library, library.limits,
            logger=EventLog(self.output/'controller.jsonl'), **extra)
        self.integration = integration_type(self.controller, config, logger=self.log, live_guard=guard)
        self.worker = RobotWorker(self.controller)
        self.stopped = threading.Event()
        self.thread = None
        self.error = None

    def start(self):
        if self.started:
            raise RuntimeError('already_started')
        self.started = True
        try:
            if self.perception_mode != 'synthetic':
                from .perception import SingleCameraPerception as Perception
                if self.config.get('tracking_loop')=='track_xy':
                    from .tracker_perception import TrackerPerception as Perception
                # Evidence is routed only to this phase. Never save upstream zones.
                self.provider = Perception(self.root, log_dir=self.output/'perception',
                                           preview_sources=self.perception_mode == 'proxy',
                                           settings_override={'rotation':{'external':self.config.get('camera_rotation',0),'wrist':0},'wrist_hz':0})
                for name in ['config/environment.yaml', 'config/phase2/zones.yaml', 'config/phase2/perception.yaml']:
                    self.initial_hashes[name] = hashlib.sha256((self.root/name).read_bytes()).hexdigest()
                self.provider.start()  # Busy cameras fail; existing owners are never stopped.
            self.worker.start()
            deadline = time.monotonic()+5
            while self.controller.status().worker_heartbeat_s == 0:
                if self.worker.error or not self.worker.thread.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError(self.worker.error or 'robot_initial_feedback_timeout')
                time.sleep(.01)
            self.thread = threading.Thread(target=self.monitor, name='followme-phase5-monitor', daemon=True)
            self.thread.start()
            self.log('session', backend=self.controller.adapter.backend, perception=self.perception_mode,
                     config=self.config, upstream_hashes=self.initial_hashes, alignment_enabled=False)
        except BaseException:
            self.close()
            raise

    def monitor(self):
        start, last_zone, since, sequence = time.monotonic(), None, time.monotonic(), 0
        try:
            while not self.stopped.is_set():
                now = time.monotonic()
                if not self.worker.health():
                    self.integration.hold('robot_worker_failure')
                    self.error = 'robot_worker_failure'
                    return
                if self.provider:
                    snapshot = self.provider.status()
                    for name, digest in self.initial_hashes.items():
                        if hashlib.sha256((self.root/name).read_bytes()).hexdigest() != digest:
                            raise RuntimeError('perception_configuration_changed')
                else:
                    zones = list(self.config['zone_to_pose'])
                    zone = zones[int((now-start)//8) % 2]
                    if zone != last_zone:
                        since, last_zone = now, zone
                    sequence += 1
                    snapshot = synthetic(now, sequence, zone, since)
                if self.integration.live and self.integration.mode == 'follow':
                    self.integration.guard()  # Outside the priority-control lock.
                self.log('observation', snapshot=snapshot)
                self.integration.tick(snapshot)
                self.stopped.wait(.05)  # Freshness/interrupt monitor independent of 5 Hz planning.
        except BaseException as exc:
            self.error = f'integration_failure:{exc}'
            self.integration.hold(self.error)
            self.controller.stop_motion(self.error)
            self.log('supervisor_failure', reason=self.error)

    def status(self):
        return {**self.integration.status(), 'perception_mode':self.perception_mode,
                'supervisor_error':self.error, 'log_directory':str(self.output), 'alignment_enabled':False,
                'live_ready':self.readiness['live_ready'],
                'live_motion_blockers':(['calibrate_camera_direction_first'] if self.controller.adapter.backend=='live' and getattr(self.controller,'response_sign',None) is None else []) if self.config.get('centering_trial') and self.controller.adapter.backend=='live' else self.readiness['blockers'] +
                    (['arm_camera_requires_motion_aware_tracking_validation']
                     if self.config.get('camera_mount') == 'arm' else []),
                'perception':self.provider.status() if self.provider else None}

    def close(self):
        self.stopped.set()
        self.integration.hold('shutdown')
        if self.thread:
            self.thread.join(1)
        if self.worker.thread.ident is not None:
            try:
                self.worker.close()
            except RuntimeError as exc:
                self.error = str(exc)
        if self.provider:
            self.provider.close()
        (self.output/'summary.json').write_text(json.dumps(self.status(), indent=2, allow_nan=False)+'\n')
