"""Continuous image centering simulation; deliberately no live motor mapping."""
import math
import queue
import statistics

from followme.robotics.controller import Controller
from followme.robotics.contracts import ARM_JOINTS, fresh
from .planner import Integration, observation


class CenteringController(Controller):
    def __init__(self, *args, bounds=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bounds = bounds or [-15.,15.]
        self.response_sign = None if self.adapter.backend == 'live' else 1
        self.calibration = None
        self.calibration_result = None
        self.center = None
        self.velocity = 0.
        self.requested = None

    def calibrate_direction(self):
        self._commands.put(('calibrate',None))

    def calibrate_tick(self, now, measured, center):
        cal = self.calibration
        angle = measured.positions['shoulder_pan']
        if cal is None:
            return False
        if self.gate(now) or not center or not fresh(center[1],now,.5):
            self.calibration = None
            self.calibration_result = {'status':'failed','reason':'tracking_or_heartbeat_lost'}
            self.adapter.hold()
            return True
        elapsed = now-cal['start']
        if center[1] != cal.get('last_frame'):
            cal['samples'].append((angle,center[0][0]))
            cal['last_frame'] = center[1]
        if elapsed < 1:
            return True
        if elapsed < 7:
            t = min(1., (elapsed-1)/6)
            value = cal['origin']+(cal['destination']-cal['origin'])*(10*t**3-15*t**4+6*t**5)
            target = dict((j,measured.positions[j]) for j in ARM_JOINTS)
            target['shoulder_pan'] = value
            self.adapter.send(target)
            self.requested = target
            return True
        if elapsed < 9:
            return True
        samples = cal['samples']
        angles, pixels = zip(*samples)
        a, x = statistics.mean(angles),statistics.mean(pixels)
        covariance = sum((v-a)*(p-x) for v,p in samples)
        avar = sum((v-a)**2 for v in angles)
        pvar = sum((p-x)**2 for p in pixels)
        correlation = covariance/(avar*pvar)**.5 if avar*pvar else 0.
        success = len(samples)>=15 and max(angles)-min(angles)>.5 and max(pixels)-min(pixels)>.01 and abs(correlation)>.8
        self.response_sign = (-1 if covariance>0 else 1) if success else None
        self.calibration_result = {'status':'passed' if success else 'failed',
            'response_sign':self.response_sign,'correlation':correlation,
            'measured_span_deg':max(angles)-min(angles),'image_span':max(pixels)-min(pixels)}
        self.log('centering_direction_calibration', **self.calibration_result)
        self.adapter.hold()
        self.calibration = None
        return True

    def update_center(self, center, timestamp):
        with self._lock:
            self.center = (center, timestamp)

    def tick(self):
        now = self.clock()
        if not self._started:
            self.adapter.bind_owner()
            self.adapter.connect(read_only=True)
            self._started = True
        measured = self.adapter.read_joint_positions()
        self.publish(measured=measured, worker_heartbeat_s=now)
        interrupted = self._stop.is_set() or self._hold.is_set()
        if self._stop.is_set():
            self._stop.clear()
            self.publish(state='STOPPED', mode='hold', motion_enabled=False)
        if self._hold.is_set():
            self._hold.clear()
            self.publish(mode='hold', motion_enabled=False)
        if interrupted:
            self.calibration = None
            if self.adapter.backend == 'live' and self.adapter.armed:
                self.adapter.hold()
            self.requested = None
            if hasattr(self,'axis_velocity'):
                self.axis_velocity=dict.fromkeys(self.axis_velocity,0.)
        while True:
            try:
                command, _ = self._commands.get_nowait()
            except queue.Empty:
                break
            if interrupted:
                continue
            if command == 'reset' and self.status().state == 'STOPPED':
                self.publish(state='HOLDING', mode='hold', motion_enabled=False)
            elif command == 'calibrate' and not interrupted and self.status().state not in ('FAULT','STOPPED'):
                if self.adapter.backend != 'live':
                    continue
                self.adapter.arm()
                self.response_sign = None
                self.calibration_result = {'status':'running'}
                origin = measured.positions['shoulder_pan']
                destination = max(self.bounds,key=lambda v:abs(v-origin))
                self.calibration = {'start':now,'origin':origin,'destination':destination,'samples':[]}
                self.publish(state='ALIGNING',mode='hold',motion_enabled=False)
            elif command == 'follow' and not interrupted and self.response_sign is not None and self.status().state not in ('FAULT', 'STOPPED'):
                if self.adapter.backend == 'live':
                    self.adapter.arm()
                self.publish(mode='follow', motion_enabled=True)
        dt = min(.05, max(0., now-(self._last_tick if self._last_tick is not None else now)))
        self._last_tick = now
        with self._lock:
            center = self.center
        reason = self.gate(now)
        if self.calibrate_tick(now,measured,center):
            return
        if self.status().mode=='follow' and reason=='stale_or_invalid_tracking' and getattr(self,'searching_allowed',False):
            if self.adapter.backend=='live' and self.status().state!='SEARCHING':
                self.adapter.hold()
            self.requested=None
            if hasattr(self,'axis_velocity'):
                self.axis_velocity=dict.fromkeys(self.axis_velocity,0.)
            self.publish(state='SEARCHING',motion_enabled=False)
            return
        if self.status().mode != 'follow' or reason or not center or not fresh(center[1], now, .5):
            self.velocity = 0.
            if hasattr(self,'axis_velocity'):
                self.axis_velocity=dict.fromkeys(self.axis_velocity,0.)
            if reason and self.status().mode == 'follow':
                if self.adapter.backend == 'live' and self.adapter.armed:
                    self.adapter.hold()
                self.publish(mode='hold', motion_enabled=False)
                self.requested=None
            if self.status().state not in ('STOPPED', 'FAULT'):
                self.publish(state='HOLDING', hold_confirmed=self.adapter.armed)
            return
        error = center[0][0]-.5
        if hasattr(self,'track_step'):
            self.track_step(now,measured,center,dt)
            return
        speed = min(5.,self.limits.velocity)
        desired = 0. if abs(error) <= .08 else self.response_sign*max(-speed, min(speed, 10.*error))
        if desired == 0.:
            self.velocity = 0.
        else:
            self.velocity += max(-self.limits.acceleration*dt, min(self.limits.acceleration*dt, desired-self.velocity))
        target = dict((j, measured.positions[j]) for j in ARM_JOINTS)
        # Simulation bounds and response sign are NOT physical calibration.
        base = self.requested['shoulder_pan'] if self.requested else target['shoulder_pan']
        target['shoulder_pan'] = max(self.bounds[0], min(self.bounds[1], base+self.velocity*dt))
        if abs(target['shoulder_pan']-measured.positions['shoulder_pan']) > 2.:
            raise RuntimeError('pan_feedback_not_following')
        if desired == 0. and self.adapter.backend == 'live':
            if self.velocity != 0. or self.requested is not None:
                self.adapter.hold()
                self.requested = None
            self.publish(state='TRACKING',hold_confirmed=True)
            return
        if abs(target['shoulder_pan']-measured.positions['shoulder_pan']) > 0.:
            self.adapter.send(target)
            self.requested = target
            self.log('centering_command', image_error_x=error, requested=target,
                     measured=dict(measured.positions), backend=self.adapter.backend)
        self.publish(state='TRACKING', hold_confirmed=desired == 0.)

    def close(self):
        self.adapter.check_owner()
        try:
            if self.adapter.backend == 'live' and self.adapter.connected and self.adapter.armed:
                self.adapter.hold()
        finally:
            self.adapter.disconnect()
            self.publish(mode='hold',motion_enabled=False)


class CenteringIntegration(Integration):
    def heartbeat(self,client):
        with self.lock:
            if self.owner is not None and not fresh(self.lease,self.clock(),3):
                self.owner=None
            super().heartbeat(client)

    def set_mode(self, mode, client):
        with self.lock:
            if mode == 'hold':
                self.hold('operator_hold')
                return
            if mode != 'follow':
                raise ValueError('invalid_mode')
            if self.controller.status().state in ('FAULT', 'STOPPED'):
                raise ValueError('fault_or_stop_latched')
            if self.controller.response_sign is None or self.controller.calibration is not None:
                raise ValueError('calibrate_camera_direction_first')
            if self.owner != client or not fresh(self.lease, self.clock(), 3):
                raise ValueError('operator_lease_required')
            self.validate(self.latest)
            self.mode = 'follow'
            self.controller.set_mode('follow')

    def validate(self, snapshot):
        state, timestamp = observation(snapshot, self.clock(), .5, self.live)
        point = state.get('tracking_center',state.get('face_center_filtered'))
        if not isinstance(point, (list, tuple)) or len(point) != 2 or any(
                isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in point):
            raise ValueError('valid_face_center_required')
        return state, point, timestamp

    def tick(self, snapshot):
        with self.lock:
            self.latest = snapshot
            now = self.clock()
            if self.mode == 'follow' and not fresh(self.lease, now, 3):
                self.hold('heartbeat_loss')
            try:
                state, point, timestamp = self.validate(snapshot)
                identity = (state['track_generation'], state['sequence'])
                if self.last_identity and (identity[0] != self.last_identity[0] or identity[1] < self.last_identity[1]):
                    self.last_identity = identity
                    raise ValueError('tracking_identity_changed')
                self.last_identity = identity
                self.controller.update_tracking(timestamp)
                self.controller.searching_allowed=False
                self.controller.update_center(point, timestamp)
                self.last_reason = 'continuous_centering' if self.mode == 'follow' else 'operator_hold'
            except (ValueError, TypeError):
                self.controller.update_tracking(now, valid=False)
                state=snapshot.get('state',{}) if isinstance(snapshot,dict) else {}
                metrics=snapshot.get('metrics',{}) if isinstance(snapshot,dict) else {}
                missing=hasattr(self.controller,'track_step') and state.get('invalid_reason')=='target_missing_or_stale' and fresh(state.get('capture_monotonic_ns',0)/1e9,now,.5) and not snapshot.get('worker_error') and not metrics.get('external',{}).get('error')
                self.controller.searching_allowed=missing
                if self.mode == 'follow' and not missing:
                    self.hold('invalid_centering_tracking')
            if self.controller.status().mode == 'hold' and self.controller.status().state in ('STOPPED','FAULT'):
                self.mode = 'hold'

    def reset(self):
        with self.lock:
            measured = self.controller.status().measured
            if measured is None or not fresh(measured.timestamp_s, self.clock(), .5):
                raise ValueError('fresh_feedback_required')
            if self.controller.status().state == 'FAULT':
                raise ValueError('fault_requires_worker_recovery')
            self.mode = 'hold'
            self.controller.reset()

    def calibrate(self, client):
        with self.lock:
            if self.mode != 'hold' or self.owner != client or not fresh(self.lease,self.clock(),3):
                raise ValueError('hold_and_operator_lease_required')
            if self.controller.status().state in ('FAULT','STOPPED'):
                raise ValueError('fault_or_stop_latched')
            self.validate(self.latest)
            self.controller.calibrate_direction()

    def reverse(self,axis):
        with self.lock:
            if not hasattr(self.controller,'tilt_sign') or axis not in ('horizontal','vertical'):
                raise ValueError('xy_axis_required')
            self.hold('operator_direction_change')
            if axis=='horizontal':self.controller.response_sign *= -1
            else:self.controller.tilt_sign *= -1

    def status(self):
        return {**super().status(), 'tracking_behavior':'continuous_centering',
                'requested_joints':self.controller.requested,
                'pan_bounds_deg':self.controller.bounds, 'centering_axes':['horizontal','vertical'] if hasattr(self.controller,'track_step') else ['horizontal'],
                'joint_bounds_deg':getattr(self.controller.adapter,'joint_bounds',None),
                'direction_calibration':self.controller.calibration_result,
                'direction_calibration_active':self.controller.calibration is not None,
                'live_response_mapping_validated':self.controller.calibration_result is not None and self.controller.calibration_result.get('status')=='passed',
                'direction_source':'track.py default signs' if hasattr(self.controller,'track_step') else 'calibration'}
