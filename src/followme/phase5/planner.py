"""Observation bridge and policy; no SDK imports, camera capture, or serial writes."""
import math
import threading
import time
import uuid
from followme.robotics.contracts import TargetRequest, fresh


def number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def observation(snapshot, now, budget, live):
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get('state'), dict):
        raise ValueError('malformed_observation')
    s = snapshot['state']
    ts = s.get('capture_monotonic_ns')
    if not number(ts) or not fresh(ts/1e9, now, budget):
        raise ValueError('stale_or_future_observation')
    if s.get('camera_role') != 'external' or s.get('visible') is not True or s.get('tracking_valid') is not True:
        raise ValueError('invalid_external_tracking')
    if s.get('ambiguous') or s.get('selection_required'):
        raise ValueError('ambiguous_or_unselected_user')
    if any(not isinstance(s.get(k), bool) for k in ['ambiguous','selection_required']):
        raise ValueError('invalid_tracking_flags')
    metrics = snapshot.get('metrics', {})
    if not isinstance(metrics, dict) or not isinstance(metrics.get('external', {}), dict):
        raise ValueError('malformed_metrics')
    if snapshot.get('worker_error') or metrics.get('external', {}).get('error'):
        raise ValueError('perception_failure')
    modes = snapshot.get('source_modes', {})
    if not isinstance(modes, dict):
        raise ValueError('malformed_source_modes')
    if live and modes.get('external') != 'native_camera':
        raise ValueError('native_capture_required')
    for key in ['sequence', 'track_generation']:
        if isinstance(s.get(key), bool) or not isinstance(s.get(key), int) or s[key] < 0:
            raise ValueError('invalid_frame_identity')
    for key in ['candidate_zone', 'stable_zone']:
        if s.get(key) is not None and not isinstance(s[key], str):
            raise ValueError('invalid_zone_type')
    return s, ts/1e9


class Integration:
    def __init__(self, controller, config, clock=time.monotonic, logger=None, live_guard=None):
        self.controller, self.config, self.clock = controller, config, clock
        self.log = logger or (lambda *a, **k: None)
        self.guard = live_guard
        self.live = controller.adapter.backend == 'live'
        self.lock = threading.RLock()
        self.mode = 'hold'
        self.owner = None
        self.lease = None
        self.latest = None
        self.last_identity = None
        self.last_capture = None
        self.generation = None
        self.candidate = None
        self.since = None
        self.history_floor = clock()
        self.cooldown_until = 0.
        self.outstanding = None
        self.was_moving = False
        self.last_reason = 'startup_hold'
        self.last_plan = float('-inf')
        self.session = uuid.uuid4().hex
        self.follow_requested_at = None
        self.decisions = []

    def clear(self):
        self.candidate = self.since = None
        self.history_floor = self.clock()

    def heartbeat(self, client):
        with self.lock:
            if not isinstance(client, str) or not 0 < len(client) <= 128:
                raise ValueError('client_identity_required')
            if self.owner is not None and self.owner != client:
                raise ValueError('operator_lease_owned')
            self.owner, self.lease = client, self.clock()
            self.controller.heartbeat()

    def set_mode(self, mode, client):
        # File/evidence verification can be slow: never hold the priority-control lock.
        if mode == 'follow' and self.live:
            if self.guard is None:
                raise ValueError('live_guard_required')
            self.guard()
        with self.lock:
            if mode == 'hold':
                self.hold('operator_hold')
                return
            if mode != 'follow':
                raise ValueError('invalid_mode')
            if self.owner != client or not fresh(self.lease, self.clock(), self.config['heartbeat_s']):
                raise ValueError('operator_lease_required')
            status = self.controller.status()
            if status.state in ('FAULT', 'STOPPED') or status.current_pose_name is None:
                raise ValueError('healthy_anchor_required')
            if self.latest is None:
                raise ValueError('tracking_required')
            observation(self.latest, self.clock(), self.config['freshness_s'], self.live)
            if self.live:
                self.controller.request_arm()
            self.clear()
            self.mode = 'follow'
            self.follow_requested_at = self.clock()
            self.controller.set_mode('follow')
            self.log('operator_mode', mode=mode)

    def hold(self, reason):
        with self.lock:
            self.mode = 'hold'
            self.clear()
            self.outstanding = None
            self.last_reason = reason
            self.controller.request_hold(reason)
            self.log('integration_hold', reason=reason)

    def stop(self):
        with self.lock:
            self.mode = 'hold'
            self.clear()
            self.outstanding = None
            self.last_reason = 'operator_stop'
            self.controller.stop_motion()
            self.log('integration_stop')

    def reset(self):
        if self.live:
            self.guard()
        with self.lock:
            status = self.controller.status()
            if status.measured is None or not fresh(status.measured.timestamp_s, self.clock(), .5) or self.controller.library.recognize(status.measured.positions) is None:
                raise ValueError('supervised_anchor_restoration_required')
            self.mode = 'hold'
            self.clear()
            self.outstanding = None
            self.controller.reset()
            self.log('integration_reset')

    def tick(self, snapshot):
        with self.lock:
            now = self.clock()
            self.latest = snapshot if isinstance(snapshot, dict) else None
            status = self.controller.status()
            if self.owner and not fresh(self.lease, now, self.config['heartbeat_s']):
                if self.mode == 'follow':
                    self.hold('heartbeat_loss')
                self.owner = None
            if status.state in ('FAULT', 'STOPPED'):
                if self.mode != 'hold':
                    self.hold(status.fault_reason or status.state.lower())
                self.last_reason = status.fault_reason or status.state.lower()
                return
            if self.mode == 'follow' and status.mode == 'hold' and self.follow_requested_at is not None and now-self.follow_requested_at > .15:
                self.hold('controller_returned_to_hold')
                return
            if self.mode == 'follow' and status.state != 'REPOSITIONING' and self.outstanding is None and status.current_pose_name is None:
                self.hold('unknown_start_pose')
                return
            try:
                state, captured = observation(snapshot, now, self.config['freshness_s'], self.live)
                identity = (state['track_generation'], state['sequence'])
                if self.generation == identity[0] and self.last_identity and identity[1] < self.last_identity[1]:
                    raise ValueError('regressing_sequence')
                if self.last_capture is not None and captured < self.last_capture:
                    raise ValueError('regressing_capture_time')
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.controller.update_tracking(now, valid=False)
                self.clear()
                self.last_reason = str(exc)
                if self.mode == 'follow':
                    self.hold(str(exc))
                return
            self.controller.update_tracking(captured, valid=True, ambiguous=False)
            changed = self.generation is not None and self.generation != identity[0]
            if changed:
                self.clear()
                if self.mode == 'follow':
                    self.hold('track_generation_changed')
            self.generation = identity[0]
            moving = status.state == 'REPOSITIONING'
            if self.outstanding and not moving and self.was_moving:
                arrived = status.current_pose_name == self.outstanding.target_pose_name and status.hold_confirmed
                if arrived:
                    self.cooldown_until = now + self.config['cooldown_s']
                    self.log('integration_arrived', request_id=self.outstanding.request_id,
                             pose=status.current_pose_name, latency_s=now-self.outstanding.timestamp_s)
                else:
                    self.hold('interrupted_without_arrival')
                self.outstanding = None
                self.clear()
            elif self.outstanding and not moving and now-self.outstanding.timestamp_s > self.config['freshness_s']:
                self.hold('request_not_started')
            self.was_moving = moving
            # Observe identities even while inhibited, so duplicates cannot recreate history.
            duplicate = identity == self.last_identity
            self.last_identity, self.last_capture = identity, captured
            if duplicate or self.mode != 'follow' or moving or self.outstanding:
                return
            candidate = state.get('candidate_zone')
            if candidate is None or candidate not in self.config['zone_to_pose']:
                self.clear()
                self.last_reason = 'neutral_or_unstable_zone'
                return
            start = state.get('candidate_since_ns')
            if not number(start) or not 0 <= start/1e9 <= captured:
                self.clear()
                self.last_reason = 'invalid_candidate_timestamp'
                return
            if candidate != self.candidate:
                self.candidate = candidate
                self.since = max(start/1e9, self.history_floor)
            if candidate != state.get('stable_zone'):
                self.last_reason = 'neutral_or_unstable_zone'
                return
            if captured-self.since + 1e-9 < self.config['dwell_s']:
                self.last_reason = 'zone_dwell'
                return
            if now < self.cooldown_until:
                self.last_reason = 'cooldown'
                return
            if now-self.last_plan < 1/self.config['planner_hz']:
                return
            self.last_plan = now
            pose = self.config['zone_to_pose'][candidate]
            if pose == status.current_pose_name:
                self.last_reason = 'already_at_pose'
                return
            if not status.motion_enabled or status.current_pose_name is None:
                self.last_reason = 'controller_not_ready'
                return
            try:
                self.controller.library.path(status.current_pose_name, pose)
            except ValueError as exc:
                self.hold(str(exc))
                return
            request = TargetRequest(f'{self.session}:{identity[0]}:{identity[1]}:{pose}', pose,
                                    f'stable_zone:{candidate}', captured)
            result = self.controller.request_safe_pose(request)
            self.last_reason = result.reason
            record = {'request_id':request.request_id, 'target':pose, 'accepted':result.accepted,
                      'reason':result.reason, 'source_timestamp_s':captured, 'timestamp_s':now}
            self.decisions.append(record)
            self.log('decision', **record)
            if result.accepted:
                self.outstanding = request
            else:
                self.hold(f'request_rejected:{result.reason}')

    def status(self):
        with self.lock:
            s = self.controller.status()
            return {'backend':s.backend, 'mode':self.mode, 'controller_mode':s.mode, 'state':s.state,
                    'current_pose':s.current_pose_name, 'target_pose':s.target_pose_name,
                    'motion_enabled':s.motion_enabled, 'fault_reason':s.fault_reason,
                    'hold_confirmed':s.hold_confirmed, 'last_reason':self.last_reason,
                    'heartbeat_age_s':None if self.lease is None else self.clock()-self.lease,
                    'cooldown_remaining_s':max(0,self.cooldown_until-self.clock()),
                    'tracking':None if not self.latest else self.latest.get('state'),
                    'measured':None if s.measured is None else dict(s.measured.positions),
                    'physical_commands':self.controller.adapter.physical_commands,
                    'decision_count':len(self.decisions)}
