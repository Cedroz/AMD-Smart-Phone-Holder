"""Single-owner control loop with priority interruption and bounded trajectories."""
from dataclasses import replace
from collections import deque
import json
import queue
import threading
import time
import uuid
from .contracts import ARM_JOINTS, Limits, RequestResult, RobotStatus, fresh


class EventLog:
    def __init__(self, path=None, clock=time.monotonic):
        self.path, self.clock = path, clock
        self.events = deque(maxlen=10000)
        self.session_id = uuid.uuid4().hex

    def __call__(self, event, **values):
        record = {'timestamp_s': self.clock(), 'session_id': self.session_id, 'event': event, **values}
        self.events.append(record)
        if self.path:
            with open(self.path, 'a') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')


class Controller:
    def __init__(self, adapter, library, limits=Limits(), clock=time.monotonic, logger=None):
        if adapter.backend != library.backend:
            raise ValueError('library_backend_mismatch')
        self.adapter, self.library, self.limits, self.clock = adapter, library, limits, clock
        self.log = logger or EventLog(clock=clock)
        self._status = RobotStatus(adapter.backend)
        self._lock = threading.Lock()
        self._pending = None
        self._ids = set()
        self._commands = queue.SimpleQueue()
        self._stop, self._hold = threading.Event(), threading.Event()
        self._stop_reason = 'operator_stop'
        self._hold_reason = 'operator_hold'
        self._tracking = (None, False, False)
        self._heartbeat = None
        self._active = None
        self._last_tick = None
        self._last_command = None
        self._deviation_since = None
        self._arrival_since = None
        self._settling_since = None
        self._fault = None
        self._started = False
        self._closed = False

    def status(self):
        with self._lock:
            return self._status

    def publish(self, **values):
        with self._lock:
            self._status = replace(self._status, **values)

    def heartbeat(self):
        with self._lock:
            self._heartbeat = self.clock()

    def update_tracking(self, timestamp_s, valid=True, ambiguous=False):
        with self._lock:
            self._tracking = (timestamp_s, valid, ambiguous)

    def set_mode(self, mode):
        if mode not in ('hold', 'follow'):
            raise ValueError('invalid_mode')
        if mode == 'hold':
            self.request_hold()
        else:
            self._commands.put(('follow', None))

    def request_hold(self, reason='operator_hold'):
        self._hold_reason = reason
        self._hold.set()

    def stop_motion(self, reason='operator_stop'):
        self._stop_reason = reason
        self._stop.set()

    def reset(self):
        self._commands.put(('reset', None))

    def request_arm(self):
        # No CLI arm command is exposed in Phase 3; Phase 4 harness supplies evidence.
        self._commands.put(('arm', None))

    def request_safe_pose(self, request):
        now = self.clock()
        with self._lock:
            status = self._status
            if not request.request_id or request.request_id in self._ids:
                reason = 'duplicate_request'
            elif not fresh(request.timestamp_s, now, self.limits.freshness_s):
                reason = 'stale_or_future_request'
            elif self._stop.is_set() or self._hold.is_set() or not status.motion_enabled:
                reason = 'motion_inhibited'
            elif self._pending is not None or status.state == 'REPOSITIONING':
                reason = 'busy_worker'
            elif request.target_pose_name not in self.library.poses:
                reason = 'unknown_pose'
            elif request.target_pose_name == status.current_pose_name:
                reason = 'already_at_pose'
            else:
                reason = 'accepted'
                self._pending = request
                self._ids.add(request.request_id)
        return RequestResult(reason == 'accepted', reason, request.request_id)

    def gate(self, now):
        with self._lock:
            timestamp, valid, ambiguous = self._tracking
            heartbeat = self._heartbeat
        if not fresh(heartbeat, now, self.limits.heartbeat_s):
            return 'heartbeat_loss'
        if ambiguous:
            return 'ambiguous_tracking'
        if not valid or not fresh(timestamp, now, self.limits.freshness_s):
            return 'stale_or_invalid_tracking'
        if not self.adapter.armed:
            return 'not_armed'
        return None

    def start(self):
        self.adapter.bind_owner()
        try:
            self.adapter.connect(read_only=True)
            self._started = True
            self.publish(measured=self.adapter.read_joint_positions(), worker_heartbeat_s=self.clock())
        except Exception as exc:
            self._started = True
            self.fault(str(exc))

    def fault(self, reason):
        self._fault = reason
        self.publish(state='FAULT', mode='hold', fault_reason=reason, motion_enabled=False,
                     hold_confirmed=False, current_pose_name=None, target_pose_name=None)
        self._active = None
        with self._lock:
            self._pending = None
        self.log('fault', reason=reason, backend=self.adapter.backend)

    def cancel(self, reason, state='HOLDING'):
        self._active = None
        self._arrival_since = self._settling_since = self._deviation_since = None
        with self._lock:
            self._pending = None
        confirmed = False
        snapshot = self.status().measured
        if self.adapter.armed:
            try:
                if snapshot is None or not fresh(snapshot.timestamp_s, self.clock(), self.limits.freshness_s):
                    raise RuntimeError('hold_unconfirmed')
                target = {j: snapshot.positions[j] for j in ARM_JOINTS}
                self.library.validate_target(target)
                actual = self.adapter.send(target)
                if actual != target:
                    raise RuntimeError('hold_clipped')
                self._last_command = target
                confirmed = True
            except Exception as exc:
                self._fault = f'hold_unconfirmed:{exc}'
        anchor = self.library.recognize(snapshot.positions) if snapshot else None
        self.publish(mode='hold', state=state if state == 'STOPPED' else ('FAULT' if self._fault else state),
                     fault_reason=self._fault, target_pose_name=None, current_pose_name=anchor,
                     motion_enabled=False, hold_confirmed=confirmed)
        self.log('cancel', reason=reason, state=self.status().state, hold_confirmed=confirmed,
                 physical_commands=self.adapter.physical_commands, backend=self.adapter.backend)

    def tick(self):
        if self._closed:
            return
        if not self._started:
            self.start()
        self.adapter.check_owner()
        now = self.clock()
        late = self._last_tick is not None and now - self._last_tick > 2/self.limits.hz + 1e-9
        self._last_tick = now
        try:
            if not self.adapter.connected:
                raise RuntimeError('transport_disconnected')
            snapshot = self.adapter.read_joint_positions()
            if not fresh(snapshot.timestamp_s, self.clock(), self.limits.freshness_s):
                raise RuntimeError('stale_feedback')
            self.publish(measured=snapshot, worker_heartbeat_s=self.clock())
        except Exception as exc:
            self.fault(str(exc))
            if self._stop.is_set():
                self._stop.clear()
                self.cancel(self._stop_reason, 'STOPPED')
            return
        if self._stop.is_set():
            self._stop.clear()
            while not self._commands.empty():
                self._commands.get()
            self.cancel(self._stop_reason, 'STOPPED')
            return
        if self._hold.is_set():
            self._hold.clear()
            while not self._commands.empty():
                self._commands.get()
            self.cancel(self._hold_reason)
            return
        if self._active and self.clock()-now > 1/self.limits.hz:
            self.fault('io_budget_exceeded')
            self.cancel('io_budget_exceeded')
            return
        if late and self._active:
            self.fault('loop_lateness')
            self.cancel('loop_lateness')
            return
        while not self._commands.empty():
            command, _ = self._commands.get()
            if command == 'reset':
                try:
                    if self.adapter.backend == 'live':
                        from .adapters import verify_live_gate
                        verify_live_gate(self.adapter.root, self.library)
                        if not self.adapter.robot.bus.is_calibrated:
                            raise RuntimeError('calibration_mismatch')
                    if self.adapter.backend == 'fake' and not self.adapter.calibration_matches:
                        raise RuntimeError('calibration_mismatch')
                    self._fault = None
                    self._active = None
                    self.publish(mode='hold', state='HOLDING', fault_reason=None, motion_enabled=False)
                    self.cancel('operator_reset')
                except Exception as exc:
                    self.fault(f'reset_rejected:{exc}')
            elif command == 'arm':
                try:
                    if self.status().state in ('FAULT', 'STOPPED'):
                        raise RuntimeError('latched_inhibition')
                    if self.adapter.backend == 'live':
                        self.adapter.arm(self.library)
                    self.log('armed', backend=self.adapter.backend)
                except Exception as exc:
                    self.fault(str(exc))
            elif command == 'follow' and self.status().state not in ('FAULT', 'STOPPED'):
                self.publish(mode='follow')
        if self.status().state in ('FAULT', 'STOPPED'):
            return
        now = self.clock()
        if self.status().mode == 'follow':
            reason = self.gate(now)
            if reason:
                self.cancel(reason)
                return
        anchor = self.library.recognize(snapshot.positions)
        if not self._active:
            self.publish(current_pose_name=anchor, state='TRACKING' if self.status().mode == 'follow' else 'HOLDING',
                         motion_enabled=self.status().mode == 'follow' and anchor is not None and self.adapter.armed)
            with self._lock:
                request, self._pending = self._pending, None
            if request:
                try:
                    if not fresh(request.timestamp_s, now, self.limits.freshness_s):
                        raise ValueError('expired_request')
                    if anchor is None:
                        raise ValueError('unknown_start_pose')
                    segments = self.library.path(anchor, request.target_pose_name)
                    self._active = [request, segments, 0, now]
                    self._last_command = {j: snapshot.positions[j] for j in ARM_JOINTS}
                    self._arrival_since = self._settling_since = self._deviation_since = None
                    self.publish(state='REPOSITIONING', target_pose_name=request.target_pose_name,
                                 current_pose_name=None, motion_enabled=True)
                    self.log('motion_start', request_id=request.request_id, destination=request.target_pose_name,
                             source_timestamp_s=request.timestamp_s, backend=self.adapter.backend)
                except Exception as exc:
                    self.log('request_rejected', request_id=request.request_id, reason=str(exc))
        if self._active:
            try:
                self.execute(snapshot, self.clock())
            except Exception as exc:
                self.fault(str(exc))
                self.cancel('execution_fault')

    def execute(self, snapshot, now):
        request, segments, index, started = self._active
        segment = segments[index]
        elapsed = now - started
        if self._last_command:
            error = max(abs(snapshot.positions[j]-self._last_command[j]) for j in ARM_JOINTS)
            if error > self.limits.deviation_deg:
                self._deviation_since = now if self._deviation_since is None else self._deviation_since
                if now-self._deviation_since >= self.limits.deviation_s:
                    raise RuntimeError('feedback_deviation')
            else:
                self._deviation_since = None
        target = segment.sample(elapsed)
        self.library.validate_target(target)
        if any(abs(target[j]-snapshot.positions[j]) > self.limits.relative+1e-9 for j in ARM_JOINTS):
            raise RuntimeError('relative_limit')
        # An interruption arriving during read/validation must precede dispatch.
        if self._stop.is_set() or self._hold.is_set():
            stopped = self._stop.is_set()
            self._stop.clear()
            self._hold.clear()
            self.cancel('priority_interrupt', 'STOPPED' if stopped else 'HOLDING')
            return
        if self.gate(self.clock()):
            self.cancel('gate_changed')
            return
        actual = self.adapter.send(target)
        if set(actual) != set(target) or any(abs(actual[j]-target[j]) > 1e-9 for j in target):
            raise RuntimeError('sdk_clipping')
        self._last_command = target
        self.log('command', request_id=request.request_id, requested=target, dispatched=actual,
                 measured=dict(snapshot.positions), feedback_timestamp_s=snapshot.timestamp_s,
                 elapsed_s=elapsed, backend=self.adapter.backend,
                 physical_commands=self.adapter.physical_commands)
        if self._stop.is_set() or self._hold.is_set():
            stopped = self._stop.is_set()
            self._stop.clear()
            self._hold.clear()
            # Read current feedback before issuing a hold after a dispatch.
            self.publish(measured=self.adapter.read_joint_positions())
            self.cancel('priority_interrupt', 'STOPPED' if stopped else 'HOLDING')
            return
        if elapsed >= segment.duration_s:
            self._settling_since = now if self._settling_since is None else self._settling_since
            converged = all(abs(snapshot.positions[j]-segment.end[j]) <= self.limits.arrival_deg for j in ARM_JOINTS)
            self._arrival_since = (now if self._arrival_since is None else self._arrival_since) if converged else None
            if self._arrival_since is not None and now-self._arrival_since >= self.limits.arrival_s:
                if index+1 == len(segments):
                    self._active = None
                    self.publish(state='TRACKING', current_pose_name=request.target_pose_name,
                                 target_pose_name=None, hold_confirmed=True)
                    self.log('motion_arrived', request_id=request.request_id, destination=request.target_pose_name,
                             latency_s=now-request.timestamp_s, backend=self.adapter.backend)
                else:
                    self._active[2:] = [index+1, now]
                    self._arrival_since = self._settling_since = None
            elif now-self._settling_since > self.limits.settling_s:
                raise RuntimeError('settling_timeout')

    def close(self):
        self.adapter.check_owner()
        try:
            if self.adapter.connected:
                try:
                    self.publish(measured=self.adapter.read_joint_positions())
                except Exception:
                    self.publish(measured=None)
                self.cancel('shutdown', 'STOPPED' if self.status().state == 'STOPPED' else 'HOLDING')
        finally:
            self.adapter.disconnect()
            self._closed = True
            self.publish(motion_enabled=False)


class RobotWorker:
    def __init__(self, controller):
        self.controller = controller
        self._quit = threading.Event()
        self.thread = threading.Thread(target=self.run, name='followme-robot', daemon=True)
        self.error = None

    def start(self):
        self.thread.start()

    def run(self):
        try:
            while not self._quit.is_set():
                start = time.monotonic()
                self.controller.tick()
                self._quit.wait(max(0., start+1/self.controller.limits.hz-time.monotonic()))
        except BaseException as exc:
            self.error = str(exc)
            self.controller.fault(f'worker_failure:{exc}')
        finally:
            try:
                self.controller.close()
            except Exception as exc:
                self.error = str(exc)
                self.controller.fault(f'cleanup_unconfirmed:{exc}')

    def health(self):
        status = self.controller.status()
        stalled = self.thread.is_alive() and self.controller.clock()-status.worker_heartbeat_s > .5
        if stalled or not self.thread.is_alive():
            reason = 'worker_stalled' if stalled else 'worker_dead'
            self.controller._fault = reason
            self.controller.stop_motion(reason)
            self.controller.publish(state='FAULT', mode='hold', motion_enabled=False,
                                    fault_reason=reason, hold_confirmed=False)
            return False
        return True

    def close(self, timeout=1.):
        self.controller.request_hold('shutdown')
        self._quit.set()
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise RuntimeError('cleanup_unconfirmed: worker still owns serial')
