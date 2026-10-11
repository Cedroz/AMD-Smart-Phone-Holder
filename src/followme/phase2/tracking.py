"""Deterministic camera-only tracking, freshness and workstation classification."""
from dataclasses import asdict, dataclass
import math


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_settings(settings):
    if settings.get('schema_version')!=1:
        raise ValueError('Unsupported perception schema')
    for key in ('capture_width','capture_height','inference_width'):
        if type(settings.get(key)) is not int:
            raise ValueError(f'{key} must be an integer')
    for key in ('capture_width', 'capture_height', 'capture_fps', 'external_hz',
                'inference_width', 'freshness_ms', 'capture_stall_s', 'dwell_s', 'max_center_jump'):
        if not finite(settings.get(key)) or settings[key] <= 0:
            raise ValueError(f'Invalid {key}')
    for key in ('alpha', 'face_threshold'):
        if not finite(settings.get(key)) or not 0 < settings[key] <= 1:
            raise ValueError(f'Invalid {key}')
    if not finite(settings.get('wrist_hz')) or settings['wrist_hz'] < 0:
        raise ValueError('Invalid wrist_hz')
    if settings.get('position_source') not in ('face', 'shoulders'):
        raise ValueError('Invalid position_source')
    if not 32<=settings['inference_width']<=settings['capture_width']:
        raise ValueError('Inference width must be 32..capture width')
    for role in ('external', 'wrist'):
        if settings.get('rotation', {}).get(role) not in (0, 90, 180, 270):
            raise ValueError('Invalid camera rotation')
    if settings['position_source'] == 'shoulders' and not all(settings.get(k) for k in ('pose_model_path','pose_model_manifest')):
        raise ValueError('Shoulder tracking requires a verified pose model')
    return settings


class Zones:
    def __init__(self, config, camera_path=None, settings=None):
        self.config = config
        self.regions = config.get('regions', {})
        self.margin = config.get('margin', .03)
        if config.get('schema_version') != 1 or not isinstance(self.regions,dict) or not finite(self.margin) or not 0 <= self.margin <= .2:
            raise ValueError('Invalid zone schema or margin')
        if config.get('status') == 'pending' and not self.regions:
            return
        if config.get('status') != 'calibrated' or set(self.regions) != {'station_a', 'station_b'}:
            raise ValueError('Two calibrated station regions required')
        if camera_path and config.get('camera_path') != camera_path:
            raise ValueError('Zone camera metadata mismatch')
        if settings:
            if config.get('rotation') != settings['rotation']['external'] or config.get('position_source') != settings['position_source']:
                raise ValueError('Zone orientation/position source mismatch')
            w, h = settings['capture_width'], settings['capture_height']
            if settings['rotation']['external'] in (90, 270):
                w, h = h, w
            if config.get('image_size') != [w, h]:
                raise ValueError('Zone image size mismatch')
        for rect in self.regions.values():
            if not isinstance(rect, list) or len(rect) != 4 or not all(finite(x) for x in rect):
                raise ValueError('Region must contain four finite bounds')
            x0, y0, x1, y1 = rect
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError('Region bounds out of range')
        a, b = [self.expanded(k) for k in self.regions]
        if max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3]):
            raise ValueError('Expanded workstation regions overlap')

    def expanded(self, name):
        a, b, c, d = self.regions[name]
        m = self.margin
        return [max(0, a-m), max(0, b-m), min(1, c+m), min(1, d+m)]

    @staticmethod
    def contains(rect, point):
        return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]

    def classify(self, point):
        return next((name for name, rect in self.regions.items() if self.contains(rect, point)), None)


@dataclass(frozen=True)
class UserState:
    camera_role: str = 'external'
    sequence: int = -1
    capture_monotonic_ns: int = 0
    processed_monotonic_ns: int = 0
    visible: bool = False
    tracking_valid: bool = False
    ambiguous: bool = False
    invalid_reason: str | None = 'waiting_for_camera'
    face_count: int = 0
    face_center_raw: tuple | None = None
    face_center_filtered: tuple | None = None
    body_center_raw: tuple | None = None
    body_center_filtered: tuple | None = None
    track_generation: int = 0
    selection_required: bool = False
    candidate_zone: str | None = None
    stable_zone: str | None = None
    candidate_since_ns: int | None = None
    head_yaw_deg: float | None = None
    head_pitch_deg: float | None = None
    position_source: str = 'face'


class Tracker:
    def __init__(self, settings, zones):
        self.settings = settings
        self.zones = zones
        self.freshness = int(settings['freshness_ms'] * 1e6)
        self.dwell = int(settings['dwell_s'] * 1e9)
        self.state = UserState(position_source=settings['position_source'])
        self.last_seq = -1
        self.last_stamp = -1
        self.last_valid = None
        self.raw = self.filtered = self.body_filtered = None
        self.acquired = False
        self.selection_required = False
        self.acquire_since = None
        self.generation = 0
        self.candidate = self.stable = self.candidate_since = None

    def reacquire(self):
        self.acquired = False
        self.selection_required = False
        self.acquire_since = None
        self.raw = self.filtered = self.body_filtered = None
        self.last_valid = None
        self.candidate = self.stable = self.candidate_since = None
        self.invalidate(self.state.processed_monotonic_ns, 'acquiring')

    def invalidate(self, now, reason, clear_selection=False, ambiguous=False, face_count=0):
        self.candidate = self.stable = self.candidate_since = None
        self.acquire_since = None
        if clear_selection:
            self.selection_required = True
            self.acquired = False
            self.raw = self.filtered = self.body_filtered = None
        self.state = UserState(sequence=self.last_seq, capture_monotonic_ns=max(0, self.last_stamp),
                               processed_monotonic_ns=now, invalid_reason=reason, ambiguous=ambiguous,
                               face_count=face_count, track_generation=self.generation,
                               selection_required=self.selection_required,
                               position_source=self.settings['position_source'])
        return self.state

    def tick(self, now):
        if self.last_valid is not None and now - self.last_valid > self.freshness:
            return self.invalidate(now, 'stale', clear_selection=self.acquired or self.selection_required)
        return self.state

    def update(self, reply, now):
        self.tick(now)
        if reply.get('camera_role') != 'external':
            return self.state
        seq, stamp = reply.get('sequence'), reply.get('capture_monotonic_ns')
        if type(seq) is not int or type(stamp) is not int or stamp < 0:
            return self.invalidate(now, 'invalid_timestamp')
        if seq <= self.last_seq or stamp <= self.last_stamp:
            return self.state  # A delayed/duplicate result cannot overwrite a newer state.
        if stamp > now or now-stamp > self.freshness:
            return self.invalidate(now, 'future_timestamp' if stamp > now else 'stale_result')
        self.last_seq, self.last_stamp = seq, stamp
        if reply.get('error'):
            return self.invalidate(now, 'inference_error', clear_selection=True)
        count = reply.get('face_count')
        if type(count) is not int or count < 0 or count > 2:
            return self.invalidate(now, 'invalid_face_count')
        if count == 2:
            return self.invalidate(now, 'multiple_faces', clear_selection=True, ambiguous=True, face_count=2)
        if count == 0:
            return self.invalidate(now, 'no_face')
        faces = reply.get('landmarks')
        if not isinstance(faces, list) or len(faces) != 1 or not faces[0]:
            return self.invalidate(now, 'invalid_landmarks')
        points = faces[0]
        if not all(isinstance(p, dict) and all(finite(p.get(k)) and 0 <= p[k] <= 1 for k in ('x','y')) for p in points):
            return self.invalidate(now, 'invalid_landmarks')
        raw = tuple((min(p[k] for p in points)+max(p[k] for p in points))/2 for k in ('x', 'y'))
        if self.selection_required:
            return self.invalidate(now, 'selection_required')
        if self.acquired and self.raw is not None and math.dist(raw, self.raw) > self.settings['max_center_jump']:
            return self.invalidate(now, 'continuity_break', clear_selection=True)
        self.last_valid = stamp
        self.raw = raw
        alpha = self.settings['alpha']
        self.filtered = raw if self.filtered is None else tuple(alpha*x+(1-alpha)*y for x,y in zip(raw,self.filtered))
        body = reply.get('shoulder_center')
        if body is not None and (not isinstance(body, list) or len(body) != 2 or not all(finite(x) and 0 <= x <= 1 for x in body)):
            body = None
        if body is not None:
            self.body_filtered = tuple(body) if self.body_filtered is None else tuple(alpha*x+(1-alpha)*y for x,y in zip(body,self.body_filtered))
        elif self.settings['position_source']=='shoulders':
            self.body_filtered = None
        if not self.acquired:
            if self.acquire_since is None:
                self.acquire_since = stamp
            if stamp-self.acquire_since >= self.dwell:
                self.acquired = True
                self.generation += 1
        point = self.filtered if self.settings['position_source'] == 'face' else self.body_filtered
        body_valid = self.settings['position_source'] == 'face' or body is not None
        valid = self.acquired and body_valid
        if valid and self.zones.regions:
            if self.stable and not self.zones.contains(self.zones.expanded(self.stable), point):
                self.stable = None
            candidate = self.zones.classify(point)
            if candidate != self.candidate:
                self.candidate, self.candidate_since = candidate, stamp if candidate else None
            if candidate and stamp-self.candidate_since >= self.dwell:
                self.stable = candidate
        else:
            self.candidate = self.stable = self.candidate_since = None
        self.state = UserState(sequence=seq, capture_monotonic_ns=stamp, processed_monotonic_ns=now,
                               visible=True, tracking_valid=valid, invalid_reason=None if valid else ('acquiring' if not self.acquired else 'missing_shoulders'),
                               face_count=1, face_center_raw=raw, face_center_filtered=self.filtered,
                               body_center_raw=tuple(body) if body is not None else None,
                               body_center_filtered=self.body_filtered if body is not None else None,
                               track_generation=self.generation, selection_required=False,
                               candidate_zone=self.candidate, stable_zone=self.stable,
                               candidate_since_ns=self.candidate_since, position_source=self.settings['position_source'])
        return self.state

    def snapshot(self, now):
        self.tick(now)
        return asdict(self.state)
