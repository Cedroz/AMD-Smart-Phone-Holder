"""Draft pose libraries, provenance and fail-closed physical approvals."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
import os

from followme.robotics.contracts import ARM_JOINTS, JOINTS, Limits, validate_joints
from followme.robotics.trajectories import PoseLibrary

STATIONS = ('station_a', 'station_b')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def content_digest(data):
    return digest({k: v for k, v in data.items() if k != 'validation'})


def save(path, data):
    import yaml
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        temporary = f.name
        yaml.safe_dump(data, f, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def load(path):
    import yaml
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('unsupported_phase4_library')
    return data


def blank(setup, calibration_sha256):
    return {'schema_version': 1, 'backend': 'live', 'units': 'degrees',
            'calibration_id': 'follower_arm', 'calibration_sha256': calibration_sha256,
            'setup': setup, 'joint_limits': {}, 'gripper': {'target': None, 'tolerance': 2.},
            'limits': vars(Limits(arrival_s=.5, settling_s=5.)),
            'poses': {}, 'transitions': {}, 'validation': {'status': 'draft'}}


def invalidate(data):
    data['validation'] = {'status': 'draft'}


def validate_bounds(data):
    if set(data['joint_limits']) != set(ARM_JOINTS):
        raise ValueError('measured_bounds_required')
    for bound in data['joint_limits'].values():
        if not isinstance(bound,list) or len(bound)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in bound) or bound[0] >= bound[1]:
            raise ValueError('invalid_bounds')


def read_samples(path):
    samples = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not samples:
        raise ValueError('empty_samples')
    previous = None
    for s in samples:
        validate_joints(s['positions'])
        t = s['timestamp_s']
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
            raise ValueError('invalid_timestamp')
        if previous is not None and (t <= previous or t - previous > .5):
            raise ValueError('unordered_or_stale_samples')
        if not 0 <= s['positions']['gripper'] <= 100:
            raise ValueError('invalid_gripper')
        previous = t
    return samples


def capture(data, name, samples, evidence):
    if name not in STATIONS:
        raise ValueError('unknown_station')
    if samples[-1]['timestamp_s'] - samples[0]['timestamp_s'] < 2:
        raise ValueError('capture_requires_two_seconds')
    values = {j: [s['positions'][j] for s in samples] for j in JOINTS}
    if any(max(v)-min(v) > (1 if j == 'gripper' else 1.) for j, v in values.items()):
        raise ValueError('unstable_capture')
    target = data['gripper']['target']
    measured_gripper = statistics.median(values['gripper'])
    if target is None:
        data['gripper']['target'] = measured_gripper
    elif abs(measured_gripper-target) > data['gripper']['tolerance']:
        raise ValueError('gripper_changed')
    data['poses'][name] = {'joints': {j: statistics.median(values[j]) for j in ARM_JOINTS},
                          'gripper_unchanged': True, 'capture_evidence': evidence,
                          'capture_duration_s': samples[-1]['timestamp_s']-samples[0]['timestamp_s']}
    invalidate(data)


def record(data, source, destination, samples, indices, evidence):
    if (source, destination) not in [('station_a','station_b'), ('station_b','station_a')]:
        raise ValueError('invalid_direction')
    if indices != sorted(set(indices)) or len(indices) < 2 or indices[0] != 0 or indices[-1] != len(samples)-1:
        raise ValueError('explicit_endpoint_waypoint_indices_required')
    target = data['gripper']['target']
    if target is None or any(abs(s['positions']['gripper']-target) > data['gripper']['tolerance'] for s in samples):
        raise ValueError('gripper_changed')
    for name, sample in [(source, samples[0]), (destination, samples[-1])]:
        if any(abs(sample['positions'][j]-data['poses'][name]['joints'][j]) > data['limits']['arrival_deg'] for j in ARM_JOINTS):
            raise ValueError('endpoint_mismatch')
    points = [{j: samples[i]['positions'][j] for j in ARM_JOINTS} for i in indices]
    points[0], points[-1] = data['poses'][source]['joints'], data['poses'][destination]['joints']
    limits = Limits(**data['limits'])
    validate_bounds(data)
    durations = []
    for a, b in zip(points, points[1:]):
        duration = max(.1, *(max(1.875*abs(b[j]-a[j])/limits.velocity,
                          math.sqrt((10/math.sqrt(3))*abs(b[j]-a[j])/limits.acceleration)) for j in ARM_JOINTS))
        durations.append(math.ceil(duration*limits.hz)/limits.hz)
    data['transitions'][f'{source}->{destination}'] = {
        'from': source, 'to': destination, 'waypoints': points,
        'durations_s': durations, 'teaching_evidence': evidence,
        'sample_indices': indices, 'approved': False}
    invalidate(data)


def check(data):
    if data.get('units') != 'degrees' or data.get('calibration_id') != 'follower_arm':
        raise ValueError('invalid_units_or_identity')
    for key in ['calibration_sha256', 'setup']:
        if not data.get(key):
            raise ValueError(f'missing_{key}')
    if len(data['calibration_sha256']) != 64:
        raise ValueError('invalid_calibration_digest')
    limits = Limits(**data['limits'])
    validate_bounds(data)
    if limits.arrival_s < .5 or limits.settling_s > 5 or limits.arrival_deg > 2 or limits.deviation_deg > 3 or limits.deviation_s > .3:
        raise ValueError('phase4_thresholds_exceeded')
    gripper = data['gripper']
    if any(isinstance(gripper[k], bool) or not isinstance(gripper[k], (float,int)) or not math.isfinite(gripper[k]) for k in ('target','tolerance')):
        raise ValueError('invalid_gripper')
    if not 0 <= gripper['target'] <= 100 or not 0 < gripper['tolerance'] <= 2:
        raise ValueError('invalid_gripper')
    if set(data['poses']) != set(STATIONS) or set(data['transitions']) != {'station_a->station_b','station_b->station_a'}:
        raise ValueError('both_poses_and_directions_required')
    projected = project(data, 'fake', 'offline')
    library = PoseLibrary(projected, limits, 'fake')
    for name in STATIONS:
        if library.recognize(data['poses'][name]['joints']) != name:
            raise ValueError('ambiguous_pose_anchors')
    for key, route in data['transitions'].items():
        source, destination = key.split('->')
        if route['from'] != source or route['to'] != destination:
            raise ValueError('direction_mismatch')
        if len(route['waypoints']) < 2:
            raise ValueError('missing_waypoints')
        if route['waypoints'][0] != data['poses'][source]['joints'] or route['waypoints'][-1] != data['poses'][destination]['joints']:
            raise ValueError('endpoint_mismatch')
        library.path(source, destination)
    if data.get('validation', {}).get('status') == 'approved' and data['validation'].get('content_sha256') != content_digest(data):
        raise ValueError('approval_invalidated')
    return library


def project(data, backend, hardware_sha256):
    """Ephemeral Phase 3 view for supervised replay; never updates production config."""
    return {'backend': backend, 'units':'degrees', 'hardware_sha256':hardware_sha256,
            'joint_limits':data['joint_limits'],
            'poses':{k:{**v, 'approved':True, 'hardware_sha256':hardware_sha256} for k,v in data['poses'].items()},
            'transitions':{k:{'approved':True, 'hardware_sha256':hardware_sha256,
                              'waypoints':v['waypoints'][1:-1], 'durations_s':v['durations_s']}
                           for k,v in data['transitions'].items()}}


def approve(data, evidence, root):
    """Require physical evidence linked to the exact tested content."""
    check(data)
    from followme.robotics.adapters import verify_live_gate
    raw = (Path(root)/'artifacts/phase-0/hardware.yaml').read_bytes()
    revision = hashlib.sha256(raw).hexdigest()
    if data['setup'].get('hardware_sha256') != revision:
        raise ValueError('setup_changed_since_teaching')
    verify_live_gate(root, PoseLibrary(project(data, 'live', revision), Limits(**data['limits']), 'live'))
    import yaml
    hardware = yaml.safe_load(raw)
    calibration = Path(hardware['devices']['arms']['follower']['calibration_path']).read_bytes()
    if hashlib.sha256(calibration).hexdigest() != data['calibration_sha256']:
        raise ValueError('calibration_changed_since_teaching')
    if evidence.get('backend') != 'live' or evidence.get('content_sha256') != content_digest(data):
        raise ValueError('physical_evidence_required')
    required = ['phase0_passed','controller_passed','limits_measured','holds_60s_passed',
                'both_paths_clear','fixed_gripper_passed','physical_stop_passed','recovery_passed']
    if not all(evidence.get(k) is True for k in required):
        raise ValueError('acceptance_gates_pending')
    if not evidence.get('operator') or not evidence.get('recorded_at'):
        raise ValueError('missing_operator_attestation')
    interrupts = evidence.get('interruptions', [])
    if {(i['direction'],i['kind'],i['position']) for i in interrupts} != {
            (d,k,p) for d in data['transitions'] for k in ['hold','stop'] for p in ['early','middle','late']}:
        raise ValueError('interruption_trials_missing')
    if any(not isinstance(i['latency_s'], (int,float)) or isinstance(i['latency_s'],bool)
           or not math.isfinite(i['latency_s']) or not 0 <= i['latency_s'] <= .25 for i in interrupts):
        raise ValueError('cancellation_latency_failed')
    cycles = evidence.get('cycles', [])
    if len(cycles) != 10 or [c['trial_id'] for c in cycles] != list(range(1,11)):
        raise ValueError('ten_cycles_required')
    records = cycles + interrupts
    if len({record['log'] for record in records}) != len(records):
        raise ValueError('reused_trial_evidence')
    attachments = evidence.get('physical_attachments', [])
    if {a.get('kind') for a in attachments} != {'holds','clearance','physical_stop'}:
        raise ValueError('physical_attachment_evidence_required')
    for attachment in attachments:
        path = (Path(root)/attachment['path']).resolve()
        if not path.is_relative_to(Path(root).resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != attachment['sha256']:
            raise ValueError('missing_or_changed_physical_attachment')
    for record in records:
        path = (Path(root)/record['log']).resolve()
        if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
            raise ValueError('missing_evidence_log')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('changed_evidence_log')
        log = json.loads(raw)
        if log.get('backend') != 'live' or log.get('content_sha256') != content_digest(data) or log.get('passed') is not True:
            raise ValueError('failed_or_simulated_trial')
        if record in cycles:
            legs = log.get('legs', [])
            if [leg.get('direction') for leg in legs] != ['station_a->station_b','station_b->station_a']:
                raise ValueError('incomplete_cycle')
            if log.get('trial_id') != record['trial_id']:
                raise ValueError('trial_identity_mismatch')
        else:
            legs = [log]
            if log.get('direction') != record['direction'] or log.get('interrupt') != record['kind']:
                raise ValueError('interruption_identity_mismatch')
            if log.get('latency_s') != record['latency_s']:
                raise ValueError('interruption_latency_mismatch')
            if log.get('interrupt_fraction') != {'early':.1,'middle':.5,'late':.8}[record['position']]:
                raise ValueError('interruption_position_mismatch')
        if any(leg.get('backend') != 'live' or leg.get('passed') is not True or
               leg.get('content_sha256') != content_digest(data) or leg.get('physical_commands',0) <= 0
               for leg in legs):
            raise ValueError('physical_motion_evidence_required')
    data['validation'] = {**evidence, 'status':'approved'}
    return data
