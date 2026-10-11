import hashlib
import json
import math
from pathlib import Path


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(root, path=None):
    import yaml
    config = yaml.safe_load(Path(path or Path(root)/'config/phase5/following.yaml').read_text())
    if config.get('schema_version') != 1 or config.get('mode') != 'hold' or config.get('dry_run') is not True:
        raise ValueError('hold_dry_run_defaults_required')
    for key, low, high in [('dwell_s', .75, 60), ('cooldown_s', 2, 60),
                           ('freshness_s', .001, .5), ('heartbeat_s', .001, 3), ('planner_hz', 1, 5)]:
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'invalid_{key}')
    mapping = config.get('zone_to_pose', {})
    if len(mapping) != 2 or set(mapping.values()) != {'station_a', 'station_b'}:
        raise ValueError('two_distinct_station_mappings_required')
    if any(not isinstance(k, str) or not k for k in mapping):
        raise ValueError('invalid_zone_name')
    if not isinstance(config.get('allowed_origins'), list) or not config['allowed_origins']:
        raise ValueError('origins_required')
    return config


def verify_approvals(root, config):
    """Revalidate evidence, not draft/projected approval flags. No device I/O."""
    from followme.phase4.library import load, approve, project
    from followme.robotics.contracts import Limits
    from followme.robotics.trajectories import PoseLibrary
    import yaml
    root = Path(root).resolve()
    if not config.get('approved_session') or not config.get('perception_acceptance'):
        raise ValueError('live_approval_paths_missing')
    data = load(root/config['approved_session'])
    if data.get('validation', {}).get('status') != 'approved':
        raise ValueError('phase4_session_not_approved')
    # approve checks content, hardware/calibration, all trial/attachment hashes.
    approve(data, data['validation'], root)
    if config.get('camera_mount') == 'arm':
        raise ValueError('arm_camera_requires_motion_aware_tracking_validation')
    acceptance = json.loads((root/config['perception_acceptance']).read_text())
    if (acceptance.get('status') != 'passed' or acceptance.get('source_mode') != 'native_camera'
            or not acceptance.get('operator') or not acceptance.get('recorded_at')):
        raise ValueError('native_workstation_acceptance_required')
    for name in ['config/phase2/zones.yaml', 'config/phase2/perception.yaml', 'config/environment.yaml']:
        if acceptance.get('config_hashes', {}).get(name) != digest_file(root/name):
            raise ValueError('perception_configuration_changed')
    if not acceptance.get('evidence'):
        raise ValueError('perception_evidence_missing')
    for item in acceptance['evidence']:
        path = (root/item['path']).resolve()
        if not path.is_relative_to(root) or digest_file(path) != item['sha256']:
            raise ValueError('perception_evidence_changed')
    zones = yaml.safe_load((root/'config/phase2/zones.yaml').read_text())
    if zones.get('status') != 'calibrated' or set(config['zone_to_pose']) != set(zones.get('regions', {})):
        raise ValueError('calibrated_zone_mapping_required')
    if acceptance.get('station_changes_correct') != 10 or acceptance.get('station_changes_total') != 10 or acceptance.get('unnecessary_repositions') != 0 or acceptance.get('stationary_duration_s', 0) < 120:
        raise ValueError('workstation_acceptance_trials_missing')
    limits = Limits(**data['limits'])
    library = PoseLibrary(project(data, 'live', data['setup']['hardware_sha256']), limits, 'live')
    return data, library


def inspect(root, config):
    try:
        verify_approvals(root, config)
        blockers = []
    except (ValueError, KeyError, TypeError, OSError) as exc:
        blockers = [str(exc)]
    return {'dry_run': True, 'mode': 'hold', 'live_ready': not blockers, 'blockers': blockers,
            'hardware_opened': False, 'config': config}
