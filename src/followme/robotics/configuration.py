"""Phase 3 configuration only; independent of environment/vision preflight."""
from pathlib import Path
from .contracts import Limits


def load_robot_config(root):
    import yaml
    config = yaml.safe_load((Path(root)/'config/robot.yaml').read_text())
    if config.get('schema_version') != 1:
        raise ValueError('unsupported_robot_config')
    if config.get('backend') != 'fake' or config.get('dry_run') is not True or config.get('mode') != 'hold':
        raise ValueError('phase3_requires_fake_hold_defaults')
    if config.get('live_movement_enabled') is not False or config.get('gripper_commands_enabled') is not False:
        raise ValueError('phase3_live_movement_disabled')
    if config.get('arm_units') != 'degrees' or config.get('gripper_units') != 'percent':
        raise ValueError('invalid_units')
    if config.get('calibration_id') != 'follower_arm' or not str(config.get('follower_port','')).endswith(
            'usb-1a86_USB_Single_Serial_5C4C128613-if00'):
        raise ValueError('unexpected_follower_identity')
    return config, Limits(**config['limits'])
