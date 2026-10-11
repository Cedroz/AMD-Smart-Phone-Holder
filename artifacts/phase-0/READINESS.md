# Phase 0 Readiness Record

Overall outcome: **blocked — physical acceptance evidence pending**.

Recorded: October 10, 2026, 15:21 America/Los_Angeles. Automated inspection performed by Codex; physical operator observations are not yet recorded.

## Verified results

- Both expected USB serial adapters were connected and had no existing process owners at inspection time.
- Follower `5C4C128613` resolves to `/dev/ttyACM1`; leader `5C4C128105` resolves to `/dev/ttyACM0`. Arm identities follow the session's unplug test.
- All six follower motors' calibration offsets and ranges match `follower_arm.json` on device.
- All six leader motors' calibration offsets and ranges match `leader_arm.json` on device.
- Calibration files were retained. Register matches establish stored calibration consistency, not mechanical alignment or payload capability.
- All twelve motors reported torque disabled. Inspection left torque unchanged and issued zero movement commands.
- Both cameras returned 640 × 480 frames. Camera `/dev/video0` reported YUYV; `/dev/video2` reported MJPG. Reported frame rates were 30 FPS, not a measured sustained throughput result.

Evidence:

- [Device, calibration, position, torque, and camera inventory](evidence/20261010T152136.086253-0700/inventory.json)
- [USB inventory](evidence/20261010T152136.086253-0700/lsusb.txt)
- [Network addresses](evidence/20261010T152136.086253-0700/network.txt)
- [Camera `/dev/video0` screenshot](evidence/20261010T152136.086253-0700/usb-icSpring_icspring_camera-video-index0.jpg)
- [Camera `/dev/video2` screenshot](evidence/20261010T152136.086253-0700/usb-icSpring_icspring_camera_202404160005-video-index0.jpg)

The first screenshot shows a room with multiple people; the second shows a close rotated view of the robot assembly. The user confirmed the static camera corresponds to port 8000 and the arm-mounted camera to port 8001. These images do not verify mounts, power ratings, workstations, or face visibility from both intended screen positions.

Both camera-only preview processes were restored using stable device paths:

- Static/external camera: http://169.233.72.8:8000/ — `/dev/video0` at inventory time.
- Arm-mounted/wrist camera: http://169.233.72.8:8001/ — `/dev/video2` at inventory time.

The previews run `tracker/track.py --dry-run`; no motor connection is opened. They occupy the cameras while running, so subsequent inspection captures will skip those busy devices. These are running terminal sessions, not automatically restarted services.

## Step outcomes

| Step | Outcome | Evidence still required |
|---|---|---|
| 0.1 Inventory | pending; automated discovery passed | Physical arm/controller markings, mount inventory, and labels |
| 0.2 Base/power/attachments | pending | Inspection, power compatibility reference, retention and cable photographs |
| 0.3 Payload | pending | Payload identity, component/total masses, dimensions, wrist offset |
| 0.4 Camera roles | pending; roles confirmed and frame capture passed | Fixed camera placement, workspace coverage, wrist visibility |
| 0.5 Workspace | pending | Marked A/B stations, robot region, obstacles and exclusions |
| 0.6 Physical stop | blocked | Mechanism identification and stationary/moving loaded test evidence |
| 0.7 Loaded positions | blocked | Stop gate first, then two supervised positions and 60-second holds |
| 0.8 Operating envelope | pending; artifact created | Physical measurements and restrictions populated in `hardware.yaml` |
| 0.9 Handoff | blocked | Required physical gates passed and operator review recorded |

## Repeatable inspection

From the project root:

```bash
source ./lerobot-env.sh
# Filesystem and process inventory; no serial connection:
python scripts/inspect_phase0.py
# Read calibration/feedback registers and capture camera evidence:
python scripts/inspect_phase0.py --read-motors --capture-cameras
# Hardware-independent inspection boundary tests:
python -m unittest discover -s test -p 'test_phase0_inspection.py' -v
```

The inspection skips busy devices, never calls `Robot.connect()`, blocks bus writes, and disconnects with `disable_torque=False`, including after handshake failure. Timestamped evidence directories preserve previous runs. No motor-ID configuration or recalibration commands are part of this workflow.

## Operator evidence to complete

Fill the corresponding fields in [hardware.yaml](hardware.yaml), attach evidence, and record operator/date for each item:

- [ ] Physical model/controller markings and secure base/holder verified.
- [ ] Power supply compatibility verified against actual markings/documentation.
- [ ] Phone/surrogate, all carried masses, dimensions, and wrist offset measured.
- [x] External and wrist camera paths confirmed by user; positions and visibility still require recording.
- [ ] Station A/B and separated robot operating region marked and photographed.
- [ ] Physical stop identified and tested according to [Step 0.6](PLAN.md#06--establish-and-test-the-physical-stop-procedure).
- [ ] Two loaded viewing locations tested according to [Step 0.7](PLAN.md#07--check-two-loaded-viewing-positions-and-cable-clearance).
- [ ] Retention, cable clearance, readability, and 60-second hold results recorded.
- [ ] Final review confirms all required gates passed; trajectories remain unapproved until Phase 4.

Do not infer a physical pass from this checklist being present. Update outcomes only after the associated observation/test has been completed. Any change to payload, mount, base, cable routing, power, stop mechanism, or boundaries invalidates the affected physical checks.
