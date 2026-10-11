# FollowMe: Hackathon MVP Specification — Phases 0–8

Individual phase specifications:

- [Phase 0 — Hardware and Operating Envelope](phase-0/PLAN.md)
- [Phase 1 — Reproducible Robotics and Vision Environment](phase-1/PLAN.md)
- [Phase 3 — Hardware Adapter and Trajectory Controller](phase-3/PLAN.md)
- [Phase 4 — Teach Screen Poses and Validate Transition Paths](phase-4/PLAN.md)
- [Phase 5 — Integrated Workstation Following](phase-5/PLAN.md)
- [Phase 2 — Face and Workstation Tracking Without Motion](phase-2/PLAN.md)
- [Phase 6 — Optional Closed-Loop Wrist-Camera Refinement](phase-6/PLAN.md)

## Summary and decisions

Build a two-workstation demonstration: a fixed camera detects sustained user location changes, and the SO-101 moves a phone between two recorded, validated viewing poses. Small head movements should leave the phone stationary.

Use the existing Python 3.12 and LeRobot 0.6.1 installation. Start in **Hold**; an operator explicitly enables **Follow**. Wrist-camera refinement is optional. Phase 8 covers evaluation and rehearsal, extending the attachment’s numbered phases 0–7.

Exclude model training, ROS, inverse kinematics, stereo reconstruction, voice commands, and personalized placements from MVP completion.

Known hardware:

- Follower adapter: `5C4C128613`; calibration ID: `follower_arm`.
- Leader adapter: `5C4C128105`; calibration ID: `leader_arm`.
- Cameras: currently `/dev/video0` and `/dev/video2`; verify physical roles and store their `/dev/v4l/by-id/` paths.
- Laptop’s current address: `169.233.72.8`; treat it as configurable.

The existing tracker remains a diagnostic tool. Its current direct camera-error-to-joint loop will not drive the new autonomous workflow.

## Architecture and interfaces

Implement a Python package under `src/followme`, with configuration, calibration utilities, a browser interface, and tests.

**Runtime ownership**

- The main application owns camera capture, planning, telemetry, and FastAPI.
- One robot worker exclusively owns the follower connection and all motor reads and writes.
- A supervised perception subprocess uses a separate vision environment, preserving the working LeRobot environment. MediaPipe declares an overlapping OpenCV dependency, so keep the installations separate. [MediaPipe package](https://pypi.org/project/mediapipe/)
- Camera capture maintains one latest frame per camera. Perception receives timestamped JPEG frames through local IPC, with at most one inference request outstanding; replace unsent frames rather than accumulating them.
- A failed perception worker inhibits autonomous motion. A failed robot worker latches a fault and requires operator recovery.

**Internal contracts**

| Contract | Required information |
|---|---|
| `CameraFrame` | Camera role, sequence, monotonic capture timestamp, image |
| `UserState` | Source timestamp, face center, tracking validity, ambiguity, candidate and stable zones |
| `TargetRequest` | Request ID, approved pose name, reason, source timestamp |
| `RobotStatus` | Operating mode, behavior state, measured joints, current/target pose, motion permission, fault reason |

Do not invent per-face confidence values absent from the detector output. Use configured detection thresholds and explicit validity/ambiguity flags.

**Operator interfaces**

- CLI: `followme inspect`, `followme calibrate`, `followme teach`, and `followme run --dry-run`.
- Modes: `hold` and `follow`; behavior states: `SEARCHING`, `TRACKING`, `REPOSITIONING`, `ALIGNING`, `HOLDING`, `FAULT`, `STOPPED`.
- HTTP: `GET /api/status`, `POST /api/mode`, `POST /api/stop`, `POST /api/reset`, `GET /api/recipe`, `POST /api/recipe/next`.
- Telemetry: `WS /ws/state`; camera previews: `GET /streams/{role}`.
- Operator controls submit requests through the same motion gate. No endpoint accepts arbitrary joint targets.

## Phase specifications and exit gates

| Phase | Implementation | Exit gate |
|---|---|---|
| **0 — Hardware and operating envelope** | Verify mounts, payload retention, cable clearance, camera roles, and two reachable viewing locations. Establish a physical stop mechanism and test its loaded-arm consequences. Record the permitted sweep and payload. | Two viewing positions are feasible; the payload remains retained during normal operation and the tested stop behavior. |
| **1 — Reproducible environment** | Preserve the project environment and calibration IDs. Pin LeRobot 0.6.1; add FastAPI and configuration dependencies. Create a separate perception environment containing MediaPipe and its OpenCV dependency. Lock both environments and download/checksum the official Face Landmarker model. | Both environments pass import checks; the supervisor can launch perception; both cameras and the follower are identified without automatic motion. |
| **2 — Perception without motion** | Capture both cameras at 640 × 480 using MJPEG where supported. Run Face Landmarker in `VIDEO` mode in the perception worker. Configure up to two faces so a second person can trigger ambiguity. Smooth face position with EMA, initially α = 0.25. Calibrate two workstation regions with a neutral boundary region. | Ten scripted station changes classify correctly; standing still and brief head turns do not produce repeated zone changes. |
| **3 — Hardware adapter and controller** | Wrap the installed SO-family LeRobot API. Implement connection, measured feedback, bounded trajectory execution, Hold, Stop, and fault reporting. Add a fake adapter for development. Keep live pose execution disabled until Phase 4 validation. | Fake-hardware tests demonstrate single I/O ownership, cancellation, limits, feedback faults, and stale-request rejection. |
| **4 — Teach poses and transitions** | Record `station_a` and `station_b` using the leader or a supervised teaching utility. Record and validate both directional waypoint paths with the intended payload. Store joint values, units, limits, durations, tolerances, and validation metadata. | Ten supervised A→B→A cycles complete without cable interference, unacceptable sag, limit violations, or feedback faults. |
| **5 — Integrated following** | Map stable zones to approved poses. Require fresh tracking, an unambiguous user, Follow mode, and a different destination. Use 0.75-second dwell and a two-second cooldown. Do not queue successive movements during an active transition. | Ten station changes select the correct pose; boundary jitter and head turns leave the phone stationary; Hold interrupts motion. |
| **6 — Optional wrist refinement** | Calibrate the preferred screen-viewing image point and one useful joint’s response direction. After arrival, observe→correct→observe using fresh wrist frames. Start with corrections ≤1°, total offset ≤3°, and at most three attempts inside a physically validated correction envelope. | Measured alignment improves in controlled trials without oscillation. Otherwise disable refinement and retain the two-pose MVP. |
| **7 — Phone interface and console** | Serve a responsive recipe page and operator console from FastAPI. Show state, zone, tracking age, pose, faults, and both previews. Provide Follow, Hold, Stop, reset, and recipe navigation. Bind to the shared LAN; use a per-session token for control requests and restrict accepted browser origins. | The phone receives recipe updates; console controls work; browser loss returns the system to Hold. |
| **8 — Evaluation and rehearsal** | Run repeatable static, reactive, and FollowMe trials. Use the same user movement script; every moving baseline uses the same approved paths and limits. Export metrics, logs, operating instructions, and backup demonstration footage. | Complete the acceptance suite below and demonstrate the system repeatedly from a cold start. |

MediaPipe’s `VIDEO` mode blocks its calling thread and requires frame timestamps; keeping it in the perception worker preserves controller responsiveness. [Face Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)

## Control rules and failure handling

- Default to dry-run and Hold. Dry-run must never open the robot serial port.
- Use explicit degree units for arm joints; retain the SDK’s separate gripper units and keep the gripper fixed during following.
- Reuse existing calibration. Recalibrate only when verification shows it is necessary; follow the documented SO-101 procedure. [SO-101 guide](https://huggingface.co/docs/lerobot/v0.6.1/so101)
- Generate quintic joint interpolation along approved waypoint segments. Check joint positions, velocity, acceleration, and per-update changes before execution.
- Initial software ceilings: 10°/s velocity, 20°/s² acceleration, and 2° relative target changes. Phase 0/4 measurements may reduce these ceilings. Absolute joint limits and path clearance must come from the actual setup.
- Target a 20 Hz robot loop, 10 Hz perception, and 5 Hz telemetry. Benchmark these rates before enabling autonomy.
- Reject observations and requests older than 500 ms. Loss of the fixed camera, ambiguous users, invalid tracking, or operator Hold cancels autonomous motion.
- Software cancellation interrupts the active trajectory and applies the tested hold procedure. It is distinct from the physical stop and cannot guarantee stopping during lost communication.
- Critical faults and Stop latch motion inhibition. Reset clears the latch only after fresh feedback and fault resolution, then returns to Hold.
- Do not automatically park or resume an interrupted trajectory. If measured position is outside an approved starting pose, require supervised restoration before following.
- Disconnect normally with holding torque retained, matching the current project behavior. Power-loss retention must be addressed mechanically.
- If the optional wrist camera fails, disable refinement; continue workstation following only while the fixed camera remains valid.
- Require an operator heartbeat while Follow is enabled; three seconds without it triggers Hold.

## Validation, metrics, and fallback

**Automated tests**

Use fake hardware, controlled timestamps, and synthetic perception events to verify dwell, cooldown, ambiguity, expired requests, stop priority, interrupted trajectories, invalid paths, unit handling, and fault recovery. Test API controls against the fake adapter.

**Physical acceptance**

- Ten successful A→B→A cycles with the validated payload.
- Two minutes of stationary work and head movements without unnecessary repositioning.
- Camera disconnect, target loss, UI disconnect, and Stop tested during both idle and movement.
- Measure software cancellation latency; require cancellation handling within 250 ms under healthy communication. Record actual loaded-arm stopping behavior separately.
- Confirm dry-run sends zero motor commands.

**Evaluation output**

Record monotonic timestamps, observations, decisions, requested and measured joints, motion events, faults, and recipe changes in JSONL. Report correct destination rate, repositioning latency, unnecessary movement count, and valid wrist-alignment error when available. Report measured results without assumed improvements.

If face-based zones fail the head-turn test, add shoulder-based Pose Landmarker tracking before enabling Follow. If refinement fails, omit it. If the robot cannot validate two positions, present the perception/dry-run demonstration and report the physical MVP gate as unmet.

MVP completion requires phases **0–5, 7, and 8**. Phase **6** remains a gated extension.

