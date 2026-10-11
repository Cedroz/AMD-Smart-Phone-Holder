# Phase 3 — Hardware Adapter and Trajectory Controller

Status: **Implemented — independent software/fake-hardware acceptance passed; live physical validation pending**.

Execution results: [READINESS.md](READINESS.md). Usage and handoff: [README.md](README.md).

Parent specification: [FollowMe MVP plan](../PLAN.md).
Inputs: [Phase 0 readiness](../phase-0/READINESS.md), [hardware evidence](../phase-0/hardware.yaml), and [Phase 1 environment spec](../phase-1/PLAN.md).

## Objective and completion boundary

Implement a single-owner SO-101 hardware adapter, bounded waypoint trajectory execution, Hold/Stop behavior, fault reporting, and a deterministic fake adapter. Demonstrate the controller's behavior through hardware-independent tests.

Phase 3 completes with fake-hardware acceptance. Optional live inspection reads calibration and feedback only. Live arming and trajectory execution remain disabled until the physical gates and individual paths are validated through Phase 4. No demonstration motion is required to finish this phase.

The current `src/followme` package contains environment and vision-worker components. Extend that package and its existing CLI; do not replace the camera diagnostics or turn the vision worker into a robot controller.

## Required outputs

| Output | Responsibility |
|---|---|
| `src/followme/robotics/` | Contracts, fake/LeRobot adapters, ownership worker, trajectory generation, and motion gate |
| `config/robot.yaml` | Device identity, units, software ceilings, feedback/watchdog settings, and disabled-live defaults |
| `config/safe_poses.yaml` | Empty production pose/transition library awaiting Phase 4 measurements |
| Test fixtures | Synthetic pose/path library clearly labeled fake-only; never accepted by the live adapter |
| `scripts/check_phase3.py` | Hardware-independent acceptance runner and timing report |
| `artifacts/phase-3/READINESS.md` and `evidence/` | Test results, cancellation measurements, unresolved physical gates, and Phase 4 handoff |

Creating these artifacts does not mark the phase passed. Record checks as `pending`, `passed`, `failed`, or `blocked` with evidence.

## Interfaces and invariants

The controller is the only entry point for motion requests. Adapter operations are private to the robot worker.

| Interface | Contract |
|---|---|
| `connect(read_only=True)` | Worker-owned connection; inspection never configures motors or changes torque |
| `read_joint_positions()` | Complete measured snapshot, monotonic sampling time, explicit units |
| `request_safe_pose(TargetRequest)` | Named destination only; accept/reject result with reason and request ID |
| `request_hold(reason)` | Cancel pending/active movement; switch to Hold without latching a stop |
| `stop_motion(reason)` | Priority interruption; latch STOPPED and clear all motion requests |
| `reset()` | Clear resolved latches only; return to Hold and never resume a path |
| `status()` | Immutable snapshot; no caller-side serial reads |
| `disconnect()` | Worker-owned cleanup preserving existing holding torque |

All timestamps use a monotonic clock. All arm joints use degrees; gripper readings use percent. Autonomous commands contain the five arm joints and omit the gripper. No arbitrary joint-target endpoint or public UI operation is introduced.

Default backend is fake, default mode is Hold, and live movement is disabled. Missing physical evidence, absolute limits, or path approval is a rejection, not a request to infer values.

## Individual steps

### 3.1 — Define contracts, configuration, and rejection reasons

1. Implement immutable `JointSnapshot`, `TargetRequest`, `RobotStatus`, pose/transition records, and request results. Include sample/source timestamps, request IDs, units, backend, and fault/rejection reason.
2. Validate the exact expected joint set: shoulder pan/lift, elbow flex, wrist flex/roll, and gripper feedback. Reject missing, extra, nonfinite, or incorrectly unit-labeled data.
3. Configure follower stable path ending `5C4C128613`, calibration ID `follower_arm`, cameras disabled inside the adapter, degree normalization, and retained torque on disconnect.
4. Set initial ceilings: 10°/s, 20°/s², 2° relative target change, 20 Hz target loop, 500 ms maximum observation/request age, and three-second operator heartbeat timeout. Configuration may reduce these ceilings, not raise them for live use without new validation.
5. Default feedback deviation tolerance to 3° sustained for 300 ms, pose arrival tolerance to 2° for 250 ms, and segment settling timeout to two seconds. These are software starting values requiring Phase 4 confirmation before live use.
6. Define stable reason codes for stale/future timestamp, duplicate request, busy worker, wrong mode, unknown pose, unapproved path, invalid limits, calibration mismatch, feedback deviation, heartbeat loss, transport failure, and worker failure.

**Pass:** invalid configurations and data fail closed; configuration and contract tests require no SDK import or hardware.

### 3.2 — Implement deterministic fake hardware

1. Provide an injectable monotonic clock and fake adapter that records every read, write, connection, and cleanup operation with owning thread identity.
2. Support perfect tracking, delayed tracking, stuck joint, invalid feedback, calibration mismatch, clipped command, serial exception, and bounded slow-I/O scenarios.
3. Provide synthetic A/B poses, directional waypoint paths, limits, and approval metadata in test fixtures. Mark their backend `fake`; the live loader must reject them.
4. Distinguish simulated command counts from physical command counts. Dry-run may simulate trajectories but must issue zero physical motor commands and never construct a serial connection.

**Pass:** tests can advance time and inject failures without sleeping or loading LeRobot; dry-run remains hardware-free.

### 3.3 — Implement read-only LeRobot inspection and explicit arming

1. Lazy-import the installed LeRobot 0.6.1 SO-family classes through `lerobot.robots.so_follower`. Use the existing Phase 0 inspection boundary as the reference for calibration verification.
2. Acquire a process-level device lock, refuse adapters already in use, and open the bus only inside the worker. Keep the lock until serial cleanup completes.
3. Read and compare on-device calibration with the saved file before any writable initialization. Verify all motors respond and report complete finite positions. A mismatch faults; never recalibrate automatically.
4. Do not use `Robot.connect(calibrate=False)` for inspection: the installed implementation still calls `configure()`, which changes motor settings and torque. Read-only mode uses bus connection/read calls and disconnects without disabling torque.
5. Implement a separate guarded arming path for later Phase 4 use. It requires explicit operator initiation, completed physical gates, matching payload/setup evidence, a recognized approved starting pose, and verified position operating mode. Refuse unsupported mode/settings rather than configuring firmware automatically.
6. For joints with torque disabled, latch their measured present positions as initial goals before enabling arm-joint torque. For torque-enabled joints, first verify goals agree with the recognized stationary pose. Do not change gripper goal or torque; retention remains a physical prerequisite.
7. Recheck measured position after arming. Any unexpected deviation faults and prevents trajectory submission. Do not run calibration, motor-ID setup, PID tuning, or general SDK configuration in this path.

**Pass:** mocked SDK tests prove inspection performs no motor-register writes, mismatched calibration prevents arming, and denied arming leaves torque unchanged. Actual arming is not executed during Phase 3 acceptance.

### 3.4 — Enforce one I/O owner and priority control

1. Run one robot worker thread. Assert its identity on every adapter call; callers receive snapshots instead of invoking SDK methods.
2. Use separate priority flags for Stop, faults, and Hold; an active-motion slot; and at most one pending target. Check interruptions before/after each I/O operation and before sending another trajectory sample.
3. Reject motion requests while repositioning; do not build a backlog or replace an active path with a new path. Repeated request IDs are rejected within a controller session.
4. Stop takes priority over all normal actions. Preserve any underlying fault reason when STOPPED is also active. Hold cancels pending requests and active trajectories.
5. Publish immutable status snapshots and worker heartbeat without requiring the API/vision thread to touch serial.
6. A supervisor may set inhibition flags and report a dead/stalled worker; it must not become a second motor writer or forcibly release the serial lock while the original worker is still alive.

**Pass:** concurrent submissions, Hold, and Stop cannot produce overlapping writers or delayed queued movements.

### 3.5 — Validate the pose library and generate bounded paths

1. Keep the production library empty until Phase 4. Pose records contain measured arm joints, explicit units, unchanged-gripper requirement, and validation metadata identifying payload/setup/calibration revision.
2. Transition records contain source/destination pose names, ordered waypoints, segment durations, joint limits, and directional approval. Reject missing approval, mixed units, fake fixtures, changed setup, and unapproved reverse paths.
3. Recognize the current pose only when measured joints are within its configured tolerances. Starting outside every approved pose requires supervised restoration; never invent a path back to an anchor.
4. Generate each segment using `s(u)=10u³−15u⁴+6u⁵`, `q(u)=q0+s(u)(q1−q0)`. Validate endpoints, all scheduled targets, per-step changes, and continuous velocity/acceleration extrema.
5. For each joint, require duration at least `1.875 × abs(delta)/velocity_limit` and `sqrt((10√3/3) × abs(delta)/acceleration_limit)`. Reject insufficient recorded durations rather than silently changing a physically approved path. Phase 4 teaching may use these formulas to propose durations before validation.
6. Check target-to-measured relative differences before SDK dispatch. Treat SDK clipping as an unexpected mismatch and fault; do not silently continue a trajectory that differs from its validated commands.
7. State explicitly that joint-space checks do not prove collision clearance. Full physical sweep/path approval comes from Phase 4.

**Pass:** analytic and sampled limits reject fast, discontinuous, out-of-range, or unapproved paths before their first command.

### 3.6 — Execute, monitor feedback, and report arrival

1. At each tick, read fresh feedback, evaluate interruption/motion gates, and send at most one scheduled sample. Use the actual monotonic elapsed time rather than assuming every tick is on schedule.
2. If a tick is more than one configured period late, interrupt the path; do not send a burst of catch-up commands or skip ahead to a distant target.
3. Record requested targets, targets actually returned by SDK dispatch, measured joints, feedback age, loop timing, request IDs, and mode changes.
4. Track persistent joint error against the commanded path with the configured tolerance/time window. Invalid feedback, transport exceptions, sustained error, or clipping latches FAULT.
5. At each waypoint endpoint, wait for measured convergence before starting the next segment. Missing convergence within the settling timeout faults.
6. Declare arrival only after measured destination convergence for the arrival dwell. Then clear the target and enter TRACKING in Follow mode or HOLDING in Hold mode.

**Pass:** fake tests verify completed movement depends on measured feedback; stalled or late execution cannot be reported as successful.

### 3.7 — Implement cancellation, watchdogs, and recovery

1. Expired tracking, ambiguity, fixed-camera loss, perception-worker failure, or operator heartbeat expiry cancels movement and returns to Hold. Critical hardware/worker failures latch FAULT; operator Stop latches STOPPED.
2. Cancellation first prevents all further trajectory samples. With fresh valid feedback and functioning transport, apply the measured-position hold procedure through the same worker. Verify relative target bounds before sending the hold target.
3. A missing/stale measurement or failed hold write latches a fault and reports `hold_unconfirmed`. Do not fall back to a previous distant target, release torque, or claim a physical stop occurred.
4. Keep the gripper unchanged. Do not automatically park, reconnect, resume, or move toward an anchor following an interruption.
5. Set finite SDK serial timeouts and retry budgets and record their measured total I/O duration. Target cancellation handling within 250 ms under healthy communication; a blocked SDK call cannot be safely interrupted by a second serial writer. Physical stop remains the fallback for lost/stalled communication.
6. Reset requires explicit operator intent, resolved failure, valid fresh feedback, and intact calibration/physical gates. Return to Hold. If current position is not an approved anchor, remain movement-inhibited pending supervised restoration.
7. On normal shutdown, request Hold and join the worker; close serial with torque retained. If shutdown cannot confirm transport/worker cleanup, report it and retain the fault indication rather than asserting a safe physical state.

**Pass:** interruption and reset tests show no queued motion, automatic parking, silent fault clearing, torque release, or automatic resumption.

### 3.8 — Add CLI diagnostics and telemetry evidence

1. Extend the existing CLI with fake-backend controller checks and read-only robot inspection. Default commands remain fake/dry-run; a live read requires an explicit inspection option.
2. Keep production live movement disabled. Reserve the arming/execution implementation for Phase 4's supervised harness, which must supply the validated gate evidence.
3. Write JSONL controller events with monotonic timestamps, session/request IDs, backend, mode, intended/dispatched targets, measured feedback, interruptions, faults, and physical-command count.
4. Add `scripts/check_phase3.py` to run the acceptance suite and write a machine-readable result plus a human summary under timestamped evidence directories.
5. Leave both camera-only previews running; the robot adapter contains no camera configuration and never competes for camera devices.

**Pass:** developers can reproduce fake acceptance and inspect status from a fresh runtime environment without opening cameras or moving hardware.

### 3.9 — Run acceptance tests and hand off to Phase 4

Required test scenarios:

| Group | Cases and expected result |
|---|---|
| Defaults | No serial construction in fake/dry-run; initial Hold; empty live pose library rejects movement |
| Ownership | Concurrent requests never create a second writer; other-thread adapter calls fail; busy device refuses connection |
| Freshness | Past age >500 ms, future timestamp, expired tracking, and heartbeat loss prevent/cancel movement |
| Calibration/units | Missing/mismatched calibration, nonfinite/incomplete joints, wrong units, and changed payload revision reject live use |
| Trajectories | Velocity/acceleration extrema, joint/relative limits, short durations, reverse-path approval, and SDK clipping handled correctly |
| Feedback | Delayed/stuck/invalid joints, arrival dwell, waypoint settling timeout, loop lateness, and serial faults |
| Interruption | Hold/Stop before dispatch, mid-segment, and during settling; no automatic park or resume; unconfirmed hold reported |
| Recovery | Reset while unresolved rejects; successful reset enters Hold; interrupted non-anchor pose remains inhibited |
| Cleanup | Failed handshake/read/dispatch closes when possible without disabling torque; dead worker never triggers competing I/O |

Run deterministic tests with controlled time. Separately run at least 100 randomized real-thread cancellation trials against fake hardware with bounded simulated healthy-I/O delays. Report maximum observed handling latency and test conditions; require no trial above 250 ms. This is a software timing result, not measured physical braking latency.

Record readiness only after the suite passes. Handoff includes adapter contracts, synthetic examples, production-empty library schema, watchdog budgets, holding/fault behavior, and exact commands. Phase 4 must supply measured poses, approved directional paths, payload/clearance evidence, physical-stop results, and live timing/feedback validation before enabling movement.

## Acceptance checklist

- [ ] Fake/dry-run is the default and opens no serial connection.
- [ ] Read-only inspection preserves calibration/settings/torque and has single ownership.
- [ ] Production live execution rejects missing physical/path evidence.
- [ ] Unit, library, freshness, velocity, acceleration, and relative-target validation pass.
- [ ] Measured feedback controls arrival and detects divergence.
- [ ] Hold, Stop, watchdogs, faults, reset, and cleanup pass the interruption suite.
- [ ] Fake cancellation trials meet the 250 ms handling target under recorded healthy-I/O conditions.
- [ ] Neither adapter nor CLI opens cameras or disrupts preview links.
- [ ] JSONL evidence distinguishes simulated motion from physical commands.
- [ ] Readiness and Phase 4 handoff recorded; no physical path is implicitly approved.

## Failure and scope policy

SDK/runtime incompatibility blocks live inspection but does not justify upgrading LeRobot or recalibrating automatically. Keep the fake controller usable and record the exact mismatch. Transport/fault behavior that cannot meet the software timing target remains failed until fixed; do not weaken the target to claim completion.

Face tracking and zone dwell are Phase 2/5 inputs, not new perception algorithms in this phase. Wrist correction and API/browser controls are Phase 6/7 work and must use this controller's same motion gate when added.
