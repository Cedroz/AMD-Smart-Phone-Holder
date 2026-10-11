# Phase 4 — Teach Screen Poses and Validate Transition Paths

Status: **Software implemented — physical pose teaching and path acceptance pending**.

Implementation workflow: [README.md](README.md). Verification and remaining physical gates: [READINESS.md](READINESS.md). The independent entrypoint is `python -m followme.phase4`; it does not modify other phases' CLI registrations or production configuration.

Parent specification: [FollowMe MVP plan](../PLAN.md).
Hardware handoff: [Phase 0 readiness](../phase-0/READINESS.md) and [hardware inventory](../phase-0/hardware.yaml).
Environment handoff: [Phase 1 specification](../phase-1/PLAN.md).

## Objective and completion boundary

Record two useful phone viewing poses, `station_a` and `station_b`, and independently validate both directional waypoint paths with the intended payload. Produce a versioned pose library that the Phase 5 planner can reference by name without supplying arbitrary joint commands.

Completion requires ten supervised A→B→A cycles and passing stop/hold, feedback, clearance, and loaded-position checks. Pose capture, mathematical interpolation, or fake-hardware tests alone cannot establish physical path clearance.

Only these two poses and their explicit transitions are in scope. Automatic parking, fine wrist-camera corrections, inverse kinematics, and automatic return from an interrupted path are excluded. The separate Phase 6 correction envelope must be validated before any correction is enabled.

## Prerequisites and current constraints

- Phase 0 must pass its mount, payload, operating-boundary, physical-stop, and loaded-position gates before powered teaching/replay. Those gates are currently pending in the recorded handoff.
- Phase 1 environment checks and the Phase 3 single-owner controller must pass. Phase 3 must supply supervised teaching/validation mode, measured feedback, interruption handling, fault latching, and a fake adapter.
- Live validation mode accepts only an explicitly selected draft route after structural/dynamic checks and operator confirmation. It does not accept tracking requests or arbitrary browser joint targets. Default mode remains Hold/dry-run.
- Follower: stable adapter ending `5C4C128613`, calibration ID `follower_arm`. Leader: adapter ending `5C4C128105`, calibration ID `leader_arm`.
- Preserve the existing matching calibrations. Do not configure motor IDs or recalibrate to make a recorded pose pass.
- One operator teaches; a second observer watches the physical stop and workspace during powered tests.
- Arm joints use degrees with `use_degrees=True`. Gripper values use the SDK's 0–100 representation; the gripper remains fixed at the measured retained-payload setting throughout teaching and replay.

Offline schema, interpolation, fake-hardware tests, and evidence tooling can proceed while physical gates are pending. Do not mark physical validation complete until it occurs.

## Required outputs and interface

| Output | Purpose |
|---|---|
| `config/safe_poses.yaml` | Named poses, directional waypoint paths, limits, timing, and approval metadata |
| `artifacts/phase-4/READINESS.md` | Per-step outcomes, test results, failures, and Phase 5 handoff |
| `artifacts/phase-4/evidence/<session>/` | Raw teaching samples, feedback logs, clearance photographs, stop videos, trial results, and checksums |
| Teaching/validation CLI | Extend the planned `followme teach` interface with capture, route recording, offline checking, supervised replay, and approval |

CLI operations:

- `followme teach capture --pose station_a` (or `station_b`): record measured follower joints after the operator settles the pose.
- `followme teach record --from station_a --to station_b`: record supervised follower movement as a draft directional path.
- `followme teach check`: validate the library without opening serial hardware.
- `followme teach replay --from station_a --to station_b --dry-run`: execute against the fake adapter.
- `followme teach replay --from station_a --to station_b --live`: run one supervised candidate transition after all live-validation prerequisites pass.
- `followme teach approve --session <id>`: approve only an unchanged library whose complete physical evidence passes the acceptance gates.

These interfaces are required implementation work, not commands currently known to exist. All live operations use the Phase 3 robot worker; no teaching utility creates a second follower connection.

### Pose-library contract

- Root fields: `schema_version: 1`, calibration ID/digest, payload/setup identity, arm-joint units, fixed gripper target, limits, pose map, transition map, and validation record.
- Each pose stores all five degree-valued arm joints, capture timestamp, sample stability results, and evidence references. Keep gripper metadata separate from degree-valued joints.
- Each transition stores explicit `from`/`to` names, an ordered list of full-joint waypoints including endpoints, segment durations, approved controller rate, and directional validation evidence.
- Limits store measured absolute joint bounds, velocity/acceleration ceilings, relative-target ceiling, start/arrival tolerances, feedback deviation threshold, and clearance restrictions.
- Validation records store draft/approved/invalidated status, operator/date, calibration/setup fingerprints, library content digest, completed trials, and evidence references. Compute the digest from canonical motion-relevant content, excluding the approval record itself.
- Missing values, non-finite numbers, unknown joints, invalid units, mismatched calibration, or missing paths fail validation. Draft libraries cannot drive autonomous following.
- Write drafts and approvals atomically; retain previous versions and raw samples. No invented joint targets, physical clearances, or payload measurements.

## Individual steps

### 4.1 — Verify prerequisites and open a teaching session

1. Read Phase 0/1 evidence and Phase 3 controller test results. Record which physical gates passed, their operator/date, and any setup changes since validation.
2. Compare current calibration hashes and device identity with the handoff. Verify exclusive follower ownership before connecting through the controller.
3. Record payload identity/mass, base location, phone/camera mounting, cable routing, operating boundaries, and physical-stop mechanism as a setup fingerprint.
4. Disable tracking requests and wrist refinement. Start in Hold and confirm the observer can access the physical stop.
5. Create a timestamped draft session and retain the preceding approved library, if any.

**Pass:** prerequisites are recorded, no conflicting robot owner exists, and the session is identified. Otherwise live operations remain blocked.

### 4.2 — Establish joint limits, fixed gripper setting, and test thresholds

1. Import the measured operating restrictions from Phase 0. Check them against calibration conversion and current joint feedback; calibration register ranges alone are not collision-free workspace limits.
2. Record per-joint absolute bounds and evidence. Do not default missing bounds to the full calibration range.
3. Begin with software ceilings of 10°/s, 20°/s², and 2° maximum relative targets; use lower values where measured restrictions require them. Keep the relative-target ceiling as an additional SDK guard, not a trajectory validator.
4. Record the retained-payload gripper setting. Suppress leader gripper changes and refuse replay if the measured gripper is outside its recorded tolerance.
5. Initial test thresholds: start and arrival error ≤2° per arm joint; gripper tolerance ≤2 percentage points; feedback deviation >3° persisting for 300 ms causes a fault. Tighten thresholds if physical clearances demand it; record any required adjustment and its evidence before approval.
6. Use a 20 Hz target controller rate. Stale motor feedback older than 500 ms faults the session. Missed ticks must not cause catch-up jumps.

**Pass:** every limit has units, a source, and a finite value; all routes will use the same frozen thresholds during their acceptance trials.

### 4.3 — Teach and capture `station_a`

1. Use the supported leader teaching path through the single-owner controller. Filter leader input through limits and keep the gripper fixed. Do not move an unpowered loaded arm by forcing joints.
2. Slowly approach the Phase 0 candidate A location while checking the full assembly and cables. The user remains outside the robot operating region.
3. Adjust the phone's viewing orientation and confirm readability from station A.
4. Once settled, record two seconds of measured follower positions. Capture their median as the pose only when each arm joint's peak-to-peak variation is ≤1° and the gripper variation is ≤1 percentage point.
5. Hold for 60 seconds. Record drift, vibration, sag, base/mount displacement, cable tension, and readability against a fixed visual reference.
6. Reject visible continued drift, mount slip, persistent vibration, or exceeded limits. Preserve failed evidence and repeat after correction.

**Pass:** a complete measured A pose, stable capture, loaded hold, and viewing/clearance evidence exist.

### 4.4 — Teach and capture `station_b`

Repeat Step 4.3 at station B using the same payload, fixed gripper setting, limits, and thresholds. Preserve both sets of raw samples. Do not extrapolate B from A or mirror joint values.

**Pass:** B has its own stable capture and 60-second loaded hold; both positions are useful from the marked user stations.

### 4.5 — Record both directional transition paths

1. Start at measured A within the start tolerance. Teach A→B slowly and record timestamped measured follower joints, requested targets, fault state, and operator-marked waypoints.
2. Add explicit intermediate waypoints at changes of direction or tight clearances. Inspect the phone corners, camera bracket, cables, and the full arm sweep throughout travel.
3. Independently teach and record B→A. A reversed list is only a candidate; approval in one direction does not approve the reverse.
4. Convert operator-marked settled waypoints into the draft library, with each route starting/ending at its captured named pose. Retain the complete raw trajectory.
5. Do not automatically simplify the recording or shortcut between points. The generated replay path must be inspected and physically tested even if the manually taught movement looked clear.

**Pass:** two explicit draft routes exist with waypoint and continuous-sweep evidence. Neither is approved yet.

### 4.6 — Generate and validate bounded trajectories offline

1. Between each pair of recorded waypoints, use quintic smoothstep `s(u)=10u³−15u⁴+6u⁵`. Stop at segment endpoints; no blending across waypoint corners in this phase.
2. Choose each segment duration from the largest joint displacement and the configured dynamic limits. Enforce `T ≥ 1.875·|Δq|/v_max` and `T ≥ sqrt((10/√3)·|Δq|/a_max)` for every moving joint, then verify the sampled target changes against the relative-update ceiling.
3. Validate all waypoint coordinates, continuous segment extrema, velocities, accelerations, and controller samples. Reject nonpositive durations, missing endpoint matches, and gripper changes.
4. Run both routes against the fake adapter. Test out-of-range starts, missing limits, calibration mismatch, stale feedback, actual/requested deviation, and simulated clipping.
5. Compare commanded and returned SDK targets; unexpected clipping invalidates the run rather than silently accepting a different trajectory.

**Pass:** mathematical and fake-hardware checks pass. This is not physical collision validation.

### 4.7 — Conduct progressive supervised replay and interruption tests

1. Start at the approved candidate endpoint within tolerance. An arbitrary current position must not be connected to the route by automatic interpolation.
2. Replay each draft direction once at reduced speed, retiming segments to preserve acceleration/update limits. Inspect the entire path and log measured feedback.
3. After reduced-speed checks pass, validate both directions at the exact intended production timing. Approval applies to those durations and controller settings.
4. Test operator Hold and software Stop at early, middle, and late route positions. Measure software cancellation handling; require ≤250 ms under healthy communication. Record actual mechanical stopping behavior separately.
5. Use the Phase 0 physical stop procedure for a supervised loaded moving stop test with retention and observer in place. Do not intentionally create a hardware fault by unplugging a loaded arm's serial cable.
6. Stop/fault latches inhibit further movement. Preserve torque on normal disconnect as specified; power interruption behavior follows the tested physical procedure.
7. After interruption, hold the measured position. Require explicit supervised restoration to A or B before another route; never automatically park, resume, or jump to an endpoint.

**Pass:** both timed paths are clear, loaded interruption behavior is documented, cancellation timing passes, and recovery remains operator-controlled.

### 4.8 — Run ten A→B→A acceptance cycles

1. Freeze the library, payload/setup fingerprint, calibration digest, thresholds, and route timings. Assign trial IDs 1–10.
2. Run each full cycle with the observer and log start checks, target/measured joints, arrival errors, duration, faults, stop events, and visual clearance observations.
3. Require every arrival to remain within the configured tolerance for 500 ms before declaring completion. Fault if settling has not completed within five seconds after the planned trajectory end.
4. Record readability, mounting stability, sag, vibration, and cable behavior in both positions. Store representative footage covering the full sweep.
5. A fault, collision/contact, mount movement, unacceptable drift, unexpected clipping, or failed arrival fails that cycle. Do not exclude it from the report. Correct the cause, retain failure evidence, and restart the ten-cycle acceptance sequence for the modified library/setup.

**Pass:** ten consecutive complete cycles pass for the unchanged library and intended payload. If a surrogate was used, approval is restricted to that exact surrogate/setup.

### 4.9 — Approve the library and hand off to Phase 5

1. Review all physical, mathematical, cancellation, feedback, and ten-cycle evidence. Verify calibration and library digests still match the tested session.
2. Write directional approvals only when every required check passes. Include operator/date, evidence links, exact timing, thresholds, and the setup fingerprint.
3. Document Phase 5's allowed pose names and routes, fixed gripper requirement, endpoint-start checks, and interruption recovery procedure.
4. Record readiness and unresolved restrictions. Update the hardware handoff's approved-route references without turning on Follow; live following still requires the Phase 5 gates and explicit operator action.

**Pass:** the unchanged library is consumable by the planner/controller, and an operator can identify exactly what was validated.

## Acceptance checklist

- [ ] Phase 0 physical gates and Phase 3 controller gates passed.
- [ ] Calibration preserved and verified; payload/setup fingerprint recorded.
- [ ] Absolute bounds, units, timings, and tolerances documented.
- [ ] Fixed gripper enforced in teaching and replay.
- [ ] A and B individually captured and held for 60 seconds.
- [ ] A→B and B→A independently recorded and physically validated.
- [ ] Continuous/sampled dynamic checks and fake-hardware failure tests pass.
- [ ] Hold/Stop interruptions and supervised recovery pass.
- [ ] Software cancellation ≤250 ms under healthy communication; physical stop behavior separately evidenced.
- [ ] Ten consecutive unchanged A→B→A cycles pass with the intended payload.
- [ ] Approved content digest, evidence, and Phase 5 handoff recorded; Follow remains disabled by default.

## Invalidation and fallback

Changing calibration, joint bounds, payload, mounts, base location, cable routing, workspace/obstacles, physical-stop mechanism, route geometry, timings, controller rate, or tolerances invalidates affected approvals. Retain old evidence as superseded and rerun the required checks; never silently reuse approval metadata.

If two loaded paths cannot pass, keep autonomous movement blocked and demonstrate camera-only/dry-run behavior. If only one direction passes, do not enable workstation following: the MVP requires both. Missing wrist-camera face visibility does not fail Phase 4; it disables refinement until Phase 6 can establish its own envelope.
