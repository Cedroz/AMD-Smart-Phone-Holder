# Phase 6 — Optional Closed-Loop Wrist-Camera Refinement

Status: **Planned — disabled until correction calibration and physical validation pass**.

Parent specification: [FollowMe MVP plan](../PLAN.md).
Related specifications: [Phase 2 perception](../phase-2/PLAN.md), [Phase 3 controller](../phase-3/PLAN.md), and [Phase 4 pose/path validation](../phase-4/PLAN.md).

## Objective and completion boundary

After the arm arrives at an approved station pose, use fresh wrist-camera observations to measure screen alignment, propose a small correction on one calibrated arm joint, execute it through the existing robot worker, and observe the result again.

Phase 6 is optional. The two-workstation MVP remains valid with refinement disabled. Implement offline/fake-hardware behavior first; do not enable physical correction simply because tracking and base transitions work.

Use one joint only in this phase, selected by measured response and clearance. No joint is selected by name before calibration. Keep the gripper and other arm joints fixed. Exclude continuous visual servoing, multi-joint optimization, depth reconstruction, and inferred gaze/attention from this phase.

## Existing implementation and prerequisites

- The Phase 3 controller and robot worker exclusively own serial I/O. Extend them; do not connect to the arm from perception or an alignment utility.
- `JointSnapshot` expresses arm joints in degrees and gripper in percent. Preserve those units and the fixed gripper setting.
- The current production `config/safe_poses.yaml` has no approved poses or paths. Live refinement is blocked until Phase 4 supplies them.
- Phase 0 physical gates, Phase 4 loaded path approval, and Phase 5 integration/interrupt tests must pass before autonomous correction.
- User-confirmed wrist camera: stable path ending `usb-icSpring_icspring_camera_202404160005-video-index0`, previously `/dev/video2`, preview port 8001. Reverify after remounting.
- The existing wrist snapshot is rotated and close to the robot assembly. Useful face visibility from both viewing positions has not been established.

## Required outputs and interfaces

| Output | Contents |
|---|---|
| `config/alignment.yaml` | Disabled-by-default feature flag, per-station target point, chosen joint, calibrated response, correction limits, and approval fingerprints |
| `artifacts/phase-6/READINESS.md` | Step outcomes, optional-feature eligibility, physical test results, and handoff |
| `artifacts/phase-6/evidence/<session>/` | Calibration samples, setup images, measured joints, pre/post alignment results, interruption trials, and approval digests |
| Alignment planner/controller extension | Observe→request bounded correction→await feedback→observe loop, with no second serial owner |
| Calibration/evaluation tooling | Offline checking, fake replay, and supervised response/envelope validation through the robot worker |

Extend the typed controller interface with an `AlignmentRequest`: request ID, station pose name, approved calibration digest, one-joint offset in degrees relative to that station's recorded joint value, source frame sequence/timestamp, and reason. The controller independently calculates and validates the complete joint target. Do not expose this as an arbitrary-joint HTTP endpoint.

Publish an `AlignmentStatus`: station, state, source timestamp, before/after normalized error, attempts, measured offset, and outcome/reason. Outcomes: `DISABLED`, `ALIGNED`, `BEST_EFFORT`, `HOLD`, `LIMIT_REACHED`, or `FAULT`.

Configuration approval binds to the Phase 4 library digest, robot calibration, payload/mount/cable setup, wrist-camera orientation, and model digest. Missing or changed fingerprints disable refinement without invalidating otherwise unchanged base routes.

## Individual steps

### 6.1 — Verify eligibility and establish a calibration session

1. Read the physical, controller, and base-transition evidence. Keep refinement disabled while any relevant gate is pending.
2. Verify wrist device identity and camera ownership. Reuse the shared capture manager/vision worker, rather than opening a second capture stream.
3. Record setup/model/library digests, camera orientation, payload, intended stations, operator, observer, and physical-stop access.
4. Create an evidence session with `pending` outcomes. Do not copy approval from the base path library to correction paths.

**Pass:** dependencies and physical prerequisites are recorded, and a calibration session exists. Missing face visibility makes the affected station ineligible, not a reason to guess a correction.

### 6.2 — Establish the screen-viewing target in image coordinates

1. At each approved station pose, have the user stand at the intended viewing location and confirm comfortable screen orientation/readability.
2. Obtain at least ten fresh wrist observations while the arm and user are stationary. Record the face-center estimator consistently with Phase 2 and its observed scatter.
3. Store the median normalized face center as that station's preferred image point. Account for camera-to-screen mounting offset through this observation; do not assume the optical image center is the correct target.
4. Use the unmirrored processed image coordinates for control. Record any crop/rotation; apply the same transform to targets and observations. Display-only mirroring must not change control signs.
5. Record whether both station views are usable. Enable eligibility per station so one usable station does not require enabling the other.

**Pass:** each eligible station has a measured target point, coordinate convention, evidence, and scatter estimate. No calibrated 3D pose or gaze claim is made.

### 6.3 — Calibrate one useful joint and its correction direction

1. In supervised calibration mode, nominate a non-gripper joint whose motion appears to change screen orientation usefully. All movement goes through the robot worker and normal safety checks.
2. Begin with a correction no larger than 0.5° from the station pose, inside a provisionally inspected local region. The operator/observer must be able to stop the test.
3. Measure settled face centers and actual joint positions before and after the test. Return along the supervised inspected local path before testing the opposite direction.
4. Repeat at least three measurements in each tested direction. Fit the local response vector `g = (dx/dq, dy/dq)` in normalized image coordinates per degree, using measured joint changes rather than commands alone.
5. Reject inconsistent response signs, response dominated by observation noise, or a joint whose useful direction requires moving the phone into an excluded region. Try another joint only as a new supervised calibration session.
6. Store response and valid offset interval separately for each station. Never infer a positive motor direction from face error alone.

**Pass:** one joint has repeatable measured response at each eligible station, with a useful signal above stationary scatter. Otherwise keep that station's refinement disabled.

### 6.4 — Validate the correction and return envelope physically

1. Define a station-local offset interval inside measured joint limits, never wider than ±3° from its approved station value. Validate smaller intervals when necessary.
2. Inspect and test the complete loaded paths throughout that interval, including phone corners, camera mount, cables, and restrained gripper. Endpoint validity alone is insufficient.
3. Validate movement from the station pose to offsets and return to the station pose. Use bounded interpolated segments respecting the existing 10°/s, 20°/s², and 2° relative-target ceilings, or lower validated limits.
4. For return travel, use intermediate offset targets no more than 1° apart. This prevents departure from a refined pose through an unvalidated base-transition start.
5. Test Hold/Stop during local movement and retention at offset positions. A stopped intermediate position does not gain automatic restart approval.
6. Record and approve the envelope only after loaded clearance, feedback, and stop checks pass. Keep raw calibration/envelope evidence and its digest.

**Pass:** all enabled local corrections and their return paths are physically approved for the actual setup. No correction outside the interval can be accepted by the controller.

### 6.5 — Implement the bounded correction policy

1. Define image-space error `e = observed_face_center - calibrated_target`, normalized by image width/height. Report full error magnitude and its component along the calibrated response.
2. Default deadbands to 0.05 on each normalized axis; increase only through documented calibration if observed scatter requires it. Do not reduce them merely to demonstrate movement.
3. Compute the one-joint proposal `dq = -dot(g, e) / dot(g, g)`. Reject a zero, non-finite, or uncalibrated response; this local projection cannot correct every two-dimensional error.
4. Clamp each requested change to ≤1°, the approved offset interval, and cumulative outward correction travel ≤3° per station visit. Limit to three correction attempts; rejected/failed requests do not enable unlimited retries.
5. If the error is within both deadbands, report `ALIGNED` without movement. If the remaining error cannot be improved by the selected joint, report `BEST_EFFORT`; if a boundary prevents a useful request, report `LIMIT_REACHED`.
6. On non-improvement or worsening after a correction, stop further refinement for that visit. Do not oscillate by reversing correction signs repeatedly.

**Pass:** synthetic observations produce bounded, correctly signed requests; uncorrectable errors, exhausted budgets, and limits terminate the loop predictably.

### 6.6 — Integrate observe→move→observe with controller gates

1. Enter `ALIGNING` only after measured arrival/settling at an approved station pose, in Follow mode with fresh external tracking, a valid operator heartbeat, and no active base transition or fault.
2. Require the wrist frame and external selected-user state to be ≤500 ms old. Take the next wrist observation only after the previous correction completes and measured settling passes, with a sequence/timestamp newer than the completed motion.
3. For this single-user MVP, accept local face association only when both views contain one unambiguous participant in the constrained demo area. Hold on uncertainty; do not silently choose the largest/nearest face or claim cross-camera identity recognition.
4. Submit at most one correction request. The robot worker independently rechecks mode, heartbeat, tracking, fingerprints, limits, full target, gripper invariance, and freshness immediately before execution.
5. External tracking loss/ambiguity, operator Hold, Stop, or hardware fault cancels local movement through the existing interruption path. Stop/fault stays latched and reset returns to Hold.
6. Wrist loss during active alignment cancels correction and retains the measured position. The planner disables refinement; it must not launch a base route from an unapproved offset position.
7. After a healthy completed refinement, retain the useful offset while stationary. Before a later station transition, return through the approved local path to the base station pose only while all base-movement gates remain valid. Check measured arrival before dispatching the original Phase 4 route.
8. After an interrupted correction/return or an unknown measured position, require supervised restoration to an approved starting pose. Never automatically return because tracking disappeared.

**Pass:** both corrections and return-to-base motions use the single worker; no base route starts at an unapproved offset; failure handling cannot bypass Hold/Stop or physical approval.

### 6.7 — Add diagnostics and deterministic tests

1. Log source frame identity/age, association validity, target point, full/projected error, response vector, proposal, accepted/sent/measured targets, offset, attempts, and termination reason.
2. Show alignment state and feature eligibility in telemetry. Render the target point and measured face center on the wrist preview without altering control coordinates.
3. Test correct sign, coordinate rotation/mirroring, deadbands, zero response, noise, non-finite values, stale/future timestamps, and duplicate/pre-motion frames.
4. Test request limits, total travel/attempt budgets, immutable gripper, invalid setup/model/library digests, and rejection of another station's correction calibration.
5. Test Stop/Hold during correction and return, wrist failure, external ambiguity/loss, heartbeat expiry, worker failure, and prevention of concurrent base/refinement execution.
6. Test corrected-pose departure: approved return precedes the base route; interrupted/unknown starts require restoration. Fake-hardware tests must issue no real serial connection.

**Pass:** the deterministic suite passes and telemetry makes every correction/rejection explainable.

### 6.8 — Measure improvement in supervised trials

1. For each eligible station, run ten trials with small, reproducible user-position variations inside the calibrated station. Start each trial at the approved base pose with fresh valid observations.
2. Measure median pre- and post-refinement error using at least five fresh settled observations per measurement. Record invalid observations separately rather than treating missing faces as zero error.
3. Record full error reduction, corrected-axis error, attempts, travel, completion latency, and any unexpected movement, limit hit, or feedback fault.
4. Approve a station only if at least eight of ten valid trials improve full error by more than the recorded stationary scatter, median improvement is positive, and all trials remain inside the validated envelope with no oscillation, clearance issue, or hardware fault.
5. Trials already inside deadband validate no-movement behavior but do not count toward the ten improvement trials. If enough trials cannot be collected, leave approval pending.
6. Repeat interruption tests in supervised physical operation and record actual behavior. Apply the parent software cancellation target under healthy communication; distinguish it from loaded mechanical stopping time.

**Pass:** measured improvement and all motion/interruption gates pass for each enabled station. These are acceptance targets, not preclaimed results.

### 6.9 — Approve, enable selectively, and hand off

1. Record configuration digest, approved stations/envelopes, setup fingerprints, physical evidence, test results, limitations, and operator/date in `READINESS.md`.
2. Keep refinement disabled by default. Enable it explicitly only for approved unchanged stations and setup; mismatches disable it with a reported reason.
3. Hand configuration/status/overlay fields to Phase 7 and trial logs/metric definitions to Phase 8. Preserve a one-command configuration switch to disable refinement while retaining base following.
4. Revalidate after camera/phone remounting, orientation changes, payload/cable/base changes, calibration changes, or pose-library changes. Retain superseded evidence.

## Acceptance and fallback

- [ ] At least one station has usable wrist face visibility, measured target point, and repeatable joint response.
- [ ] Local envelope and return paths pass loaded physical validation.
- [ ] One-degree steps, three-degree outward travel, and three-attempt ceilings are enforced independently by the controller.
- [ ] Gripper stays fixed; observations and requests satisfy freshness/identity gates.
- [ ] Stop/Hold, failure handling, and corrected-pose departure tests pass.
- [ ] Supervised trials meet measured improvement criteria without oscillation or physical faults.
- [ ] Configuration approvals, logs, readiness, and downstream handoff exist.

If any required station gate fails, disable refinement for that station and continue the unchanged two-pose MVP. If a failed test leaves the arm at an offset, Hold and perform supervised restoration before base following resumes. Do not enlarge limits, assume response direction, introduce multi-joint corrections, or relax tracking gates to force a demonstration.
