# Phase 5 — Integrated Workstation Following

Status: **Software implemented and fake-hardware acceptance passed — live following blocked until perception and physical path acceptance pass**.

Execution record: [READINESS.md](READINESS.md). Independent commands and handoff: [README.md](README.md).

Parent specification: [FollowMe MVP plan](../PLAN.md).
Prerequisites: [Phase 2 perception](../phase-2/READINESS.md), [Phase 3 controller](../phase-3/READINESS.md), [Phase 4 pose validation](../phase-4/READINESS.md), and [Phase 0 physical readiness](../phase-0/READINESS.md).

## Objective and completion boundary

Connect fixed-camera workstation tracking to approved screen-pose selection through the existing single-owner robot controller. A sustained change between two calibrated stations should request one validated transition. Small head movements, boundary jitter, stale images, multiple people, and repeated observations should not produce unnecessary motion.

Start in Hold and dry-run. Require an explicit operator Follow request and an operator heartbeat. Implement and verify fake-hardware integration before enabling live operation.

Phase 5 covers integration, deterministic movement decisions, interruption/recovery, operator controls, and acceptance evidence. It does not implement wrist alignment, recipe UI, learned policies, arbitrary joint control, or new pose teaching. Those remain in their respective phases.

## Current integration points and prerequisites

- Phase 2 supplies `Runtime.status()` with tracker state, camera source modes, and worker errors. Its state contains capture timestamps, sequence, track generation, validity, ambiguity, selection requirements, and candidate/stable zone fields.
- Phase 3 supplies `Controller`, `RobotWorker`, immutable `TargetRequest`/`RobotStatus`, controlled-time fake hardware, and physical-motion gates. Serial I/O remains exclusively in `RobotWorker`.
- Phase 4 supplies draft sessions, content/setup/calibration fingerprints, directional paths, and live approval evidence. Its `project()` helper also supports supervised validation; projection alone is **not** proof that a draft is approved.
- Phase 2 workstation acceptance is currently deferred. Its zones are not yet approved for live following. Phase 4 physical acceptance remains blocked by pending Phase 0 gates.
- Existing preview-proxy timestamps measure delivery, not original camera capture. Proxy observations may support diagnostics/fake replay, but must never enable live following.

## Required outputs and interfaces

| Output | Purpose |
|---|---|
| `src/followme/phase5` | Integration runtime, observation adapter, planner, operator controls, diagnostics, and CLI |
| `config/following.yaml` | Zone-to-pose mapping, conservative timing, backend, approved-library location, and control-server settings |
| `artifacts/phase-5/READINESS.md` | Software and physical outcomes recorded separately |
| `artifacts/phase-5/evidence/<session>/` | Input snapshots, decisions, controller events, acceptance results, and cancellation measurements |
| Phase 5 test/check tooling | Controlled-time planner tests, fake integration trials, API tests, and evaluation replay |

Add the standalone command `python -m followme.phase5` with `inspect`, `replay`, and `run` subcommands. `run` defaults to dry-run; live mode is an explicit option that still requires all approval gates. Preserve existing phase CLIs and preview processes.

Expose a small operator console on configurable port **8020**, avoiding existing ports 8000/8001 and Phase 2 ports 8010/8011. Provide `GET /api/status`, `POST /api/mode`, `POST /api/heartbeat`, `POST /api/stop`, `POST /api/reset`, and `WS /ws/state`. Phase 7 can reuse this control service and add the recipe interface.

All control requests require a per-session token and accepted same-origin browser requests. Heartbeats renew only the active operator's lease; viewing telemetry does not renew it. No request accepts arbitrary joint values or grants physical approval.

## Individual steps

### 5.1 — Validate prerequisites and configuration

1. Read upstream readiness records and store their relevant content/setup/calibration digests in the integration session.
2. Define exactly two workstation-to-pose mappings using approved Phase 2 zone names and Phase 4 `station_a`/`station_b` poses. Reject missing zones, unknown poses, duplicate station assignments, or absent directional paths.
3. Default to `dry_run: true`, `mode: hold`, dwell 0.75 seconds, cooldown 2 seconds, freshness 0.5 seconds, heartbeat expiry 3 seconds, planner 5 Hz, and controller at its configured rate no higher than 20 Hz.
4. Timing configuration may tighten freshness/heartbeat limits or increase dwell/cooldown. It must not weaken the master plan's gates silently.
5. Inspect mode opens neither cameras nor serial ports. Fake replay can use explicitly synthetic zones/paths, tagged as such; never write them into production configuration.

**Pass:** invalid/incomplete live prerequisites produce named failures; dry-run configuration is explicit; no unavailable calibration, limits, or paths are invented.

### 5.2 — Adapt perception snapshots without changing their meaning

1. Convert `capture_monotonic_ns` to controller seconds using the same system monotonic clock domain. Preserve original frame identity and timestamp; do not stamp an old image with current processing time.
2. Consume immutable snapshots. Require the external-camera role, valid tracking, visible user, no ambiguity, no selection requirement, native capture for live operation, and a healthy perception worker.
3. Verify observations are neither future-dated nor older than 500 ms. Invalid/missing timestamps fail closed.
4. Recognize repeated frame sequences and track-generation changes. Only a newly observed frame can advance or establish a station decision; repeated snapshots can expire validity but cannot create a new dwell history.
5. Clear planner dwell/candidate history after invalid tracking, changed track generation, user reacquisition, or zone/configuration changes.
6. Bridge freshness/ambiguity to `Controller.update_tracking()`. Do not replace the controller's independent watchdog checks with planner-only checks.

**Pass:** stale, replayed, ambiguous, proxy, and worker-failure observations cannot produce live requests; the original timestamps remain visible in logs.

### 5.3 — Implement deterministic station selection

1. Use Phase 2's candidate/stable-zone evidence and candidate start timestamp. Require at least 0.75 seconds of continuously valid evidence for the current candidate; do not add a second full dwell after the tracker has already satisfied that interval.
2. Require current candidate and stable zone to agree. Neutral/unknown regions do not choose a destination or continue accumulating dwell for an earlier candidate.
3. Request a pose only when Follow is active, the operator lease is fresh, the robot is at a recognized approved anchor, all controller gates pass, and the mapped pose differs from the measured current pose.
4. Treat changing to the mapped station pose as the MVP improvement criterion. Do not calculate fictitious 3D viewing costs or infer gaze from face location.
5. Start the two-second cooldown at measured arrival/settling completion, not request submission. Clear old candidate history at completion so observations from the move cannot immediately trigger the opposite transition.
6. Deduplicate using session, track generation, source sequence, and target identity. A controller rejection is logged and consumes that decision; reevaluate only fresh evidence rather than retrying the same request indefinitely.

**Pass:** one stable station change yields at most one request; identical pose, jitter, head turns, neutral regions, and cooldown yield no unnecessary request.

### 5.4 — Connect the planner to the single-owner robot worker

1. Create one Phase 3 controller/worker and one perception lifecycle under the integration supervisor. Reuse an in-process Phase 2 runtime rather than launching a competing native camera capture service.
2. Call thread-safe controller methods only; the planner, console, and perception adapter never construct an independent motor connection or call SDK send methods.
3. Submit `TargetRequest(request_id, target_pose_name, reason, timestamp_s)` using the triggering frame's capture timestamp and record `RequestResult`.
4. While pending or REPOSITIONING, continue freshness monitoring but discard normal movement proposals. Do not queue an opposite destination or retain a deferred motion request.
5. Determine completion from controller feedback and settled pose recognition, not elapsed duration alone. The gripper setting and software motion ceilings remain those enforced by Phase 3/4.
6. Monitor worker health independently. A failed worker latches a fault; no replacement serial writer may attempt emergency commands.

**Pass:** fake integration verifies one serial owner, one outstanding movement, independently checked request freshness, measured arrival, and no backlog.

### 5.5 — Implement interruption and explicit recovery

1. Operator Hold, invalid/stale external tracking, ambiguity, camera/perception failure, and heartbeat expiry request priority Hold and clear pending planner decisions.
2. These events return the integration mode to Hold. Restored images or heartbeats do not automatically resume following; a new explicit Follow request is required.
3. Stop overrides Hold/Follow and latches STOPPED. Hardware/worker failures latch FAULT. Both clear candidate history and outstanding intentions.
4. Reset requires the existing controller's healthy fresh-feedback and recognized-anchor checks; it returns to Hold only. It never arms, moves to an anchor, resumes a trajectory, or enables Follow.
5. An interruption between anchors requires supervised restoration under the Phase 4 recovery procedure. Do not interpolate from an arbitrary stopped pose to the next target.
6. Shutdown requests Hold, waits for the bounded controller shutdown/confirmation, then closes resources. Report unconfirmed hold or lost communication explicitly; preserve the existing holding-torque disconnect policy.

**Pass:** every interruption stops issuing trajectory updates and requires explicit recovery where applicable. Software hold results are distinguished from physical braking and retention results.

### 5.6 — Add minimal operator controls and session telemetry

1. Console shows backend, mode, behavior state, current/target pose, zone, capture age, ambiguity, heartbeat age, worker health, physical approval blockers, and last decision/rejection reason.
2. Follow validates prerequisites before enabling the mode. A live run may request arming only through the worker after approved-library checks; startup, heartbeat, and reset never arm automatically.
3. Browser heartbeat runs at one-second intervals while the active operator console is present. Expiry at three seconds requests Hold. Reject another client's attempt to silently take over the current lease.
4. Keep control handlers independent of camera/inference calls. Stop/Hold invoke priority controller methods immediately, not the next 5 Hz planner iteration.
5. Log observations, decision IDs/reasons, accepted/rejected requests, measured arrival, cooldown, operator mode changes, and fault/reset events with monotonic timestamps. Never log control tokens.
6. Provide sequence-linked JSONL evidence joining planner and controller records. Mark fake, proxy, and live runs explicitly.

**Pass:** authenticated controls operate with fake hardware; unauthorized controls fail; browser loss enters Hold; diagnostics cannot bypass the worker or motion gates.

### 5.7 — Run deterministic and concurrent software acceptance

1. Controlled-clock tests cover dwell boundaries, cooldown from arrival, neutral-zone resets, head-turn jitter, same-pose suppression, duplicates, generation changes, future/stale timestamps, and delayed inference.
2. Test Hold/Stop during request submission and at early/middle/late trajectory stages. Test loss of camera/perception/robot worker and heartbeat expiry while idle and moving.
3. Test requests rejected due to busy worker, invalid route, expired source image, changed calibration/setup, and unrecognized start pose. Confirm no automatic retries/resumption.
4. Run ten synthetic station changes with the same single-owner controller used in production. Require ten correct destination decisions and zero physical commands.
5. Run at least 100 randomized real-thread fake-I/O interruption trials. Measure gate-trigger-to-controller-handling latency separately from the configured detection/dwell/expiry interval. Require handling within 250 ms under documented healthy fake-I/O conditions.
6. Test the operator API, lease behavior, dry-run serial nonaccess, and malformed observations. Run existing Phase 2–4 checks affected by integration changes.

**Pass:** automated integration acceptance succeeds with recorded timing conditions. Simulation does not mark physical tests passed.

### 5.8 — Enable gated live acceptance

1. Proceed only with passed Phase 0 physical gates, completed Phase 2 native workstation acceptance, and an approved Phase 4 library matching the current payload/setup/calibration and evidence hashes.
2. Verify approval rather than accepting a draft's `approved` booleans. Load the approved session through a verification helper that checks the evidence/log hashes and current fingerprints before producing the controller library view.
3. Retain Hold on startup. Require the operator at the physical stop, clear operating region, recognized starting anchor, and explicit Follow.
4. Conduct ten deliberate alternating workstation changes. Require the correct settled destination for every accepted change, with no queued or redundant movement.
5. Conduct two minutes of stationary work, brief head turns, and boundary jitter. Require no workstation repositioning when the working station has not changed.
6. Supervise tracking-loss, camera-disconnect, ambiguity, browser-loss, Hold, and Stop tests during idle and motion. Record trigger handling and actual loaded-arm behavior separately; do not simulate a transport fault by introducing a second serial writer.
7. If face-derived zones fail the head-turn test, return to Phase 2 shoulder-based zone tracking and repeat acceptance. If any physical gate fails, return to Hold, record the failure, and keep live following disabled.

**Pass:** physical acceptance evidence supports useful two-station following and the required interruption behavior with the exact tested setup.

### 5.9 — Publish readiness and downstream handoff

1. Record each step as `pending`, `passed`, `failed`, or `blocked`; distinguish software/fake acceptance from native-camera and physical acceptance.
2. Export correct destination counts, accepted/rejected decisions, unnecessary repositionings, dwell-to-arrival latency, controller cancellation latency, and setup/library/calibration fingerprints.
3. Document startup, Follow/Hold/Stop, recovery, shutdown, camera ownership, and exact commands for fake and live operation. No run command bypasses approval.
4. Hand the settled-arrival event and interruption/control interfaces to optional Phase 6; leave ALIGNING disabled until its own approval passes. Hand the service/status interface to Phase 7 and trial logs to Phase 8.

## Final acceptance checklist

- [ ] Configuration mappings, perception freshness, and approved-library loading validate.
- [ ] Default Hold/dry-run does not open serial hardware.
- [ ] Only the existing robot worker performs motor I/O.
- [ ] Stable station changes produce one request; no movement backlog develops.
- [ ] Dwell, cooldown, duplicate, neutral-region, and track-generation rules pass.
- [ ] Tracking/heartbeat loss, operator Hold, Stop, and faults interrupt through priority controls.
- [ ] Recovery/reset never moves, resumes, or grants Follow automatically.
- [ ] Fake tests and documented cancellation timing pass; API/session controls pass.
- [ ] Native perception and unchanged physical approval gates pass before live trials.
- [ ] Ten live station changes and two stationary minutes pass with evidence.
- [ ] Software latency and actual physical stopping behavior are reported separately.
- [ ] Readiness and Phase 6–8 handoffs are complete.

## Failure and rollback policy

Keep upstream calibration, environment locks, taught sessions, and preview services unchanged unless an integration requirement explicitly needs a coordinated change. A camera owned by another service is reported busy; never terminate an unrelated owner automatically. Existing previews may support fake diagnostics through Phase 2 proxy mode but cannot qualify live freshness.

If live prerequisites remain incomplete, finish the integration and fake acceptance, mark live acceptance blocked, and present the dry-run demonstration. Changes to zones, camera placement/rotation, payload, paths, calibration, or hardware setup invalidate the affected acceptance and require revalidation before Follow.
