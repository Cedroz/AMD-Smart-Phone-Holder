# Phase 2 — Face and Workstation Tracking Without Motion

Status: **Software implemented — live workstation calibration/trials deferred by user**.

Execution results: [READINESS.md](READINESS.md). Running commands: [OPERATIONS.md](OPERATIONS.md).

Implementation isolation: Phase 2 lives in `followme.phase2` and `config/phase2`, uses `followme-phase2`, and defaults to ports 8010/8011. This replaces the shared module/config paths and preview-port takeover originally proposed below so other phase implementations retain their own code, configuration, and camera ownership.

Parent: [FollowMe MVP plan](../PLAN.md).
Environment contract: [Phase 1 specification](../phase-1/PLAN.md).
Hardware handoff: [Phase 0 readiness](../phase-0/READINESS.md).

## Objective and completion boundary

Track a single demonstration participant using the fixed camera, publish fresh filtered observations, and recognize sustained changes between two calibrated workstations. Provide live diagnostic overlays and measurable camera-only acceptance evidence.

Capture both cameras, but derive workstation decisions only from the fixed camera. The wrist camera supplies a preview and optional landmark diagnostics for later alignment work. It must never override the external camera's user selection or zone classification.

This phase never opens a robot serial connection, creates motor targets, or enables Follow. Calibration remains unchanged. Physical Phase 0 checks may remain pending while camera-only work proceeds. Phase 5 consumes the observations produced here after the required movement gates pass.

## Existing implementation and prerequisites

The repository currently contains `VisionClient`, the JSON socket protocol, and a vision worker. The worker already maintains separate `VIDEO` Face Landmarker instances and timestamp histories for external/wrist streams. Extend these interfaces instead of introducing another inference process or IPC format.

`config/environment.yaml` identifies both stable camera paths and keeps dry-run/Hold defaults. Its vision launch settings must satisfy the existing client's startup requirements, including `mediapipe_version`. Presence of source files is not proof that Phase 1 rebuild, model, and launch checks passed: run those checks before live inference and record any failures.

Confirmed camera mapping:

| Role | Stable capture path | Existing preview |
|---|---|---|
| `external` | `/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0` | Port 8000 |
| `wrist` | `/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0` | Port 8001 |

## Required outputs and interfaces

- Camera capture and latest-frame buffers under `src/followme/cameras`.
- Perception validation, selected-user continuity, filtering, and zone classification under `src/followme/perception`.
- `config/perception.yaml` for acquisition, thresholds, freshness, smoothing, and rates; `config/zones.yaml` for measured workstation regions.
- CLI entrypoints `followme perceive`, `followme calibrate-zones`, and `followme evaluate-perception`. All are camera-only. These names distinguish workstation calibration from robot calibration.
- Diagnostic preview endpoints at the existing ports, JSONL observation logs, and deterministic replay/test fixtures.
- `artifacts/phase-2/READINESS.md` and timestamped evidence containing configuration/model hashes, performance measurements, trial annotations, and results.

### Observation contract

Publish immutable `UserState` snapshots using these fields:

| Field | Meaning |
|---|---|
| `camera_role`, `sequence`, `capture_monotonic_ns` | Original source identity and application acquisition timestamp |
| `processed_monotonic_ns` | Result handling time; never used to refresh source age |
| `visible`, `tracking_valid`, `ambiguous`, `invalid_reason` | Explicit validity without invented confidence scores |
| `face_count` | Number detected, capped by the configured detector; two means ambiguity |
| `face_center_raw`, `face_center_filtered` | Normalized x/y coordinates, or null when invalid |
| `track_generation`, `selection_required` | Session continuity identity and whether operator reacquisition is needed; not biometric identity |
| `candidate_zone`, `stable_zone`, `candidate_since_ns` | Pending and accepted workstation classifications |
| `head_yaw_deg`, `head_pitch_deg` | Null in the initial implementation; head-orientation estimates are not needed for the phase gate |
| `position_source` | `face`, or `shoulders` if the tested fallback is enabled |

Diagnostic history may retain the last successful point/zone, but actionable fields become invalid/null when tracking fails. No motor-oriented `TargetRequest` is published in this phase.

## Individual steps

### 2.1 — Verify the handoff and establish camera ownership

1. Run Phase 1 environment/model/worker smoke checks and default hardware-free preflight. Verify both calibration hashes are unchanged.
2. Record current camera owners and preview process commands. Do not open a second reader on an occupied capture device.
3. Implement the replacement camera-only preview before switching ownership. Stop only the identified legacy preview processes when the replacement is ready; retain their exact restart commands for rollback.
4. Start one capture owner per physical camera. Downstream inference, overlays, and preview clients consume its buffer rather than reopening the device.
5. Verify physical roles with the confirmed stable paths. A missing device or contradictory role assignment is an explicit error, not a reason to guess another index.

**Pass:** Phase 1 smoke checks pass and each camera has one known capture owner; no serial adapter is opened.

### 2.2 — Implement timestamped latest-frame capture

1. Request 640 × 480 and 30 FPS using V4L2 on Linux. Request MJPEG and read back the actual format/size/rate. The external camera previously reported YUYV: accept that supported format and report it rather than assuming MJPEG succeeded.
2. Capture each camera on its own worker and publish a copied immutable frame, strictly increasing sequence, and `time.monotonic_ns()` timestamp immediately after successful acquisition.
3. Maintain exactly one latest-frame slot per role, replaced atomically. Do not create an unbounded frame queue.
4. Track capture errors and frame age independently. Failed reads never republish the old image with a new timestamp.
5. Configure a one-second capture-stall deadline. A device backend that can block indefinitely must run in a supervised capture process that can be terminated; do not claim thread cancellation releases a blocked native read.
6. On external-camera failure, invalidate user state immediately when observed, or by the 500 ms freshness deadline if capture stops silently. Wrist failure disables wrist diagnostics while external tracking continues.

**Pass:** both streams publish independently; disconnect/stall tests invalidate the affected data; buffer memory stays bounded. Log actual capture throughput separately from requested FPS.

### 2.3 — Schedule fresh inference through the existing worker

1. Feed JPEGs through `VisionClient.submit_latest` and infer using the existing request/response identity fields. Keep the one-outstanding-request rule.
2. Prioritize external-camera inference at a target 10 Hz. Capture/preview the wrist camera continuously; optional wrist landmark diagnostics target 2 Hz only when they do not delay external work.
3. Configure Face Landmarker detection, presence, and tracking thresholds initially to 0.7, `num_faces=2`, `VIDEO` mode, CPU delegate, and transformation-matrix output. Keep independent model instances per camera.
4. Validate timestamps before dispatch and after response. Reject future, non-increasing, wrong-role, wrong-sequence, and older-than-500-ms results; do not clamp malformed source timestamps to make them acceptable.
5. Keep model millisecond timestamps strictly increasing per role. If two frames fall in the same millisecond, skip the duplicate for inference while preserving actual capture timestamps.
6. Evaluate freshness independently of a blocked inference call, at least every 50 ms. The Phase 1 five-second inference timeout is a worker failure deadline, not permission to display a stale observation as valid.
7. Treat worker errors, invalid landmark payloads, or worker exits as invalid tracking with a reason. Require explicit operator restart after worker failure; restart does not restore the previous selected user.

**Pass:** outdated results cannot overwrite newer state; slow inference drops unsent frames; external freshness expires even while the client waits for a response.

### 2.4 — Validate detections and establish participant continuity

1. Accept a face measurement only when exactly one face is detected, the response has no error, and landmark x/y coordinates and resulting center are finite and inside the image.
2. Define the face center as the midpoint of the axis-aligned bounding box of the returned face landmarks. Record this definition for repeatable metrics.
3. Zero faces makes tracking invalid. Two detected faces makes it ambiguous; never choose the largest or nearest face during ambiguity.
4. On startup, acquire a single participant after 0.75 seconds of continuous fresh single-face observations. Assign a new `track_generation`.
5. While acquired, reject a raw-center jump greater than 0.20 normalized image distance between consecutive accepted observations as a continuity break. This is a conservative continuity heuristic, not proof of identity.
6. Brief missed detections invalidate actionable state and reset zone dwell. Loss exceeding 500 ms, a continuity break, or multiple-face ambiguity clears selection and requires operator `reacquire` before the acquisition dwell runs again.
7. Expose reacquisition through the camera-only diagnostic CLI/API. It clears filters and history; it cannot enable motion. For the MVP, require one participant in view rather than implement biometric cross-camera matching.

**Pass:** ambiguity/loss cannot silently select another person or reuse an old zone; reacquisition starts a new continuity generation with cleared history.

### 2.5 — Add filtering and temporal validity

1. Apply EMA independently to external x/y: `filtered = 0.25 * raw + 0.75 * previous`. Initialize from the first accepted measurement.
2. Filter only new, valid single-face measurements. Duplicate, stale, out-of-order, ambiguous, or missing observations never update the filter.
3. Clear the filter when selection is cleared or its last accepted source is older than 500 ms. Brief losses reset dwell even when the previous filter remains eligible for continuity.
4. Report both raw/filtered values and source age. Avoid predicting or extrapolating a person into a different workstation during missing data.
5. Use monotonic time for freshness/dwell and wall-clock time only for artifact labels. Reject non-finite numeric values in logs and configuration.

**Pass:** jitter is reduced in deterministic fixtures; gaps and participant changes cannot combine unrelated measurements or refresh expired state.

### 2.6 — Calibrate workstation regions and hysteresis

1. Add a calibration screen using the external camera's unmirrored normalized coordinate frame. Allow the operator to mark two non-overlapping rectangular core regions named `station_a` and `station_b`, leaving a neutral gap.
2. Record camera path, orientation/rotation, image size, screenshot reference, operator/date, and configuration digest alongside region bounds. Display mirroring must not alter classification coordinates.
3. Build an exit/retention region around each core with an initial 0.03 normalized margin, clamped to image bounds. Reject expanded-region overlap; reduce the margin or adjust the measured cores before saving.
4. A core-region measurement starts that zone's candidate dwell. Accept it only after 0.75 seconds of uninterrupted fresh valid observations inside the same core.
5. Keep the accepted zone while measurements remain in its exit region. Outside it, clear actionable `stable_zone`; the neutral gap yields no destination.
6. A different core requires a full new dwell. Missing/ambiguous/stale tracking or selection reset clears candidate/stable actionable zones immediately. Keep the last confirmed zone only as diagnostic history.
7. Validate bounds, names, overlap, margins, and camera metadata at startup. Missing/invalid calibration permits preview but publishes no stable workstation.

**Pass:** reproducible fixtures verify entry dwell, retained-zone hysteresis, neutral-region behavior, and reset semantics. Camera relocation/rotation invalidates zone calibration until repeated.

### 2.7 — Publish overlays, telemetry, and replay evidence

1. Preserve preview URLs on ports 8000/8001. Serve overlays from captured frames with role, sequence, age, measured rates, landmarks/center, zones, candidate dwell, selection state, and invalid reason.
2. Show stale streams explicitly. Do not repeatedly serve a stale JPEG as if it were live; new clients receive a clear unavailable response until frames return.
3. Publish immutable `UserState` snapshots and JSONL events for detections, invalidation, acquisition/reacquisition, candidate changes, and accepted zone changes.
4. Log source/processing timestamps, model/config hashes, inference duration, dropped-frame counts, and validity. Raw video recording is opt-in for the demonstration participant; default logs contain derived observations.
5. Implement deterministic replay of logged observations with a virtual monotonic clock. Store manually annotated expected workstation intervals separately from the algorithm's output.
6. Keep the phone UI, recipe API, operator motion controls, and robot connection out of this phase.

**Pass:** both previews remain usable; replay produces the same state transitions; performance/validity can be measured independently of live hardware.

### 2.8 — Test performance, faults, and live workstation behavior

1. Automated tests use synthetic landmarks, fake clocks/capture, and fake IPC. Cover EMA, duplicate/out-of-order results, future/stale timestamps, malformed landmarks, neutral zones, boundary jitter, dwell interruptions, ambiguity, continuity breaks, worker failure, and missing camera/configuration.
2. Patch serial constructors and robot connection methods to raise if invoked. Run startup, calibration, replay, fault paths, and shutdown against those guards.
3. Run a two-minute dual-camera benchmark. Report capture FPS, external/wrist inference FPS, p50/p95 source-to-result age, dropped frames, stalls, and valid observation fraction. Require external inference ≥10 Hz and p95 age ≤500 ms on this machine before marking the target met.
4. If rates fail, first disable optional wrist inference, then reduce inference image size to 320 × 240 while keeping capture/preview at 640 × 480 and normalized coordinates unchanged. Re-run and record the actual operating settings; do not mark an unmet target as passed.
5. Conduct ten alternating A↔B station changes with the participant remaining at each destination for at least three seconds. Require all ten correct zone acceptances, no extra accepted transitions, and acceptance within two seconds of annotated entry into the destination core.
6. Conduct two minutes of stationary work at each station, including looking down and brief head turns. Require no accepted changes to the other station. Loss of face visibility may invalidate state without constituting a false workstation transition.
7. Test a second person entering view, participant loss/reacquisition, external camera disconnect, wrist disconnect, and inference delay. Verify invalidation and recovery behavior in logs without motor access.

**Pass:** automated boundaries and measured live acceptance gates pass. Synthetic replay alone does not establish the live gate.

### 2.9 — Apply the body-tracking fallback and record readiness

1. If the head-turn trial produces false workstation changes, add a Pose Landmarker model using the Phase 1 download/manifest verification procedure. Update and recheck vision dependencies only if required.
2. In the same isolated vision worker, compute the external shoulder midpoint from valid left/right shoulder landmarks, requiring visibility/presence ≥0.7 where supplied. Run pose inference at a target 10 Hz; retain face detection for single-user/ambiguity gating.
3. Keep `face_center_*` fields as face measurements. Add optional `body_center_raw/filtered` fields and set `position_source=shoulders` for workstation classification. Missing shoulders invalidate zone classification; do not silently switch between body and face coordinates.
4. Recalibrate zone regions for shoulder coordinates and rerun the full head-turn, ten-transition, freshness, and throughput tests. Disable optional wrist inference if necessary to meet the external budget.
5. If neither position source passes, mark automated workstation tracking blocked; retain camera previews and diagnostics. Do not bypass dwell/validity to make the demonstration move.
6. Write readiness with per-step outcomes, model/configuration hashes, benchmark results, trial annotations, and limitations. Handoff observation contracts and replay fixtures to Phases 3/5; wrist diagnostics inform Phase 6 without implying alignment is implemented.

## Final acceptance checklist

- [ ] Phase 1 launch/model checks pass; configurations match the tested environment.
- [ ] One capture owner per camera; both original preview links work.
- [ ] Independent latest-frame buffers and bounded inference transport verified.
- [ ] Face validity, ambiguity, continuity, and explicit reacquisition behavior tested.
- [ ] EMA, 500 ms freshness, hysteresis, and 0.75-second dwell behave deterministically.
- [ ] Calibrated workstation regions include a neutral gap and camera metadata.
- [ ] Ten live alternating changes classify correctly with no extra transitions.
- [ ] Head-turn/stationary trials produce no false workstation changes.
- [ ] Throughput/latency targets and disconnect/worker-failure cases are measured.
- [ ] Serial-access guards pass; calibration unchanged; no motor command generated.
- [ ] Readiness artifacts distinguish synthetic tests, live evidence, and pending physical gates.

## Rollback and revalidation

If replacement previews fail, release their cameras and restart the recorded legacy `--dry-run` previews with the same stable paths/ports. Preserve failed-run logs. Do not fall back to the legacy tracker's motor mode.

Camera mount/orientation changes, model changes, filter thresholds, zone boundaries, participant-continuity rules, or position-source changes require affected acceptance trials to be repeated. Missing/stale tracking always produces invalid observations, never an inferred movement destination.
