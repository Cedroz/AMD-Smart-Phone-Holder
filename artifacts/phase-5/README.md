# Phase 5 operations and handoff

Phase 5 is an independent integration package. It reads Phase 2 perception interfaces and Phase 3/4 control/approval interfaces without modifying their implementations. Its configuration lives in `config/phase5/following.yaml`; it does not change the shared CLI, packaging, environments, calibrations, zones, taught sessions, or existing camera services.

## Offline inspection and acceptance

From the repository root:

```bash
source ./lerobot-env.sh
python -m followme.phase5 inspect
python -m followme.phase5 replay --output artifacts/phase-5/evidence/my-new-replay
python -m unittest discover -s test -p 'test_phase5.py' -v
python -m followme.phase5.acceptance --trials 100 --output artifacts/phase-5/evidence/my-interruptions.json
```

Inspection opens no hardware. Default replay uses synthetic observations and the existing fake controller library: ten alternating station changes, one motor I/O owner, and zero physical commands. Replay refuses an existing evidence output directory containing run files rather than mixing trials.

`replay --input recording.jsonl` accepts canonical observation/control records with original monotonic event times:

```json
{"timestamp_s":100.0,"snapshot":{"state":{"camera_role":"external","sequence":0,"track_generation":0,"capture_monotonic_ns":100000000000,"visible":true,"tracking_valid":true,"ambiguous":false,"selection_required":false,"candidate_zone":"station_a","stable_zone":"station_a","candidate_since_ns":100000000000},"source_modes":{"external":"synthetic"}}}
{"timestamp_s":100.0,"action":"heartbeat"}
{"timestamp_s":100.0,"action":"follow"}
```

Subsequent records provide fresh frames and heartbeat events. Supported actions are heartbeat, follow, hold, stop, and reset. Recorded-input replay starts in Hold and never implicitly resumes. It advances the controller between event times and preserves capture timestamps, so replay can demonstrate stale tracking and heartbeat loss. Phase 2 raw inference JSONL is not this canonical format; adapt its tracker-state events explicitly, including scripted operator events.

## Independent console

```bash
python -m followme.phase5 run --dry-run
```

Open http://127.0.0.1:8020/. This uses synthetic perception and fake hardware; it owns no camera or serial device. The synthetic participant switches stations every eight seconds. The browser sends an operator heartbeat; press Follow to begin simulated transitions. Closing/hiding the active console expires the three-second lease and returns to Hold. Reset returns to Hold and never enables Follow.

The console includes static and arm-mounted camera panels. Camera-backed runs stream the existing Phase 2 annotated latest frames through `/streams/external` and `/streams/wrist`. Synthetic runs display the existing diagnostic feeds on 8010/8011, falling back to camera-only previews on 8000/8001. Viewing these feeds does not change which observations drive the robot. Unavailable/stale sources show a retry status; previews never open additional camera devices or renew the operator lease.

To serve the console on the configured LAN address:

```bash
python -m followme.phase5 run --dry-run --host 0.0.0.0
```

Only origins in this phase's configuration can issue controls. Update its `allowed_origins` when the laptop address or port changes. The console contains the session control token; do not expose it to untrusted clients. Tokens are sent in headers, and server access logging is disabled. A second browser cannot silently take the active operator lease. Hold/Stop are priority commands available to a token-authorized operator; telemetry does not renew the lease.

Read-only status: `GET /api/status`. Controls: `POST /api/mode`, `/api/heartbeat`, `/api/stop`, `/api/reset`. Requests require `Origin`, `Authorization: Bearer <session-token>`, and an `X-Operator-ID` for lease/follow commands. Mode accepts only `{"mode":"hold"}` or `{"mode":"follow"}`. No arbitrary-joint endpoint exists. WebSocket telemetry uses protocols `followme` and the session token, with an accepted origin.

## Camera-backed diagnostics

```bash
python -m followme.phase5 run --dry-run --perception proxy
```

This delegates camera management to an in-process Phase 2 runtime, writing its evidence into the Phase 5 session. Occupied cameras can be consumed through the existing 8000/8001 preview URLs in proxy mode. Delivery timestamps cannot qualify live freshness. Native diagnostics use `--perception native` and refuse busy cameras; they never kill another owner. Start camera-backed runs only when ownership has been coordinated. Synthetic mode is the default and was the mode used for this implementation's acceptance.

The supervisor watches upstream environment/zone/perception fingerprints. A change stops the integration rather than adopting new zones mid-run. Operator generation changes, stale or ambiguous observations, provider failure, heartbeat loss, unknown anchors, controller inhibition, and rejected requests return to Hold. Fresh images or restored heartbeats alone do not resume movement.

## Live approval prerequisites

Live execution was **not performed**. Before enabling it, complete Phase 0 physical evidence, Phase 2 native workstation acceptance, and Phase 4 physical path approval. Then set the independent configuration's `approved_session` and `perception_acceptance` paths.

The Phase 4 session must have approved validation. Phase 5 reuses `approve()` to recheck physical attachments, trial hashes, payload/setup fingerprint, calibration, and controller limits; projected/draft `approved` booleans are insufficient. A gripper-guard adapter preserves Phase 4's taught gripper tolerance.

The Phase 2 acceptance JSON must contain:

- `status: "passed"`, `source_mode: "native_camera"`, operator, and recorded date.
- `config_hashes` keyed by `config/phase2/zones.yaml`, `config/phase2/perception.yaml`, and `config/environment.yaml`, with current file SHA-256 values.
- `evidence`: nonempty records containing repository-relative `path` and actual `sha256` for completed native trials.
- `station_changes_correct: 10`, `station_changes_total: 10`, `unnecessary_repositions: 0`, and `stationary_duration_s` of at least 120.

This record is an operator attestation tied to evidence, not an automatically inferred physical result. Do not create a passing record for incomplete trials. The configured zone keys must match the calibrated workstation regions exactly.

After those prerequisites pass, the supervised command is:

```bash
python -m followme.phase5 run --live --perception native
```

It starts read-only/Hold and requires explicit Follow, which requests arming only through the robot worker. Runtime approval/config checks run outside the priority-control lock. Healthy fake-I/O cancellation measurements do not establish live communication or braking budgets; benchmark those with the physical operator before autonomous acceptance. Software Hold/Stop cannot replace the physical stop on lost communication.

Never restart a competing serial owner to recover a stalled worker. An interrupted off-anchor pose requires Phase 4 supervised restoration. Changing payload, poses, calibration, camera geometry, or zones requires new affected approval evidence.

## Evidence and later phases

Every run records observation snapshots, decision identity/reason, controller events, measured arrival, operator mode transitions, and faults. Controller logs retain requested/measured joints and command counters. `summary.json` and replay `results.json` identify fake/live backend; wrist alignment remains disabled.

Phase 6 may consume `integration_arrived` events but cannot bypass the robot worker. Phase 7 may extend this console/service with the recipe interface. Phase 8 may evaluate canonical replay/decision logs and physical trials. No physical acceptance results are inferred from fake replay.

## Current single-camera setup

The former static camera is now mounted on the robot and is the only Phase 5 input. Its internal role remains `external` for contract compatibility. Phase 5 uses the original orientation (0-degree rotation) for both inference and annotated preview, configured in `config/phase5/following.yaml`. The previous wrist camera is not captured or displayed by Phase 5. Other phase services and configuration are unchanged.

Run real diagnostic tracking with `python -m followme.phase5 run --dry-run --perception proxy`. Moving this camera invalidates the fixed-camera workstation assumption: camera movement can resemble user movement. Live following remains inhibited until a motion-aware tracking approach and the new camera geometry are validated. Workstation regions still require calibration.

## Continuous centering prototype

`behavior: continuous_centering` replaces zone-to-pose planning in the running Phase 5 application. Real perception drives a horizontal shoulder-pan simulation with an 8% image deadband, 5 degrees/second maximum speed, 20 degrees/second squared acceleration ceiling, and illustrative +/-15 degree simulation bounds. Gripper and other joints remain fixed. These bounds and the simulated response sign are not hardware measurements. Follow requires valid single-user tracking and an operator heartbeat; Hold, Stop, stale frames, and ambiguity inhibit updates.

Physical continuous centering is explicitly unavailable until joint bounds, payload retention/stop behavior, and camera-to-joint response direction are validated. A vertical axis also requires measured joint/image response; it is not implemented. Requested simulated joints appear in `/api/status` and commands are logged in `controller.jsonl`.

## Live supervised centering trial

The latest implementation supports the real follower through `live_pan.py`, with one worker owning all serial I/O. `tracker/track.py` is a read-only reference and remains untouched. Start with `python -m followme.phase5 run --live --perception native --config config/phase5/live-trial.yaml`.

This separate trial configuration starts in Hold, uses only shoulder pan, preserves the other joints and holding torque, and limits its target to +/-2 degrees from fresh startup feedback. Maximum target velocity is 1 degree/second. This local window is a conservative supervised trial constraint, not completed workspace or payload approval. Existing Phase 0/4 physical acceptance records remain incomplete.

On the console, stand still as the only visible person. Reacquire user in Hold if needed. Select Calibrate direction (small jog), keep the physical cutoff reachable, and watch the arm. Calibration compares camera position with measured pan feedback; insufficient or inconsistent response leaves Follow inhibited. Successful calibration stays in Hold. Select Follow to begin live horizontal centering. Hold and Stop cancel the trial; heartbeat or tracking loss inhibits movement. Restart or geometry changes require new direction calibration. Vertical movement is not enabled.

The trial checks fresh native frames, valid single-user tracking, measured motor feedback, command acknowledgement, local bounds, and motor status. Forty-eight hardware-independent tests cover centering and the prior Phase 5 behavior. Physical cancellation latency and wider operating bounds are still unmeasured.

## Current continuous X/Y loop

Launch the live page with `./scripts/run_followme_tracking.sh`, then select Follow. Default `python -m followme.phase5 run --dry-run` remains simulation. Finish any tracker motor tests first: the follower serial port has one owner.

`tracking_loop: track_xy` imports the existing tracker module without modifying it. It reuses face/body detection, largest-then-nearest target selection (including the reference jump rule when present), aim point, smoothing, and annotated camera preview. The controller accumulates normalized X/Y image-error corrections with the reference gain/FOV/deadband rule, applying them to shoulder pan and wrist flex at its bounded update rate. Gripper and the remaining joints are not commanded.

Direction calibration is no longer a prerequisite in this mode. Initial signs match track.py; Reverse X/Y changes the corresponding sign and returns to Hold before Follow is selected again. A brief detection gap stops sending correction commands while staying in Follow/Searching, and tracking resumes on fresh detections. Camera failure, stale frames, operator Hold/Stop, heartbeat loss, and feedback faults still inhibit motion.

The current live configuration uses track.py's startup-relative travel bounds: +/-90 degrees pan and +/-25 degrees wrist flex. Each range is intersected with that motor's saved calibrated travel range before any motion is enabled; the degree-mode SDK does not automatically clamp to calibration endpoints. These replace the earlier +/-2 and +/-1 trial windows. Speed remains 1 degree/second, acceleration 2 degrees/second squared, and target lead over measured feedback is limited to 2 degrees. Restart starts in Hold with fresh origin measurements. This change does not certify payload clearance or physical acceptance over the wider range. Fifty-two hardware-independent tests pass, including calibrated range intersection and zero startup motor commands. For LAN access, launch `./scripts/run_followme_tracking.sh --host 0.0.0.0`.
