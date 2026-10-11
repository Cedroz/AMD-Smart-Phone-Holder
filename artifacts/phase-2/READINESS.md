# Phase 2 readiness

Outcome: **software implemented and tested; live workstation acceptance deferred at the user's request**.

Executed October 10, 2026. Neither this implementation nor its camera-only diagnostics authorize robot movement. Existing arm calibrations remain unchanged.

## Implementation and isolation

- Independent package: `src/followme/phase2`; independent configuration: `config/phase2`; independent command: `followme-phase2`.
- Phase 1 environment, worker/client source, dependency locks, and smoke checks are unchanged. Phase 2 subclasses the read-only launch/IPC interface and uses its own inference worker. Phase 3 controller implementations are not imported or modified.
- The only shared packaging additions are the new CLI entrypoint and its HTML package data. The local editable package registration was refreshed without dependency resolution.
- Default Phase 2 ports are **8010/8011**, avoiding other phases' original **8000/8001** previews. Native acquisition refuses cameras already owned by another process. No unrelated owner is stopped.
- `--preview-sources` reads occupied cameras through their existing local previews and captures unowned cameras natively. Proxy timestamps measure delivery and cannot pass native freshness acceptance.
- Camera capture runs in supervised processes with latest-frame buffers and stall termination. Inference runs in the isolated vision environment; a separate monitor expires tracking while inference is pending.
- Face tracking includes strict freshness, single-participant acquisition, ambiguity/continuity gating, explicit reacquisition, EMA, measured-zone hysteresis/dwell, and immutable snapshots.
- Console includes live overlays, diagnostic JSON, region drawing/saving, and reacquisition. Calibration changes only Phase 2 zone configuration and retains prior versions in evidence.
- JSONL replay and independent trial evaluation implemented. Compact logs retain face bounding-box extrema required for identical reducer results; overlays retain full landmarks. Raw video is not automatically recorded.
- Optional verified Pose Landmarker support is implemented and smoke-tested; the live head-turn fallback gate has not been exercised.

## Measured validation

| Check | Outcome | Evidence |
|---|---|---|
| Phase 2 deterministic tracking/configuration tests | passed | [Unit test output](evidence/unit-tests.txt) |
| Camera-only API, origin controls, stale streams and invalid calibration | passed | Same test output |
| Capture stall termination and deterministic replay | passed | Same test output |
| Real isolated face-worker handshake and synthetic inference | passed | [Worker smoke](evidence/worker-smoke.json) |
| Optional pose-worker/model synthetic smoke | passed | [Pose smoke](evidence/pose-worker-smoke.json) |
| Dual native capture and inference, two minutes | completed with throughput gate unmet | [Metrics](evidence/native-benchmark/metrics.json) |
| Replay of recorded native observations | passed, no mismatches | [Native replay](evidence/native-replay.json) |
| Station calibration, ten live transitions, stationary/head-turn trials | deferred by user | No fabricated station bounds or annotations |
| Physical camera-disconnect/worker-fault demonstration | pending | Unit fault behavior does not replace physical trials |

The native benchmark recorded 2,272 external frames and 1,201 external inferences over 121.19 seconds including startup. External capture averaged **18.75 FPS**, inference **9.91 Hz**, and p95 acquisition-to-result age **75.70 ms**. Wrist capture averaged **10.12 FPS**; wrist inference was intentionally off. No worker fault occurred. The strict ≥10 Hz aggregate inference threshold was not met; requested 30 FPS is not a sustained capture result.

An attempted lower-resolution native benchmark correctly refused a static camera that another implementation had reacquired. That process was retained. The running independent diagnostics use 320-pixel inference on the static camera's existing preview and native wrist capture. Their proxy rate/latency is diagnostic only; it does not close the native throughput gate. Workstation visibility/identity acceptance was not established by the ambient room benchmark; valid-observation fraction was zero.

## Step outcomes

| Step | Status |
|---|---|
| 2.1 Handoff / ownership | passed; known busy devices refused, proxy alternative available |
| 2.2 Latest-frame capture | implemented and benchmarked; lower actual camera rates recorded |
| 2.3 Inference / freshness | implemented and unit/smoke tested; strict live throughput target remains unmet |
| 2.4 Participant continuity | implemented/tested; live acceptance deferred |
| 2.5 EMA / temporal validity | passed deterministic tests |
| 2.6 Workstation calibration | tooling implemented; physical calibration deferred |
| 2.7 Console / telemetry / replay | implemented and live HTTP/JPEG verified |
| 2.8 Live acceptance | partial benchmark only; participant trials deferred |
| 2.9 Fallback / handoff | optional model/worker prepared; live fallback evaluation deferred |

## Current consoles and next steps

- External console: http://169.233.72.8:8010/
- Wrist console: http://169.233.72.8:8011/
- Original static camera preview remains on port 8000 under its existing owner.
- The Phase 2 session remains camera-only, started with `--preview-sources --inference-width 320`; it is a running terminal session, not an automatically restarted service.

Detailed commands, calibration, annotation schema, rollback, and shoulder fallback are in [OPERATIONS.md](OPERATIONS.md). Phase 2 zone regions remain empty with status `pending`. The user said **“Skip for now”** for workstation calibration/participant acceptance; leave those gates deferred until resumed.

When ready: give native camera ownership to Phase 2 explicitly, repeat the lower-resolution benchmark, calibrate the two measured regions, and record the participant trials with independent timestamps. Only then decide whether the shoulder fallback is necessary. Phase 5 must not treat proxy delivery times or uncalibrated zones as validated workstation observations.
