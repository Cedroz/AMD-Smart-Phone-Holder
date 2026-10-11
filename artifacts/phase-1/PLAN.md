# Phase 1 — Reproducible Robotics and Vision Environment

Status: **Implemented — software environment and clean rebuild checks passed; physical gates remain pending**.

Execution record: [READINESS.md](READINESS.md).

Parent specification: [FollowMe MVP plan](../PLAN.md).
Prerequisite evidence: [Phase 0 readiness](../phase-0/READINESS.md) and [hardware inventory](../phase-0/hardware.yaml).

## Objective and completion boundary

Produce reproducible Python 3.12 runtime and vision environments, obtain the official Face Landmarker model, and verify that the main application can communicate with an isolated vision worker without opening a robot serial connection.

Preserve LeRobot 0.6.1, working calibration files, and the camera previews. Phase 0 physical gates remain pending; Phase 1 may pass its software checks without granting permission for live movement. Autonomous motion remains disabled until the later physical and trajectory gates pass.

Phase 1 implements environment/bootstrap tooling, configuration validation, and a worker launch/smoke-test contract. Face tracking, zone classification, motion control, and the product dashboard belong to subsequent phases.

## Current baseline

Observed in the project's `.venv` when this spec was drafted:

| Component | Installed metadata version / state |
|---|---|
| Python | 3.12 project environment |
| LeRobot | 0.6.1 |
| PyTorch / torchvision | 2.11.0 / 0.26.0; capture actual wheel/runtime build during implementation |
| NumPy | 2.2.6 |
| OpenCV runtime distribution | `opencv-python-headless` 4.13.0.92 |
| Feetech SDK | 1.0.0 |
| Datasets / PyAV / TorchCodec | 4.8.5 / 15.1.0 / 0.11.1 |
| PyYAML | 6.0.3 |
| MediaPipe, FastAPI, Uvicorn | Not installed |

`lerobot-env.sh` activates `.venv` and supplies `/home/aup/miniforge3/lib` for FFmpeg library discovery. That external dependency must be documented and included in rebuild verification. GPU training is outside scope; CPU perception must work.

## Required outputs

| Output | Purpose |
|---|---|
| `artifacts/phase-1/READINESS.md` | Step results, checks run, limitations, and handoff |
| `artifacts/phase-1/evidence/<timestamp>/` | Baseline inventories, dependency reports, calibration hashes, smoke-test results, and rebuild evidence |
| `pyproject.toml` | Installable `followme` package using a `src` layout; dependencies managed through the environment requirements |
| `requirements/runtime.in`, `requirements/vision.in` | Direct dependencies and compatibility constraints |
| `requirements/runtime.lock`, `requirements/vision.lock` | Exact, hashed transitive requirements for this Linux/Python 3.12 platform |
| `scripts/bootstrap_phase1.sh` | Repeatable staged installation and rebuild verification |
| `scripts/check_phase1.py` | Environment, model, configuration, and worker smoke checks |
| `config/environment.yaml` | Interpreter, model, FFmpeg, device, and disabled-motion configuration |
| `models/manifest.json` | Official model URL, digest, size, version identifier, and retrieval date |

Environment directories, package caches, raw tokens, and model binaries are ignored by Git. Commit requirements, the model manifest, setup instructions, and non-sensitive test evidence. Keep camera images in Phase 0; Phase 1 does not need additional images of people.

## Individual steps

### 1.1 — Snapshot the working baseline and define rollback

1. Read the Phase 0 hardware record. Confirm both calibration matches and the user-confirmed camera role mapping; retain the pending physical checks.
2. Record interpreter path/version, OS/architecture, installed package versions, `pip freeze --all`, and `pip check` using the project interpreter explicitly.
3. Record PyTorch runtime version/build, CUDA/HIP availability, OpenCV build information, FFmpeg version/library location, and the packages supplying those native libraries.
4. Hash and back up both calibration JSON files into the timestamped evidence directory. Keep their original locations and IDs unchanged.
5. Record camera preview process IDs and commands. Do not stop them merely to install dependencies.
6. Capture baseline requirements as constraints for staged runtime resolution. Never silently upgrade/downgrade an installed baseline dependency to accommodate a new package.

**Pass:** baseline and calibration hashes are saved; the existing `.venv` remains unchanged; a failed candidate installation can be discarded without affecting previews.

### 1.2 — Establish package and bootstrap structure

1. Add a minimal `src/followme` package and build metadata with Python `>=3.12,<3.13`. Avoid importing LeRobot, cameras, or MediaPipe from the package initializer.
2. Add runtime and vision dependency input files, the bootstrap script, and separate activation/launch instructions.
3. Create candidate `.venv-runtime-candidate` and `.venv-vision` environments using the discovered Python 3.12 interpreter. Use a disposable tooling environment for dependency locking.
4. The bootstrap script resolves paths from the repository root, uses explicit interpreters, fails on errors, and refuses to replace a populated working environment automatically.
5. Keep the existing `lerobot-env.sh` functional. Add native-library discovery checks and actionable errors; do not modify global shell startup files or the base Conda environment.

**Pass:** both candidate interpreters report Python 3.12; package import performs no hardware access; repeated bootstrap invocation verifies/reuses compatible environments.

### 1.3 — Resolve and lock the runtime environment

1. Pin `lerobot[feetech,dataset]==0.6.1` and preserve the observed core dependency versions through baseline constraints.
2. Add FastAPI, Uvicorn, and PyYAML to the runtime inputs. Resolve their exact compatible versions during execution, then lock the resulting complete graph with hashes.
3. Install only `opencv-python-headless` in the runtime. Reject installations containing another OpenCV distribution sharing the `cv2` namespace.
4. Install the local package without asking the package installer to resolve a second independent dependency graph.
5. Verify SDK imports through `lerobot.robots.so_follower` and `lerobot.teleoperators.so_leader`, matching this installed LeRobot release.
6. Keep any incompatible resolution as an explicit failed check. Do not migrate LeRobot, replace PyTorch builds, or repair compatibility by modifying motor firmware/calibration.

**Pass:** runtime candidate passes `pip check`, imports the SDK/API/configuration dependencies, and retains the baseline versions and a single OpenCV provider.

### 1.4 — Resolve and lock the isolated vision environment

1. Install MediaPipe with its declared dependencies, including its OpenCV distribution, only in `.venv-vision`. Pin the resolved versions in the hashed vision lock.
2. Do not install LeRobot or the runtime requirements into this environment. Verify neither environment inherits the other through `PYTHONPATH`, system-site packages, or mixed activation.
3. Use the same small local `followme` package in both environments; install it without dependencies. Worker imports must stay within vision/stdlib modules.
4. Verify imports of MediaPipe Tasks, OpenCV, and NumPy under the vision interpreter, and verify exactly one OpenCV provider there.
5. Save the resolved MediaPipe version and wheel compatibility in the evidence. The lock applies to the recorded OS/architecture/Python platform, not every machine.

**Pass:** both environments independently pass dependency checks; the runtime has no MediaPipe/contrib OpenCV contamination; vision has no LeRobot dependency.

### 1.5 — Obtain and verify the official model

1. Select the Face Landmarker `.task` bundle linked by the official [MediaPipe Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python).
2. Resolve a versioned official download URL when available. Download to a temporary file, verify successful response/nonempty content, calculate SHA-256 and size, and atomically place it in `models/face_landmarker.task`.
3. Record the source URL, upstream version if provided, observed SHA-256, byte count, and retrieval date in `models/manifest.json`. An observed digest supports repeatability; do not label it an upstream signature.
4. On subsequent bootstrap runs, accept a cached model only when its hash matches the manifest. Fail on a mismatch; never silently replace the expected digest.
5. Load the model with the vision interpreter in `VIDEO` mode, CPU delegate, up to two faces, and transformation-matrix output enabled. Run inference on a synthetic blank RGB image with increasing timestamps. Zero faces is an expected valid result.

**Pass:** manifest/hash checks and model construction/inference succeed. This proves model operability, not human detection quality.

### 1.6 — Add configuration and hardware-free preflight checks

1. Store the absolute vision interpreter path or resolve a repository-relative path to it at launch. Verify its identity before starting the worker.
2. Configure model path/digest, FFmpeg library location, and stable camera paths from Phase 0. Keep camera roles `external` and `wrist`; retain `follower_arm`/`leader_arm` calibration IDs and known stable serial paths.
3. Set defaults to `dry_run: true`, `mode: hold`, and `autonomous_motion_approved: false`. The incomplete physical gate cannot be overridden by an environment variable or dependency check.
4. Keep LAN host/address configurable. Reuse the existing preview links; do not bind a new smoke-test server to ports 8000/8001.
5. The default preflight checks filesystem/device existence and calibration hashes without opening cameras or serial adapters. Report busy devices and changed mappings explicitly.

**Pass:** configuration validates, known device paths resolve, calibration hashes remain unchanged, and default preflight opens no robot connection.

### 1.7 — Implement the vision-worker smoke interface

1. Launch the worker with the configured vision interpreter through `subprocess`, using a socket pair for local IPC and sending diagnostic output to stderr. Do not launch a network vision service.
2. Use newline-delimited JSON envelopes, with fields `type`, `request_id`, `camera_role`, `sequence`, `capture_monotonic_ns`, and `jpeg_base64` for frame requests. Use protocol version `1` in the startup handshake.
3. Startup response reports protocol version, interpreter, MediaPipe version, and model digest. Reject mismatches before accepting frames.
4. A frame response echoes identity/timestamp fields and returns `face_count`, normalized landmark results, optional transformation matrices, and an explicit error when inference fails. Do not invent per-face confidence scores.
5. Permit one outstanding frame request. The caller retains only the newest unsent frame per camera. Reject oversized messages above 4 MiB and malformed inputs.
6. Use a 30-second model-start deadline and five-second smoke-inference deadline. Production observation freshness remains 500 ms in subsequent phases; these startup/test deadlines do not relax it.
7. On worker failure or timeout, close IPC, terminate/reap the child, and report failure. Do not automatically enable Follow or open serial as recovery behavior.

**Pass:** the runtime completes handshake and synthetic-JPEG inference with the isolated worker; request timestamps survive round trips; worker failures cannot initiate hardware I/O.

### 1.8 — Verify rebuilds and promote the runtime

1. Recreate both environments in disposable directories from their locks; install the package without dependency resolution.
2. Repeat dependency, import, FFmpeg/TorchCodec, model-hash, and worker smoke checks. Record native-library dependencies separately: pip locks alone do not reproduce Miniforge's FFmpeg shared libraries.
3. Test missing interpreter/model, corrupt model, invalid JSON/JPEG, worker exit, startup timeout, and inference timeout. Assert each produces a clear failure with no serial access.
4. Test default preflight with serial/camera constructors patched to fail if invoked. Assert both calibration hashes remain unchanged.
5. Install the same locked additions into the existing runtime `.venv` only after candidate/rebuild checks pass and its baseline still matches. Preserve the environment path used by running previews; validate it again after installation.
6. Verify both preview URLs still deliver frames. If inspection would need exclusive camera access, use the previews to check streaming rather than opening the cameras again.

**Pass:** both clean rebuilds and the preserved project runtime pass checks; previews are functioning; calibration is unchanged; no motion command was issued.

### 1.9 — Record readiness and hand off

1. Write `READINESS.md` with per-step outcomes (`pending`, `passed`, `failed`, `blocked`), interpreter paths, lock/model digests, native-library assumptions, evidence links, and exact bootstrap/check commands.
2. Document launching from a fresh terminal without relying on an already activated base/robot environment.
3. Hand the vision launch contract, official model, stable external-camera mapping, and lock files to Phase 2. Hand the runtime SDK verification and retained calibration identities to Phase 3.
4. Record Phase 0 physical gates separately. A software-environment pass does not change their outcomes or authorize powered movement.

## Acceptance checklist

- [ ] Baseline and calibration evidence saved; both original calibration files unchanged.
- [ ] Python 3.12 runtime and isolated vision environments identified and independently verified.
- [ ] LeRobot 0.6.1 and working runtime dependencies preserved.
- [ ] Exact hashed locks reproduce both environments on the recorded platform.
- [ ] One OpenCV provider per environment; no mixed dependency graph.
- [ ] FFmpeg/TorchCodec native-library checks pass and rebuild prerequisites are documented.
- [ ] Official model source, size, digest, cached verification, and CPU inference pass.
- [ ] Worker handshake/inference/failure tests pass without hardware access.
- [ ] Stable device paths and Hold/dry-run defaults validate.
- [ ] Existing camera preview links remain functional.
- [ ] Readiness and Phase 2/3 handoff are recorded; physical gates remain accurately pending.

## Failure and rollback policy

Candidate environment failure leaves `.venv`, previews, and calibration untouched. Do not bypass dependency conflicts with forced installs into the working runtime. Correct the candidate inputs, regenerate locks, and repeat failed checks.

If network/model download fails, reuse only a verified cached bundle; otherwise mark model setup blocked. If native FFmpeg libraries are unavailable, document the missing dependency and fail video-decoding verification rather than claiming pip installation is sufficient.

If final runtime installation fails, restore the baseline package versions using the recorded baseline requirements and repeat smoke checks. Preserve failed-run evidence. Camera/serial absence is a device-readiness failure, not a reason to trigger recalibration or motor setup.
