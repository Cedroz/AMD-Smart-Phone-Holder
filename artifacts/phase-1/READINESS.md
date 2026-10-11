# Phase 1 Readiness Record

Overall outcome: **passed — software environment ready**.

Executed October 10, 2026, on this Linux x86_64 machine with Python 3.12. Phase 0 physical checks remain pending; this result authorizes no robot motion.

## Results by step

| Step | Outcome | Evidence / result |
|---|---|---|
| 1.1 Baseline / rollback | passed | [Baseline inventory](evidence/20261010T153404-0700/baseline.json), [package snapshot](evidence/20261010T153404-0700/baseline-freeze.txt), [calibration hashes](evidence/20261010T153404-0700/calibration-hashes.json); both calibration JSON backups retained beside them |
| 1.2 Package / bootstrap | passed | Minimal installed `followme` package; explicit-interpreter bootstrap handles missing system `ensurepip`; unmanaged environment directories are refused |
| 1.3 Runtime | passed | LeRobot 0.6.1 baseline retained, FastAPI 0.143.0 and Uvicorn 0.54.0 added; 97 locked external packages; only headless OpenCV |
| 1.4 Vision isolation | passed | MediaPipe 1.1.0, contrib OpenCV 5.0.0.93, NumPy 2.5.4 in `.venv-vision`; 22 locked external packages; no LeRobot |
| 1.5 Model | passed | Official version-1 bundle downloaded; recorded SHA-256/size verified; CPU VIDEO inference succeeds |
| 1.6 Configuration / preflight | passed | Stable paths, original calibration IDs, Hold/dry-run defaults; filesystem/ownership checks open no hardware |
| 1.7 Worker interface | passed | Protocol-v1 socket-pair handshake, two independent VIDEO trackers, timestamp-preserving inference, bounded messages, latest buffers, child cleanup |
| 1.8 Rebuild / promotion | passed | Both clean temporary environments rebuilt from hashed locks and smoke-tested; working `.venv` then promoted and checked; both previews verified |
| 1.9 Handoff | passed | Commands, environment assumptions, limitations, and next-phase interfaces recorded below |

Complete check reports:

- [Candidate runtime and vision](evidence/20261010T154155.895394-0700/candidate.json)
- [Clean runtime and vision rebuild](evidence/20261010T154155.895394-0700/rebuild.json)
- [Working runtime after promotion](evidence/20261010T154155.895394-0700/working-runtime.json)
- [Final hardware-free preflight](evidence/final-preflight.json)
- [Final runtime and worker smoke check](evidence/final-smoke.json)
- [Unit tests](evidence/unit-tests.txt)
- [Live preview checks](evidence/previews.json)

Rebuild paths in the historical report were temporary and have been removed after successful verification.

## Environment and model identity

- Runtime: project `.venv/bin/python`, Python 3.12; activation through `lerobot-env.sh`.
- Vision: project `.venv-vision/bin/python`, Python 3.12. The supervisor uses `-I` and removes inherited `PYTHONPATH`, `PYTHONHOME`, `VIRTUAL_ENV`, and `LD_LIBRARY_PATH`.
- Candidate runtime remains in `.venv-runtime-candidate` for inspection. It is separate from the working runtime.
- Model: `models/face_landmarker.task`, 3,758,596 bytes.
- Model SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
- Official source: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`.
- Digest is locally recorded from the official download, not a publisher signature. Bootstrap verifies it without silently replacing the expected digest.

Dependency artifact filenames, source URLs, and PyPI SHA-256 digests are committed in `requirements/runtime-artifacts.json` and `requirements/vision-artifacts.json`. Lock file digests are recorded in [release hashes](evidence/release-hashes.json).

## Reproducibility detail

The initial pip-tools hashing attempt fetched foreign-platform wheels unnecessarily. The implemented generator instead locks the pip-checked resolved candidate environments to artifacts compatible with this platform and obtains their hashes from PyPI release metadata. Bootstrap verified the selected archives through pip's `--require-hashes` mode during clean rebuilds.

Source-only Feetech SDK installation uses the hashed source archive. Separate hashed build-tool locks seed pip, setuptools, wheel, and packaging before full installation with build isolation disabled. The local package is installed editable with no dependency resolution.

Native FFmpeg libraries remain outside pip environments. The runtime uses `/home/aup/miniforge3/lib`; FFmpeg and the explicit native Conda inventory are in the baseline evidence directory. This is a documented native prerequisite, not a claim that pip recreates the OS. Vision does not inherit the runtime library path. GPU acceleration is unavailable and unnecessary for these CPU smoke checks.

## Fresh terminal commands

```bash
cd /home/aup/AMD-Smart-Phone-Holder
source ./lerobot-env.sh
followme inspect
python scripts/check_phase1.py --output artifacts/phase-1/evidence/manual-check.json
python -m unittest discover -s test -p 'test_phase*.py' -v
```

Repeat locked candidate checks and clean rebuilds:

```bash
bash scripts/bootstrap_phase1.sh --verify-rebuild
```

Promotion is explicit through `--promote`, checks the preserved baseline, and does not replace the working environment directory. It adds only locked packages and the local project. Future bootstrap runs create new timestamped reports; `latest-run.txt` points to the newest run.

## Phase 2 and 3 handoff

Phase 2 can use `VisionClient` from `followme.vision_client`, passing repository root and `config/environment.yaml`. Start validates the worker handshake; `submit_latest` replaces pending frames per role; `infer_latest` or `infer` exchanges one outstanding request. Model calls use capture timestamps and return explicit errors. A caller must treat invalid, stale, ambiguous, or failed results as motion-inhibiting conditions when movement is introduced.

External camera maps to `/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0`; wrist maps to `/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0`. Current camera-only previews use ports 8000 and 8001 respectively and occupy their devices. Phase 2 must take camera ownership deliberately rather than opening competing captures.

Phase 3 can import the SO-101 aliases from `lerobot.robots.so_follower` and `lerobot.teleoperators.so_leader`. Calibration identities and stable serial paths remain unchanged. SDK import success does not establish physical stop, payload capability, or approved trajectories.

No camera frame capture or serial connection is performed by the Phase 1 preflight/smoke checker. Synthetic frames yielded zero faces as expected; human tracking quality, sustained throughput, and production 500 ms freshness enforcement belong to subsequent phases.
