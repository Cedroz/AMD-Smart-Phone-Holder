# AMD-Smart-Phone-Holder

FollowMe uses a LeRobot SO-101 follower and two cameras to select viewing positions
for a phone. Hardware feasibility and movement validation are tracked in
[the phase specifications](artifacts/PLAN.md).

## Phase 1 environment setup

The runtime uses Python 3.12 and LeRobot 0.6.1 with `opencv-python-headless`.
MediaPipe runs in a separate Python 3.12 environment with `opencv-contrib-python`.
The environments must not be merged.

From a new Bash terminal on this machine:

```bash
cd /home/aup/AMD-Smart-Phone-Holder
source ./lerobot-env.sh
followme inspect
python scripts/check_phase1.py --output artifacts/phase-1/evidence/manual-check.json
```

`followme inspect` checks files, configuration, device presence/ownership, and
calibration hashes. It never opens a camera or serial port. Phase 1 checks use
synthetic JPEGs; zero detected faces is expected.

Bootstrap installs into managed candidate/vision environments and checks them
before an explicit promotion into the working runtime:

```bash
bash scripts/bootstrap_phase1.sh
bash scripts/bootstrap_phase1.sh --verify-rebuild --promote
```

Existing unmanaged directories are refused. Remove or rename only disposable
candidate environments if recreating them is needed; preserve the working
`.venv`. The bootstrap uses the working pip to seed environments because this
machine's system Python does not include `ensurepip`.

For a machine without the existing runtime, first create Python 3.12 with pip,
then use that interpreter to run `scripts/bootstrap_phase1.py`. Requirements
locks target Linux x86_64 and Python 3.12. Installing the locks requires the
corresponding binary wheels and sufficient disk space; it does not install OS
libraries or recreate the physical hardware.

TorchCodec requires FFmpeg shared libraries. On this machine these come from
`/home/aup/miniforge3/lib`, set in `config/environment.yaml` and the activation
script. Baseline evidence includes FFmpeg version and an explicit Conda package
inventory for native dependency reconstruction. On another machine, supply
compatible FFmpeg 8 shared libraries and update that path before checking.

The official Face Landmarker bundle is recorded in `models/manifest.json`.
Bootstrap downloads it when absent and verifies its recorded SHA-256 and size;
a corrupt cached model fails rather than silently changing the manifest.
The model is ignored by Git. The source and digest are documented, not claimed
to be a publisher signature.

To regenerate locks deliberately, first resolve the input requirements in the
candidate environments using Python 3.12 and verify `pip check`. Then use an
isolated Python 3.12 tooling environment with `packaging` installed:

```bash
.venv-tools/bin/python scripts/lock_phase1.py runtime .venv-runtime-candidate/bin/python
.venv-tools/bin/python scripts/lock_phase1.py vision .venv-vision/bin/python
```

The generator selects a compatible wheel per package from PyPI release metadata
and records its published SHA-256, URL, and filename in the artifact manifests.
For source-only packages it locks the source archive. The bootstrap seeds pip,
setuptools, wheel, and packaging from separate hashed build locks, then installs
the full locks without build isolation. This avoids unconstrained build tools
for the source-only Feetech SDK. The locks are platform-specific rather than a
collection of hashes for every architecture.

Keep `requirements/baseline.txt` unchanged unless an explicit runtime migration
is intended. Rebuilding and checking both environments is required after a lock
change. The local package installs with `--no-deps --no-build-isolation`; the lock
files supply its environment dependencies. A clean checkout needs the local
package installed in each environment before launching the worker with `-I`.

## Verification and remaining hardware gates

```bash
python -m unittest discover -s test -p 'test_phase*.py' -v
```

The vision supervisor validates interpreter, protocol version, MediaPipe version,
and model hash at startup. It uses a socket pair, bounded JSON messages, per-camera
latest buffers, one outstanding inference request, and reaps failed workers.
Frames echo their monotonic capture timestamps; no per-face confidence is invented.

Phase 1 starts in dry-run/Hold with motion disabled. Calibration register matches
were verified in Phase 0 and are preserved. Physical stop, payload, workspace,
and loaded-position checks remain pending; environment readiness does not pass
those checks or authorize robot movement.
