# Phase 2 operations

Phase 2 is independent: its modules are in `followme.phase2`, configuration in `config/phase2`, CLI in `followme-phase2`, and evidence in this directory. Phase 1's worker, environment config, locks and checks are unchanged; Phase 3's controller is never imported. Zone calibration writes only Phase 2 configuration and preserves the previous version beside the run log.

## Start

```bash
cd /home/aup/AMD-Smart-Phone-Holder
source ./lerobot-env.sh
followme-phase2 perceive
```

Default consoles: `http://169.233.72.8:8010/` (external) and `http://169.233.72.8:8011/` (wrist). The host and ports can be changed through `--host`, `--external-port`, and `--wrist-port`. Servers have no motor/mode/stop endpoints. Ctrl+C shuts down only this session's worker/capture processes and servers.

Native capture refuses busy cameras; it does not kill their existing owners. When the legacy previews on 8000/8001 are running, use this diagnostic mode instead:

```bash
followme-phase2 perceive --preview-sources
```

With `--preview-sources`, occupied devices are read through their existing HTTP previews and unowned devices are captured natively. The status reports the source mode for each role. Proxy mode timestamps delivery of the legacy feed, not sensor capture. Legacy overlays and inference can influence observations/throughput. It is useful for noninterfering diagnostics but does not prove native freshness or capture-rate acceptance. If legacy streams stop, the proxy reports camera failure and requires an explicit restart. Native capture uses one supervised process per device, with bounded latest-frame buffers; stalled processes are terminated.

## Calibrate workstation regions

```bash
followme-phase2 calibrate-zones
```

This starts the same camera-only console. Open the external console, choose **Draw station A**, drag its rectangle, and repeat for B. Leave a neutral gap large enough for the 0.03 exit margins. Save regions; overlapping, malformed or mismatched metadata is rejected. The screenshot/configuration are recorded in the session directory. No station bounds are supplied in advance.

Use one participant. Startup acquires a single face after 0.75 seconds. A sustained loss, ambiguity, or a large continuity jump requires **Reacquire participant**; it does not silently switch to another person. Face reacquisition is a session continuity heuristic, not biometric identification. Zone acceptance requires a further uninterrupted 0.75-second dwell.

Landmark overlays, raw/filtered centers, age, sequence, invalid reasons, status JSON, and measured rates are available in the console. Wrist inference is off by default, preserving the external inference budget. Both camera previews continue. Each stream uses `/stream`; stale streams return 503/close rather than repeatedly presenting an old JPEG as current.

## Benchmark and evaluate

```bash
followme-phase2 perceive --max-seconds 120
# If inference is too slow, run a separate session with lower inference resolution:
followme-phase2 perceive --max-seconds 120 --inference-width 320
```

The session directory is printed at startup. `observations.jsonl` contains derived face bounding-box extrema/state/config metadata; `metrics.json` is written at shutdown. Full landmarks remain available to overlays but are reduced to extrema for compact deterministic replay logs. Raw video recording is not performed. Requested camera FPS is distinct from measured throughput. Five-Hz state sampling reports valid-observation fraction; rate counters and bounded age samples report measured latency/drop counts.

Replay and evaluate a completed log:

```bash
followme-phase2 evaluate-perception --log artifacts/phase-2/evidence/<session>/observations.jsonl
followme-phase2 evaluate-perception --log artifacts/phase-2/evidence/<session>/observations.jsonl --annotations path/to/actual-trials.json --output path/to/results.json
```

Use `annotations.example.json` as a schema template only. Annotate core-region entry/exit independently using monotonic timestamps from the diagnostic state/log. Record ten alternating station changes, keeping each destination for at least three seconds, and two minutes of head turns at each station. The evaluator does not manufacture these trials from classifier outputs. Without annotations, live acceptance stays pending.

Fault trials: second participant enters view, selected participant disappears, each camera is disconnected separately, and the worker fails. Logs must show invalidation/reacquisition and wrist-only degradation. Physically disconnect cameras only when ready to perform those trials; unit tests do not establish the physical acceptance gate.

## Optional shoulder-based fallback

The independent worker supports Pose Landmarker when face-based zones fail the live head-turn trial. Its shoulder landmark indices/thresholded visibility follow the [official Pose Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker).

```bash
python scripts/prepare_phase2_pose.py
```

Then edit only `config/phase2/perception.yaml`:

```yaml
position_source: shoulders
pose_model_path: models/phase2/pose_landmarker_lite.task
pose_model_manifest: models/phase2/pose_manifest.json
```

The optional model is independently checksummed; Phase 1's face model and manifest remain intact. Reset Phase 2 zone status to `pending` with empty regions and matching `position_source`, then recalibrate for shoulder coordinates and repeat all live tests. Missing shoulder data inhibits zone classification; it never silently falls back to face coordinates. The face count still gates ambiguity. This heuristic is intended for one participant and does not establish cross-camera biometric association.

## Verification

```bash
python scripts/check_phase2.py
python -m unittest discover -s test -p 'test_phase2.py' -v
```

Neither smoke nor reducer tests open robot serial. The optional models and environments are ignored by Git; do not commit tokens or raw personal video. Phase 0 physical gates and later motion approval remain independent of camera-only results.
