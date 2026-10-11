# Phase 5 implementation readiness

Outcome: **passed — independent software/fake-hardware acceptance; live acceptance blocked by upstream physical/perception gates**.

Recorded October 10, 2026, America/Los_Angeles. No physical serial connection, arming, motor movement, calibration change, environment installation, production library publication, or camera service replacement was performed.

## Delivered

- Independent `followme.phase5` package, standalone module CLI, and `config/phase5/following.yaml`.
- Original capture-timestamp bridge, generation/sequence checks, validity/ambiguity gating, 0.75-second continuous dwell, neutral-zone reset, arrival-based two-second cooldown, and one outstanding transition.
- Existing Phase 3 worker/controller reused as the sole motor I/O owner. Perception/UI/planning never send SDK joint commands.
- Explicit Follow, three-second active-operator lease, priority Hold/Stop, latched controller faults, anchor-checked reset, and no automatic resumption.
- Independent token/origin-protected console and HTTP/WebSocket interface on default loopback port 8020. Default synthetic perception/fake backend consumes no camera or serial device.
- Optional in-process Phase 2 perception integration with evidence directed into this phase; no zone saving or upstream modifications. Native ownership failures never terminate existing services; proxy delivery timestamps cannot qualify live following.
- Live approval verifier rechecks Phase 4 content, trial/attachment hashes, setup/calibration, and native workstation evidence. Draft/projected approval flags do not bypass it.
- Canonical controlled-time replay, JSONL integration/controller evidence, supervised startup/shutdown, and operational documentation.

## Acceptance evidence

| Check | Result |
|---|---|
| Phase 5 automated tests | **35 passed**, including early/middle/late Hold/Stop subcases |
| Existing Phase 2–4 tests | **75 passed**, unchanged |
| Synthetic station changes | **10/10 correct**, exactly ten accepted requests |
| Physical commands | **Zero** |
| Motor I/O owners | **One** in synthetic and threaded trials |
| Final real-thread interruption trials | **100 passed**, alternating Hold/Stop |
| Healthy fake-I/O delay | Random fixed per-trial delay between 0 and 2 ms; seed 5101 |
| Maximum recorded final cancellation handling latency | **0.050242 s**, approximately 50.24 ms; target 0.250 s |
| HTTP smoke | Console and status served on temporary loopback port 18020; startup Hold/fake, zero physical commands; process stopped afterward |
| Live preflight | Correctly reports `live_approval_paths_missing`; opens no hardware |

The timing trials establish software handling with the recorded fake I/O conditions. They do not establish loaded braking, hardware communication latency, native perception quality, or payload retention. API telemetry does not renew the operator lease. Tests verify no automatic recovery from invalid tracking, changed user generation, rejected requests, or independent controller inhibition.

Evidence:

- [Phase 5 tests](evidence/unit-tests.txt)
- [Upstream regression tests](evidence/upstream-tests.txt)
- [Final synthetic acceptance](evidence/final-synthetic-acceptance/results.json)
- [Final threaded interruption trials](evidence/final-cancellation-trials.json)
- [HTTP console/status smoke](evidence/http-smoke/http-result.json)
- [Hardware-free preflight](evidence/preflight.json)
- [Upstream isolation verification](evidence/isolation.json)

An earlier interruption run is retained separately in `evidence/cancellation-trials.json`. Final results above use the final named run. Earlier replay logs are retained; the replay tool now refuses to mix run evidence in an existing output directory.

## Independence verification

Hash comparison covered 119 existing upstream files. Upstream implementation and configuration files were unchanged. The running Phase 2 diagnostic process continued appending its own `active-diagnostics/observations.jsonl`; that dynamic service log is identified separately from code/config changes. Phase 5 did not write to it.

No shared CLI, package metadata/entrypoints, locks, Phase 0–4 artifacts/configuration, or Phase 6 implementation was edited. Tests and tooling require no environment reinstall because the source package is already editable. Existing listening services remained running; Phase 5's temporary HTTP smoke did not use their ports.

## Step outcomes

| Step | Outcome | Notes |
|---|---|---|
| 5.1 Configuration/prerequisites | passed for software | Conservative defaults; live gate fails with a named blocker |
| 5.2 Observation bridge | passed for software | Preserved timestamps, duplicate/generation checks, malformed input and proxy rejection |
| 5.3 Station selection | passed for software | Dwell, neutral reset, same-pose suppression, arrival cooldown |
| 5.4 Single-owner integration | passed for software | Ten transitions, no backlog, one owner, no physical commands |
| 5.5 Interruption/recovery | passed for software | Hold/Stop, expiry, unknown-anchor restoration, explicit Follow/reset |
| 5.6 Console/telemetry | passed for software | HTTP/WebSocket controls, token/origin/lease tests, independent port |
| 5.7 Automated acceptance | passed | 35 new tests, 75 upstream tests, 100 final threaded trials |
| 5.8 Live acceptance | blocked | Phase 0 physical, Phase 2 native workstation, and Phase 4 loaded-path acceptance still required |
| 5.9 Handoff | passed for software | [Operations and approval contracts](README.md); physical results explicitly pending |

## Remaining physical acceptance

Camera-panel addition: the console now shows static and wrist previews, reuses owned perception overlays when available, and falls back to existing diagnostic/camera-only feeds in synthetic mode. Two additional tests verify source selection, MJPEG forwarding, stale/missing responses, and no heartbeat renewal by preview viewing; the current Phase 5 suite has **37 passing tests**. Both feeds were verified in [camera-preview-smoke.json](evidence/camera-preview-smoke.json). The stopped static camera-only preview was restored on port 8000 after verifying its camera was unowned. The running Phase 2 service was unchanged. The Phase 5 server was restarted in dry-run/Hold to load the page update.

Before live operation, populate this phase's approval-path configuration with real, unchanged Phase 4 evidence and a Phase 2 native acceptance record matching the documented schema. Native zone/camera acceptance, ten live station changes, two stationary minutes, and supervised camera/target/browser loss tests remain unperformed.

Wrist alignment remains disabled. Live operation cannot be enabled by the console, default configuration, fake test evidence, or an unapproved projected pose library. Changing the tested setup or upstream zone configuration inhibits operation and requires revalidation.
