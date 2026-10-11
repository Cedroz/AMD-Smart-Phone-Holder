# Phase 4 implementation

Phase 4 runs independently through `python -m followme.phase4`. It adds no shared console entrypoint and does not modify Phase 0 evidence, Phase 1/2 configuration, Phase 3 code, camera servers, or `config/safe_poses.yaml`. It reuses Phase 3's controller and adapters through imports.

Every session has a separate draft library and append-only uniquely named evidence files. Approval is stored in a separate file in that session. Integration into production is a later explicit handoff, not a side effect of teaching.

## Start a session

```bash
source ./lerobot-env.sh
python -m followme.phase4 init
```

Use the printed directory as `PHASE4_SESSION` below. The snapshot includes the Phase 0 hardware-file hash and saved follower calibration hash. Changing the hardware record requires a new session/revalidation before live replay.

```bash
PHASE4_SESSION=/absolute/path/printed/by/init
python -m followme.phase4 --session "$PHASE4_SESSION" capture --pose station_a --read-motors
python -m followme.phase4 --session "$PHASE4_SESSION" capture --pose station_b --read-motors
```

Position the arm through a supported supervised teaching tool before each capture, and release its serial ownership before this read-only inspection. The capture command does not move the arm or enable torque. Do not force loaded unpowered joints. It reads 2.2 seconds of follower feedback, requires stable samples, and preserves torque on disconnect. It will refuse a busy serial adapter.

Alternatively import measurements with `capture --pose station_a --samples measurements.jsonl`. Each JSONL row must contain `timestamp_s` (strictly increasing monotonic seconds) and `positions` containing all five named arm joints in degrees plus `gripper` in percent. Imported samples are not physical approval evidence.

## Measured limits and route recording

Populate the session's `safe_poses.yaml` `joint_limits` from measured Phase 0 restrictions. Each of the five joints requires `[minimum_degrees, maximum_degrees]`. No physical limits are generated automatically. Keep the gripper target from capture and limits within the parent software ceilings.

Record measured follower feedback during supported supervised teaching through the existing motor owner. Export the same JSONL format; do not open a second serial connection to record a moving arm. This Phase 4 tool imports that recording rather than implementing another live leader/follower teleoperation loop.

```bash
python -m followme.phase4 --session "$PHASE4_SESSION" record \
  --source station_a --destination station_b --samples a-to-b.jsonl --indices 0,40,80
python -m followme.phase4 --session "$PHASE4_SESSION" record \
  --source station_b --destination station_a --samples b-to-a.jsonl --indices 0,40,80
python -m followme.phase4 --session "$PHASE4_SESSION" check
```

Indices are examples: use the actual first sample, operator-selected interior waypoints, and actual last sample. Each direction is independently recorded. Original measurements are copied into the session. Endpoints must agree with captured poses; interpolation uses explicit waypoints, analytic quintic derivative limits and sampled update checks. Generated replay geometry needs its own physical inspection.

## Simulation and supervised replay

```bash
python -m followme.phase4 --session "$PHASE4_SESSION" replay \
  --source station_a --destination station_b --dry-run
python -m followme.phase4 --session "$PHASE4_SESSION" cycles
python -m followme.phase4 --session "$PHASE4_SESSION" replay \
  --source station_a --destination station_b --interrupt hold --at 0.5
```

These operations use fake hardware, open no serial connection, and send zero physical commands. Synthetic starting positions are simulation inputs only; they are never written into physical approval evidence. Ten cycle logs are generated with both directional legs; the sequence stops on failure.

Live replay additionally requires all recorded Phase 0 physical gates passed, unchanged hardware/calibration fingerprints, stable gripper, known source pose, exclusive motor ownership, and an operator/observer:

```bash
python -m followme.phase4 --session "$PHASE4_SESSION" replay \
  --source station_a --destination station_b --live --confirm-supervised
```

The harness constructs an in-memory candidate library for the supervised controller only; it does not publish a draft as a production approval. It seeds disabled arm joints before enabling their torque, leaves the gripper untouched, and retains torque on disconnect. `--confirm-supervised` cannot bypass the physical evidence gate. Follow/tracking signals inside this harness are supplied by the supervising trial loop; this is not an autonomous perception runtime.

Use validated reduced-speed durations first, then restore/freeze intended timings and repeat acceptance on that unchanged digest. A changed duration invalidates previous trial provenance.

For each direction, test `--interrupt hold` and `--interrupt stop` with `--at 0.1`, `0.5`, and `0.8` for early/middle/late trials. The command measures software handling latency and logs hold confirmation. Ctrl+C requests Stop during live replay. After interruption, restore an endpoint under supervised control; the harness never resumes automatically. A communication failure still requires the established physical stop.

## Physical approval

```bash
python -m followme.phase4 --session "$PHASE4_SESSION" evidence-template
```

Fill the generated JSON template only after physical checks pass. It requires:

- Operator, date, exact content digest, and explicit acceptance attestations.
- Ten successful live cycle records with trial IDs 1–10, `log` paths relative to the repository and each file's `sha256`; each log must contain both live directional legs.
- Twelve unique live interruption records: each `direction`, `kind` (`hold`/`stop`), `position` (`early`/`middle`/`late`), `latency_s`, `log`, and `sha256`. Latencies must match the logs and be ≤250 ms.
- `physical_attachments` for `holds`, `clearance`, and `physical_stop`, each with `kind`, repository-relative `path`, and `sha256`. These are operator-collected video/photo/report evidence; they are not generated by simulation.

```bash
python -m followme.phase4 --session "$PHASE4_SESSION" cycles --live --confirm-supervised
python -m followme.phase4 --session "$PHASE4_SESSION" approve --evidence completed-acceptance.json
```

Approval checks current Phase 0 gates, unchanged setup/calibration, evidence existence/digests, live-only successful records, unique interruption trials, cycle identities, and recorded physical motion. Operator attestations and attachments establish physical observations; software cannot independently verify readability, mass, collision clearance, or the physical stop from JSON alone.

Editing a draft resets its approval state. Editing approved motion content fails its approval digest check. Approved files remain isolated in the session for Phase 5 handoff. No production library is overwritten.

## Checks

```bash
python -m unittest discover -s test -p 'test_phase4.py' -v
python -m unittest discover -s test -p 'test_phase3*.py' -v
```

Physical readiness is still pending in Phase 0. No live teaching/replay, loaded holds, physical stop tests, or physical acceptance cycles were performed during software implementation.
