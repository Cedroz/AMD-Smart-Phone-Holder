# Phase 0 — Hardware and Operating Envelope

Status: **In progress — device/calibration evidence collected; physical checks pending**

Execution record: [READINESS.md](READINESS.md). Measured inventory and pending physical fields: [hardware.yaml](hardware.yaml).

Parent specification: [FollowMe MVP plan](../PLAN.md).

## Objective and completion boundary

Establish that the SO-101 setup can support the intended payload at two useful viewing locations, identify both cameras, and demonstrate a practical physical stop and retained payload. Produce measured evidence for the later controller and pose-teaching work.

Completing this phase establishes hardware feasibility. It does **not** approve autonomous motion or an interpolated path between the two locations. Phase 4 records and validates those paths with the intended payload.

No hardware inspection, payload test, or stop test is considered complete merely because this document exists. Unknown measurements remain `pending`; failed checks remain `failed` until repeated successfully.

## Required outputs

Keep Phase 0 evidence in this directory:

| Artifact | Contents |
|---|---|
| `READINESS.md` | Completed step checklist, measured values, operator, date, failures, and final outcome |
| `hardware.yaml` | Verified device mapping, payload, candidate viewing locations, operating boundaries, and stop procedure |
| `evidence/` | Annotated setup photographs, camera screenshots, and short physical-test recordings |

These are evidence artifacts for later phases, not production runtime configuration. Create and populate them while executing this spec. Do not populate joint limits or pose coordinates with estimates presented as measurements.

Use outcome values `pending`, `passed`, `failed`, or `blocked` for each check. Name the supporting evidence file beside each completed check.

## Preconditions and operating rules

- One operator controls the hardware. A second person observes the physical stop during powered movement tests.
- Work on a stable surface with the arm sweep separated from people. Keep hands clear during powered movement; support the arm and payload before removing power or changing attachments.
- Inspect the physical stop before performing powered movement tests. Test with no payload first, then a retained surrogate, then the intended payload only after preceding checks pass.
- Existing calibration is retained. Do not change motor IDs, baud rates, calibration, or joint limits as part of device identification.
- Do not run the existing tracker in motion mode for these checks. `--test-pan` performs movement and is not a device-inspection command.
- Before any supervised powered movement, establish who can stop it, confirm the physical stopping procedure, and verify no second program owns the motor connection.
- If calibrated supervised movement cannot be performed with an existing supported tool, mark the powered checks blocked. Complete the software adapter in Phase 3, then return to these checks before approving Phase 4 motion.

## Individual steps

### 0.1 — Inventory the hardware and identify its current owners

1. Record arm model, controller board, motor/power-supply markings, mount type, phone holder, camera mounts, and all attached cables.
2. Record the follower and leader mapping below as the session-established mapping, then verify labels against the physical arms. Use stable paths rather than `ttyACM` numbering.
3. Inspect existing processes using the cameras and serial adapters. Coordinate their shutdown before any test requiring exclusive ownership; do not kill unrelated processes.
4. Record calibration file locations and whether each file exists. Their presence does not prove calibration accuracy.

Read-only discovery commands, run from the project root:

```bash
ls -l /dev/serial/by-id/
ls -l /dev/v4l/by-id/
lsusb
fuser -v /dev/video0 /dev/video2 /dev/ttyACM0 /dev/ttyACM1
ls -l /home/aup/.cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm.json
ls -l /home/aup/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/leader_arm.json
```

| Device | Expected stable identity | Verification |
|---|---|---|
| Follower | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128613-if00` | Physical label and previous follower unplug test |
| Leader | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128105-if00` | Physical label and remaining adapter |
| Camera candidate 1 | `/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0` | Currently resolves to `/dev/video0`; role pending |
| Camera candidate 2 | `/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0` | Currently resolves to `/dev/video2`; role pending |

**Pass:** device identities and process ownership are recorded without issuing movement commands.

### 0.2 — Inspect the base, power, and attachment hardware

1. With power removed and the arm supported, inspect the base fasteners, supporting surface, joints, controller mounting, phone holder, and camera attachment.
2. Check for loose hardware, damaged insulation/connectors, unsecured cables, and attachments that can slide into a joint.
3. Verify the power supply matches the actual motor/controller requirements using their markings and manufacturer documentation. Record the markings and reference; do not infer compatibility from connector fit.
4. Confirm the phone holder positively retains the payload in the proposed orientations. Add a secondary restraint or catch arrangement that cannot enter the arm sweep.
5. Photograph the base, payload retention, power markings, and cable routing.

**Pass:** no loose or damaged components; power compatibility is documented; payload retention and any secondary restraint are independent of software behavior.

### 0.3 — Measure and select the demonstration payload

1. Weigh the phone or surrogate, holder, wrist camera, and other attachments carried by the arm. Record individual and total masses in grams.
2. Record payload dimensions, mounting orientation, and approximate offset from the wrist mounting reference in millimeters. Include cable drag and protruding phone corners in the physical inspection.
3. Choose a retained surrogate with comparable mass and mounting leverage for initial tests. If comparable loading cannot be established, use a lighter demonstration surrogate and explicitly restrict the validation to that payload.
4. State which payload will be used in the demo. If the eventual phone differs, flag the need to repeat loaded tests and Phase 4 validation.

**Pass:** the tested payload is identified and measured; its loading is not generalized to a different phone or mounting arrangement.

### 0.4 — Assign camera roles using physical observations

1. View each camera independently through a camera-only preview. Use the existing preview only if its process and mode are confirmed; otherwise use a non-motor camera viewer.
2. Place an identifiable object in front of one lens at a time and match its image to its stable device path.
3. Assign `external` to the fixed workspace camera and `wrist` to the arm-mounted camera. Label the physical cameras and cables.
4. Position the external camera so both candidate workstations and their boundary are visible. Record camera height, orientation, mounting location, and representative screenshots.
5. Check whether the wrist camera can see the user's face while the phone faces the user at both candidate positions. Record any phone obstruction or unusable direction.
6. Mark wrist refinement as `eligible` or `disabled`. An unusable wrist view does not fail the basic two-workstation MVP if external tracking remains feasible.

**Pass:** both device roles are verified; the fixed camera sees both workstations. Wrist-camera limitations are documented.

### 0.5 — Mark the permitted demonstration workspace

1. Define `station_a` and `station_b` as two adjacent user standing/working locations. Mark them on the table or floor and label them in the external-camera screenshot.
2. Mark a separate robot operating region containing the full arm, phone, camera, and cable sweep. Keep user locations outside that region.
3. Record obstacles and excluded areas, including table edges, the controller, power cables, and other equipment.
4. Choose two candidate screen locations that appear readable from their corresponding user stations. Record positions and orientation descriptions using a photographed physical reference and explicit units.
5. Photograph an overview with workstation labels, proposed phone positions, boundaries, and physical-stop access.

**Pass:** the two locations are distinct and useful; operating boundaries and obstacles are visible and recorded. Camera detection is not treated as a substitute for physical separation.

### 0.6 — Establish and test the physical stop procedure

1. Identify the installed physical stop or accessible actuator-power interruption mechanism. Record its location, the power circuit it affects, and whether stopping it removes holding torque. If no suitable mechanism exists, stop powered testing and mark this gate blocked.
2. Document the procedure for stopping, retaining/supporting the load, resolving the cause, and restarting in Hold. Neither USB disconnection nor a browser button constitutes the physical stop test.
3. With no payload and the arm protected from an uncontrolled fall, test the stop while stationary. Verify what physically happens; do not assume torque remains enabled.
4. After that succeeds, repeat with the retained surrogate while stationary.
5. Only after stationary tests pass, perform a small supervised movement using an established calibrated tool and test stopping during movement. Use the observer, secondary retention, and clear operating region.
6. Record trigger-to-stop behavior on video, any residual movement or sag, whether power was removed, and payload retention. Mark any latency measurement method and uncertainty. Do not apply the later 250 ms software-cancellation criterion to this physical test.
7. Repeat the applicable tests with the intended demo payload before approving that payload.

**Pass:** the mechanism is accessible and works as observed, the payload remains retained, and stopping does not create an uncontrolled collapse into the user area. A failed or untested physical stop blocks powered feasibility approval.

### 0.7 — Check two loaded viewing positions and cable clearance

1. Start without a payload. Using an established supervised calibrated tool, approach each candidate position slowly within known hardware limits. Do not force unpowered joints through unknown limits or command guessed joint targets.
2. Check clearance for the entire assembly at intermediate positions, including the phone corners, wrist mount, and cables. Observe both travel directions; endpoint clearance alone is insufficient.
3. Repeat with the retained surrogate only after unloaded checks pass. Use the intended payload only after surrogate and stop tests pass.
4. Hold each loaded candidate position for 60 seconds. Record visible settling, drift, vibration, mount displacement, unusual sounds, and cable tension. Photograph a fixed reference before and after the hold.
5. Verify screen readability from the corresponding user station while the user remains outside the robot operating region.
6. Adjust placement or reduce the payload if the mount slips, the base moves, joints visibly continue to drift after settling, the assembly vibrates persistently, or cables become taut. Repeat affected checks.
7. Record any measured joint values with their units as **candidate observations**, not approved autonomous poses. Phase 4 must teach and validate the actual trajectories.

**Pass:** both locations are reachable and readable with the specified payload; the 60-second holds show no mount slip, base movement, persistent vibration, or continuing visible drift. Observed clearances and limitations are recorded.

### 0.8 — Record the operating envelope and restrictions

Populate `hardware.yaml` with these sections:

| Section | Required fields |
|---|---|
| `devices` | Arm model, stable serial paths, calibration IDs and file locations, verified camera roles and paths |
| `payload` | Identity, component/total masses, dimensions, mount reference and offset, retention arrangement |
| `workspace` | Reference frame, units, workstation labels, operating region, obstacles, excluded areas, evidence links |
| `candidate_locations` | A/B physical descriptions, measured coordinates if available, observed joints with units if recorded, hold results |
| `physical_stop` | Mechanism, location, affected circuit, observed torque behavior, test results, restart procedure, evidence |
| `restrictions` | Validated payload only, camera limitations, clearances, unresolved items, mandatory revalidation triggers |
| `assessment` | Operator, date, per-step outcomes, overall outcome, reason |

Keep gripper observations explicitly separate from degree-valued arm joints. Do not turn the parent plan's software speed ceilings into claims of measured mechanical limits. Record lower observed restrictions when needed; controller/path validation follows in later phases.

**Pass:** another operator can identify the setup, recreate the permitted arrangement, and understand which checks remain unvalidated from these artifacts alone.

### 0.9 — Review and hand off

1. Review the evidence against every acceptance item below.
2. Set the overall outcome to `passed` only when all required physical gates pass. Otherwise record `blocked` or `failed` and the specific next action.
3. Hand verified device paths and calibration IDs to Phase 1, camera roles and workstation images to Phase 2, stop/holding constraints to Phase 3, and payload/candidate-location evidence to Phase 4.
4. Preserve the distinction between hardware feasibility and autonomous path approval in the handoff.

## Final acceptance checklist

- [ ] Follower, leader, camera roles, and stable paths are verified.
- [ ] Base, mounts, power compatibility, and retention pass inspection.
- [ ] Demo payload mass, dimensions, mounting arrangement, and offsets are recorded.
- [ ] Both workstation regions are visible in the external camera.
- [ ] Two useful loaded screen positions pass supervised reachability and 60-second hold checks.
- [ ] Cables and protruding attachments clear the observed supervised sweep.
- [ ] Physical stop works in the tested stationary and moving conditions; payload remains retained.
- [ ] Stop/restart behavior, operating boundaries, exclusions, and evidence are recorded.
- [ ] Unmeasured limits and unapproved trajectories are clearly identified.
- [ ] No autonomous movement has been authorized by completing Phase 0.

## Revalidation and fallback

Repeat affected checks after changing the payload, holder, camera mount, cable routing, base location, power setup, operating boundaries, or physical stop mechanism. Mark previous evidence superseded rather than silently reusing it.

- If the intended phone cannot be supported, validate a lighter surrogate and label the demo accordingly.
- If wrist visibility is unusable, disable Phase 6 and retain fixed-camera workstation following.
- If either viewing location or the physical stop cannot pass, keep live robot movement blocked. Continue environment and perception work in camera-only/dry-run mode and report the physical MVP gate as unmet.
- Ten repeated A→B→A cycles, exact waypoint trajectories, velocity/acceleration validation, software cancellation latency, and automatic tracking-loss tests remain the responsibility of Phases 3–5 and 8.
